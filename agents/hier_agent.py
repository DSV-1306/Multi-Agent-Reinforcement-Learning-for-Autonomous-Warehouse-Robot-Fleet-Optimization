"""
Hierarchical method (Phase 4): an RL policy decides TASK ALLOCATION (which pending order an idle robot
takes); the existing Space-Time A* controller does all MOVEMENT.

Decision points are exactly where the baseline's nearest-order rule runs (env reset, start of each
step, after each delivery; see WarehouseEnv._auto_dispatch_orders with dispatch_mode="external"). For
every idle robot, in robot-id order, while orders are pending, the policy picks one of up to K candidate
orders (the `candidates_nearest` closest to the robot + the `candidates_oldest` oldest, de-duplicated),
or "defer" (stay idle until the next decision point) if allowed.

Candidate features (all scaled to about [0, 1]):
  0 Manhattan distance robot -> pick cell / (W + H)
  1 shortest-path distance pick cell -> the order's station / (W + H)
  2 order age / max_steps
  3 fraction of OTHER idle robots that are closer to this order than this robot
  4 robots within 2 cells of the pick cell / num_robots
  5 1 if this is the candidate nearest to the robot
  6 1 if this is the oldest pending order
Context features: pending orders / queue_norm (clipped at 1), idle robots / num_robots,
                  step / max_steps, orders in the system / queue_norm (clipped at 1).

Learning signal, hier.reward_mode:
  "fleet": r_t = delivery_reward * deliveries_t - queue_cost * orders_in_system_t (Little's law: the summed
      queue term is total order latency). Decisions are irregular in time, so this is a semi-MDP: decision j
      collects R_j = sum_{s in [t_j, t_{j+1})} gamma^(s - t_j) r_s and bootstraps with gamma^(t_{j+1}-t_j) V_{j+1}.
      Dev finding: one decision's fleet return is dominated by other robots' events, the advantage is
      mostly noise and the policy did not move (approx_kl ~ 1e-5).
  "order_time": contextual bandit. Each assignment gets R_j = -(steps until ITS order is delivered) / max_steps
      (censored at the episode end), no bootstrapping. Low variance, and it rewards what greedy-nearest ignores
      (the shelf -> station leg, congestion). "defer" is disabled in this mode (it would earn 0 for free).
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from agents.models import layer_init, MASKED_LOGIT
from agents.ippo_agent import policy_loss_fn
from baselines.heuristic_agent import HeuristicFleetController
from environment.warehouse_env import WarehouseEnv
from training.vec_env import episode_stats

CAND_FEATURES = 7
CTX_FEATURES = 4
NEAR_RADIUS = 2


class AllocatorNet(nn.Module):
    """Scores each candidate with a shared MLP (permutation-equivariant); a context head scores "defer"."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.scorer = nn.Sequential(
            layer_init(nn.Linear(CAND_FEATURES + CTX_FEATURES, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=0.01))
        self.defer = nn.Sequential(
            layer_init(nn.Linear(CTX_FEATURES, hidden)), nn.Tanh(), layer_init(nn.Linear(hidden, 1), std=0.01))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(CTX_FEATURES + 2 * CAND_FEATURES, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0))

    def forward(self, cand: torch.Tensor, ctx: torch.Tensor, mask: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        """cand (B, K, F), ctx (B, C), mask (B, K+1) bool -> distribution over K+1 actions, value (B,)."""
        K = cand.shape[1]
        x = torch.cat([cand, ctx.unsqueeze(1).expand(-1, K, -1)], dim=-1)
        logits = torch.cat([self.scorer(x).squeeze(-1), self.defer(ctx)], dim=-1)
        logits = logits.masked_fill(~mask, MASKED_LOGIT)
        valid = mask[:, :K].unsqueeze(-1).float()
        mean = (cand * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        mx = cand.masked_fill(valid == 0, 0.0).max(1).values
        value = self.critic(torch.cat([ctx, mean, mx], dim=-1)).squeeze(-1)
        return Categorical(logits=logits, validate_args=False), value


class _DistanceCache:
    """Shortest-path distances from each station, computed once per layout."""

    def __init__(self):
        self._cache: Dict[Tuple, Dict] = {}

    def to_station(self, env: WarehouseEnv, cell, station) -> float:
        key = (id(env.grid), station)
        if key not in self._cache:
            self._cache[key] = env.grid.bfs_distances(station)
        return self._cache[key].get(cell, env.grid.width + env.grid.height)


def candidates_and_features(env: WarehouseEnv, robot, hcfg: Dict[str, Any], dist_cache: _DistanceCache):
    """Returns (candidate pending-order indices, cand features (K, F), context (C,), mask (K+1,))."""
    K = hcfg["candidates_nearest"] + hcfg["candidates_oldest"]
    pend = env.order_generator.pending_orders
    span = float(env.grid.width + env.grid.height)
    d_robot = [env.grid.manhattan_distance(robot.pos, o.shelf_access_pos) for o in pend]
    by_dist = sorted(range(len(pend)), key=lambda i: (d_robot[i], i))
    by_age = sorted(range(len(pend)), key=lambda i: (pend[i].spawn_step, i))
    chosen: List[int] = []
    for i in by_dist[:hcfg["candidates_nearest"]] + by_age[:hcfg["candidates_oldest"]]:
        if i not in chosen:
            chosen.append(i)

    idle = [r for r in env.robots if r.is_idle and r.id != robot.id]
    oldest = by_age[0] if by_age else -1
    feats = np.zeros((K, CAND_FEATURES), dtype=np.float32)
    for k, i in enumerate(chosen):
        o = pend[i]
        feats[k, 0] = d_robot[i] / span
        feats[k, 1] = dist_cache.to_station(env, o.shelf_access_pos, o.station_pos) / span
        feats[k, 2] = (env.current_step - o.spawn_step) / env.max_steps
        feats[k, 3] = (sum(env.grid.manhattan_distance(r.pos, o.shelf_access_pos) < d_robot[i] for r in idle)
                       / max(len(idle), 1))
        feats[k, 4] = sum(env.grid.manhattan_distance(r.pos, o.shelf_access_pos) <= NEAR_RADIUS
                          for r in env.robots) / env.num_robots
        feats[k, 5] = float(i == by_dist[0])
        feats[k, 6] = float(i == oldest)
    qn = hcfg["queue_norm"]
    in_system = len(pend) + len(env.order_generator.active_orders)
    ctx = np.array([min(len(pend) / qn, 1.0), (len(idle) + 1) / env.num_robots,
                    env.current_step / env.max_steps, min(in_system / qn, 1.0)], dtype=np.float32)
    mask = np.zeros(K + 1, dtype=bool)
    mask[:len(chosen)] = True
    mask[K] = hcfg["allow_defer"] and hcfg.get("reward_mode", "fleet") == "fleet"
    return chosen, feats, ctx, mask


class RLDispatcher:
    """env.dispatcher implementation. Records (features, action, logprob, value, time) when recording."""

    def __init__(self, net: AllocatorNet, hcfg: Dict[str, Any], deterministic: bool = False):
        self.net, self.hcfg, self.deterministic = net, hcfg, deterministic
        self.dist_cache = _DistanceCache()
        self.record: Optional[List[Dict[str, Any]]] = None

    @torch.no_grad()
    def __call__(self, env: WarehouseEnv):
        for robot in env.robots:
            if not robot.is_idle or not env.order_generator.pending_orders:
                continue
            chosen, cand, ctx, mask = candidates_and_features(env, robot, self.hcfg, self.dist_cache)
            dist, value = self.net(torch.as_tensor(cand)[None], torch.as_tensor(ctx)[None], torch.as_tensor(mask)[None])
            a = dist.probs.argmax(-1) if self.deterministic else dist.sample()
            k = int(a)
            order_id = env.order_generator.pending_orders[chosen[k]].id if k < len(chosen) else None
            if self.record is not None:
                self.record.append({"cand": cand, "ctx": ctx, "mask": mask, "action": k, "order_id": order_id,
                                    "logprob": float(dist.log_prob(a)), "value": float(value), "t": env.current_step})
            if k < len(chosen):
                env.assign_order(robot.id, chosen[k])


class GreedyNearestDispatcher:
    """Re-implements the built-in rule (each idle robot takes its nearest pending order) via the hook."""

    def __call__(self, env: WarehouseEnv):
        for robot in env.robots:
            pend = env.order_generator.pending_orders
            if not robot.is_idle or not pend:
                continue
            best = min(range(len(pend)), key=lambda i: env.grid.manhattan_distance(robot.pos, pend[i].shelf_access_pos))
            env.assign_order(robot.id, best)


def allocation_reward(env: WarehouseEnv, deliveries: int, hcfg: Dict[str, Any]) -> float:
    in_system = len(env.order_generator.pending_orders) + len(env.order_generator.active_orders)
    return hcfg["reward"]["delivery"] * deliveries - hcfg["reward"]["queue_cost"] * in_system


def make_hier_env(env_kwargs: Dict[str, Any], dispatcher) -> Tuple[WarehouseEnv, HeuristicFleetController]:
    env = WarehouseEnv(**{**env_kwargs, "dispatch_mode": "external"})
    env.dispatcher = dispatcher
    return env, HeuristicFleetController(env)


def run_episode(env: WarehouseEnv, ctrl: HeuristicFleetController, seed: Optional[int],
                on_step=None) -> Dict[str, float]:
    """Runs one episode with A* routing; on_step(env, deliveries) is called after every env step."""
    env.reset(seed=seed)
    ctrl.reset()
    ep_return = 0.0
    while True:
        before = env.total_orders_completed
        _, rewards, term, trunc, _ = env.step(ctrl.get_actions())
        ep_return += float(np.mean([rewards[a] for a in env.possible_agents]))
        if on_step is not None:
            on_step(env, env.total_orders_completed - before)
        if all(trunc.values()) or all(term.values()):
            return episode_stats(env, ep_return)


class HierAllocator:
    """Network + its config, trained with PPO on semi-MDP allocation decisions."""

    def __init__(self, hcfg: Dict[str, Any], gamma: float, gae_lambda: float, clip_eps: float,
                 entropy_coef: float, value_coef: float, max_grad_norm: float):
        self.hcfg, self.gamma, self.gae_lambda = hcfg, gamma, gae_lambda
        self.clip_eps, self.entropy_coef = clip_eps, entropy_coef
        self.value_coef, self.max_grad_norm = value_coef, max_grad_norm
        self.net = AllocatorNet(hcfg["hidden_dim"])
        self.optimizer = optim.Adam(self.net.parameters(), lr=hcfg["lr"], eps=1e-5)
        # Actor and critic gradients are clipped SEPARATELY. With one shared clip, the large critic gradient
        # (unnormalised semi-MDP returns) used the whole norm budget and the policy never moved (approx_kl = 0
        # in the first Phase 5 run), as in MAPPO the two parts are kept apart.
        self.actor_params = list(self.net.scorer.parameters()) + list(self.net.defer.parameters())
        self.critic_params = list(self.net.critic.parameters())
        self.base_lr = hcfg["lr"]

    def set_lr_fraction(self, frac: float):
        for g in self.optimizer.param_groups:
            g["lr"] = self.base_lr * frac

    def advantages(self, trans: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Semi-MDP GAE over a list of closed transitions (each has R, disc, done, value, next_value)."""
        n = len(trans)
        adv = np.zeros(n, dtype=np.float64)
        last = 0.0
        for j in reversed(range(n)):
            tr = trans[j]
            nonterm = 0.0 if tr["done"] else 1.0
            delta = tr["R"] + tr["disc"] * tr["next_value"] * nonterm - tr["value"]
            last = delta + tr["disc"] * self.gae_lambda * nonterm * last
            adv[j] = last
        returns = adv + np.array([t["value"] for t in trans])
        return adv, returns

    def update(self, trans: List[Dict[str, Any]], num_epochs: int, batch_size: int) -> Dict[str, float]:
        adv, ret = self.advantages(trans)
        cand = torch.as_tensor(np.stack([t["cand"] for t in trans]))
        ctx = torch.as_tensor(np.stack([t["ctx"] for t in trans]))
        mask = torch.as_tensor(np.stack([t["mask"] for t in trans]))
        act = torch.as_tensor([t["action"] for t in trans])
        old_lp = torch.as_tensor([t["logprob"] for t in trans], dtype=torch.float32)
        adv_t = torch.as_tensor(adv, dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.as_tensor(ret, dtype=torch.float32)
        logs = {k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac")}
        for _ in range(num_epochs):
            perm = np.random.permutation(len(trans))
            for s in range(0, len(trans), batch_size):
                idx = perm[s:s + batch_size]
                dist, value = self.net(cand[idx], ctx[idx], mask[idx])
                pl, kl, cf = policy_loss_fn(dist.log_prob(act[idx]), old_lp[idx], adv_t[idx], self.clip_eps)
                vl = 0.5 * ((value - ret_t[idx]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pl + self.value_coef * vl - self.entropy_coef * ent
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm)
                self.optimizer.step()
                for k, v in zip(logs, (pl, vl, ent, kl, cf)):
                    logs[k].append(v.item())
        out = {k: float(np.mean(v)) for k, v in logs.items()}
        out["explained_variance"] = float(1 - np.var(ret - np.array([t["value"] for t in trans])) / max(np.var(ret), 1e-12))
        return out

    def save(self, path: str, extra: Optional[Dict[str, Any]] = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"net": self.net.state_dict(), "hcfg": self.hcfg, "extra": extra or {}}, path)


def load_allocator(path: str) -> RLDispatcher:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = AllocatorNet(ckpt["hcfg"]["hidden_dim"])
    net.load_state_dict(ckpt["net"])
    return RLDispatcher(net, ckpt["hcfg"])


def run_hier_episodes(dispatcher: RLDispatcher, env_kwargs: Dict[str, Any], seeds: List[int],
                      torch_seed: int) -> List[Dict[str, float]]:
    env, ctrl = make_hier_env(env_kwargs, dispatcher)
    episodes = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(torch_seed)
        for seed in seeds:
            episodes.append({"seed": seed, **run_episode(env, ctrl, seed)})
    return episodes
