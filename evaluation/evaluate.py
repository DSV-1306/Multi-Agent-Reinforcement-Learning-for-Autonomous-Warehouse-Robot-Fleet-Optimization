"""
Comprehensive Head-to-Head Benchmarking Suite:
Compares Heuristic Baseline vs IPPO vs MAPPO across controlled benchmark seeds.

Usage (Phase 1):
    python evaluation/evaluate.py --split dev --ippo results/runs/<run>/best_dev.pt --mappo ...
Settings and seeds come from the YAML; policy sampling is seeded (evaluation.torch_seed).
evaluate_baseline / evaluate_ippo / evaluate_mappo below are the original (pre-Phase-1)
routines, kept so evaluation/phase0_audit.py can reproduce the committed numbers.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.warehouse_env import WarehouseEnv
from baselines.heuristic_agent import HeuristicFleetController
from agents.ippo_agent import IPPOAgent
from agents.mappo_agent import MAPPOAgent
from evaluation.metrics import MetricsTracker


def evaluate_baseline(env: WarehouseEnv, seeds: list) -> MetricsTracker:
    tracker = MetricsTracker()
    controller = HeuristicFleetController(env)

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        controller.reset()
        ep_return = 0.0

        for step in range(env.max_steps):
            actions = controller.get_actions()
            obs, rewards, term, trunc, _ = env.step(actions)
            ep_return += float(np.mean([rewards[a] for a in env.possible_agents]))
            if all(trunc.values()):
                break

        tracker.record_episode(
            orders=env.order_generator.completed_orders,
            robots=env.robots,
            fleet_collisions=env.total_fleet_collisions,
            orders_completed=env.total_orders_completed,
            total_steps=env.current_step,
            mean_return=ep_return,
        )

    return tracker


def evaluate_ippo(env: WarehouseEnv, agent: IPPOAgent, seeds: list) -> MetricsTracker:
    tracker = MetricsTracker()
    device = agent.device

    for seed in seeds:
        obs_dict, _ = env.reset(seed=seed)
        ep_return = 0.0

        for step in range(env.max_steps):
            obs_array = np.stack([obs_dict[a] for a in env.possible_agents])
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=device)

            with torch.no_grad():
                actions, _, _ = agent.get_actions_and_values(obs_tensor)

            action_dict = {f"robot_{i}": int(actions[i]) for i in range(env.num_robots)}
            obs_dict, rewards, term, trunc, _ = env.step(action_dict)
            ep_return += float(np.mean([rewards[a] for a in env.possible_agents]))

            if all(trunc.values()):
                break

        tracker.record_episode(
            orders=env.order_generator.completed_orders,
            robots=env.robots,
            fleet_collisions=env.total_fleet_collisions,
            orders_completed=env.total_orders_completed,
            total_steps=env.current_step,
            mean_return=ep_return,
        )

    return tracker


def evaluate_mappo(env: WarehouseEnv, agent: MAPPOAgent, seeds: list) -> MetricsTracker:
    tracker = MetricsTracker()
    device = agent.device

    for seed in seeds:
        obs_dict, _ = env.reset(seed=seed)
        ep_return = 0.0

        for step in range(env.max_steps):
            obs_array = np.stack([obs_dict[a] for a in env.possible_agents])
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=device)
            state_tensor = torch.tensor(env.state(), dtype=torch.float32, device=device)

            with torch.no_grad():
                actions, _, _ = agent.get_actions_and_values(obs_tensor, state_tensor)

            action_dict = {f"robot_{i}": int(actions[i]) for i in range(env.num_robots)}
            obs_dict, rewards, term, trunc, _ = env.step(action_dict)
            ep_return += float(np.mean([rewards[a] for a in env.possible_agents]))

            if all(trunc.values()):
                break

        tracker.record_episode(
            orders=env.order_generator.completed_orders,
            robots=env.robots,
            fleet_collisions=env.total_fleet_collisions,
            orders_completed=env.total_orders_completed,
            total_steps=env.current_step,
            mean_return=ep_return,
        )

    return tracker




# ---------------------------------------------------------------------------
# Phase 1: config-driven, seeded, vectorised evaluation
# ---------------------------------------------------------------------------

from typing import Any, Dict, List, Optional

from training.config_utils import load_config, env_kwargs_from_cfg
from training.normalization import ObsNormalizer
from training.vec_env import SyncVectorWarehouseEnv, episode_stats
from evaluation.metrics import summarize_episodes

METHODS = ("baseline", "ippo", "mappo")


class PolicyRunner:
    """Wraps a trained agent with the normalizers and masking setting it was trained with."""

    def __init__(self, agent, method: str, obs_norm: Optional[ObsNormalizer] = None,
                 state_norm: Optional[ObsNormalizer] = None, use_mask: bool = False,
                 deterministic: bool = False, state_max_robots: Optional[int] = None):
        """state_max_robots: MAPPO global-state padding the critic was built for (None = use env default)."""
        self.agent, self.method = agent, method
        self.state_max_robots = state_max_robots
        self.obs_norm, self.state_norm = obs_norm, state_norm
        self.use_mask, self.deterministic = use_mask, deterministic

    def act(self, obs: np.ndarray, states: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """obs (E, N, D), states (E, S), masks (E, N, A) -> actions (E, N)."""
        o = self.obs_norm(obs, update=False) if self.obs_norm else obs
        o_t = torch.as_tensor(o, dtype=torch.float32)
        m_t = torch.as_tensor(masks) if self.use_mask else None
        if self.method == "mappo":
            s = self.state_norm(states, update=False) if self.state_norm else states
            actions, _, _ = self.agent.get_actions_and_values(
                o_t, torch.as_tensor(s, dtype=torch.float32), action_mask=m_t, deterministic=self.deterministic)
        else:
            actions, _, _ = self.agent.get_actions_and_values(
                o_t, action_mask=m_t, deterministic=self.deterministic)
        return actions.numpy()


def load_policy(method: str, path: str, obs_dim: int, state_dim: int, action_dim: int = 6,
                deterministic: bool = False) -> PolicyRunner:
    """Builds a PolicyRunner from a checkpoint. Pre-Phase-1 checkpoints load without normalizers/masks."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    extra0 = ckpt.get("extra", {})
    if method == "mappo":
        # Critic input size fixes the state padding: (in_features [- obs_dim]) = max_robots * obs_dim + 10
        in_features = ckpt["critic"]["network.0.weight"].shape[1]
        critic_input = ckpt.get("critic_input", "global")
        state_dim = in_features - (obs_dim if critic_input == "agent_specific" else 0)
    state_max_robots = extra0.get("state_max_robots", (state_dim - 10) // obs_dim if method == "mappo" else None)
    if method == "ippo":
        agent = IPPOAgent(obs_dim=obs_dim, action_dim=action_dim)
    elif method == "mappo":
        agent = MAPPOAgent(obs_dim=obs_dim, state_dim=state_dim, action_dim=action_dim,
                           critic_input=MAPPOAgent.checkpoint_critic_input(path))
    else:
        raise ValueError(f"Unknown RL method {method!r}")
    extra = agent.load(path, load_optimizer=False)
    obs_norm = ObsNormalizer.from_state_dict(extra["obs_norm"]) if "obs_norm" in extra else None
    state_norm = ObsNormalizer.from_state_dict(extra["state_norm"]) if "state_norm" in extra else None
    return PolicyRunner(agent, method, obs_norm, state_norm,
                        use_mask=bool(extra.get("action_masking", False)), deterministic=deterministic,
                        state_max_robots=state_max_robots)


def run_policy_episodes(runner: PolicyRunner, env_kwargs: Dict[str, Any], seeds: List[int],
                        torch_seed: int) -> List[Dict[str, float]]:
    """One episode per seed, all seeds in parallel. The global torch RNG is restored afterwards."""
    if runner.method == "mappo" and runner.state_max_robots is not None:
        if env_kwargs["num_robots"] > runner.state_max_robots:
            raise ValueError(f"MAPPO critic supports at most {runner.state_max_robots} robots")
        env_kwargs = {**env_kwargs, "state_max_robots": runner.state_max_robots}
    venv = SyncVectorWarehouseEnv(env_kwargs, num_envs=len(seeds))
    obs, masks, states = venv.reset(seeds=list(seeds))
    episodes: List[Optional[Dict[str, float]]] = [None] * len(seeds)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(torch_seed)
        while any(ep is None for ep in episodes):
            actions = runner.act(obs, states, masks)
            obs, _, dones, masks, states, infos = venv.step(actions)
            for e in np.flatnonzero(dones):
                if episodes[e] is None:
                    episodes[e] = {"seed": seeds[e], **infos[e]["episode"]}
    return episodes


def run_baseline_episodes(env_kwargs: Dict[str, Any], seeds: List[int]) -> List[Dict[str, float]]:
    env = WarehouseEnv(**env_kwargs)
    controller = HeuristicFleetController(env)
    episodes = []
    for seed in seeds:
        env.reset(seed=seed)
        controller.reset()
        ep_return = 0.0
        while True:
            _, rewards, term, trunc, _ = env.step(controller.get_actions())
            ep_return += float(np.mean([rewards[a] for a in env.possible_agents]))
            if all(trunc.values()) or all(term.values()):
                break
        episodes.append({"seed": seed, **episode_stats(env, ep_return)})
    return episodes


def split_seeds(cfg: Dict[str, Any], split: str) -> List[int]:
    if split == "dev":
        return list(cfg["seeds"]["dev_eval"])
    if split == "test":
        return list(cfg["evaluation"]["seeds"])
    raise ValueError(f"split must be 'dev' or 'test', got {split!r}")


def main():
    parser = argparse.ArgumentParser(description="Seeded, config-driven evaluation.")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--set", action="append", default=[], help="Config override key.sub=value")
    parser.add_argument("--split", default="dev", choices=["dev", "test"],
                        help="dev = tuning seeds; test = held-out seeds (use only for final results)")
    parser.add_argument("--methods", default="baseline,ippo,mappo")
    parser.add_argument("--ippo", default=None, help="IPPO checkpoint path")
    parser.add_argument("--mappo", default=None, help="MAPPO checkpoint path")
    parser.add_argument("--episodes", type=int, default=None, help="Use only the first N seeds of the split")
    parser.add_argument("--output", default=None, help="Summary CSV (per-episode CSV is written next to it)")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    eval_cfg = cfg["evaluation"]
    seeds = split_seeds(cfg, args.split)[: args.episodes]
    env_kwargs = env_kwargs_from_cfg(cfg["environment"])
    probe = WarehouseEnv(**env_kwargs)
    paths = {"ippo": args.ippo, "mappo": args.mappo}

    rows = []
    for method in args.methods.split(","):
        if method == "baseline":
            eps = run_baseline_episodes(env_kwargs, seeds)
        else:
            if not paths[method] or not os.path.exists(paths[method]):
                print(f"(!) No checkpoint for {method} (pass --{method} PATH), skipping.")
                continue
            runner = load_policy(method, paths[method], probe.obs_dim, probe.state_dim,
                                 deterministic=eval_cfg["deterministic"])
            eps = run_policy_episodes(runner, env_kwargs, seeds, eval_cfg["torch_seed"])
        for ep in eps:
            rows.append({"method": method, **ep})
        print(f"--> {method}: {summarize_episodes(eps)['throughput_mean']:.2f} orders/episode")

    per_ep = pd.DataFrame(rows)
    summary = pd.DataFrame([
        {"method": m, "split": args.split, "lambda": env_kwargs["order_lambda"],
         "layout": env_kwargs["layout_type"], **summarize_episodes(g.to_dict("records"))}
        for m, g in per_ep.groupby("method", sort=False)
    ])
    output = args.output or os.path.join(eval_cfg["output_dir"], f"eval_{args.split}.csv")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    summary.to_csv(output, index=False)
    per_ep.to_csv(output.replace(".csv", "_ep.csv"), index=False)  # short suffix: Windows MAX_PATH
    cols = ["method", "n", "throughput_mean", "throughput_std", "collisions_mean",
            "robot_collisions_mean", "wall_bumps_mean", "mean_latency_mean", "return_mean"]
    print(summary[cols].to_string(index=False))
    print(f"--> Saved {output}")


if __name__ == "__main__":
    main()
