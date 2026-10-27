"""
Phase 0 audit: reproduce the committed evaluation, measure simulator speed on this CPU,
and collect evidence for (or against) each suspected design problem.

Read-only with respect to the project: it never trains or overwrites checkpoints and
writes only into audit.output_dir (default results/phase0/).

Usage:
    python evaluation/phase0_audit.py [--config configs/default_config.yaml] [--skip-profile]
"""

import os
import sys
import json
import time
import argparse
import platform
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.warehouse_env import WarehouseEnv
from environment.robot import RobotStatus
from baselines.heuristic_agent import HeuristicFleetController
from agents.ippo_agent import IPPOAgent
from agents.mappo_agent import MAPPOAgent
from training.buffer import MultiAgentRolloutBuffer
from evaluation.evaluate import evaluate_baseline, evaluate_ippo, evaluate_mappo

# Same mapping as WarehouseEnv.step (0=N, 1=S, 2=E, 3=W, 4=Wait, 5=Interact)
ACTION_DELTAS = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0), 4: (0, 0), 5: (0, 0)}
ACTION_NAMES = ["N", "S", "E", "W", "Wait", "Interact"]
NUM_ACTIONS = len(ACTION_NAMES)


def make_env(env_cfg: dict, **overrides) -> WarehouseEnv:
    kwargs = dict(
        num_robots=env_cfg["num_robots"],
        grid_width=env_cfg["grid_width"],
        grid_height=env_cfg["grid_height"],
        layout_type=env_cfg["layout_type"],
        order_lambda=env_cfg["order_lambda"],
        max_steps=env_cfg["max_steps"],
    )
    kwargs.update(overrides)
    return WarehouseEnv(**kwargs)


# ---------------------------------------------------------------------------
# Policies wrapped behind one interface: policy(env, obs_dict) -> action dict
# ---------------------------------------------------------------------------

class RandomPolicy:
    def __init__(self, env: WarehouseEnv, seed: int):
        self.env = env
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def __call__(self, obs_dict):
        return {a: int(self.rng.integers(NUM_ACTIONS)) for a in self.env.possible_agents}


class BaselinePolicy:
    def __init__(self, env: WarehouseEnv):
        self.env = env
        self.ctrl = HeuristicFleetController(env)

    def reset(self):
        self.ctrl.reset()

    def __call__(self, obs_dict):
        return self.ctrl.get_actions()


class RLPolicy:
    def __init__(self, env: WarehouseEnv, agent, centralized: bool):
        self.env = env
        self.agent = agent
        self.centralized = centralized

    def reset(self):
        pass

    def __call__(self, obs_dict):
        obs = torch.tensor(
            np.stack([obs_dict[a] for a in self.env.possible_agents]), dtype=torch.float32
        )
        if self.centralized:
            state = torch.tensor(self.env.state(), dtype=torch.float32)
            actions, _, _ = self.agent.get_actions_and_values(obs, state)
        else:
            actions, _, _ = self.agent.get_actions_and_values(obs)
        return {f"robot_{i}": int(actions[i]) for i in range(self.env.num_robots)}


def load_rl_agents(env: WarehouseEnv, audit_cfg: dict) -> Dict[str, object]:
    agents = {}
    if os.path.exists(audit_cfg["ippo_checkpoint"]):
        ippo = IPPOAgent(obs_dim=env.obs_dim, action_dim=NUM_ACTIONS)
        ippo.load(audit_cfg["ippo_checkpoint"])
        agents["ippo"] = ippo
    if os.path.exists(audit_cfg["mappo_checkpoint"]):
        mappo = MAPPOAgent(obs_dim=env.obs_dim, state_dim=env.state_dim, action_dim=NUM_ACTIONS)
        mappo.load(audit_cfg["mappo_checkpoint"])
        agents["mappo"] = mappo
    return agents


# ---------------------------------------------------------------------------
# 1. Reproduce committed evaluation
# ---------------------------------------------------------------------------

def reproduce_eval(audit_cfg: dict, rl_agents: dict) -> pd.DataFrame:
    """Re-runs evaluate.py's exact protocol, with seeded policy sampling."""
    seeds = [100 + i for i in range(audit_cfg["repro_num_episodes"])]  # as in evaluate.py
    env = WarehouseEnv(
        num_robots=5, grid_width=10, grid_height=10,
        order_lambda=audit_cfg["repro_order_lambda"], max_steps=audit_cfg["repro_max_steps"],
    )
    rows = []
    torch.manual_seed(audit_cfg["repro_torch_seed"])
    rows.append({"Method": "Baseline (A*)", **evaluate_baseline(env, seeds).summary()})
    if "ippo" in rl_agents:
        torch.manual_seed(audit_cfg["repro_torch_seed"])
        rows.append({"Method": "IPPO", **evaluate_ippo(env, rl_agents["ippo"], seeds).summary()})
    if "mappo" in rl_agents:
        torch.manual_seed(audit_cfg["repro_torch_seed"])
        rows.append({"Method": "MAPPO", **evaluate_mappo(env, rl_agents["mappo"], seeds).summary()})
    cols = ["Method", "avg_throughput", "std_throughput", "avg_collisions", "std_collisions",
            "avg_completion_time", "avg_travel_distance", "avg_utilization", "avg_return"]
    return pd.DataFrame(rows)[cols]


# ---------------------------------------------------------------------------
# 2. Speed profiling
# ---------------------------------------------------------------------------

def time_rollout(env: WarehouseEnv, policy, n_steps: int, seed: int) -> float:
    """Env steps per second for a given policy, including policy inference."""
    obs, _ = env.reset(seed=seed)
    policy.reset()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        obs, _, _, trunc, _ = env.step(policy(obs))
        if all(trunc.values()):
            obs, _ = env.reset()
            policy.reset()
    return n_steps / (time.perf_counter() - t0)


def time_ppo_update(env: WarehouseEnv, kind: str, cfg: dict, n_updates: int, seed: int) -> float:
    """Seconds per PPO update (collection excluded) at the configured buffer/epochs/batch size."""
    hp, algo = cfg["hyperparameters"], cfg[kind]
    if kind == "ippo":
        agent = IPPOAgent(obs_dim=env.obs_dim, action_dim=NUM_ACTIONS, lr=algo["lr"])
    else:
        agent = MAPPOAgent(obs_dim=env.obs_dim, state_dim=env.state_dim, action_dim=NUM_ACTIONS,
                           lr_actor=algo["lr_actor"], lr_critic=algo["lr_critic"])
    buf = MultiAgentRolloutBuffer(algo["buffer_size"], env.num_robots, env.obs_dim, env.state_dim,
                                  hp["gamma"], hp["gae_lambda"])
    gen = torch.Generator().manual_seed(seed)
    n = env.num_robots
    for _ in range(algo["buffer_size"]):
        buf.insert(torch.rand(n, env.obs_dim, generator=gen), torch.rand(env.state_dim, generator=gen),
                   torch.randint(0, NUM_ACTIONS, (n,), generator=gen), torch.zeros(n) - 1.79,
                   torch.randn(n, generator=gen), torch.zeros(n), torch.zeros(n))
    buf.compute_returns_and_advantages(torch.zeros(n), torch.zeros(n))
    t0 = time.perf_counter()
    for _ in range(n_updates):
        agent.train_step(buf, num_epochs=algo["num_epochs"], batch_size=algo["batch_size"])
    return (time.perf_counter() - t0) / n_updates


def profile(cfg: dict, audit_cfg: dict) -> dict:
    env_cfg = cfg["environment"]
    n_steps, seed = audit_cfg["profile_env_steps"], audit_cfg["profile_seed"]
    out = {"layouts": {}}
    for layout in audit_cfg["profile_layouts"]:
        env = make_env(env_cfg, layout_type=layout, seed=seed)
        torch.manual_seed(seed)
        ippo = IPPOAgent(obs_dim=env.obs_dim, action_dim=NUM_ACTIONS)
        mappo = MAPPOAgent(obs_dim=env.obs_dim, state_dim=env.state_dim, action_dim=NUM_ACTIONS)
        out["layouts"][layout] = {
            "sps_random_env_only": time_rollout(env, RandomPolicy(env, seed), n_steps, seed),
            "sps_ippo_collect": time_rollout(env, RLPolicy(env, ippo, False), n_steps, seed),
            "sps_mappo_collect": time_rollout(env, RLPolicy(env, mappo, True), n_steps, seed),
            "sps_baseline": time_rollout(env, BaselinePolicy(env), n_steps, seed),
        }
    env = make_env(env_cfg, seed=seed)
    for kind in ("ippo", "mappo"):
        out[f"{kind}_sec_per_update"] = time_ppo_update(env, kind, cfg, audit_cfg["profile_ppo_updates"], seed)

    # Wall-clock estimate per budget (default layout): collection time + update time
    lay = out["layouts"][env_cfg["layout_type"]]
    out["estimates_hours"] = {}
    for kind in ("ippo", "mappo"):
        buf = cfg[kind]["buffer_size"]
        sec_per_env_step = 1.0 / lay[f"sps_{kind}_collect"] + out[f"{kind}_sec_per_update"] / buf
        out[f"{kind}_effective_train_sps"] = 1.0 / sec_per_env_step
        out["estimates_hours"][kind] = {
            str(b): b * sec_per_env_step / 3600.0 for b in audit_cfg["budget_env_steps"]
        }
    return out


# ---------------------------------------------------------------------------
# 3. Diagnostics: run episodes and classify what happens
# ---------------------------------------------------------------------------

def run_diagnostics(env: WarehouseEnv, policy, seeds: List[int], gamma: float) -> dict:
    n = env.num_robots
    action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
    wall_bumps = robot_collisions = overlap_steps = 0
    interact_total = interact_useful = 0
    idle_robot_steps = robot_steps = 0
    throughput, collisions, spawned, pending_end = [], [], [], []
    obs_log, state_log = [], []
    returns_all, within_t_var, total_var = [], [], []

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        policy.reset()
        ep_rewards = []
        while True:
            actions = policy(obs)
            obs_log.append(np.stack([obs[a] for a in env.possible_agents]))
            state_log.append(env.state())
            before = [(r.pos, r.status, r.target_pos, r.collisions) for r in env.robots]
            obs, rew, _, trunc, _ = env.step(actions)
            ep_rewards.append([rew.get(a, 0.0) for a in env.possible_agents])

            for i, r in enumerate(env.robots):
                act = actions[f"robot_{i}"]
                action_counts[act] += 1
                pos0, status0, target0, coll0 = before[i]
                robot_steps += 1
                idle_robot_steps += int(target0 is None)
                dx, dy = ACTION_DELTAS[act]
                invalid = not env.grid.is_walkable(pos0[0] + dx, pos0[1] + dy)
                if r.collisions > coll0:
                    if invalid:
                        wall_bumps += 1
                    else:
                        robot_collisions += 1
                if act == 5:
                    interact_total += 1
                    # Interact only changes the outcome when picking up from an adjacent cell;
                    # standing on the target already triggers pickup/delivery automatically.
                    if (status0 == RobotStatus.EN_ROUTE_TO_SHELF and target0 is not None
                            and pos0 != target0 and env.grid.manhattan_distance(pos0, target0) <= 1
                            and r.status == RobotStatus.CARRYING_TO_STATION):
                        interact_useful += 1
            if len({r.pos for r in env.robots}) < n:
                overlap_steps += 1
            if all(trunc.values()):
                break

        throughput.append(env.total_orders_completed)
        collisions.append(env.total_fleet_collisions)
        spawned.append(env.order_generator.order_counter)
        pending_end.append(len(env.order_generator.pending_orders))

        # Per-agent discounted returns-to-go, and how much of their variance is between agents
        r = np.asarray(ep_rewards)  # (T, N)
        g = np.zeros_like(r)
        running = np.zeros(n)
        for t in reversed(range(len(r))):
            running = r[t] + gamma * running
            g[t] = running
        returns_all.append(g.ravel())
        within_t_var.append(g.var(axis=1).mean())
        total_var.append(g.var())

    obs_arr = np.concatenate(obs_log).reshape(-1, env.obs_dim)
    state_arr = np.asarray(state_log)
    g_all = np.concatenate(returns_all)
    total_coll = wall_bumps + robot_collisions
    return {
        "episodes": len(seeds),
        "throughput_mean": float(np.mean(throughput)),
        "orders_spawned_mean": float(np.mean(spawned)),
        "pending_orders_at_end_mean": float(np.mean(pending_end)),
        "collisions_mean": float(np.mean(collisions)),
        "wall_bump_share_of_collisions": wall_bumps / total_coll if total_coll else 0.0,
        "robot_robot_collisions_per_ep": robot_collisions / len(seeds),
        "wall_bumps_per_ep": wall_bumps / len(seeds),
        "steps_with_two_robots_in_one_cell": overlap_steps,
        "action_freq": {ACTION_NAMES[k]: float(action_counts[k] / action_counts.sum()) for k in range(NUM_ACTIONS)},
        "interact_actions": interact_total,
        "interact_actions_that_changed_outcome": interact_useful,
        "idle_robot_step_fraction": idle_robot_steps / robot_steps,
        "return_to_go": {
            "mean": float(g_all.mean()), "std": float(g_all.std()),
            "min": float(g_all.min()), "max": float(g_all.max()),
        },
        "between_agent_share_of_return_variance": float(np.mean(within_t_var) / np.mean(total_var)),
        "obs_dims_constant": [int(d) for d in np.where(obs_arr.std(axis=0) < 1e-6)[0]],
        "obs_dim_std": [round(float(s), 4) for s in obs_arr.std(axis=0)],
        "state_dims_constant": [int(d) for d in np.where(state_arr.std(axis=0) < 1e-6)[0]],
        "state_dims_constant_global_block": [
            int(d - env.num_robots * env.obs_dim)
            for d in np.where(state_arr.std(axis=0) < 1e-6)[0] if d >= env.num_robots * env.obs_dim
        ],
    }


# ---------------------------------------------------------------------------
# 4. Targeted checks of specific code paths
# ---------------------------------------------------------------------------

def check_chain_collision_overlap() -> dict:
    """
    Robot A follows B east while C moves west into the cell B wants. B is blocked by the
    vertex conflict, but A's move was grouped before B's revert, so A may enter B's cell.
    """
    env = WarehouseEnv(num_robots=3, grid_width=10, grid_height=10, layout_type="kiva_standard", seed=0)
    env.reset(seed=0)
    y = 0  # top perimeter aisle, free in kiva_standard
    for r, x in zip(env.robots, (0, 1, 3)):
        r.reset(start_pos=(x, y))
    env.step({"robot_0": 2, "robot_1": 2, "robot_2": 3})  # A:E, B:E, C:W
    positions = [r.pos for r in env.robots]
    return {"positions_after": positions, "overlap": len(set(positions)) < len(positions)}


def check_gae_episode_boundary(gamma: float, gae_lambda: float) -> dict:
    """
    Transition 0 ends an episode (done stored at index 0, as the training loops do) and
    the next stored state belongs to a fresh episode with value 100. A correct GAE must
    not bootstrap across the boundary, so advantage[0] should equal r0 - V0 = 0.
    """
    buf = MultiAgentRolloutBuffer(buffer_size=3, num_agents=1, obs_dim=1, state_dim=1,
                                  gamma=gamma, gae_lambda=gae_lambda)
    for t, (done, value) in enumerate([(1.0, 0.0), (0.0, 100.0), (0.0, 0.0)]):
        buf.insert(torch.zeros(1, 1), torch.zeros(1), torch.zeros(1, dtype=torch.int64),
                   torch.zeros(1), torch.zeros(1), torch.tensor([done]), torch.tensor([value]))
    buf.compute_returns_and_advantages(last_values=torch.zeros(1), last_dones=torch.zeros(1))
    adv0 = float(buf.advantages[0, 0])
    return {"advantage_at_terminal_step": adv0, "expected": 0.0, "leaks_across_episodes": abs(adv0) > 1e-6}


def static_findings(cfg: dict) -> dict:
    """Facts read from config and code paths (no simulation needed)."""
    env_cfg = cfg["environment"]
    out = {}
    for kind in ("ippo", "mappo"):
        algo = cfg[kind]
        out[kind] = {
            "config_total_timesteps": algo["total_timesteps"],
            "cli_default_timesteps_that_overrides_config": 15000,
            "episodes_at_config_budget": algo["total_timesteps"] / env_cfg["max_steps"],
            "episodes_at_cli_default": 15000 / env_cfg["max_steps"],
            "agent_transitions_at_config_budget": algo["total_timesteps"] * env_cfg["num_robots"],
            "updates_at_config_budget": algo["total_timesteps"] // algo["buffer_size"],
            "buffer_env_steps": algo["buffer_size"],
            "buffer_fraction_of_episode": algo["buffer_size"] / env_cfg["max_steps"],
            "samples_per_update": algo["buffer_size"] * env_cfg["num_robots"],
            "minibatches_per_epoch": algo["buffer_size"] * env_cfg["num_robots"] // algo["batch_size"],
        }
    out["value_clip_range"] = cfg["hyperparameters"]["clip_eps"]
    out["eval_mismatch"] = {
        "train_max_steps": env_cfg["max_steps"], "eval_max_steps_hardcoded": 300,
        "train_order_lambda": env_cfg["order_lambda"], "readme_eval_order_lambda": 0.25,
        "config_eval_seeds_used": False, "eval_seeds_hardcoded": "100..(100+episodes-1)",
        "eval_policy_sampling_seeded": False,
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--skip-profile", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    audit_cfg = cfg["audit"]
    os.makedirs(audit_cfg["output_dir"], exist_ok=True)
    hp = cfg["hyperparameters"]

    report = {
        "machine": {
            "python": platform.python_version(), "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(), "cpu": platform.processor(),
            "os": platform.platform(),
        }
    }

    print("[1/4] Reproducing committed evaluation ...")
    repro_env = WarehouseEnv(num_robots=5, grid_width=10, grid_height=10)
    rl_agents = load_rl_agents(repro_env, audit_cfg)
    df = reproduce_eval(audit_cfg, rl_agents)
    df.to_csv(os.path.join(audit_cfg["output_dir"], "repro_eval.csv"), index=False)
    print(df.to_string(index=False))

    if not args.skip_profile:
        print("\n[2/4] Profiling simulator and PPO update speed ...")
        report["profile"] = profile(cfg, audit_cfg)
        for layout, d in report["profile"]["layouts"].items():
            print(f"  {layout:16s} " + "  ".join(f"{k}={v:,.0f}" for k, v in d.items()))
        for kind in ("ippo", "mappo"):
            p = report["profile"]
            print(f"  {kind}: {p[f'{kind}_sec_per_update']:.2f}s/update, effective train "
                  f"{p[f'{kind}_effective_train_sps']:,.0f} env-steps/s, hours per budget "
                  + ", ".join(f"{b}:{h:.2f}h" for b, h in p["estimates_hours"][kind].items()))

    print("\n[3/4] Running diagnostics episodes ...")
    diag_env = WarehouseEnv(num_robots=5, grid_width=10, grid_height=10,
                            order_lambda=audit_cfg["repro_order_lambda"],
                            max_steps=audit_cfg["repro_max_steps"])
    seeds = [audit_cfg["diag_seed_start"] + i for i in range(audit_cfg["diag_num_episodes"])]
    policies = {"random": RandomPolicy(diag_env, audit_cfg["diag_seed_start"]),
                "baseline": BaselinePolicy(diag_env)}
    for name, agent in load_rl_agents(diag_env, audit_cfg).items():
        policies[name] = RLPolicy(diag_env, agent, centralized=(name == "mappo"))
    report["diagnostics"] = {}
    for name, pol in policies.items():
        torch.manual_seed(audit_cfg["repro_torch_seed"])
        d = run_diagnostics(diag_env, pol, seeds, hp["gamma"])
        report["diagnostics"][name] = d
        print(f"  {name:8s} thru={d['throughput_mean']:.1f}/{d['orders_spawned_mean']:.1f} spawned  "
              f"coll={d['collisions_mean']:.1f} (wall-bump share {d['wall_bump_share_of_collisions']:.0%})  "
              f"overlap_steps={d['steps_with_two_robots_in_one_cell']}  "
              f"interact useful {d['interact_actions_that_changed_outcome']}/{d['interact_actions']}  "
              f"idle={d['idle_robot_step_fraction']:.0%}  "
              f"G std={d['return_to_go']['std']:.1f}  between-agent var share="
              f"{d['between_agent_share_of_return_variance']:.0%}")
        print(f"           actions: " + ", ".join(f"{k}={v:.0%}" for k, v in d["action_freq"].items()))

    print("\n[4/4] Targeted code-path checks ...")
    report["checks"] = {
        "chain_collision_overlap": check_chain_collision_overlap(),
        "gae_episode_boundary": check_gae_episode_boundary(hp["gamma"], hp["gae_lambda"]),
        "static": static_findings(cfg),
    }
    for k in ("chain_collision_overlap", "gae_episode_boundary"):
        print(f"  {k}: {report['checks'][k]}")
    print(f"  constant obs dims (baseline run): {report['diagnostics']['baseline']['obs_dims_constant']}")
    print(f"  constant global-state dims (offset into global block): "
          f"{report['diagnostics']['baseline']['state_dims_constant_global_block']}")

    out_path = os.path.join(audit_cfg["output_dir"], "audit_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n--> Wrote {out_path} and repro_eval.csv")


if __name__ == "__main__":
    main()
