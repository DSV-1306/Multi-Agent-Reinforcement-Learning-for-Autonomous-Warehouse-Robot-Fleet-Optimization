"""
Training script for the hierarchical method (Phase 4): PPO allocation policy + Space-Time A* routing.

    python training/train_hier.py --seed 1
    python training/train_hier.py --seed 1 --timesteps 20000

Trains on the same task distribution as the last curriculum stage of IPPO/MAPPO (5 robots, training
layouts alternating per episode, order rate ~ U(stage range) per episode, synthetic orders). The robot
count is fixed, so no robot-count curriculum is needed. Episode ends are treated as terminal for the
allocator (no bootstrap): the learning signal is per decision and gamma^100 ~ 0.37 makes the cut small.

Outputs: results/runs/<run_name>/{learning_curve.csv, best_dev.pt, final.pt, config.json, summary.json}
"""

import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.hier_agent import (HierAllocator, RLDispatcher, allocation_reward, make_hier_env, run_episode,
                               run_hier_episodes)
from evaluation.metrics import summarize_episodes
from training.config_utils import load_config, env_kwargs_from_cfg
from training.trainer import CURVE_METRICS, dev_scenarios
from training.train_ippo import parse_args


def episode_transitions(decisions: List[Dict[str, Any]], step_rewards: Dict[int, float], last_step: int,
                        gamma: float) -> List[Dict[str, Any]]:
    """
    Turns one episode's decisions into semi-MDP transitions. Decision j at time t_j collects the rewards
    of steps s in [t_j, t_{j+1}) discounted by gamma^(s - t_j); disc_j = gamma^(t_{j+1} - t_j).
    The last decision of the episode collects rewards up to last_step and is terminal.
    """
    out = []
    for j, d in enumerate(decisions):
        end = decisions[j + 1]["t"] if j + 1 < len(decisions) else last_step + 1
        R = sum(gamma ** (s - d["t"]) * step_rewards.get(s, 0.0) for s in range(d["t"], end))
        last = j + 1 == len(decisions)
        out.append({**d, "R": R, "disc": gamma ** (end - d["t"]), "done": last,
                    "next_value": 0.0 if last else decisions[j + 1]["value"]})
    return out


def order_time_transitions(decisions: List[Dict[str, Any]], delivered: Dict[int, int], last_step: int,
                           max_steps: int) -> List[Dict[str, Any]]:
    """Bandit transitions: R_j = -(delivery step of the assigned order - t_j) / max_steps, censored at the end."""
    out = []
    for d in decisions:
        done_at = delivered.get(d["order_id"], last_step + 1)
        out.append({**d, "R": -(done_at - d["t"]) / max_steps, "disc": 0.0, "done": True, "next_value": 0.0})
    return out


def train_hier(cfg: Dict[str, Any], seed: int, run_name: str = None, total_timesteps: int = None,
               variant: str = None) -> Dict[str, Any]:
    hcfg, hp, tcfg = cfg["hier"], cfg["hyperparameters"], cfg["training"]
    total_timesteps = total_timesteps or hcfg["total_timesteps"]
    torch.set_num_threads(tcfg["torch_threads"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    run_name = run_name or f"hier_seed{seed}"
    run_dir = os.path.join(tcfg["runs_dir"], run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"method": "hier", "variant": variant or "hier", "seed": seed,
                   "total_timesteps": total_timesteps, "cfg": cfg}, f, indent=2)

    alloc = HierAllocator(hcfg, hp["gamma"], hp["gae_lambda"], hp["clip_eps"], hcfg["entropy_coef"],
                          hp["value_coef"], hp["max_grad_norm"])
    dispatcher = RLDispatcher(alloc.net, hcfg)
    stage = cfg["curriculum"]["stages"][-1]
    stride = tcfg["env_seed_stride"]
    rng = np.random.default_rng(seed * stride)
    envs = [make_hier_env(env_kwargs_from_cfg(cfg["environment"], num_robots=stage["num_robots"], layout_type=l),
                          dispatcher) for l in tcfg["layouts"]]
    dev_kw = dev_scenarios(cfg)
    dev_seeds = list(cfg["seeds"]["dev_eval"])

    def dev_eval() -> Dict[str, float]:
        evald = RLDispatcher(alloc.net, hcfg, deterministic=cfg["evaluation"]["deterministic"])
        out, all_eps = {}, []
        for kw in dev_kw:
            eps = run_hier_episodes(evald, kw, dev_seeds, cfg["evaluation"]["torch_seed"])
            out[f"dev_{kw['layout_type']}_throughput_mean"] = summarize_episodes(eps)["throughput_mean"]
            all_eps += eps
        s = summarize_episodes(all_eps)
        out.update({f"dev_{k}": s[k] for k in CURVE_METRICS})
        return out

    curve_rows, recent, transitions = [], [], []
    train_metrics: Dict[str, float] = {}
    global_step, next_eval, best_dev, episode, decisions_total = 0, 0, -float("inf"), 0, 0
    start = time.time()
    curve_path = os.path.join(run_dir, "learning_curve.csv")
    print(f"--> [HIER] seed={seed} budget={total_timesteps:,} env steps -> {run_dir}", flush=True)

    def log_dev():
        nonlocal best_dev, recent
        dev = dev_eval()
        row = {"env_steps": global_step, "decisions": decisions_total, "episodes": episode,
               "wall_time_s": time.time() - start, **dev,
               **({"train_throughput": float(np.mean([e["throughput"] for e in recent]))} if recent else {}),
               **train_metrics}
        curve_rows.append(row)
        pd.DataFrame(curve_rows).to_csv(curve_path, index=False)
        if dev["dev_throughput_mean"] > best_dev:
            best_dev = dev["dev_throughput_mean"]
            alloc.save(os.path.join(run_dir, "best_dev.pt"), {"seed": seed})
        print(f"  step {global_step:>9,} | decisions {decisions_total:>7,} | dev thru {dev['dev_throughput_mean']:5.1f} "
              f"coll {dev['dev_collisions_mean']:5.1f} lat {dev['dev_mean_latency_mean']:5.1f} | train thru "
              f"{row.get('train_throughput', float('nan')):5.1f} | {global_step / max(time.time() - start, 1e-9):,.0f} "
              f"env-steps/s", flush=True)
        recent = []

    while global_step < total_timesteps:
        if global_step >= next_eval:
            log_dev()
            next_eval += tcfg["eval_interval"]
        env, ctrl = envs[episode % len(envs)]
        lam = float(rng.uniform(*stage["order_lambda_range"]))
        env.order_lambda = lam
        env.order_generator.lambda_rate = lam
        dispatcher.record = []
        step_rewards: Dict[int, float] = {}
        stats = run_episode(env, ctrl, seed * stride + episode,
                            on_step=lambda e, d: step_rewards.__setitem__(e.current_step, allocation_reward(e, d, hcfg)))
        recent.append(stats)
        if hcfg["reward_mode"] == "order_time":
            delivered = {o.id: o.delivered_step for o in env.order_generator.completed_orders}
            trans = order_time_transitions(dispatcher.record, delivered, env.current_step, env.max_steps)
        else:
            trans = episode_transitions(dispatcher.record, step_rewards, env.current_step, hp["gamma"])
        transitions += trans
        decisions_total += len(trans)
        global_step += env.current_step
        episode += 1
        if len(transitions) >= hcfg["rollout_decisions"]:
            alloc.set_lr_fraction(max(1.0 - global_step / total_timesteps, 0.0))
            train_metrics = alloc.update(transitions, hcfg["num_epochs"], hcfg["batch_size"])
            transitions = []

    dispatcher.record = None
    log_dev()
    alloc.save(os.path.join(run_dir, "final.pt"), {"seed": seed})
    summary = {"method": "hier", "seed": seed, "env_steps": global_step, "decisions": decisions_total,
               "wall_time_s": time.time() - start, "env_steps_per_s": global_step / (time.time() - start),
               "final_dev_throughput": curve_rows[-1]["dev_throughput_mean"], "best_dev_throughput": best_dev,
               "run_dir": run_dir}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"--> [HIER] done in {summary['wall_time_s'] / 60:.1f} min | best dev thru {best_dev:.1f}", flush=True)
    return summary


if __name__ == "__main__":
    args = parse_args("hier")
    train_hier(load_config(args.config, args.set), args.seed, args.run_name, args.timesteps, args.variant)
