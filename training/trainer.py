"""
Shared PPO training loop for IPPO and MAPPO (Phases 1-2).

Per update:
  1. Collect `buffer_size` steps from each of `training.num_envs` envs (one batched forward pass for all
     envs x robots), with optional action masks, observation normalization and reward scaling.
  2. On time-limit truncation, add gamma * V(final observation) to the last reward: the episode was cut by
     the clock, not ended by the task (time is not in the observation).
  3. GAE + PPO update, optional linear learning-rate annealing (over the whole run).
  4. Every `training.eval_interval` env steps: evaluate on the DEV seeds of every dev scenario (never the
     test seeds), append a row to learning_curve.csv, keep the checkpoint with the best mean dev throughput.

Curriculum (Phase 2): the budget is split into stages (curriculum.stages, fractions of total_timesteps).
Each stage fixes the robot count and an order-rate range; every episode reset draws lambda ~ U(range).
Envs alternate over training.layouts. With curriculum.enabled=false only the LAST stage is used for the
whole budget, so the ablation removes the ramp but keeps the same final task distribution.

"Timesteps" are environment steps summed over all parallel envs; one env step = num_robots transitions.
"""

import json
import math
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from agents.ippo_agent import IPPOAgent
from agents.mappo_agent import MAPPOAgent
from evaluation.evaluate import PolicyRunner, run_policy_episodes
from evaluation.metrics import summarize_episodes
from training.buffer import MultiAgentRolloutBuffer
from training.config_utils import env_kwargs_from_cfg
from training.normalization import ObsNormalizer, RewardScaler
from training.vec_env import SyncVectorWarehouseEnv

CURVE_METRICS = ("throughput_mean", "throughput_std", "collisions_mean", "robot_collisions_mean",
                 "wall_bumps_mean", "mean_latency_mean", "return_mean")


def build_agent(method: str, cfg: Dict[str, Any], obs_dim: int, state_dim: int, action_dim: int):
    hp, algo = cfg["hyperparameters"], cfg[method]
    common = dict(
        obs_dim=obs_dim, action_dim=action_dim, clip_eps=hp["clip_eps"], value_coef=hp["value_coef"],
        entropy_coef=hp["entropy_coef"], max_grad_norm=hp["max_grad_norm"], value_clip=hp["value_clip"],
    )
    if method == "ippo":
        return IPPOAgent(lr=algo["lr"], **common)
    if method == "mappo":
        return MAPPOAgent(state_dim=state_dim, lr_actor=algo["lr_actor"], lr_critic=algo["lr_critic"],
                          critic_input=algo["critic_input"], **common)
    raise ValueError(f"Unknown method {method!r}")


def curriculum_stages(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    stages = cfg["curriculum"]["stages"]
    if not cfg["curriculum"]["enabled"]:
        return [{**stages[-1], "fraction": 1.0}]
    total = sum(s["fraction"] for s in stages)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"curriculum stage fractions must sum to 1, got {total}")
    return stages


def dev_scenarios(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Env kwargs of each dev scenario used for learning curves and best-checkpoint selection."""
    return [env_kwargs_from_cfg(cfg["environment"], **sc) for sc in cfg["training"]["dev_scenarios"]]


def _explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    var_y = returns.var()
    return float(1.0 - (returns - values).var() / var_y) if var_y > 0 else float("nan")


def train(method: str, cfg: Dict[str, Any], seed: int, run_name: Optional[str] = None,
          total_timesteps: Optional[int] = None, variant: Optional[str] = None) -> Dict[str, Any]:
    tcfg, hp, algo = cfg["training"], cfg["hyperparameters"], cfg[method]
    total_timesteps = total_timesteps if total_timesteps is not None else algo["total_timesteps"]
    torch.set_num_threads(tcfg["torch_threads"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_name = run_name or f"{method}_seed{seed}"
    run_dir = os.path.join(tcfg["runs_dir"], run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"method": method, "variant": variant or method, "seed": seed,
                   "total_timesteps": total_timesteps, "cfg": cfg}, f, indent=2)

    E, T = tcfg["num_envs"], algo["buffer_size"]
    use_mask = tcfg["action_masking"]
    centralized = method == "mappo"
    env_cfg = cfg["environment"]
    layouts = tcfg["layouts"]
    stages = curriculum_stages(cfg)

    probe_kwargs = env_kwargs_from_cfg(env_cfg)
    from environment.warehouse_env import WarehouseEnv, NUM_ACTIONS
    probe = WarehouseEnv(**probe_kwargs)
    D, S, A = probe.obs_dim, probe.state_dim, NUM_ACTIONS

    agent = build_agent(method, cfg, D, S, A)
    obs_norm = ObsNormalizer(D, tcfg["normalize_obs"], tcfg["obs_clip"], tcfg["norm_epsilon"])
    state_norm = ObsNormalizer(S, tcfg["normalize_obs"] and centralized, tcfg["obs_clip"], tcfg["norm_epsilon"])
    reward_rms = None  # shared across stages so the reward scale does not restart

    def values_of(o_n: np.ndarray, s_n: np.ndarray) -> torch.Tensor:
        o_t = torch.as_tensor(o_n, dtype=torch.float32)
        return agent.get_values(o_t, torch.as_tensor(s_n, dtype=torch.float32)) if centralized \
            else agent.get_values(o_t)

    def checkpoint_extra() -> Dict[str, Any]:
        return {"method": method, "seed": seed, "action_masking": use_mask,
                "obs_norm": obs_norm.state_dict(), "state_norm": state_norm.state_dict(),
                "state_max_robots": probe.state_max_robots, "env_kwargs": probe_kwargs}

    dev_seeds = list(cfg["seeds"]["dev_eval"])
    dev_kwargs = dev_scenarios(cfg)

    def dev_eval() -> Dict[str, float]:
        runner = PolicyRunner(agent, method, obs_norm, state_norm, use_mask,
                              deterministic=cfg["evaluation"]["deterministic"])
        out, all_eps = {}, []
        for kw in dev_kwargs:
            eps = run_policy_episodes(runner, kw, dev_seeds, cfg["evaluation"]["torch_seed"])
            out[f"dev_{kw.get('layout_type', 'custom')}_throughput_mean"] = summarize_episodes(eps)["throughput_mean"]
            all_eps += eps
        summ = summarize_episodes(all_eps)
        out.update({f"dev_{k}": summ[k] for k in CURVE_METRICS})
        return out

    steps_per_update = T * E
    num_updates = math.ceil(total_timesteps / steps_per_update)
    stage_ends = np.cumsum([s["fraction"] for s in stages]) * num_updates
    eval_interval = tcfg["eval_interval"]
    global_step, next_eval, best_dev = 0, 0, -float("inf")
    recent_eps: List[Dict[str, float]] = []
    curve_rows: List[Dict[str, Any]] = []
    curve_path = os.path.join(run_dir, "learning_curve.csv")
    train_metrics: Dict[str, float] = {}
    start = time.time()

    print(f"--> [{method.upper()}] seed={seed} envs={E} rollout={T} updates={num_updates} "
          f"budget={num_updates * steps_per_update:,} env steps, {len(stages)} stage(s) -> {run_dir}")

    def log_dev(update_idx: int, stage_idx: int):
        nonlocal best_dev, recent_eps
        dev = dev_eval()
        row = {"env_steps": global_step, "agent_transitions": global_step * stages[stage_idx]["num_robots"],
               "update": update_idx, "stage": stage_idx, "wall_time_s": time.time() - start, **dev,
               **({f"train_{k}": float(np.mean([e[k] for e in recent_eps]))
                   for k in ("throughput", "collisions", "return")} if recent_eps else {}),
               "train_episodes_in_window": len(recent_eps), **train_metrics}
        curve_rows.append(row)
        pd.DataFrame(curve_rows).to_csv(curve_path, index=False)
        if dev["dev_throughput_mean"] > best_dev:
            best_dev = dev["dev_throughput_mean"]
            agent.save(os.path.join(run_dir, "best_dev.pt"), checkpoint_extra())
        print(f"  step {global_step:>9,} st{stage_idx} | dev thru {dev['dev_throughput_mean']:5.1f} "
              f"coll {dev['dev_collisions_mean']:5.1f} (wall {dev['dev_wall_bumps_mean']:.1f}) | train thru "
              f"{row.get('train_throughput', float('nan')):5.1f} | "
              f"{global_step / max(time.time() - start, 1e-9):,.0f} env-steps/s", flush=True)
        recent_eps = []

    update = 0
    for stage_idx, stage in enumerate(stages):
        n = stage["num_robots"]
        kw_list = [env_kwargs_from_cfg(env_cfg, num_robots=n, layout_type=layouts[k % len(layouts)])
                   for k in range(E)]
        venv = SyncVectorWarehouseEnv(kw_list, E, lambda_range=tuple(stage["order_lambda_range"]),
                                      seed=seed * tcfg["env_seed_stride"] + stage_idx)
        rew_scaler = RewardScaler((E, n), hp["gamma"], tcfg["normalize_reward"], tcfg["reward_clip"],
                                  tcfg["norm_epsilon"])
        if reward_rms is not None:
            rew_scaler.rms = reward_rms
        reward_rms = rew_scaler.rms
        buffer = MultiAgentRolloutBuffer(T, n, D, S, hp["gamma"], hp["gae_lambda"], num_envs=E, action_dim=A)
        obs, masks, states = venv.reset(
            seeds=[seed * tcfg["env_seed_stride"] + 100 * stage_idx + k for k in range(E)])

        while update < round(stage_ends[stage_idx]):
            update += 1
            if global_step >= next_eval:
                log_dev(update - 1, stage_idx)
                next_eval += eval_interval
            if tcfg["anneal_lr"]:
                agent.set_lr_fraction(1.0 - (update - 1) / num_updates)
            buffer.reset()

            for _ in range(T):
                o_n = obs_norm(obs, update=tcfg["normalize_obs"])
                s_n = state_norm(states, update=state_norm.enabled)
                o_t = torch.as_tensor(o_n, dtype=torch.float32)
                s_t = torch.as_tensor(s_n, dtype=torch.float32)
                m_t = torch.as_tensor(masks)
                if centralized:
                    actions, logprobs, values = agent.get_actions_and_values(o_t, s_t, m_t if use_mask else None)
                else:
                    actions, logprobs, values = agent.get_actions_and_values(o_t, m_t if use_mask else None)

                obs, rewards, dones, masks, states, infos = venv.step(actions.numpy())
                r = rew_scaler(rewards, dones)
                for e in np.flatnonzero(dones):
                    recent_eps.append(infos[e]["episode"])
                    if tcfg["bootstrap_on_truncation"] and infos[e]["truncated"]:
                        v_final = values_of(obs_norm(infos[e]["final_obs"], update=False),
                                            state_norm(infos[e]["final_state"], update=False))
                        r[e] += hp["gamma"] * v_final.numpy()

                buffer.insert(o_t, s_t, actions, logprobs, torch.as_tensor(r), torch.as_tensor(dones),
                              values, m_t if use_mask else None)
                global_step += E

            last_values = values_of(obs_norm(obs, update=False), state_norm(states, update=False))
            buffer.compute_returns_and_advantages(last_values)
            train_metrics = agent.train_step(buffer, num_epochs=algo["num_epochs"], batch_size=algo["batch_size"])
            train_metrics["explained_variance"] = _explained_variance(buffer.values.flatten(),
                                                                      buffer.returns.flatten())
            train_metrics["reward_scale"] = rew_scaler.scale

    log_dev(num_updates, len(stages) - 1)
    agent.save(os.path.join(run_dir, "final.pt"), checkpoint_extra())

    final = curve_rows[-1]
    summary = {"method": method, "seed": seed, "env_steps": global_step,
               "wall_time_s": time.time() - start, "env_steps_per_s": global_step / (time.time() - start),
               "final_dev_throughput": final["dev_throughput_mean"], "best_dev_throughput": best_dev,
               "curriculum": cfg["curriculum"]["enabled"], "run_dir": run_dir}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"--> [{method.upper()}] done in {summary['wall_time_s'] / 60:.1f} min | final dev thru "
          f"{final['dev_throughput_mean']:.1f} | best dev thru {best_dev:.1f}", flush=True)
    return summary
