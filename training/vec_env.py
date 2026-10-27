"""
Synchronous vector of WarehouseEnv instances in one process.

Running E environments side by side lets the policy act for all E*N robots in one batched
forward pass, which amortises the per-call torch overhead measured in Phase 0 (~6.6 ms/step).
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from environment.warehouse_env import WarehouseEnv, NUM_ACTIONS


def episode_stats(env: WarehouseEnv, episode_return: float) -> Dict[str, float]:
    """Episode-level metrics read from a finished env (before it is reset)."""
    latencies = [o.latency for o in env.order_generator.completed_orders if o.latency is not None]
    robot_steps = env.num_robots * env.current_step
    moves = sum(r.total_steps for r in env.robots)
    return {
        "throughput": env.total_orders_completed,
        "collisions": env.total_fleet_collisions,
        "robot_collisions": env.total_robot_collisions,
        "wall_bumps": env.total_wall_bumps,
        "mean_latency": float(np.mean(latencies)) if latencies else float("nan"),
        "travel_distance": moves,
        "utilization": moves / robot_steps if robot_steps else 0.0,
        "orders_spawned": env.order_generator.order_counter,
        "pending_at_end": len(env.order_generator.pending_orders),
        "return": episode_return,
        "steps": env.current_step,
        "order_lambda": env.order_lambda,
    }


class SyncVectorWarehouseEnv:
    """
    step() returns arrays:
        obs (E, N, obs_dim), rewards (E, N), dones (E,), masks (E, N, 6), states (E, state_dim), infos
    Finished envs are reset automatically; their infos[e] carry "final_obs", "final_state" and
    "episode" (episode_stats) so callers can bootstrap truncated episodes and log metrics.
    """

    def __init__(self, env_kwargs, num_envs: int, lambda_range: Optional[Tuple[float, float]] = None,
                 seed: Optional[int] = None):
        """
        env_kwargs: one dict for all envs, or a list with one dict per env (e.g. alternating layouts).
        lambda_range: if given, every reset draws order_lambda ~ Uniform(lo, hi) (domain randomisation).
        """
        kw_list = env_kwargs if isinstance(env_kwargs, list) else [env_kwargs] * num_envs
        if len(kw_list) != num_envs:
            raise ValueError("need one env_kwargs dict per env")
        self.envs = [WarehouseEnv(**kw) for kw in kw_list]
        if len({env.num_robots for env in self.envs}) != 1:
            raise ValueError("all envs in a vector must have the same number of robots")
        self.lambda_range = lambda_range
        self._rng = np.random.default_rng(seed)
        self.num_envs = num_envs
        first = self.envs[0]
        self.num_robots = first.num_robots
        self.obs_dim = first.obs_dim
        self.state_dim = first.state_dim
        self.action_dim = NUM_ACTIONS
        self.agents = first.possible_agents
        self._ep_returns = np.zeros(num_envs, dtype=np.float64)

    def _sample_lambda(self, env: WarehouseEnv):
        if self.lambda_range is not None:
            lam = float(self._rng.uniform(*self.lambda_range))
            env.order_lambda = lam
            env.order_generator.lambda_rate = lam

    def _obs(self, env: WarehouseEnv, obs_dict) -> np.ndarray:
        return np.stack([obs_dict[a] for a in self.agents])

    def _masks(self, info_dict) -> np.ndarray:
        return np.stack([info_dict[a]["action_mask"] for a in self.agents])

    def reset(self, seeds: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        seeds = seeds if seeds is not None else [None] * self.num_envs
        obs, masks, states = [], [], []
        for env, seed in zip(self.envs, seeds):
            self._sample_lambda(env)
            o, info = env.reset(seed=seed)
            obs.append(self._obs(env, o))
            masks.append(self._masks(info))
            states.append(env.state(o))
        self._ep_returns[:] = 0.0
        return np.stack(obs), np.stack(masks), np.stack(states)

    def step(self, actions: np.ndarray):
        E, N = self.num_envs, self.num_robots
        obs = np.zeros((E, N, self.obs_dim), dtype=np.float32)
        rewards = np.zeros((E, N), dtype=np.float32)
        dones = np.zeros(E, dtype=bool)
        masks = np.zeros((E, N, self.action_dim), dtype=bool)
        states = np.zeros((E, self.state_dim), dtype=np.float32)
        infos: List[Dict[str, Any]] = [{} for _ in range(E)]

        for e, env in enumerate(self.envs):
            act = {a: int(actions[e, i]) for i, a in enumerate(self.agents)}
            o, r, term, trunc, info = env.step(act)
            rewards[e] = [r.get(a, 0.0) for a in self.agents]
            self._ep_returns[e] += float(rewards[e].mean())
            done = all(term.values()) or all(trunc.values())
            dones[e] = done
            if done:
                infos[e]["final_obs"] = self._obs(env, o)
                infos[e]["final_state"] = env.state(o)
                infos[e]["truncated"] = all(trunc.values()) and not all(term.values())
                infos[e]["episode"] = episode_stats(env, self._ep_returns[e])
                self._ep_returns[e] = 0.0
                self._sample_lambda(env)
                o, info = env.reset()
            obs[e] = self._obs(env, o)
            masks[e] = self._masks(info)
            states[e] = env.state(o)
        return obs, rewards, dones, masks, states, infos
