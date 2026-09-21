"""
Gymnasium Vectorized/Multi-Agent Wrapper for WarehouseEnv.
Enables running the multi-agent warehouse environment with standard RL frameworks.
"""

from typing import Dict, Tuple, Any, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from environment.warehouse_env import WarehouseEnv


class MultiAgentWarehouseGymEnv(gym.Env):
    """
    Gymnasium compatibility wrapper for WarehouseEnv.
    Exposes joint observations (or per-agent observations) and MultiDiscrete actions.
    Observation: Box of shape (num_robots, 28)
    Action: MultiDiscrete([6, 6, ..., 6])
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, **env_kwargs):
        super().__init__()
        self.parallel_env = WarehouseEnv(**env_kwargs)
        self.num_robots = self.parallel_env.num_robots
        self.obs_dim = self.parallel_env.obs_dim
        self.state_dim = self.parallel_env.state_dim

        # Joint observation space: (num_robots, 28)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_robots, self.obs_dim), dtype=np.float32
        )

        # Multi-discrete action space: 6 choices per robot
        self.action_space = spaces.MultiDiscrete([6] * self.num_robots)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs_dict, info_dict = self.parallel_env.reset(seed=seed, options=options)
        joint_obs = np.stack([obs_dict[agent] for agent in self.parallel_env.possible_agents], axis=0)
        return joint_obs, {"agents_info": info_dict}

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict[str, Any]]:
        # Convert action array to dict
        action_dict = {
            agent: int(actions[i])
            for i, agent in enumerate(self.parallel_env.possible_agents)
        }

        obs_dict, rewards_dict, term_dict, trunc_dict, info_dict = self.parallel_env.step(action_dict)

        joint_obs = np.stack([obs_dict[agent] for agent in self.parallel_env.possible_agents], axis=0)
        joint_rewards = np.array([rewards_dict[agent] for agent in self.parallel_env.possible_agents], dtype=np.float32)

        terminated = all(term_dict.values()) if term_dict else False
        truncated = all(trunc_dict.values()) if trunc_dict else False

        info = {
            "agents_info": info_dict,
            "mean_reward": float(np.mean(joint_rewards)),
            "fleet_throughput": self.parallel_env.total_orders_completed,
            "fleet_collisions": self.parallel_env.total_fleet_collisions,
            "global_state": self.parallel_env.state(),
        }

        return joint_obs, joint_rewards, terminated, truncated, info

    def get_global_state(self) -> np.ndarray:
        """Returns centralized state for MAPPO Critic."""
        return self.parallel_env.state()
