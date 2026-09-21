"""
Multi-Agent Rollout Buffer with Generalized Advantage Estimation (GAE).
Supports both IPPO (decentralized critic) and MAPPO (centralized critic).
"""

from typing import Generator, Tuple
import torch
import numpy as np


class MultiAgentRolloutBuffer:
    """
    Stores multi-agent trajectory transitions and computes GAE advantages.
    """

    def __init__(
        self,
        buffer_size: int,
        num_agents: int,
        obs_dim: int,
        state_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device = torch.device("cpu"),
    ):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        self.reset()

    def reset(self):
        self.obs = torch.zeros((self.buffer_size, self.num_agents, self.obs_dim), dtype=torch.float32, device=self.device)
        self.states = torch.zeros((self.buffer_size, self.state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.buffer_size, self.num_agents), dtype=torch.int64, device=self.device)
        self.logprobs = torch.zeros((self.buffer_size, self.num_agents), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros((self.buffer_size, self.num_agents), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.buffer_size, self.num_agents), dtype=torch.float32, device=self.device)
        self.values = torch.zeros((self.buffer_size, self.num_agents), dtype=torch.float32, device=self.device)
        self.returns = torch.zeros((self.buffer_size, self.num_agents), dtype=torch.float32, device=self.device)
        self.advantages = torch.zeros((self.buffer_size, self.num_agents), dtype=torch.float32, device=self.device)

        self.step_idx = 0
        self.is_full = False

    def insert(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ):
        idx = self.step_idx
        self.obs[idx] = obs
        self.states[idx] = state
        self.actions[idx] = action
        self.logprobs[idx] = logprob
        self.rewards[idx] = reward
        self.dones[idx] = done
        self.values[idx] = value

        self.step_idx += 1
        if self.step_idx == self.buffer_size:
            self.is_full = True

    def compute_returns_and_advantages(self, last_values: torch.Tensor, last_dones: torch.Tensor):
        """
        Computes GAE advantages:
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
        A_t = delta_t + gamma * lambda * (1 - done) * A_{t+1}
        """
        last_gae = torch.zeros(self.num_agents, device=self.device)
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]

            delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def get_batches(
        self, batch_size: int
    ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        Flattens transitions across time and agents to yield training minibatches.
        """
        total_samples = self.buffer_size * self.num_agents

        # Flatten (T, N, D) -> (T*N, D)
        flat_obs = self.obs.view(total_samples, self.obs_dim)
        # States are replicated for each agent: (T, S) -> (T, N, S) -> (T*N, S)
        flat_states = self.states.unsqueeze(1).repeat(1, self.num_agents, 1).view(total_samples, self.state_dim)
        flat_actions = self.actions.view(total_samples)
        flat_logprobs = self.logprobs.view(total_samples)
        flat_advantages = self.advantages.view(total_samples)
        flat_returns = self.returns.view(total_samples)
        flat_values = self.values.view(total_samples)

        # Normalize advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        indices = np.random.permutation(total_samples)
        for start in range(0, total_samples, batch_size):
            end = start + batch_size
            mb_idx = indices[start:end]
            yield (
                flat_obs[mb_idx],
                flat_states[mb_idx],
                flat_actions[mb_idx],
                flat_logprobs[mb_idx],
                flat_advantages[mb_idx],
                flat_returns[mb_idx],
                flat_values[mb_idx],
            )
