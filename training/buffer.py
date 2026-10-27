"""
Multi-Agent Rollout Buffer with Generalized Advantage Estimation (GAE).
Supports both IPPO (decentralized critic) and MAPPO (centralized critic), one or several
parallel environments, and per-step action masks.

Done convention: dones[t] = 1 means transition t (s_t, a_t -> r_t, s_{t+1}) ended its episode,
which is what the training loops store (the done flag returned by env.step for that action).
"""

from typing import Generator, NamedTuple, Optional
import torch
import numpy as np


class Minibatch(NamedTuple):
    obs: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    logprobs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor
    masks: torch.Tensor


class MultiAgentRolloutBuffer:
    """
    Stores multi-agent trajectory transitions and computes GAE advantages.
    Internal layout: (buffer_size, num_envs, num_agents, ...). With num_envs=1 the old
    (num_agents, ...) per-step shapes are accepted unchanged.
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
        num_envs: int = 1,
        action_dim: int = 6,
    ):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        self.reset()

    def reset(self):
        T, E, N = self.buffer_size, self.num_envs, self.num_agents
        kw = dict(dtype=torch.float32, device=self.device)
        self.obs = torch.zeros((T, E, N, self.obs_dim), **kw)
        self.states = torch.zeros((T, E, self.state_dim), **kw)
        self.actions = torch.zeros((T, E, N), dtype=torch.int64, device=self.device)
        self.logprobs = torch.zeros((T, E, N), **kw)
        self.rewards = torch.zeros((T, E, N), **kw)
        self.dones = torch.zeros((T, E, N), **kw)
        self.values = torch.zeros((T, E, N), **kw)
        self.returns = torch.zeros((T, E, N), **kw)
        self.advantages = torch.zeros((T, E, N), **kw)
        self.masks = torch.ones((T, E, N, self.action_dim), dtype=torch.bool, device=self.device)

        self.step_idx = 0
        self.is_full = False

    def _per_agent(self, x: torch.Tensor) -> torch.Tensor:
        """Accepts (E,), (E*N,) or (E, N) and returns (E, N)."""
        E, N = self.num_envs, self.num_agents
        x = torch.as_tensor(x, device=self.device)
        if x.numel() == E and N != 1:
            return x.reshape(E, 1).expand(E, N)
        return x.reshape(E, N)

    def insert(
        self,
        obs: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
        logprob: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ):
        E, N = self.num_envs, self.num_agents
        idx = self.step_idx
        self.obs[idx] = obs.reshape(E, N, self.obs_dim)
        self.states[idx] = state.reshape(E, self.state_dim)
        self.actions[idx] = action.reshape(E, N)
        self.logprobs[idx] = logprob.reshape(E, N)
        self.rewards[idx] = self._per_agent(reward)
        self.dones[idx] = self._per_agent(done).float()
        self.values[idx] = value.reshape(E, N)
        if action_mask is not None:
            self.masks[idx] = action_mask.reshape(E, N, self.action_dim)

        self.step_idx += 1
        if self.step_idx == self.buffer_size:
            self.is_full = True

    def compute_returns_and_advantages(self, last_values: torch.Tensor, last_dones: Optional[torch.Tensor] = None):
        """
        GAE with the post-step done convention (see module docstring):
            delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
        last_values: V(s_T) for the observation after the last stored step.
        last_dones is accepted for backward compatibility and unused: done_{T-1} is already stored.
        Phase 0 finding F4: the old code used done_{t+1}, which leaked advantages across episodes.
        """
        last_values = last_values.reshape(self.num_envs, self.num_agents)
        last_gae = torch.zeros((self.num_envs, self.num_agents), device=self.device)
        for t in reversed(range(self.buffer_size)):
            next_values = last_values if t == self.buffer_size - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int) -> Generator[Minibatch, None, None]:
        """
        Flattens transitions across time, envs and agents to yield training minibatches.
        """
        total_samples = self.buffer_size * self.num_envs * self.num_agents

        flat_obs = self.obs.reshape(total_samples, self.obs_dim)
        # States are replicated for each agent: (T, E, S) -> (T, E, N, S) -> (T*E*N, S)
        flat_states = (
            self.states.unsqueeze(2)
            .expand(-1, -1, self.num_agents, -1)
            .reshape(total_samples, self.state_dim)
        )
        flat_actions = self.actions.reshape(total_samples)
        flat_logprobs = self.logprobs.reshape(total_samples)
        flat_advantages = self.advantages.reshape(total_samples)
        flat_returns = self.returns.reshape(total_samples)
        flat_values = self.values.reshape(total_samples)
        flat_masks = self.masks.reshape(total_samples, self.action_dim)

        # Normalize advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        indices = np.random.permutation(total_samples)
        for start in range(0, total_samples, batch_size):
            mb_idx = indices[start:start + batch_size]
            yield Minibatch(
                flat_obs[mb_idx],
                flat_states[mb_idx],
                flat_actions[mb_idx],
                flat_logprobs[mb_idx],
                flat_advantages[mb_idx],
                flat_returns[mb_idx],
                flat_values[mb_idx],
                flat_masks[mb_idx],
            )
