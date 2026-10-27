"""
PyTorch Neural Network Models for Warehouse RL and MARL:
- Actor (Decentralized policy)
- DecentralizedCritic (for IPPO)
- CentralizedCritic (for MAPPO with CTDE)
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

# Logit assigned to masked actions: probability exp(-1e8) == 0 in float32, still finite.
MASKED_LOGIT = -1e8


def layer_init(layer: nn.Linear, std: float = 1.414, bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal layer initialization for stable PPO training."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    """
    Decentralized Actor Network:
    Maps agent local observation (28-dim) to categorical action distribution (6 actions).
    An optional boolean action mask (True = allowed) removes invalid actions from the distribution.
    """

    def __init__(self, obs_dim: int = 28, action_dim: int = 6, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )

    def forward(self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> Categorical:
        logits = self.network(obs)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, MASKED_LOGIT)
        # validate_args=False: argument checks cost ~1 ms per call on this CPU (Phase 0 profile)
        return Categorical(logits=logits, validate_args=False)

    def get_action_and_logprob(
        self,
        obs: torch.Tensor,
        action: torch.Tensor = None,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ):
        dist = self.forward(obs, action_mask)
        if action is None:
            action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy()


class DecentralizedCritic(nn.Module):
    """
    Decentralized Value Network (used in IPPO):
    Evaluates state value using only the agent's local observation (28-dim).
    """

    def __init__(self, obs_dim: int = 28, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class CentralizedCritic(nn.Module):
    """
    Centralized Value Network (used in MAPPO with CTDE):
    Evaluates expected return from the global warehouse state (150-dim), or from the
    agent-specific global state (global state + the agent's own observation, 178-dim).
    """

    def __init__(self, state_dim: int = 150, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim // 2, 1), std=1.0),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)
