"""
PyTorch Neural Network Models for Warehouse RL and MARL:
- Actor (Decentralized policy)
- DecentralizedCritic (for IPPO)
- CentralizedCritic (for MAPPO with CTDE)
"""

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(layer: nn.Linear, std: float = 1.414, bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal layer initialization for stable PPO training."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(nn.Module):
    """
    Decentralized Actor Network:
    Maps agent local observation (28-dim) to categorical action distribution (6 actions).
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

    def forward(self, obs: torch.Tensor) -> Categorical:
        logits = self.network(obs)
        return Categorical(logits=logits)

    def get_action_and_logprob(
        self, obs: torch.Tensor, action: torch.Tensor = None
    ):
        dist = self.forward(obs)
        if action is None:
            action = dist.sample()
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
    Evaluates expected return using the global state of the entire warehouse (150-dim).
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
