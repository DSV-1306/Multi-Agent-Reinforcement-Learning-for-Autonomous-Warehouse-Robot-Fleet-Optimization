"""
Multi-Agent PPO (MAPPO) Agent with Centralized Training and Decentralized Execution (CTDE).
"""

from typing import Tuple, Dict, Any, Optional
import os
import torch
import torch.nn as nn
import torch.optim as optim

from agents.models import Actor, CentralizedCritic
from training.buffer import MultiAgentRolloutBuffer


class MAPPOAgent:
    """
    Multi-Agent PPO (MAPPO) with Centralized Critic (CTDE):
    - Actor: Decentralized, conditions on local robot observation (28-dim).
    - Critic: Centralized, conditions on full global warehouse state (150-dim).
    """

    def __init__(
        self,
        obs_dim: int = 28,
        state_dim: int = 150,
        action_dim: int = 6,
        lr_actor: float = 3e-4,
        lr_critic: float = 5e-4,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: torch.device = torch.device("cpu"),
    ):
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        self.actor = Actor(obs_dim=obs_dim, action_dim=action_dim).to(device)
        self.critic = CentralizedCritic(state_dim=state_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)

    @torch.no_grad()
    def get_actions_and_values(
        self, obs_tensor: torch.Tensor, state_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        obs_tensor shape: (num_agents, obs_dim)
        state_tensor shape: (state_dim,)
        Returns:
            actions: (num_agents,)
            logprobs: (num_agents,)
            values: (num_agents,)  -- Centralized value broadcasted to all agents
        """
        num_agents = obs_tensor.shape[0]
        actions, logprobs, _ = self.actor.get_action_and_logprob(obs_tensor)
        central_val = self.critic(state_tensor.unsqueeze(0)).squeeze()
        values = central_val.repeat(num_agents)
        return actions, logprobs, values

    def train_step(
        self, buffer: MultiAgentRolloutBuffer, num_epochs: int = 4, batch_size: int = 64
    ) -> Dict[str, float]:
        """Performs PPO training updates on Actor and Centralized Critic."""
        policy_losses = []
        value_losses = []
        entropy_losses = []

        for _ in range(num_epochs):
            for mb_obs, mb_states, mb_actions, mb_old_logprobs, mb_advantages, mb_returns, mb_old_values in buffer.get_batches(batch_size):
                # 1. Update Decentralized Actor
                _, new_logprobs, entropy = self.actor.get_action_and_logprob(mb_obs, mb_actions)
                ratio = torch.exp(new_logprobs - mb_old_logprobs)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                entropy_loss = entropy.mean()
                total_actor_loss = policy_loss - self.entropy_coef * entropy_loss

                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # 2. Update Centralized Critic using global warehouse state
                new_values = self.critic(mb_states).squeeze(-1)
                v_loss_unclipped = (new_values - mb_returns) ** 2
                v_clipped = mb_old_values + torch.clamp(new_values - mb_old_values, -self.clip_eps, self.clip_eps)
                v_loss_clipped = (v_clipped - mb_returns) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                self.critic_optimizer.zero_grad()
                (self.value_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

        return {
            "policy_loss": float(torch.tensor(policy_losses).mean()),
            "value_loss": float(torch.tensor(value_losses).mean()),
            "entropy": float(torch.tensor(entropy_losses).mean()),
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
