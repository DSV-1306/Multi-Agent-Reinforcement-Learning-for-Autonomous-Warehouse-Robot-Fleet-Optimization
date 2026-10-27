"""
Independent PPO (IPPO) Agent implementation with parameter sharing across robots.
"""

from typing import Tuple, Dict, Any, Optional
import os
import torch
import torch.nn as nn
import torch.optim as optim

from agents.models import Actor, DecentralizedCritic
from training.buffer import MultiAgentRolloutBuffer


def value_loss_fn(new_values, old_values, returns, value_clip: Optional[float]) -> torch.Tensor:
    """0.5 * MSE, optionally with the PPO value-clipping objective (value_clip=None disables it)."""
    v_loss_unclipped = (new_values - returns) ** 2
    if value_clip is None:
        return 0.5 * v_loss_unclipped.mean()
    v_clipped = old_values + torch.clamp(new_values - old_values, -value_clip, value_clip)
    v_loss_clipped = (v_clipped - returns) ** 2
    return 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()


def policy_loss_fn(new_logprobs, old_logprobs, advantages, clip_eps: float):
    """Clipped surrogate loss plus diagnostics (approx. KL, fraction of clipped ratios)."""
    log_ratio = new_logprobs - old_logprobs
    ratio = torch.exp(log_ratio)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    with torch.no_grad():
        approx_kl = ((ratio - 1.0) - log_ratio).mean()  # unbiased low-variance estimator
        clipfrac = ((ratio - 1.0).abs() > clip_eps).float().mean()
    return -torch.min(surr1, surr2).mean(), approx_kl, clipfrac


class IPPOAgent:
    """
    Independent PPO (IPPO):
    Each robot shares an actor and a decentralized critic.
    The critic evaluates state values solely from the individual robot's local observation.
    """

    def __init__(
        self,
        obs_dim: int = 28,
        action_dim: int = 6,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: torch.device = torch.device("cpu"),
        value_clip: Optional[float] = None,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.clip_eps = clip_eps
        self.value_clip = value_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.base_lr = lr

        self.actor = Actor(obs_dim=obs_dim, action_dim=action_dim).to(device)
        self.critic = DecentralizedCritic(obs_dim=obs_dim).to(device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr, eps=1e-5
        )

    def set_lr_fraction(self, frac: float):
        """Linear learning-rate annealing: lr = base_lr * frac."""
        for group in self.optimizer.param_groups:
            group["lr"] = self.base_lr * frac

    @torch.no_grad()
    def get_actions_and_values(
        self,
        obs_tensor: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        obs_tensor shape: (..., num_agents, obs_dim); action_mask: (..., num_agents, action_dim) bool
        Returns actions, logprobs, values, each (..., num_agents)
        """
        actions, logprobs, _ = self.actor.get_action_and_logprob(
            obs_tensor, action_mask=action_mask, deterministic=deterministic
        )
        values = self.critic(obs_tensor).squeeze(-1)
        return actions, logprobs, values

    @torch.no_grad()
    def get_values(self, obs_tensor: torch.Tensor, state_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.critic(obs_tensor).squeeze(-1)

    def train_step(
        self, buffer: MultiAgentRolloutBuffer, num_epochs: int = 4, batch_size: int = 64
    ) -> Dict[str, float]:
        """Runs PPO gradient optimization over collected rollouts."""
        logs = {k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac")}

        for _ in range(num_epochs):
            for mb in buffer.get_batches(batch_size):
                # Actor loss (the stored mask is re-applied so old and new log-probs match)
                _, new_logprobs, entropy = self.actor.get_action_and_logprob(
                    mb.obs, mb.actions, action_mask=mb.masks
                )
                policy_loss, approx_kl, clipfrac = policy_loss_fn(
                    new_logprobs, mb.logprobs, mb.advantages, self.clip_eps
                )

                new_values = self.critic(mb.obs).squeeze(-1)
                value_loss = value_loss_fn(new_values, mb.values, mb.returns, self.value_clip)

                entropy_loss = entropy.mean()

                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                for k, v in zip(logs, (policy_loss, value_loss, entropy_loss, approx_kl, clipfrac)):
                    logs[k].append(v.item())

        return {k: float(torch.tensor(v).mean()) for k, v in logs.items()}

    def save(self, filepath: str, extra: Optional[Dict[str, Any]] = None):
        """extra: run metadata (normalizer statistics, masking flag, ...) needed to evaluate the policy."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "extra": extra or {},
        }, filepath)

    def load(self, filepath: str, load_optimizer: bool = True) -> Dict[str, Any]:
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint.get("extra", {})
