"""
Multi-Agent PPO (MAPPO) Agent with Centralized Training and Decentralized Execution (CTDE).
"""

from typing import Tuple, Dict, Any, Optional
import os
import torch
import torch.nn as nn
import torch.optim as optim

from agents.models import Actor, CentralizedCritic
from agents.ippo_agent import policy_loss_fn, value_loss_fn
from training.buffer import MultiAgentRolloutBuffer

CRITIC_INPUTS = ("global", "agent_specific")


class MAPPOAgent:
    """
    Multi-Agent PPO (MAPPO) with Centralized Critic (CTDE):
    - Actor: Decentralized, conditions on local robot observation (28-dim).
    - Critic: Centralized. critic_input selects its input:
        "global"         -> V(s): one fleet value copied to every robot (original design).
        "agent_specific" -> V(s, o_i): global state concatenated with robot i's own observation,
                            so each robot gets its own value for its own per-robot reward
                            (the "agent-specific global state" of Yu et al., 2022).
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
        value_clip: Optional[float] = None,
        critic_input: str = "global",
    ):
        if critic_input not in CRITIC_INPUTS:
            raise ValueError(f"critic_input must be one of {CRITIC_INPUTS}, got {critic_input!r}")
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clip_eps = clip_eps
        self.value_clip = value_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.critic_input = critic_input
        self.base_lr_actor = lr_actor
        self.base_lr_critic = lr_critic

        critic_dim = state_dim + obs_dim if critic_input == "agent_specific" else state_dim
        self.actor = Actor(obs_dim=obs_dim, action_dim=action_dim).to(device)
        self.critic = CentralizedCritic(state_dim=critic_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)

    def set_lr_fraction(self, frac: float):
        for group in self.actor_optimizer.param_groups:
            group["lr"] = self.base_lr_actor * frac
        for group in self.critic_optimizer.param_groups:
            group["lr"] = self.base_lr_critic * frac

    def _values(self, obs: torch.Tensor, state: torch.Tensor, per_sample: bool = False) -> torch.Tensor:
        """
        Rollout mode: obs (..., N, obs_dim), state (..., state_dim) -> values (..., N).
        per_sample mode (training minibatches): obs (B, obs_dim), state (B, state_dim) -> (B,).
        """
        if per_sample:
            x = torch.cat([state, obs], dim=-1) if self.critic_input == "agent_specific" else state
            return self.critic(x).squeeze(-1)
        state_per_agent = state.unsqueeze(-2).expand(*obs.shape[:-1], state.shape[-1])
        if self.critic_input == "agent_specific":
            return self.critic(torch.cat([state_per_agent, obs], dim=-1)).squeeze(-1)
        return self.critic(state).expand(*obs.shape[:-1])

    @torch.no_grad()
    def get_actions_and_values(
        self,
        obs_tensor: torch.Tensor,
        state_tensor: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        obs_tensor shape: (..., num_agents, obs_dim)
        state_tensor shape: (..., state_dim)
        Returns:
            actions: (..., num_agents)
            logprobs: (..., num_agents)
            values: (..., num_agents) -- copied fleet value ("global") or per-robot ("agent_specific")
        """
        actions, logprobs, _ = self.actor.get_action_and_logprob(
            obs_tensor, action_mask=action_mask, deterministic=deterministic
        )
        values = self._values(obs_tensor, state_tensor)
        return actions, logprobs, values

    @torch.no_grad()
    def get_values(self, obs_tensor: torch.Tensor, state_tensor: torch.Tensor) -> torch.Tensor:
        return self._values(obs_tensor, state_tensor)

    def train_step(
        self, buffer: MultiAgentRolloutBuffer, num_epochs: int = 4, batch_size: int = 64
    ) -> Dict[str, float]:
        """Performs PPO training updates on Actor and Centralized Critic."""
        logs = {k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac")}

        for _ in range(num_epochs):
            for mb in buffer.get_batches(batch_size):
                # 1. Update Decentralized Actor
                _, new_logprobs, entropy = self.actor.get_action_and_logprob(
                    mb.obs, mb.actions, action_mask=mb.masks
                )
                policy_loss, approx_kl, clipfrac = policy_loss_fn(
                    new_logprobs, mb.logprobs, mb.advantages, self.clip_eps
                )

                entropy_loss = entropy.mean()
                total_actor_loss = policy_loss - self.entropy_coef * entropy_loss

                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # 2. Update Centralized Critic
                new_values = self._values(mb.obs, mb.states, per_sample=True)
                value_loss = value_loss_fn(new_values, mb.values, mb.returns, self.value_clip)

                self.critic_optimizer.zero_grad()
                (self.value_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                for k, v in zip(logs, (policy_loss, value_loss, entropy_loss, approx_kl, clipfrac)):
                    logs[k].append(v.item())

        return {k: float(torch.tensor(v).mean()) for k, v in logs.items()}

    def save(self, filepath: str, extra: Optional[Dict[str, Any]] = None):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "critic_input": self.critic_input,
            "extra": extra or {},
        }, filepath)

    def load(self, filepath: str, load_optimizer: bool = True) -> Dict[str, Any]:
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        saved_input = checkpoint.get("critic_input", "global")  # checkpoints before Phase 1
        if saved_input != self.critic_input:
            raise ValueError(
                f"Checkpoint critic_input={saved_input!r} but agent was built with {self.critic_input!r}; "
                f"use MAPPOAgent.checkpoint_critic_input(path) to construct a matching agent."
            )
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if load_optimizer:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        return checkpoint.get("extra", {})

    @staticmethod
    def checkpoint_critic_input(filepath: str) -> str:
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        return checkpoint.get("critic_input", "global")
