"""
Unit tests for IPPO and MAPPO agent models, rollout buffers, and optimization loops.
"""

import pytest
import torch
import numpy as np

from environment.warehouse_env import WarehouseEnv
from agents.models import Actor, DecentralizedCritic, CentralizedCritic
from agents.ippo_agent import IPPOAgent
from agents.mappo_agent import MAPPOAgent
from training.buffer import MultiAgentRolloutBuffer


def test_models_forward():
    obs_dim = 28
    state_dim = 150
    num_agents = 5
    batch_size = 8

    actor = Actor(obs_dim=obs_dim, action_dim=6)
    dec_critic = DecentralizedCritic(obs_dim=obs_dim)
    cent_critic = CentralizedCritic(state_dim=state_dim)

    dummy_obs = torch.randn(batch_size, num_agents, obs_dim)
    flat_obs = dummy_obs.view(-1, obs_dim)

    # Actor forward
    actions, logprobs, entropy = actor.get_action_and_logprob(flat_obs)
    assert actions.shape == (batch_size * num_agents,)
    assert logprobs.shape == (batch_size * num_agents,)
    assert entropy.shape == (batch_size * num_agents,)

    # Decentralized Critic forward
    values = dec_critic(flat_obs)
    assert values.shape == (batch_size * num_agents, 1)

    # Centralized Critic forward
    dummy_states = torch.randn(batch_size, state_dim)
    cent_values = cent_critic(dummy_states)
    assert cent_values.shape == (batch_size, 1)


def test_ippo_step_and_buffer():
    env = WarehouseEnv(num_robots=5, grid_width=10, grid_height=10, max_steps=50)
    agent = IPPOAgent(obs_dim=28, action_dim=6)
    buffer = MultiAgentRolloutBuffer(
        buffer_size=16, num_agents=5, obs_dim=28, state_dim=env.state_dim
    )

    obs_dict, _ = env.reset(seed=42)
    obs_array = np.stack([obs_dict[a] for a in env.possible_agents])
    obs_tensor = torch.tensor(obs_array, dtype=torch.float32)

    for _ in range(16):
        actions, logprobs, values = agent.get_actions_and_values(obs_tensor)
        action_dict = {f"robot_{i}": int(actions[i]) for i in range(5)}
        next_obs, rewards, term, trunc, _ = env.step(action_dict)

        reward_tensor = torch.tensor([rewards[a] for a in env.possible_agents], dtype=torch.float32)
        done_tensor = torch.tensor([float(trunc[a]) for a in env.possible_agents], dtype=torch.float32)
        state_tensor = torch.tensor(env.state(), dtype=torch.float32)

        buffer.insert(
            obs=obs_tensor,
            state=state_tensor,
            action=actions,
            logprob=logprobs,
            reward=reward_tensor,
            done=done_tensor,
            value=values,
        )
        obs_array = np.stack([next_obs[a] for a in env.possible_agents])
        obs_tensor = torch.tensor(obs_array, dtype=torch.float32)

    assert buffer.is_full
    _, _, last_values = agent.get_actions_and_values(obs_tensor)
    buffer.compute_returns_and_advantages(last_values, done_tensor)

    metrics = agent.train_step(buffer, num_epochs=2, batch_size=32)
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics


def test_mappo_step_and_buffer():
    env = WarehouseEnv(num_robots=5, grid_width=10, grid_height=10, max_steps=50)
    agent = MAPPOAgent(obs_dim=28, state_dim=env.state_dim, action_dim=6)
    buffer = MultiAgentRolloutBuffer(
        buffer_size=16, num_agents=5, obs_dim=28, state_dim=env.state_dim
    )

    obs_dict, _ = env.reset(seed=42)
    obs_array = np.stack([obs_dict[a] for a in env.possible_agents])
    obs_tensor = torch.tensor(obs_array, dtype=torch.float32)
    state_tensor = torch.tensor(env.state(), dtype=torch.float32)

    for _ in range(16):
        actions, logprobs, values = agent.get_actions_and_values(obs_tensor, state_tensor)
        action_dict = {f"robot_{i}": int(actions[i]) for i in range(5)}
        next_obs, rewards, term, trunc, _ = env.step(action_dict)

        reward_tensor = torch.tensor([rewards[a] for a in env.possible_agents], dtype=torch.float32)
        done_tensor = torch.tensor([float(trunc[a]) for a in env.possible_agents], dtype=torch.float32)
        curr_state = torch.tensor(env.state(), dtype=torch.float32)

        buffer.insert(
            obs=obs_tensor,
            state=curr_state,
            action=actions,
            logprob=logprobs,
            reward=reward_tensor,
            done=done_tensor,
            value=values,
        )
        obs_array = np.stack([next_obs[a] for a in env.possible_agents])
        obs_tensor = torch.tensor(obs_array, dtype=torch.float32)
        state_tensor = curr_state

    assert buffer.is_full
    _, _, last_values = agent.get_actions_and_values(obs_tensor, state_tensor)
    buffer.compute_returns_and_advantages(last_values, done_tensor)

    metrics = agent.train_step(buffer, num_epochs=2, batch_size=32)
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
