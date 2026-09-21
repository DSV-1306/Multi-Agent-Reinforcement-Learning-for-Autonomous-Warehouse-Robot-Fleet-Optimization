"""
Training script for Independent PPO (IPPO) in Warehouse Multi-Agent Environment.
"""

import os
import time
import argparse
import yaml
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.warehouse_env import WarehouseEnv
from agents.ippo_agent import IPPOAgent
from training.buffer import MultiAgentRolloutBuffer


def train_ippo(config_path: str = "configs/default_config.yaml", total_timesteps: int = None, seed: int = 42):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["environment"]
    hp_cfg = cfg["hyperparameters"]
    ippo_cfg = cfg["ippo"]

    if total_timesteps is None:
        total_timesteps = ippo_cfg["total_timesteps"]

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = f"results/logs/ippo_seed_{seed}_{int(time.time())}"
    writer = SummaryWriter(log_dir)
    print(f"--> [IPPO] Starting training on device: {device}")
    print(f"--> [IPPO] Logging to: {log_dir}")

    env = WarehouseEnv(
        num_robots=env_cfg["num_robots"],
        grid_width=env_cfg["grid_width"],
        grid_height=env_cfg["grid_height"],
        layout_type=env_cfg["layout_type"],
        order_lambda=env_cfg["order_lambda"],
        max_steps=env_cfg["max_steps"],
        seed=seed,
    )

    agent = IPPOAgent(
        obs_dim=env.obs_dim,
        action_dim=6,
        lr=ippo_cfg["lr"],
        clip_eps=hp_cfg["clip_eps"],
        value_coef=hp_cfg["value_coef"],
        entropy_coef=hp_cfg["entropy_coef"],
        max_grad_norm=hp_cfg["max_grad_norm"],
        device=device,
    )

    buffer_size = ippo_cfg["buffer_size"]
    buffer = MultiAgentRolloutBuffer(
        buffer_size=buffer_size,
        num_agents=env.num_robots,
        obs_dim=env.obs_dim,
        state_dim=env.state_dim,
        gamma=hp_cfg["gamma"],
        gae_lambda=hp_cfg["gae_lambda"],
        device=device,
    )

    obs_dict, _ = env.reset(seed=seed)
    obs_array = np.stack([obs_dict[a] for a in env.possible_agents])
    obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=device)

    episode_rewards = []
    episode_collisions = []
    episode_throughputs = []
    curr_ep_reward = 0.0

    global_step = 0
    num_updates = total_timesteps // buffer_size

    best_mean_reward = -float("inf")
    save_path = "results/models/ippo_best.pt"
    os.makedirs("results/models", exist_ok=True)

    start_time = time.time()

    for update in range(1, num_updates + 1):
        buffer.reset()

        for step in range(buffer_size):
            global_step += 1
            actions, logprobs, values = agent.get_actions_and_values(obs_tensor)
            action_dict = {f"robot_{i}": int(actions[i]) for i in range(env.num_robots)}

            next_obs_dict, rewards_dict, term_dict, trunc_dict, infos = env.step(action_dict)

            reward_tensor = torch.tensor(
                [rewards_dict[a] for a in env.possible_agents], dtype=torch.float32, device=device
            )
            done_tensor = torch.tensor(
                [float(trunc_dict[a]) for a in env.possible_agents], dtype=torch.float32, device=device
            )
            state_tensor = torch.tensor(env.state(), dtype=torch.float32, device=device)

            buffer.insert(
                obs=obs_tensor,
                state=state_tensor,
                action=actions,
                logprob=logprobs,
                reward=reward_tensor,
                done=done_tensor,
                value=values,
            )

            curr_ep_reward += float(reward_tensor.mean().item())

            if all(trunc_dict.values()):
                episode_rewards.append(curr_ep_reward)
                episode_collisions.append(env.total_fleet_collisions)
                episode_throughputs.append(env.total_orders_completed)

                writer.add_scalar("rollout/ep_reward", curr_ep_reward, global_step)
                writer.add_scalar("rollout/ep_throughput", env.total_orders_completed, global_step)
                writer.add_scalar("rollout/ep_collisions", env.total_fleet_collisions, global_step)

                curr_ep_reward = 0.0
                next_obs_dict, _ = env.reset()

            obs_array = np.stack([next_obs_dict[a] for a in env.possible_agents])
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=device)

        # Compute advantages
        _, _, last_values = agent.get_actions_and_values(obs_tensor)
        buffer.compute_returns_and_advantages(last_values, done_tensor)

        # Optimize network
        metrics = agent.train_step(
            buffer, num_epochs=ippo_cfg["num_epochs"], batch_size=ippo_cfg["batch_size"]
        )

        # Tensorboard logging
        writer.add_scalar("train/policy_loss", metrics["policy_loss"], global_step)
        writer.add_scalar("train/value_loss", metrics["value_loss"], global_step)
        writer.add_scalar("train/entropy", metrics["entropy"], global_step)

        if update % 5 == 0 or update == num_updates:
            avg_rew = np.mean(episode_rewards[-5:]) if episode_rewards else 0.0
            avg_thru = np.mean(episode_throughputs[-5:]) if episode_throughputs else 0.0
            avg_coll = np.mean(episode_collisions[-5:]) if episode_collisions else 0.0
            sps = int(global_step / (time.time() - start_time))

            print(
                f"Update [{update}/{num_updates}] Step {global_step}/{total_timesteps} | "
                f"Reward: {avg_rew:.2f} | Throughput: {avg_thru:.1f} | Collisions: {avg_coll:.1f} | SPS: {sps}"
            )

            if avg_rew > best_mean_reward:
                best_mean_reward = avg_rew
                agent.save(save_path)

    agent.save("results/models/ippo_final.pt")
    writer.close()
    print(f"--> [IPPO] Training complete! Best model saved to: {save_path}")
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=15000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_ippo(total_timesteps=args.timesteps, seed=args.seed)
