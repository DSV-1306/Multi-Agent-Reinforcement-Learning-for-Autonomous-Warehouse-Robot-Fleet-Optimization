"""
Comprehensive Head-to-Head Benchmarking Suite:
Compares Heuristic Baseline vs IPPO vs MAPPO across controlled benchmark seeds.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.warehouse_env import WarehouseEnv
from baselines.heuristic_agent import HeuristicFleetController
from agents.ippo_agent import IPPOAgent
from agents.mappo_agent import MAPPOAgent
from evaluation.metrics import MetricsTracker


def evaluate_baseline(env: WarehouseEnv, seeds: list) -> MetricsTracker:
    tracker = MetricsTracker()
    controller = HeuristicFleetController(env)

    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        controller.reset()
        ep_return = 0.0

        for step in range(env.max_steps):
            actions = controller.get_actions()
            obs, rewards, term, trunc, _ = env.step(actions)
            ep_return += float(np.mean([rewards[a] for a in env.possible_agents]))
            if all(trunc.values()):
                break

        tracker.record_episode(
            orders=env.order_generator.completed_orders,
            robots=env.robots,
            fleet_collisions=env.total_fleet_collisions,
            orders_completed=env.total_orders_completed,
            total_steps=env.current_step,
            mean_return=ep_return,
        )

    return tracker


def evaluate_ippo(env: WarehouseEnv, agent: IPPOAgent, seeds: list) -> MetricsTracker:
    tracker = MetricsTracker()
    device = agent.device

    for seed in seeds:
        obs_dict, _ = env.reset(seed=seed)
        ep_return = 0.0

        for step in range(env.max_steps):
            obs_array = np.stack([obs_dict[a] for a in env.possible_agents])
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=device)

            with torch.no_grad():
                actions, _, _ = agent.get_actions_and_values(obs_tensor)

            action_dict = {f"robot_{i}": int(actions[i]) for i in range(env.num_robots)}
            obs_dict, rewards, term, trunc, _ = env.step(action_dict)
            ep_return += float(np.mean([rewards[a] for a in env.possible_agents]))

            if all(trunc.values()):
                break

        tracker.record_episode(
            orders=env.order_generator.completed_orders,
            robots=env.robots,
            fleet_collisions=env.total_fleet_collisions,
            orders_completed=env.total_orders_completed,
            total_steps=env.current_step,
            mean_return=ep_return,
        )

    return tracker


def evaluate_mappo(env: WarehouseEnv, agent: MAPPOAgent, seeds: list) -> MetricsTracker:
    tracker = MetricsTracker()
    device = agent.device

    for seed in seeds:
        obs_dict, _ = env.reset(seed=seed)
        ep_return = 0.0

        for step in range(env.max_steps):
            obs_array = np.stack([obs_dict[a] for a in env.possible_agents])
            obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=device)
            state_tensor = torch.tensor(env.state(), dtype=torch.float32, device=device)

            with torch.no_grad():
                actions, _, _ = agent.get_actions_and_values(obs_tensor, state_tensor)

            action_dict = {f"robot_{i}": int(actions[i]) for i in range(env.num_robots)}
            obs_dict, rewards, term, trunc, _ = env.step(action_dict)
            ep_return += float(np.mean([rewards[a] for a in env.possible_agents]))

            if all(trunc.values()):
                break

        tracker.record_episode(
            orders=env.order_generator.completed_orders,
            robots=env.robots,
            fleet_collisions=env.total_fleet_collisions,
            orders_completed=env.total_orders_completed,
            total_steps=env.current_step,
            mean_return=ep_return,
        )

    return tracker


def run_benchmarks(
    num_episodes: int = 10,
    order_lambda: float = 0.2,
    ippo_model_path: str = "results/models/ippo_best.pt",
    mappo_model_path: str = "results/models/mappo_best.pt",
    output_csv: str = "results/evaluation_results.csv",
):
    seeds = [100 + i for i in range(num_episodes)]
    env = WarehouseEnv(num_robots=5, grid_width=10, grid_height=10, order_lambda=order_lambda, max_steps=300)

    results = []

    # 1. Evaluate Baseline
    print("--> Evaluating Heuristic Baseline (Greedy + Space-Time A*)...")
    baseline_tracker = evaluate_baseline(env, seeds)
    b_summary = baseline_tracker.summary()
    b_summary["Method"] = "Baseline (A*)"
    b_summary["Lambda"] = order_lambda
    results.append(b_summary)

    # 2. Evaluate IPPO
    if os.path.exists(ippo_model_path):
        print(f"--> Evaluating IPPO (Loading: {ippo_model_path})...")
        ippo_agent = IPPOAgent(obs_dim=env.obs_dim, action_dim=6)
        ippo_agent.load(ippo_model_path)
        ippo_tracker = evaluate_ippo(env, ippo_agent, seeds)
        i_summary = ippo_tracker.summary()
        i_summary["Method"] = "IPPO"
        i_summary["Lambda"] = order_lambda
        results.append(i_summary)
    else:
        print(f"(!) IPPO checkpoint not found at {ippo_model_path}, skipping.")

    # 3. Evaluate MAPPO
    if os.path.exists(mappo_model_path):
        print(f"--> Evaluating MAPPO (Loading: {mappo_model_path})...")
        mappo_agent = MAPPOAgent(obs_dim=env.obs_dim, state_dim=env.state_dim, action_dim=6)
        mappo_agent.load(mappo_model_path)
        mappo_tracker = evaluate_mappo(env, mappo_agent, seeds)
        m_summary = mappo_tracker.summary()
        m_summary["Method"] = "MAPPO"
        m_summary["Lambda"] = order_lambda
        results.append(m_summary)
    else:
        print(f"(!) MAPPO checkpoint not found at {mappo_model_path}, skipping.")

    df = pd.DataFrame(results)
    # Reorder columns
    cols = ["Method", "Lambda", "avg_throughput", "std_throughput", "avg_completion_time", "std_completion_time", "avg_collisions", "std_collisions", "avg_travel_distance", "avg_utilization", "avg_return"]
    df = df[[c for c in cols if c in df.columns]]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print("\n========================= BENCHMARK RESULTS =========================")
    print(df.to_string(index=False))
    print(f"\n--> Saved full results to: {output_csv}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10, help="Number of test episodes per method")
    parser.add_argument("--order_lambda", type=float, default=0.2, help="Order arrival rate")
    args = parser.parse_args()

    run_benchmarks(num_episodes=args.episodes, order_lambda=args.order_lambda)
