"""
Unit test and evaluation for the Space-Time A* Heuristic Baseline.
"""

import pytest
from environment.warehouse_env import WarehouseEnv
from baselines.heuristic_agent import HeuristicFleetController


def test_heuristic_baseline_execution():
    env = WarehouseEnv(num_robots=5, grid_width=10, grid_height=10, max_steps=100)
    obs, _ = env.reset(seed=42)

    controller = HeuristicFleetController(env)
    controller.reset()

    total_rewards = {agent: 0.0 for agent in env.possible_agents}

    for step in range(100):
        actions = controller.get_actions()
        obs, rewards, term, trunc, infos = env.step(actions)

        for agent, r in rewards.items():
            total_rewards[agent] += r

        if all(trunc.values()):
            break

    assert env.current_step == 100
    # Heuristic baseline should successfully deliver multiple orders
    assert env.total_orders_completed > 0
    # Collisions should be minimal or 0 due to reservation table
    print(f"Baseline completed {env.total_orders_completed} orders with {env.total_fleet_collisions} collisions.")
