"""
Unit tests for the warehouse multi-agent environment, collision logic, and renderer.
"""

import pytest
import numpy as np
from environment.grid import WarehouseGrid
from environment.warehouse_env import WarehouseEnv
from environment.gym_wrapper import MultiAgentWarehouseGymEnv
from visualization.renderer import WarehouseVisualizer


def test_grid_initialization():
    grid = WarehouseGrid(width=10, height=10, layout_type="kiva_standard")
    assert grid.width == 10
    assert grid.height == 10
    assert len(grid.shelves) > 0
    assert len(grid.stations) >= 2
    
    # Station coordinates should be at bottom
    for sx, sy in grid.stations:
        assert sy == 9
        assert grid.is_station(sx, sy)

    # Shelf cells should not be walkable for transit
    sample_shelf = list(grid.shelves)[0]
    assert not grid.is_walkable(sample_shelf[0], sample_shelf[1])

    # Shelf access cells should be walkable
    access_cells = grid.get_shelf_access_cells(sample_shelf)
    assert len(access_cells) > 0
    for ax, ay in access_cells:
        assert grid.is_walkable(ax, ay)


def test_environment_reset_and_dimensions():
    env = WarehouseEnv(num_robots=5, grid_width=10, grid_height=10)
    obs, infos = env.reset(seed=42)

    assert len(env.agents) == 5
    for agent in env.possible_agents:
        assert agent in obs
        assert obs[agent].shape == (28,)
        # Normalized values check
        assert np.all(obs[agent] >= -1.0) and np.all(obs[agent] <= 1.0)

    # Centralized state dimension check
    state = env.state()
    assert state.shape == (5 * 28 + 10,)


def test_environment_step_random_actions():
    env = WarehouseEnv(num_robots=5, grid_width=10, grid_height=10, max_steps=50)
    obs, _ = env.reset(seed=42)

    for step in range(20):
        actions = {agent: np.random.randint(0, 6) for agent in env.agents}
        obs, rewards, term, trunc, infos = env.step(actions)
        
        for agent in env.agents:
            assert agent in obs
            assert obs[agent].shape == (28,)
            assert isinstance(rewards[agent], (float, np.floating))
            assert not term[agent]
            assert not trunc[agent]

    assert env.current_step == 20


def test_gym_wrapper():
    gym_env = MultiAgentWarehouseGymEnv(num_robots=5, grid_width=10, grid_height=10)
    joint_obs, info = gym_env.reset(seed=123)

    assert joint_obs.shape == (5, 28)
    
    # Take step with joint actions
    actions = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    next_obs, rewards, term, trunc, info = gym_env.step(actions)

    assert next_obs.shape == (5, 28)
    assert rewards.shape == (5,)
    assert "mean_reward" in info
    assert "global_state" in info
    assert info["global_state"].shape == (5 * 28 + 10,)


def test_renderer_rgb_array():
    env = WarehouseEnv(num_robots=5, grid_width=10, grid_height=10, render_mode="rgb_array")
    env.reset(seed=1)
    visualizer = WarehouseVisualizer(env, cell_size=40)
    frame = visualizer.render()
    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3
    assert frame.shape[2] == 3  # RGB channels
    visualizer.close()
