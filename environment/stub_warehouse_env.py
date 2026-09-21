import numpy as np
import gymnasium as gym
from gymnasium import spaces

R_COMPLETE = 10.0
R_STEP_PROGRESS = 0.1
R_COLLISION = -1.0
R_IDLE_PENALTY = -0.05
R_TIME_PENALTY = -0.01

STATUS_IDLE = 0
STATUS_CARRYING = 1
STATUS_EN_ROUTE = 2


class StubWarehouseEnv(gym.Env):
    def __init__(self, grid_size=8, max_steps=200, seed=None):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps

        # TODO 1: fixed delivery station in the bottom-right corner
        self.station_pos = np.array([grid_size - 1, grid_size - 1], dtype=np.int32)

        # TODO 2: vertical wall down the middle, with one gap so robots can pass through
        self.walls = {(grid_size // 2, y) for y in range(1, grid_size - 1)}
        self.walls.discard((grid_size // 2, grid_size // 2))

        # TODO 3: observation is 13 numbers, each normalized to the range [0, 1]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(13,), dtype=np.float32)

        # TODO 4: 6 discrete actions: up, down, left, right, wait, accept
        self.action_space = spaces.Discrete(6)

        self._rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    def reset(self, *, seed=None, options=None):
        # TODO 5: pick a random free robot start position (not on a wall/station)
        # TODO 6: set self.status = STATUS_IDLE
        # TODO 7: pick a random free task_target (the shelf location)
        # TODO 8: set self.carrying = False, self.steps = 0
        # TODO 9: store self._prev_dist = manhattan distance from robot to task_target
        pass

    def step(self, action):
        # TODO 10-16: see previous message for full breakdown
        pass

    def _get_obs(self):
        # TODO 17: build and return the 13-dim numpy array
        pass