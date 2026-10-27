"""
Warehouse Multi-Agent Environment conforming to PettingZoo ParallelEnv standard.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv

from environment.grid import WarehouseGrid
from environment.robot import WarehouseRobot, RobotStatus
from environment.order import OrderGenerator, Order

# Reward terms per robot per step. Values equal the original hardcoded ones; configs override them.
DEFAULT_REWARDS: Dict[str, Any] = {
    "time": -0.02,             # every step
    "collision_wall": -5.0,    # move blocked by shelf / wall / boundary
    "collision_robot": -5.0,   # move blocked by a vertex or swap conflict with another robot
    "progress": 0.2,           # x (previous Manhattan distance - current) to the active target
    "idle": -0.05,             # waiting while a target is assigned
    "pickup": 3.0,
    "delivery": 10.0,
    # Phase 0 finding F1: action 5 without effect used to be a penalty-free wait.
    # True charges it the idle penalty; False restores the original behaviour.
    "penalize_noop_interact": True,
}

ACTION_WAIT = 4
ACTION_INTERACT = 5
NUM_ACTIONS = 6
# Movement direction mappings: 0=N, 1=S, 2=E, 3=W, 4=Wait, 5=Interact
ACTION_DELTAS = {0: (0, -1), 1: (0, 1), 2: (1, 0), 3: (-1, 0), 4: (0, 0), 5: (0, 0)}


class WarehouseEnv(ParallelEnv):
    """
    PettingZoo Parallel Multi-Agent Warehouse Environment.
    Simulates autonomous mobile robots (AMRs) transporting items from shelves to packing stations.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "warehouse_marl_v1"}

    def __init__(
        self,
        num_robots: int = 5,
        grid_width: int = 10,
        grid_height: int = 10,
        layout_type: str = "kiva_standard",
        order_lambda: float = 0.2,
        max_steps: int = 400,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
        rewards: Optional[Dict[str, Any]] = None,
        mask_useless_interact: bool = True,
        layout_spec: Optional[Dict[str, Any]] = None,
        state_max_robots: Optional[int] = None,
        dispatch_mode: str = "auto",
        order_config: Optional[Dict[str, Any]] = None,
    ):
        """
        layout_spec: explicit layout (see WarehouseGrid); overrides layout_type/grid_width/grid_height.
        state_max_robots: pad the MAPPO global state to this many robots so the state size stays fixed
            when the robot count changes (curriculum, held-out tests). Default: num_robots (original size).
        dispatch_mode: "auto" = env gives each idle robot its nearest pending order (original);
            "external" = orders wait until assign_order() is called (hierarchical RL allocator).
        order_config: extra OrderGenerator options, e.g. {"mode": "empirical", "empirical": {...}}.
        """
        super().__init__()
        if dispatch_mode not in ("auto", "external"):
            raise ValueError(f"dispatch_mode must be 'auto' or 'external', got {dispatch_mode!r}")
        self.dispatch_mode = dispatch_mode
        # External allocator: callable(env) that calls env.assign_order(); set as an attribute so env
        # kwargs stay plain data.
        self.dispatcher = None
        self.reward_cfg = {**DEFAULT_REWARDS, **(rewards or {})}
        unknown = set(self.reward_cfg) - set(DEFAULT_REWARDS)
        if unknown:
            raise ValueError(f"Unknown reward keys: {sorted(unknown)}")
        self.mask_useless_interact = mask_useless_interact
        self.num_robots = num_robots
        if layout_spec:
            grid_width, grid_height = layout_spec["width"], layout_spec["height"]
            layout_type = layout_spec.get("name", "custom")
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.layout_type = layout_type
        self.order_lambda = order_lambda
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Core grid
        self.grid = WarehouseGrid(width=grid_width, height=grid_height, layout_type=layout_type, spec=layout_spec)
        
        # Agents setup
        self.possible_agents = [f"robot_{i}" for i in range(num_robots)]
        self.agents = list(self.possible_agents)
        self.agent_name_to_idx = {name: i for i, name in enumerate(self.possible_agents)}

        # Observation space: 28 dimensions per robot
        self.obs_dim = 28
        self._observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self.observation_spaces = {agent: self._observation_space for agent in self.possible_agents}

        # Action space: 6 discrete actions
        # 0: North, 1: South, 2: East, 3: West, 4: Wait, 5: Interact (Pickup/Dropoff)
        self._action_space = spaces.Discrete(NUM_ACTIONS)
        self.action_spaces = {agent: self._action_space for agent in self.possible_agents}

        # Global state dimension for centralized critic (MAPPO)
        # state_max_robots * obs_dim + 10 global features (robots beyond num_robots are zero padding)
        self.state_max_robots = state_max_robots or num_robots
        if self.state_max_robots < num_robots:
            raise ValueError("state_max_robots must be >= num_robots")
        self.state_dim = self.state_max_robots * self.obs_dim + 10

        # Robot instances
        self.robots: List[WarehouseRobot] = []
        walkable = self.grid.get_walkable_cells()
        rng = np.random.default_rng(seed)
        start_indices = rng.choice(len(walkable), size=num_robots, replace=False)
        for i in range(num_robots):
            pos = walkable[start_indices[i]]
            self.robots.append(WarehouseRobot(robot_id=i, initial_pos=pos))

        # Order generator
        self.order_generator = OrderGenerator(
            shelves=list(self.grid.shelves),
            stations=self.grid.stations,
            grid_access_fn=self.grid.get_shelf_access_cells,
            lambda_rate=order_lambda,
            seed=seed,
            max_steps=max_steps,
            **(order_config or {}),
        )

        self.current_step = 0
        self.total_fleet_collisions = 0
        self.total_wall_bumps = 0
        self.total_robot_collisions = 0
        self.total_orders_completed = 0
        self._rng = np.random.default_rng(seed)

        # Pygame renderer placeholder
        self._visualizer = None

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.agents = list(self.possible_agents)
        self.current_step = 0
        self.total_fleet_collisions = 0
        self.total_wall_bumps = 0
        self.total_robot_collisions = 0
        self.total_orders_completed = 0

        # Reset robots to free walkable cells
        walkable = [c for c in self.grid.get_walkable_cells() if not self.grid.is_station(c[0], c[1])]
        start_indices = self._rng.choice(len(walkable), size=self.num_robots, replace=False)
        for i, robot in enumerate(self.robots):
            pos = walkable[start_indices[i]]
            robot.reset(start_pos=pos)

        # Reset order generator
        self.order_generator.reset(seed=seed)

        # Generate initial batch of orders and assign to idle robots
        self.order_generator.generate_orders_for_step(current_step=0)
        self._auto_dispatch_orders()

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {"action_mask": self.action_mask(agent)} for agent in self.agents}

        return observations, infos

    def action_mask(self, agent: str) -> np.ndarray:
        """
        Boolean mask over the 6 actions (True = allowed).
        Moves into static obstacles (shelves, walls, outside the grid) are masked; other robots are
        NOT masked, so avoiding them remains the policy's job. Wait is always allowed. Interact is
        allowed only when it can pick up from an adjacent access cell (if mask_useless_interact).
        """
        robot = self.robots[self.agent_name_to_idx[agent]]
        mask = np.ones(NUM_ACTIONS, dtype=bool)
        for act in range(ACTION_WAIT):
            dx, dy = ACTION_DELTAS[act]
            mask[act] = self.grid.is_walkable(robot.x + dx, robot.y + dy)
        if self.mask_useless_interact:
            mask[ACTION_INTERACT] = self._interact_can_pick_up(robot)
        return mask

    def action_masks(self) -> np.ndarray:
        """Masks for all robots, shape (num_robots, 6)."""
        return np.stack([self.action_mask(a) for a in self.possible_agents])

    def _interact_can_pick_up(self, robot: WarehouseRobot) -> bool:
        return (
            robot.status == RobotStatus.EN_ROUTE_TO_SHELF
            and robot.target_pos is not None
            and robot.pos != robot.target_pos
            and self.grid.manhattan_distance(robot.pos, robot.target_pos) <= 1
        )

    def _resolve_moves(self, actions: Dict[str, int]):
        """
        Returns (final positions, wall-bump flags, robot-conflict flags).
        Blocked robots stay in place. Vertex and swap checks are repeated until no robot changes,
        because reverting one robot can create a new conflict (Phase 0 finding F3).
        """
        current = {i: self.robots[i].pos for i in range(self.num_robots)}
        proposed: Dict[int, Tuple[int, int]] = {}
        wall_bump = {i: False for i in range(self.num_robots)}
        robot_conflict = {i: False for i in range(self.num_robots)}

        for agent in self.agents:
            idx = self.agent_name_to_idx[agent]
            dx, dy = ACTION_DELTAS.get(actions.get(agent, ACTION_WAIT), (0, 0))
            target = (current[idx][0] + dx, current[idx][1] + dy)
            if not self.grid.is_walkable(*target):
                wall_bump[idx] = True
                target = current[idx]
            proposed[idx] = target
        for idx in range(self.num_robots):
            proposed.setdefault(idx, current[idx])

        changed = True
        while changed:
            changed = False
            # Vertex conflicts: several robots end in one cell -> every mover in it is blocked
            occupants: Dict[Tuple[int, int], List[int]] = {}
            for idx, pos in proposed.items():
                occupants.setdefault(pos, []).append(idx)
            for group in occupants.values():
                if len(group) > 1:
                    for idx in group:
                        if proposed[idx] != current[idx]:
                            robot_conflict[idx] = True
                            proposed[idx] = current[idx]
                            changed = True
            # Swap (edge) conflicts: two robots exchange cells
            for i in range(self.num_robots):
                for j in range(i + 1, self.num_robots):
                    if (
                        proposed[i] == current[j]
                        and proposed[j] == current[i]
                        and proposed[i] != current[i]
                    ):
                        robot_conflict[i] = robot_conflict[j] = True
                        proposed[i], proposed[j] = current[i], current[j]
                        changed = True

        return proposed, wall_bump, robot_conflict

    def _auto_dispatch_orders(self):
        """Assigns oldest pending orders to nearest available idle robots."""
        if self.dispatch_mode == "external":
            # The allocator is called at exactly the moments the built-in nearest-order rule would run
            # (reset, start of each step, after each delivery), so the two are directly comparable.
            if self.dispatcher is not None:
                self.dispatcher(self)
            return
        idle_robots = [r for r in self.robots if r.is_idle]
        if not idle_robots or not self.order_generator.pending_orders:
            return

        for robot in idle_robots:
            if not self.order_generator.pending_orders:
                break
            # Find closest unassigned pending order
            best_idx = -1
            best_dist = float("inf")
            for idx, order in enumerate(self.order_generator.pending_orders):
                d = self.grid.manhattan_distance(robot.pos, order.shelf_access_pos)
                if d < best_dist:
                    best_dist = d
                    best_idx = idx

            if best_idx >= 0:
                order = self.order_generator.pending_orders.pop(best_idx)
                order.assigned_robot_id = robot.id
                self.order_generator.active_orders[order.id] = order
                robot.assign_order(order.id, order.shelf_access_pos)

    def assign_order(self, robot_idx: int, pending_idx: int) -> None:
        """External dispatch: give pending order `pending_idx` to idle robot `robot_idx`."""
        robot = self.robots[robot_idx]
        if not robot.is_idle:
            raise ValueError(f"robot {robot_idx} is not idle")
        order = self.order_generator.pending_orders.pop(pending_idx)
        order.assigned_robot_id = robot.id
        self.order_generator.active_orders[order.id] = order
        robot.assign_order(order.id, order.shelf_access_pos)

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        self.current_step += 1

        # 1. Spawn dynamic orders
        self.order_generator.generate_orders_for_step(self.current_step)
        self._auto_dispatch_orders()

        # 2. Movement and collision resolution
        action_intent = {
            self.agent_name_to_idx[agent]: actions.get(agent, ACTION_WAIT) for agent in self.agents
        }
        final_pos, wall_bump, robot_conflict = self._resolve_moves(actions)
        collision_detected = {i: wall_bump[i] or robot_conflict[i] for i in range(self.num_robots)}

        for idx in range(self.num_robots):
            new_pos = final_pos[idx]
            if wall_bump[idx]:
                self.robots[idx].log_collision("wall")
                self.total_wall_bumps += 1
            elif robot_conflict[idx]:
                self.robots[idx].log_collision("robot")
                self.total_robot_collisions += 1
            if collision_detected[idx]:
                self.total_fleet_collisions += 1
            if new_pos != self.robots[idx].pos:
                self.robots[idx].move_to(new_pos[0], new_pos[1])
            else:
                self.robots[idx].wait()

        # 3. Task interactions & rewards
        rc = self.reward_cfg
        rewards: Dict[str, float] = {agent: 0.0 for agent in self.agents}

        for agent in self.agents:
            idx = self.agent_name_to_idx[agent]
            robot = self.robots[idx]
            act = action_intent[idx]
            step_reward = rc["time"]

            if wall_bump[idx]:
                step_reward += rc["collision_wall"]
            elif robot_conflict[idx]:
                step_reward += rc["collision_robot"]

            # Shaping towards the active target
            had_target = robot.target_pos is not None
            if had_target:
                curr_dist = self.grid.manhattan_distance(robot.pos, robot.target_pos)
                if robot.prev_target_dist is not None:
                    step_reward += rc["progress"] * (robot.prev_target_dist - curr_dist)
                robot.prev_target_dist = curr_dist

            interact_useful = False
            # Pickup at shelf: automatic on the access cell, or Interact from an adjacent cell
            if robot.status == RobotStatus.EN_ROUTE_TO_SHELF and robot.target_pos is not None:
                interact_useful = act == ACTION_INTERACT and self._interact_can_pick_up(robot)
                if robot.pos == robot.target_pos or interact_useful:
                    order = self.order_generator.active_orders.get(robot.current_order_id)
                    if order:
                        robot.pickup_item(station_pos=order.station_pos)
                        self.order_generator.mark_picked_up(order.id, self.current_step)
                        step_reward += rc["pickup"]

            # Delivery at workstation: automatic on arrival
            elif robot.status == RobotStatus.CARRYING_TO_STATION and robot.target_pos is not None:
                if robot.pos == robot.target_pos:
                    delivered_order = self.order_generator.mark_delivered(robot.current_order_id, self.current_step)
                    if delivered_order:
                        robot.complete_delivery()
                        self.total_orders_completed += 1
                        step_reward += rc["delivery"]
                        # Auto-dispatch any pending order to this newly freed robot
                        self._auto_dispatch_orders()

            # Idle penalty: waiting (or a no-effect Interact) while a target is assigned
            is_wait = act == ACTION_WAIT or (
                act == ACTION_INTERACT and rc["penalize_noop_interact"] and not interact_useful
            )
            if had_target and is_wait and not collision_detected[idx]:
                step_reward += rc["idle"]

            rewards[agent] = step_reward

        # 4. Check Termination and Truncation
        truncated = self.current_step >= self.max_steps
        terminated = False  # Continuous fleet operation until max_steps

        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}

        if truncated:
            self.agents = []

        observations = {agent: self._get_obs(agent) for agent in self.possible_agents}
        infos = {
            agent: {
                "orders_completed": self.robots[self.agent_name_to_idx[agent]].deliveries_completed,
                "collisions": self.robots[self.agent_name_to_idx[agent]].collisions,
                "fleet_throughput": self.total_orders_completed,
                "fleet_collisions": self.total_fleet_collisions,
                "fleet_wall_bumps": self.total_wall_bumps,
                "fleet_robot_collisions": self.total_robot_collisions,
                "action_mask": self.action_mask(agent),
            }
            for agent in self.possible_agents
        }

        return observations, rewards, terminations, truncations, infos

    def _get_obs(self, agent: str) -> np.ndarray:
        """Constructs 28-dimensional normalized observation vector for robot."""
        idx = self.agent_name_to_idx[agent]
        robot = self.robots[idx]
        w, h = float(self.grid_width), float(self.grid_height)

        obs = np.zeros(self.obs_dim, dtype=np.float32)

        # 1. Self status (6 dims)
        obs[0] = robot.x / w
        obs[1] = robot.y / h
        obs[2] = float(robot.status) / 2.0  # 0.0=Idle, 0.5=En route, 1.0=Carrying
        if robot.target_pos is not None:
            tx, ty = robot.target_pos
            obs[3] = tx / w
            obs[4] = ty / h
            dist = self.grid.manhattan_distance(robot.pos, robot.target_pos)
            obs[5] = dist / (w + h)
        else:
            obs[3] = robot.x / w
            obs[4] = robot.y / h
            obs[5] = 0.0

        # 2. Local 4-direction obstacle raycasts (4 dims: N, S, E, W)
        # 1.0 if impassable, 0.0 if walkable
        directions = [(0, -1), (0, 1), (1, 0), (-1, 0)]
        for d_i, (dx, dy) in enumerate(directions):
            nx, ny = robot.x + dx, robot.y + dy
            obs[6 + d_i] = 1.0 if not self.grid.is_walkable(nx, ny) else 0.0

        # 3. Relative vectors to 3 closest other robots (12 dims: 3 * 4)
        other_robots = [r for r in self.robots if r.id != robot.id]
        other_robots.sort(key=lambda r: self.grid.manhattan_distance(robot.pos, r.pos))
        
        for k in range(3):
            base_k = 10 + k * 4
            if k < len(other_robots):
                other = other_robots[k]
                obs[base_k] = (other.x - robot.x) / w
                obs[base_k + 1] = (other.y - robot.y) / h
                obs[base_k + 2] = 1.0 if other.is_carrying else 0.0
                # Proximity alert: 1.0 if adjacent or within distance 2
                dist_other = self.grid.manhattan_distance(robot.pos, other.pos)
                obs[base_k + 3] = 1.0 if dist_other <= 2 else 0.0
            else:
                obs[base_k:base_k + 4] = 0.0

        # 4. Target direction unit vector (2 dims)
        if robot.target_pos is not None:
            dx = robot.target_pos[0] - robot.x
            dy = robot.target_pos[1] - robot.y
            norm = np.hypot(dx, dy)
            if norm > 1e-5:
                obs[22] = dx / norm
                obs[23] = dy / norm
        
        # 5. Nearest packing station relative vector (2 dims)
        best_station = min(self.grid.stations, key=lambda s: self.grid.manhattan_distance(robot.pos, s))
        obs[24] = (best_station[0] - robot.x) / w
        obs[25] = (best_station[1] - robot.y) / h

        # 6. Congestion gauge (2 dims)
        nearby_count = sum(
            1 for r in other_robots if self.grid.manhattan_distance(robot.pos, r.pos) <= 2
        )
        obs[26] = nearby_count / float(self.num_robots)
        obs[27] = min(robot.wait_steps / 20.0, 1.0)

        return obs

    def state(self, observations: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """
        Global state vector for MAPPO Centralized Critic.
        Contains concatenation of all agent observations + global warehouse status.
        observations: the dict just returned by step()/reset() for the current state, passed to avoid
        recomputing every robot's observation (pure speed-up; results are identical).
        """
        global_state = np.zeros(self.state_dim, dtype=np.float32)
        # 1. Observations of all robots (num_robots * obs_dim)
        for i, agent in enumerate(self.possible_agents):
            start = i * self.obs_dim
            global_state[start : start + self.obs_dim] = (
                observations[agent] if observations is not None else self._get_obs(agent)
            )

        # 2. Global warehouse metrics (10 dims)
        offset = self.state_max_robots * self.obs_dim
        global_state[offset] = min(len(self.order_generator.pending_orders) / 20.0, 1.0)
        global_state[offset + 1] = len(self.order_generator.active_orders) / float(self.num_robots)
        global_state[offset + 2] = min(self.total_orders_completed / 50.0, 1.0)
        global_state[offset + 3] = min(self.total_fleet_collisions / 50.0, 1.0)
        global_state[offset + 4] = self.current_step / float(self.max_steps)

        # Station status (2 stations)
        for s_idx, station in enumerate(self.grid.stations[:2]):
            station_occupied = any(r.pos == station for r in self.robots)
            global_state[offset + 5 + s_idx] = 1.0 if station_occupied else 0.0

        # Fleet spatial dispersion variance
        xs = [r.x for r in self.robots]
        ys = [r.y for r in self.robots]
        var_pos = (np.var(xs) + np.var(ys)) / float(self.grid_width * self.grid_height)
        global_state[offset + 7] = float(np.clip(var_pos, 0.0, 1.0))

        return global_state
