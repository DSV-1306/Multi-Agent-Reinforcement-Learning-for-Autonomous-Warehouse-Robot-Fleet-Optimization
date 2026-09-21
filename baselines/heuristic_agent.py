"""
Classical Heuristic Fleet Controller: Nearest Robot Allocation + Space-Time A* Routing.
"""

from typing import Dict, List, Tuple, Optional
from baselines.astar import ReservationTable, space_time_astar
from environment.warehouse_env import WarehouseEnv


class HeuristicFleetController:
    """
    Decentralized/Cooperative Heuristic Controller:
    - Queries active tasks from the warehouse environment.
    - Plans conflict-free 4D space-time trajectories using Space-Time A*.
    - Employs a shared reservation table to prevent collisions.
    """

    def __init__(self, env: WarehouseEnv):
        self.env = env
        self.reservation_table = ReservationTable()
        self.planned_paths: Dict[int, List[Tuple[int, int]]] = {
            i: [] for i in range(env.num_robots)
        }
        self.current_goals: Dict[int, Optional[Tuple[int, int]]] = {
            i: None for i in range(env.num_robots)
        }

    def reset(self):
        self.reservation_table.clear()
        for i in range(self.env.num_robots):
            self.planned_paths[i].clear()
            self.current_goals[i] = None

    def get_actions(self) -> Dict[str, int]:
        """
        Computes the next discrete action for all robots in the environment.
        """
        actions: Dict[str, int] = {}
        curr_step = self.env.current_step

        # Gather current positions of all robots
        current_positions = {i: self.env.robots[i].pos for i in range(self.env.num_robots)}

        # Prioritize robots with active tasks
        robot_order = sorted(
            range(self.env.num_robots),
            key=lambda i: (0 if self.env.robots[i].target_pos is not None else 1, i),
        )

        for i in robot_order:
            robot = self.env.robots[i]
            agent_name = f"robot_{i}"
            goal = robot.target_pos

            # If goal changed or path is exhausted or stuck, re-plan
            needs_replan = (goal != self.current_goals[i]) or (not self.planned_paths[i] and goal is not None and robot.pos != goal)

            if needs_replan:
                self.reservation_table.release_robot(i)
                self.current_goals[i] = goal

                if goal is not None and robot.pos != goal:
                    other_positions = {pos for idx, pos in current_positions.items() if idx != i}
                    path = space_time_astar(
                        grid=self.env.grid,
                        start=robot.pos,
                        goal=goal,
                        start_time=curr_step,
                        reservation_table=self.reservation_table,
                        robot_id=i,
                        other_robot_positions=other_positions,
                        max_time_horizon=80,
                    )
                    if path and len(path) > 1:
                        self.planned_paths[i] = path[1:]  # Exclude start pos
                        self.reservation_table.reserve_path(
                            robot_id=i, path=path, start_time=curr_step
                        )
                    else:
                        self.planned_paths[i] = []
                        self.reservation_table.reserve_stationary(i, robot.pos, curr_step, horizon=10)
                else:
                    self.planned_paths[i] = []
                    self.reservation_table.reserve_stationary(i, robot.pos, curr_step, horizon=10)

            # Determine next action
            if self.planned_paths[i]:
                next_pos = self.planned_paths[i].pop(0)
                dx = next_pos[0] - robot.x
                dy = next_pos[1] - robot.y

                if dx == 0 and dy == -1:
                    act = 0  # North
                elif dx == 0 and dy == 1:
                    act = 1  # South
                elif dx == 1 and dy == 0:
                    act = 2  # East
                elif dx == -1 and dy == 0:
                    act = 3  # West
                else:
                    act = 4  # Wait / same cell

            elif goal is not None and robot.pos == goal:
                # At goal: trigger interact (pickup or dropoff)
                act = 5
            else:
                # Idle: wait and stay in place
                act = 4
                self.reservation_table.reserve_stationary(i, robot.pos, curr_step, horizon=5)

            actions[agent_name] = act

        return actions
