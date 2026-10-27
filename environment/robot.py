"""
Robot Agent physical state and movement representation.
"""

from enum import IntEnum
from typing import Tuple, Optional, List


class RobotStatus(IntEnum):
    IDLE = 0
    EN_ROUTE_TO_SHELF = 1
    CARRYING_TO_STATION = 2


class WarehouseRobot:
    """
    Encapsulates the state and metrics of an autonomous mobile robot (AMR).
    """

    def __init__(self, robot_id: int, initial_pos: Tuple[int, int]):
        self.id = robot_id
        self.x = initial_pos[0]
        self.y = initial_pos[1]
        self.initial_pos = initial_pos

        self.status = RobotStatus.IDLE
        self.current_order_id: Optional[int] = None
        self.target_pos: Optional[Tuple[int, int]] = None
        
        # Metrics tracking
        self.total_steps = 0
        self.collisions = 0
        self.wall_bumps = 0
        self.robot_collisions = 0
        self.deliveries_completed = 0
        self.wait_steps = 0
        self.prev_target_dist: Optional[int] = None

        # Position trail for visualizer
        self.history: List[Tuple[int, int]] = [(self.x, self.y)]

    @property
    def pos(self) -> Tuple[int, int]:
        return (self.x, self.y)

    @property
    def is_carrying(self) -> bool:
        return self.status == RobotStatus.CARRYING_TO_STATION

    @property
    def is_idle(self) -> bool:
        return self.status == RobotStatus.IDLE

    def reset(self, start_pos: Optional[Tuple[int, int]] = None):
        if start_pos is not None:
            self.initial_pos = start_pos
        self.x = self.initial_pos[0]
        self.y = self.initial_pos[1]
        self.status = RobotStatus.IDLE
        self.current_order_id = None
        self.target_pos = None
        self.total_steps = 0
        self.collisions = 0
        self.wall_bumps = 0
        self.robot_collisions = 0
        self.deliveries_completed = 0
        self.wait_steps = 0
        self.prev_target_dist = None
        self.history = [(self.x, self.y)]

    def assign_order(self, order_id: int, shelf_access_pos: Tuple[int, int]):
        self.current_order_id = order_id
        self.target_pos = shelf_access_pos
        self.status = RobotStatus.EN_ROUTE_TO_SHELF
        self.wait_steps = 0
        self.prev_target_dist = abs(self.x - shelf_access_pos[0]) + abs(self.y - shelf_access_pos[1])

    def pickup_item(self, station_pos: Tuple[int, int]):
        self.status = RobotStatus.CARRYING_TO_STATION
        self.target_pos = station_pos
        self.wait_steps = 0
        self.prev_target_dist = abs(self.x - station_pos[0]) + abs(self.y - station_pos[1])

    def complete_delivery(self):
        self.status = RobotStatus.IDLE
        self.current_order_id = None
        self.target_pos = None
        self.prev_target_dist = None
        self.deliveries_completed += 1
        self.wait_steps = 0

    def move_to(self, new_x: int, new_y: int):
        self.x = new_x
        self.y = new_y
        self.total_steps += 1
        self.wait_steps = 0
        self.history.append((self.x, self.y))
        if len(self.history) > 20:
            self.history.pop(0)

    def wait(self):
        self.wait_steps += 1
        self.history.append((self.x, self.y))
        if len(self.history) > 20:
            self.history.pop(0)

    def log_collision(self, kind: str = "robot"):
        """kind: "wall" (static obstacle / out of bounds) or "robot" (vertex or swap conflict)."""
        self.collisions += 1
        if kind == "wall":
            self.wall_bumps += 1
        else:
            self.robot_collisions += 1
