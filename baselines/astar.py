"""
Space-Time A* Pathfinding with 3D Reservation Table for Multi-Robot Coordination.
"""

from typing import List, Tuple, Dict, Set, Optional
import heapq


class ReservationTable:
    """
    Maintains space-time occupancy reservations across time steps:
    - vertex_reservations: (x, y, t) -> robot_id
    - edge_reservations: ((from_x, from_y), (to_x, to_y), t) -> robot_id
    """

    def __init__(self):
        self.vertex_reservations: Dict[Tuple[int, int, int], int] = {}
        self.edge_reservations: Dict[Tuple[Tuple[int, int], Tuple[int, int], int], int] = {}

    def clear(self):
        self.vertex_reservations.clear()
        self.edge_reservations.clear()

    def reserve_path(self, robot_id: int, path: List[Tuple[int, int]], start_time: int, hold_end_steps: int = 15):
        for i, pos in enumerate(path):
            t = start_time + i
            self.vertex_reservations[(pos[0], pos[1], t)] = robot_id
            if i > 0:
                prev_pos = path[i - 1]
                self.edge_reservations[(prev_pos, pos, t - 1)] = robot_id

        # Hold final destination reserved for a few steps to prevent immediate collisions upon arrival
        if path:
            last_pos = path[-1]
            end_t = start_time + len(path)
            for dt in range(hold_end_steps):
                self.vertex_reservations[(last_pos[0], last_pos[1], end_t + dt)] = robot_id

    def reserve_stationary(self, robot_id: int, pos: Tuple[int, int], start_time: int, horizon: int = 40):
        for dt in range(horizon):
            self.vertex_reservations[(pos[0], pos[1], start_time + dt)] = robot_id

    def release_robot(self, robot_id: int):
        self.vertex_reservations = {
            k: v for k, v in self.vertex_reservations.items() if v != robot_id
        }
        self.edge_reservations = {
            k: v for k, v in self.edge_reservations.items() if v != robot_id
        }

    def is_vertex_reserved(self, x: int, y: int, t: int, robot_id: int) -> bool:
        occupant = self.vertex_reservations.get((x, y, t))
        return occupant is not None and occupant != robot_id

    def is_edge_reserved(
        self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], t: int, robot_id: int
    ) -> bool:
        # Check swap conflict: another robot moving from to_pos to from_pos at time t
        reverse_occupant = self.edge_reservations.get((to_pos, from_pos, t))
        return reverse_occupant is not None and reverse_occupant != robot_id


def space_time_astar(
    grid,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    start_time: int,
    reservation_table: ReservationTable,
    robot_id: int,
    other_robot_positions: Optional[Set[Tuple[int, int]]] = None,
    max_time_horizon: int = 60,
) -> Optional[List[Tuple[int, int]]]:
    """
    Computes shortest collision-free path in (x, y, t) space avoiding reserved cells.
    """
    if start == goal:
        return [start]

    def heuristic(pos: Tuple[int, int]) -> int:
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    # Priority queue item: (f_score, g_score, (x, y, t))
    open_heap = []
    initial_f = heuristic(start)
    heapq.heappush(open_heap, (initial_f, 0, (start[0], start[1], start_time)))

    came_from: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
    g_scores: Dict[Tuple[int, int, int], int] = {(start[0], start[1], start_time): 0}
    closed_set: Set[Tuple[int, int, int]] = set()

    moves = [(0, -1), (0, 1), (1, 0), (-1, 0), (0, 0)]  # 4-cardinal + wait

    while open_heap:
        current_f, current_g, current_node = heapq.heappop(open_heap)
        cx, cy, ct = current_node

        if (cx, cy) == goal:
            # Reconstruct path
            path = []
            curr = current_node
            while curr in came_from:
                path.append((curr[0], curr[1]))
                curr = came_from[curr]
            path.append(start)
            path.reverse()
            return path

        if current_node in closed_set:
            continue
        closed_set.add(current_node)

        if current_g >= max_time_horizon:
            continue

        next_t = ct + 1
        for dx, dy in moves:
            nx, ny = cx + dx, cy + dy

            # Bounds and static obstacle check
            if not grid.is_walkable(nx, ny):
                continue

            # Check immediate collision with other stationary robots at step 1
            if current_g == 0 and other_robot_positions and (nx, ny) in other_robot_positions:
                continue

            # Vertex conflict check at next_t
            if reservation_table.is_vertex_reserved(nx, ny, next_t, robot_id):
                continue

            # Edge / swap conflict check
            if reservation_table.is_edge_reserved((cx, cy), (nx, ny), ct, robot_id):
                continue

            next_node = (nx, ny, next_t)
            tentative_g = current_g + 1

            if next_node not in g_scores or tentative_g < g_scores[next_node]:
                g_scores[next_node] = tentative_g
                f_score = tentative_g + heuristic((nx, ny))
                came_from[next_node] = current_node
                heapq.heappush(open_heap, (f_score, tentative_g, next_node))

    # Fallback if no path: wait in place
    return [start, start]
