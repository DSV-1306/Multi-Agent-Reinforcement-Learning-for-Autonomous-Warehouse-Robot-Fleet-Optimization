"""
Warehouse Grid module.
Defines the 2D layout, shelf pods, transit aisles, and packing workstations.
"""

from typing import List, Tuple, Set, Optional
import numpy as np


class WarehouseGrid:
    """
    Represents the physical warehouse floor.
    Standard layout: 
      - Perimeter aisles for transit.
      - Dual-column shelf pods separated by vertical and horizontal aisles.
      - Packing/delivery workstations along the southern boundary.
    """

    def __init__(
        self,
        width: int = 10,
        height: int = 10,
        layout_type: str = "kiva_standard",
    ):
        self.width = width
        self.height = height
        self.layout_type = layout_type

        self.shelves: Set[Tuple[int, int]] = set()
        self.stations: List[Tuple[int, int]] = []
        self.walls: Set[Tuple[int, int]] = set()

        self._build_layout(layout_type)

    def _build_layout(self, layout_type: str) -> None:
        """Constructs layout features based on selected layout preset."""
        self.shelves.clear()
        self.stations.clear()
        self.walls.clear()

        if layout_type == "kiva_standard":
            # 10x10 or scalable standard layout
            # Packing stations at bottom boundary
            station_xs = [self.width // 4, (3 * self.width) // 4]
            for sx in station_xs:
                self.stations.append((sx, self.height - 1))

            # Shelves in 2-column pods
            # Example for 10x10:
            # Columns x in [2, 3] and x in [6, 7]
            # Rows y in [2, 3, 4, 5, 6]
            shelf_cols = []
            c = 2
            while c + 1 < self.width - 2:
                shelf_cols.extend([c, c + 1])
                c += 4  # 2 columns of shelves, 2 columns of aisle

            for x in shelf_cols:
                for y in range(2, self.height - 3):
                    self.shelves.add((x, y))

        elif layout_type == "narrow_corridor":
            # Stress-test layout: single central bottleneck aisle
            # Stations at bottom
            self.stations.append((self.width // 2, self.height - 1))
            # Wall dividing left and right with a 1-cell corridor at the center
            mid_y = self.height // 2
            for x in range(self.width):
                if x != self.width // 2:
                    self.walls.add((x, mid_y))
            # Shelves in the top half
            for x in range(1, self.width - 1, 2):
                for y in range(1, mid_y - 1):
                    self.shelves.add((x, y))

        else:
            raise ValueError(f"Unknown layout type: {layout_type}")

    def is_in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_wall(self, x: int, y: int) -> bool:
        return (x, y) in self.walls

    def is_shelf(self, x: int, y: int) -> bool:
        return (x, y) in self.shelves

    def is_station(self, x: int, y: int) -> bool:
        return (x, y) in self.stations

    def is_walkable(self, x: int, y: int) -> bool:
        """
        Walkable transit cells: inside bounds, not a wall, and not a shelf.
        Robots navigate through aisles to reach shelves and delivery stations.
        """
        return self.is_in_bounds(x, y) and not self.is_wall(x, y) and not self.is_shelf(x, y)

    def get_walkable_cells(self) -> List[Tuple[int, int]]:
        """Returns all walkable cells (including station cells)."""
        cells = []
        for x in range(self.width):
            for y in range(self.height):
                if self.is_walkable(x, y):
                    cells.append((x, y))
        return cells

    def get_shelf_access_cells(self, shelf_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Returns adjacent walkable cells from which a robot can pick from a shelf.
        """
        sx, sy = shelf_pos
        access_cells = []
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = sx + dx, sy + dy
            if self.is_walkable(nx, ny):
                access_cells.append((nx, ny))
        return access_cells

    def manhattan_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
