"""
Pygame 2D Warehouse Visualizer and Live Simulation Renderer.
"""

from typing import Optional, Dict, Tuple
import numpy as np
import pygame

from environment.warehouse_env import WarehouseEnv
from environment.robot import RobotStatus


class WarehouseVisualizer:
    """
    Renders the multi-agent warehouse floor in real-time or exports frames.
    """

    # Palette
    COLOR_BG = (30, 34, 42)           # Dark modern background
    COLOR_GRID_LINE = (45, 50, 62)     # Subtle grid lines
    COLOR_SHELF = (209, 154, 102)      # Amber/wood tone for shelf racks
    COLOR_SHELF_BORDER = (160, 110, 60)
    COLOR_STATION = (152, 195, 121)    # Emerald for packing stations
    COLOR_STATION_BORDER = (90, 140, 70)
    COLOR_WALL = (75, 82, 99)          # Steel gray walls
    COLOR_TEXT = (220, 223, 230)
    COLOR_HUD_BG = (22, 25, 31)

    # Distinct robot colors (up to 10 robots)
    ROBOT_COLORS = [
        (97, 175, 239),   # Robot 0: Cyan
        (224, 108, 117),  # Robot 1: Coral red
        (229, 192, 123),  # Robot 2: Amber gold
        (198, 120, 221),  # Robot 3: Purple
        (86, 182, 194),   # Robot 4: Teal
        (152, 195, 121),  # Robot 5: Mint green
        (235, 104, 65),   # Robot 6: Orange
        (120, 150, 240),  # Robot 7: Royal blue
        (255, 110, 180),  # Robot 8: Pink
        (180, 220, 100),  # Robot 9: Lime
    ]

    def __init__(self, env: WarehouseEnv, cell_size: int = 55, fps: int = 15):
        self.env = env
        self.cell_size = cell_size
        self.fps = fps

        self.grid_w = env.grid_width
        self.grid_h = env.grid_height

        self.screen_w = self.grid_w * self.cell_size
        self.hud_height = 80
        self.screen_h = self.grid_h * self.cell_size + self.hud_height

        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font: Optional[pygame.font.Font] = None
        self.font_small: Optional[pygame.font.Font] = None
        self._is_initialized = False

    def _init_pygame(self):
        if not self._is_initialized:
            pygame.init()
            pygame.font.init()
            if self.env.render_mode == "human":
                self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
                pygame.display.set_caption("Warehouse MARL Fleet Simulator")
            else:
                self.screen = pygame.Surface((self.screen_w, self.screen_h))

            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Segoe UI", 16, bold=True)
            self.font_small = pygame.font.SysFont("Segoe UI", 12)
            self._is_initialized = True

    def render(self) -> Optional[np.ndarray]:
        """Renders one frame of the simulation."""
        self._init_pygame()

        # Handle UI quit event
        if self.env.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

        self.screen.fill(self.COLOR_BG)

        # 1. Draw Grid Lines
        for x in range(self.grid_w + 1):
            px = x * self.cell_size
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (px, 0), (px, self.grid_h * self.cell_size))
        for y in range(self.grid_h + 1):
            py = y * self.cell_size
            pygame.draw.line(self.screen, self.COLOR_GRID_LINE, (0, py), (self.screen_w, py))

        # 2. Draw Shelves
        for sx, sy in self.env.grid.shelves:
            rect = pygame.Rect(
                sx * self.cell_size + 4,
                sy * self.cell_size + 4,
                self.cell_size - 8,
                self.cell_size - 8,
            )
            pygame.draw.rect(self.screen, self.COLOR_SHELF, rect, border_radius=4)
            pygame.draw.rect(self.screen, self.COLOR_SHELF_BORDER, rect, width=2, border_radius=4)
            # Rack slats
            mid_y = rect.centery
            pygame.draw.line(self.screen, self.COLOR_SHELF_BORDER, (rect.left + 4, mid_y), (rect.right - 4, mid_y), 2)

        # 3. Draw Stations
        for st_x, st_y in self.env.grid.stations:
            rect = pygame.Rect(
                st_x * self.cell_size + 2,
                st_y * self.cell_size + 2,
                self.cell_size - 4,
                self.cell_size - 4,
            )
            pygame.draw.rect(self.screen, self.COLOR_STATION, rect, border_radius=6)
            pygame.draw.rect(self.screen, self.COLOR_STATION_BORDER, rect, width=2, border_radius=6)
            txt = self.font_small.render("PACK", True, (20, 40, 20))
            self.screen.blit(txt, txt.get_rect(center=rect.center))

        # 4. Draw Robot Trails and Target Lines
        for robot in self.env.robots:
            # Target dashed indicator
            if robot.target_pos is not None:
                r_center = (
                    robot.x * self.cell_size + self.cell_size // 2,
                    robot.y * self.cell_size + self.cell_size // 2,
                )
                t_center = (
                    robot.target_pos[0] * self.cell_size + self.cell_size // 2,
                    robot.target_pos[1] * self.cell_size + self.cell_size // 2,
                )
                color = self.ROBOT_COLORS[robot.id % len(self.ROBOT_COLORS)]
                pygame.draw.line(self.screen, color, r_center, t_center, 1)

        # 5. Draw Robots
        for robot in self.env.robots:
            color = self.ROBOT_COLORS[robot.id % len(self.ROBOT_COLORS)]
            cx = robot.x * self.cell_size + self.cell_size // 2
            cy = robot.y * self.cell_size + self.cell_size // 2
            radius = self.cell_size // 2 - 6

            # Collision halo
            if robot.wait_steps > 0 and robot.collisions > 0:
                pygame.draw.circle(self.screen, (220, 50, 50), (cx, cy), radius + 4, width=2)

            # Robot body
            pygame.draw.circle(self.screen, color, (cx, cy), radius)
            pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), radius, width=2)

            # Robot ID number
            id_txt = self.font.render(str(robot.id), True, (15, 15, 20))
            self.screen.blit(id_txt, id_txt.get_rect(center=(cx, cy)))

            # Cargo indicator (yellow parcel on top)
            if robot.is_carrying:
                cargo_rect = pygame.Rect(cx + 4, cy - radius - 2, 10, 10)
                pygame.draw.rect(self.screen, (255, 215, 0), cargo_rect, border_radius=2)
                pygame.draw.rect(self.screen, (100, 80, 0), cargo_rect, width=1, border_radius=2)

        # 6. Draw HUD
        hud_y = self.grid_h * self.cell_size
        hud_rect = pygame.Rect(0, hud_y, self.screen_w, self.hud_height)
        pygame.draw.rect(self.screen, self.COLOR_HUD_BG, hud_rect)
        pygame.draw.line(self.screen, self.COLOR_WALL, (0, hud_y), (self.screen_w, hud_y), 2)

        # Metrics text
        step_str = f"Step: {self.env.current_step}/{self.env.max_steps}"
        comp_str = f"Orders Done: {self.env.total_orders_completed}"
        coll_str = f"Collisions: {self.env.total_fleet_collisions}"
        pend_str = f"Pending Orders: {len(self.env.order_generator.pending_orders)}"

        t1 = self.font.render(f"{step_str}   |   {comp_str}", True, self.COLOR_TEXT)
        t2 = self.font.render(f"{coll_str}   |   {pend_str}", True, (220, 180, 120))
        self.screen.blit(t1, (16, hud_y + 12))
        self.screen.blit(t2, (16, hud_y + 42))

        # Update display
        if self.env.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.fps)
            return None
        else:
            # Return RGB array (height, width, 3)
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), (1, 0, 2)
            )

    def close(self):
        if self._is_initialized:
            pygame.quit()
            self._is_initialized = False
