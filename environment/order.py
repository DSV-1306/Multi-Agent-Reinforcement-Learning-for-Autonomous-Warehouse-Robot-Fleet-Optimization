"""
Order management and dynamic Poisson arrival generation with Pareto 80/20 skew.
"""

from typing import Tuple, List, Dict, Optional, Set
import numpy as np


class Order:
    """Represents a discrete warehouse fulfillment order."""

    def __init__(
        self,
        order_id: int,
        shelf_pos: Tuple[int, int],
        shelf_access_pos: Tuple[int, int],
        station_pos: Tuple[int, int],
        spawn_step: int,
    ):
        self.id = order_id
        self.shelf_pos = shelf_pos
        self.shelf_access_pos = shelf_access_pos
        self.station_pos = station_pos
        self.spawn_step = spawn_step
        
        self.assigned_robot_id: Optional[int] = None
        self.picked_up_step: Optional[int] = None
        self.delivered_step: Optional[int] = None
        self.is_completed: bool = False

    @property
    def latency(self) -> Optional[int]:
        """Total turnaround time from creation to delivery."""
        if self.delivered_step is not None:
            return self.delivered_step - self.spawn_step
        return None


class OrderGenerator:
    """
    Simulates dynamic customer orders:
    - Poisson arrival process with rate lambda_rate.
    - Pareto 80/20 popularity skew across shelves.
    """

    def __init__(
        self,
        shelves: List[Tuple[int, int]],
        stations: List[Tuple[int, int]],
        grid_access_fn,
        lambda_rate: float = 0.2,
        seed: Optional[int] = None,
    ):
        self.shelves = sorted(list(shelves))
        self.stations = stations
        self.grid_access_fn = grid_access_fn
        self.lambda_rate = lambda_rate

        self.rng = np.random.default_rng(seed)
        self.order_counter = 0

        # Build shelf probabilities: 20% of shelves get 80% weight
        n_shelves = len(self.shelves)
        if n_shelves == 0:
            raise ValueError("No shelves provided to OrderGenerator.")

        n_popular = max(1, int(0.2 * n_shelves))
        weights = np.zeros(n_shelves, dtype=np.float64)
        weights[:n_popular] = 0.8 / n_popular
        if n_shelves > n_popular:
            weights[n_popular:] = 0.2 / (n_shelves - n_popular)
        self.shelf_weights = weights

        self.pending_orders: List[Order] = []
        self.active_orders: Dict[int, Order] = {}
        self.completed_orders: List[Order] = []

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.order_counter = 0
        self.pending_orders.clear()
        self.active_orders.clear()
        self.completed_orders.clear()

    def generate_orders_for_step(self, current_step: int) -> List[Order]:
        """Generates new orders according to the Poisson distribution."""
        n_arrivals = self.rng.poisson(self.lambda_rate)
        new_orders = []

        for _ in range(n_arrivals):
            self.order_counter += 1
            # Sample shelf according to Pareto skew
            shelf_idx = self.rng.choice(len(self.shelves), p=self.shelf_weights)
            shelf_pos = self.shelves[shelf_idx]
            
            # Select access cell for the shelf
            access_cells = self.grid_access_fn(shelf_pos)
            if access_cells:
                access_pos = access_cells[self.rng.choice(len(access_cells))]
            else:
                access_pos = shelf_pos

            # Pick a station
            station_pos = self.stations[self.rng.choice(len(self.stations))]

            order = Order(
                order_id=self.order_counter,
                shelf_pos=shelf_pos,
                shelf_access_pos=access_pos,
                station_pos=station_pos,
                spawn_step=current_step,
            )
            self.pending_orders.append(order)
            new_orders.append(order)

        return new_orders

    def get_next_unassigned_order(self) -> Optional[Order]:
        """Retrieves and removes the oldest pending order."""
        if self.pending_orders:
            return self.pending_orders.pop(0)
        return None

    def mark_picked_up(self, order_id: int, step: int):
        if order_id in self.active_orders:
            self.active_orders[order_id].picked_up_step = step

    def mark_delivered(self, order_id: int, step: int) -> Optional[Order]:
        if order_id in self.active_orders:
            order = self.active_orders.pop(order_id)
            order.delivered_step = step
            order.is_completed = True
            self.completed_orders.append(order)
            return order
        return None
