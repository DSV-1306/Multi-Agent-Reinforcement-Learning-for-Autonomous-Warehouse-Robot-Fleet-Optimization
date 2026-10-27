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
    Simulates dynamic customer orders. Each order is one item (one shelf -> one station trip).

    mode="synthetic" (default, original behaviour):
      - Poisson(lambda_rate) orders per step.
      - Pareto 80/20 popularity skew: the first 20% of the sorted shelf list gets 80% of picks.
    mode="empirical" (Phase 3, statistics from Online Retail II, see data/preprocess_online_retail.py):
      - Shelf popularity: the top-n_shelves SKUs by order-line count, renormalised, mapped to shelves by a
        seeded random permutation (shelf_assignment_seed), i.e. no slotting optimisation is assumed.
      - Arrivals: invoices ~ Poisson(lambda_t / E[lines]) per step, each invoice spawning L order lines
        (L ~ empirical lines-per-invoice pmf, all to one station). The expected number of order lines per
        step therefore equals lambda_t, so load is comparable with synthetic mode, only burstier.
      - lambda_t = lambda_rate * hourly_profile[hour(t)] / mean(profile over the simulated hours), where step t
        maps to hour start_hour + t / steps_per_hour.
    """

    def __init__(
        self,
        shelves: List[Tuple[int, int]],
        stations: List[Tuple[int, int]],
        grid_access_fn,
        lambda_rate: float = 0.2,
        seed: Optional[int] = None,
        max_steps: Optional[int] = None,
        mode: str = "synthetic",
        empirical: Optional[Dict] = None,
    ):
        if mode not in ("synthetic", "empirical"):
            raise ValueError(f"order mode must be 'synthetic' or 'empirical', got {mode!r}")
        self.shelves = sorted(list(shelves))
        self.stations = stations
        self.grid_access_fn = grid_access_fn
        self.lambda_rate = lambda_rate
        self.mode = mode

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

        if mode == "empirical":
            if not empirical:
                raise ValueError("empirical mode needs the 'empirical' statistics dict")
            self._setup_empirical(empirical, max_steps)

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

    def _setup_empirical(self, emp: Dict, max_steps: Optional[int]):
        n_shelves = len(self.shelves)
        sku_probs = np.asarray(emp["sku_probs"], dtype=np.float64)[:n_shelves]
        if len(sku_probs) < n_shelves:
            raise ValueError(f"Need >= {n_shelves} SKUs in the empirical stats, got {len(sku_probs)}")
        perm = np.random.default_rng(emp["shelf_assignment_seed"]).permutation(n_shelves)
        weights = np.zeros(n_shelves, dtype=np.float64)
        weights[perm] = sku_probs / sku_probs.sum()  # SKU rank k -> shelf perm[k]
        self.shelf_weights = weights

        pmf = np.asarray(emp["lines_pmf"], dtype=np.float64)
        self.lines_pmf = pmf / pmf.sum()
        self.mean_lines = float(np.sum(np.arange(1, len(pmf) + 1) * self.lines_pmf))

        profile = np.asarray(emp["hourly_profile"], dtype=np.float64)
        self.steps_per_hour = float(emp["steps_per_hour"])
        self.start_hour = int(emp["start_hour"])
        horizon = max_steps or int(self.steps_per_hour * 24)
        hours = [(self.start_hour + int(t / self.steps_per_hour)) % 24 for t in range(horizon + 1)]
        mean_rate = profile[hours].mean()
        if mean_rate <= 0:
            raise ValueError("Hourly profile is zero over the simulated hours; change start_hour")
        self.hour_multiplier = profile / mean_rate

    def current_lambda(self, current_step: int) -> float:
        if self.mode == "synthetic":
            return self.lambda_rate
        hour = (self.start_hour + int(current_step / self.steps_per_hour)) % 24
        return self.lambda_rate * float(self.hour_multiplier[hour])

    def generate_orders_for_step(self, current_step: int) -> List[Order]:
        """Generates new orders according to the Poisson distribution."""
        if self.mode == "empirical":
            return self._generate_empirical(current_step)
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

    def _make_order(self, current_step: int, station_pos: Tuple[int, int]) -> Order:
        self.order_counter += 1
        shelf_pos = self.shelves[self.rng.choice(len(self.shelves), p=self.shelf_weights)]
        access_cells = self.grid_access_fn(shelf_pos)
        access_pos = access_cells[self.rng.choice(len(access_cells))] if access_cells else shelf_pos
        order = Order(self.order_counter, shelf_pos, access_pos, station_pos, current_step)
        self.pending_orders.append(order)
        return order

    def _generate_empirical(self, current_step: int) -> List[Order]:
        new_orders = []
        n_invoices = self.rng.poisson(self.current_lambda(current_step) / self.mean_lines)
        for _ in range(n_invoices):
            n_lines = 1 + int(self.rng.choice(len(self.lines_pmf), p=self.lines_pmf))
            station_pos = self.stations[self.rng.choice(len(self.stations))]
            new_orders.extend(self._make_order(current_step, station_pos) for _ in range(n_lines))
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
