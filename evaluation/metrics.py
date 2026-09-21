"""
Evaluation Metrics calculation and statistical tracking.
"""

from typing import List, Dict, Any
import numpy as np


class MetricsTracker:
    """
    Collects episode-level performance metrics:
    - Average Order Completion Time (latency in simulation steps)
    - Total Travel Distance (sum of all robot steps)
    - Collision Count (total fleet collisions)
    - Fleet Throughput (orders delivered per episode)
    - Fleet Utilization Rate (ratio of active carrying/moving steps to total steps)
    - Mean Episode Return
    """

    def __init__(self):
        self.completion_times: List[float] = []
        self.travel_distances: List[int] = []
        self.collision_counts: List[int] = []
        self.throughputs: List[int] = []
        self.utilization_rates: List[float] = []
        self.episode_returns: List[float] = []

    def record_episode(
        self,
        orders: List[Any],
        robots: List[Any],
        fleet_collisions: int,
        orders_completed: int,
        total_steps: int,
        mean_return: float,
    ):
        # Order completion latency
        latencies = [o.latency for o in orders if o.is_completed and o.latency is not None]
        if latencies:
            self.completion_times.append(float(np.mean(latencies)))
        else:
            self.completion_times.append(float(total_steps))  # Cap at max steps if none finished

        # Travel distance
        total_dist = sum(r.total_steps for r in robots)
        self.travel_distances.append(total_dist)

        # Collisions & throughput
        self.collision_counts.append(fleet_collisions)
        self.throughputs.append(orders_completed)

        # Utilization: proportion of steps where robots were not idle
        total_possible_steps = len(robots) * total_steps
        active_steps = sum(r.total_steps for r in robots)
        utilization = active_steps / float(total_possible_steps) if total_possible_steps > 0 else 0.0
        self.utilization_rates.append(utilization)

        self.episode_returns.append(mean_return)

    def summary(self) -> Dict[str, float]:
        """Returns mean and standard deviation for all tracked metrics."""
        return {
            "avg_completion_time": float(np.mean(self.completion_times)) if self.completion_times else 0.0,
            "std_completion_time": float(np.std(self.completion_times)) if self.completion_times else 0.0,
            "avg_travel_distance": float(np.mean(self.travel_distances)) if self.travel_distances else 0.0,
            "std_travel_distance": float(np.std(self.travel_distances)) if self.travel_distances else 0.0,
            "avg_collisions": float(np.mean(self.collision_counts)) if self.collision_counts else 0.0,
            "std_collisions": float(np.std(self.collision_counts)) if self.collision_counts else 0.0,
            "avg_throughput": float(np.mean(self.throughputs)) if self.throughputs else 0.0,
            "std_throughput": float(np.std(self.throughputs)) if self.throughputs else 0.0,
            "avg_utilization": float(np.mean(self.utilization_rates)) if self.utilization_rates else 0.0,
            "std_utilization": float(np.std(self.utilization_rates)) if self.utilization_rates else 0.0,
            "avg_return": float(np.mean(self.episode_returns)) if self.episode_returns else 0.0,
        }
