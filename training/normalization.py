"""
Running observation normalization and reward scaling (the VecNormalize scheme of
Stable-Baselines3 / the PPO implementation details of Huang et al., 2022).

- Observations: (x - running_mean) / sqrt(running_var + eps), clipped to +/- obs_clip.
  Statistics update only during training and are frozen and saved with the checkpoint.
- Rewards: divided by the running std of each robot's discounted return (not mean-centred,
  so the sign of every reward is preserved), clipped to +/- reward_clip.
"""

from typing import Dict, Tuple

import numpy as np


class RunningMeanStd:
    """Parallel (Chan et al.) running mean/variance over the leading axis."""

    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon  # tiny prior count avoids division by zero on the first update

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64).reshape(-1, *self.mean.shape)
        batch_mean, batch_var, batch_count = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta ** 2 * self.count * batch_count / total
        self.mean, self.var, self.count = new_mean, m2 / total, total

    def state_dict(self) -> Dict:
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": float(self.count)}

    def load_state_dict(self, d: Dict):
        self.mean = np.asarray(d["mean"], dtype=np.float64)
        self.var = np.asarray(d["var"], dtype=np.float64)
        self.count = float(d["count"])


class ObsNormalizer:
    def __init__(self, dim: int, enabled: bool, clip: float, epsilon: float):
        self.enabled = enabled
        self.clip = clip
        self.epsilon = epsilon
        self.rms = RunningMeanStd(shape=(dim,))

    def __call__(self, x: np.ndarray, update: bool) -> np.ndarray:
        if not self.enabled:
            return np.asarray(x, dtype=np.float32)
        if update:
            self.rms.update(x)
        z = (x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon)
        return np.clip(z, -self.clip, self.clip).astype(np.float32)

    def state_dict(self) -> Dict:
        return {"enabled": self.enabled, "clip": self.clip, "epsilon": self.epsilon, "rms": self.rms.state_dict()}

    @classmethod
    def from_state_dict(cls, d: Dict) -> "ObsNormalizer":
        norm = cls(dim=len(d["rms"]["mean"]), enabled=d["enabled"], clip=d["clip"], epsilon=d["epsilon"])
        norm.rms.load_state_dict(d["rms"])
        return norm


class RewardScaler:
    """One discounted-return accumulator per (env, robot) stream; one shared scale."""

    def __init__(self, shape: Tuple[int, ...], gamma: float, enabled: bool, clip: float, epsilon: float):
        self.enabled = enabled
        self.gamma = gamma
        self.clip = clip
        self.epsilon = epsilon
        self.returns = np.zeros(shape, dtype=np.float64)
        self.rms = RunningMeanStd(shape=())

    def __call__(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """rewards: (E, N); dones: (E,) bool. Returns scaled rewards (E, N) float32."""
        if not self.enabled:
            return np.asarray(rewards, dtype=np.float32)
        self.returns = self.returns * self.gamma + rewards
        self.rms.update(self.returns.ravel())
        scaled = np.clip(rewards / np.sqrt(self.rms.var + self.epsilon), -self.clip, self.clip)
        self.returns[np.asarray(dones, dtype=bool)] = 0.0
        return scaled.astype(np.float32)

    @property
    def scale(self) -> float:
        return float(np.sqrt(self.rms.var + self.epsilon)) if self.enabled else 1.0
