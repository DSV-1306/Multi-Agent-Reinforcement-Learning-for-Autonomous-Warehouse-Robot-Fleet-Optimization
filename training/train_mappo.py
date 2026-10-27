"""
Training script for Multi-Agent PPO (MAPPO) with Centralized Critic (CTDE).

    python training/train_mappo.py --seed 1
    python training/train_mappo.py --seed 1 --set mappo.critic_input=global

All settings come from the YAML (--config); --timesteps and --set only override when given.
Outputs: results/runs/<run_name>/{learning_curve.csv, best_dev.pt, final.pt, config.json, summary.json}
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.config_utils import load_config
from training.trainer import train
from training.train_ippo import parse_args


def train_mappo(config_path: str = "configs/default_config.yaml", total_timesteps: int = None, seed: int = 42,
                overrides=None, run_name: str = None, variant: str = None):
    cfg = load_config(config_path, overrides)
    return train("mappo", cfg, seed=seed, run_name=run_name, total_timesteps=total_timesteps, variant=variant)


if __name__ == "__main__":
    args = parse_args("mappo")
    train_mappo(args.config, args.timesteps, args.seed, args.set, args.run_name, args.variant)
