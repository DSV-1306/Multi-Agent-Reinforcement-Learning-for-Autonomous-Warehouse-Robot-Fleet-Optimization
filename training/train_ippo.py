"""
Training script for Independent PPO (IPPO) in Warehouse Multi-Agent Environment.

    python training/train_ippo.py --seed 1
    python training/train_ippo.py --seed 1 --timesteps 200000 --set training.action_masking=false

All settings come from the YAML (--config); --timesteps and --set only override when given.
Outputs: results/runs/<run_name>/{learning_curve.csv, best_dev.pt, final.pt, config.json, summary.json}
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.config_utils import load_config
from training.trainer import train


def train_ippo(config_path: str = "configs/default_config.yaml", total_timesteps: int = None, seed: int = 42,
               overrides=None, run_name: str = None, variant: str = None):
    cfg = load_config(config_path, overrides)
    return train("ippo", cfg, seed=seed, run_name=run_name, total_timesteps=total_timesteps, variant=variant)


def parse_args(method: str):
    parser = argparse.ArgumentParser(description=f"Train {method.upper()}")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--timesteps", type=int, default=None,
                        help=f"Env steps (summed over parallel envs). Default: {method}.total_timesteps in the YAML")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-name", default=None, help="Default: <method>_seed<seed>")
    parser.add_argument("--set", action="append", default=[], help="Config override key.sub=value (repeatable)")
    parser.add_argument("--variant", default=None, help="Label stored in config.json (ablation name)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args("ippo")
    train_ippo(args.config, args.timesteps, args.seed, args.set, args.run_name, args.variant)
