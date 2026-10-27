"""
Scenario-matrix evaluation (Phase 2 / Phase 5).

    # own layouts x low/medium/high load, DEV seeds (allowed for tuning)
    python evaluation/scenario_matrix.py --matrix own --split dev --runs results/runs/ippo_s1 ... --output results/eval/own_dev.csv
    # held-out MAPF layouts + empirical orders, TEST seeds (final results only)
    python evaluation/scenario_matrix.py --matrix test --split test --runs ... --output results/eval/test.csv

Writes one row per (method, training run, scenario, evaluation seed) - the long format that
evaluation/stats.py turns into means and 95% confidence intervals.

Held-out order rate rule (fixed before any test result is seen):
    lambda = load_per_5_robots * (num_robots / 5) * (trip_ref / trip_layout)
where trip_* = WarehouseGrid.mean_trip_length() and trip_ref is that of the training layout
`kiva_standard`. It keeps the offered load per robot comparable when trips get longer or robots are added.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.grid import WarehouseGrid
from environment.warehouse_env import WarehouseEnv
from evaluation.evaluate import load_policy, run_policy_episodes, run_baseline_episodes, split_seeds
from training.config_utils import load_config, env_kwargs_from_cfg


def own_scenarios(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    sc = cfg["scenarios"]
    return [{"scenario": f"{layout}/{load}", "layout_type": layout, "load": load,
             "order_lambda": lam, "num_robots": sc["num_robots"], "order_mode": "synthetic"}
            for layout in sc["layouts"] for load, lam in sc["loads"].items()]


def test_scenarios(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    tc = cfg["test_scenarios"]
    env_cfg = cfg["environment"]
    ref = WarehouseGrid(env_cfg["grid_width"], env_cfg["grid_height"], tc["reference_layout"]).mean_trip_length()
    out = []
    for layout in tc["layouts"]:
        grid = WarehouseEnv(**env_kwargs_from_cfg(env_cfg, layout_type=layout)).grid
        trip = grid.mean_trip_length()
        for n in tc["num_robots"]:
            lam = tc["load_per_5_robots"] * (n / 5.0) * (ref / trip)
            out.append({"scenario": f"{layout}/n{n}", "layout_type": layout, "load": tc["load_name"],
                        "order_lambda": round(lam, 4), "num_robots": n, "order_mode": tc["order_mode"],
                        "trip_length": round(trip, 2), "trip_ref": round(ref, 2)})
    return out


def run_info(run_dir: str) -> Dict[str, Any]:
    with open(os.path.join(run_dir, "config.json")) as f:
        c = json.load(f)
    return {"method": c["method"], "train_seed": c["seed"], "run": os.path.basename(os.path.normpath(run_dir)),
            "variant": c.get("variant", c["method"])}


def evaluate_run(run_dir: str, checkpoint: str, env_kwargs: Dict[str, Any], seeds: List[int],
                 cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    info = run_info(run_dir)
    path = os.path.join(run_dir, checkpoint)
    if info["method"] == "hier":
        from agents.hier_agent import load_allocator, run_hier_episodes
        return run_hier_episodes(load_allocator(path), env_kwargs, seeds, cfg["evaluation"]["torch_seed"])
    probe = WarehouseEnv(**env_kwargs)
    runner = load_policy(info["method"], path, probe.obs_dim, probe.state_dim,
                         deterministic=cfg["evaluation"]["deterministic"])
    return run_policy_episodes(runner, env_kwargs, seeds, cfg["evaluation"]["torch_seed"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--matrix", choices=["own", "test"], default="own")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--runs", nargs="*", default=[], help="Training run directories")
    parser.add_argument("--checkpoint", default="best_dev.pt")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    if args.matrix == "test" and args.split != "test":
        print("(!) Held-out layouts evaluated on dev seeds: fine for smoke tests, not for reporting.")
    seeds = split_seeds(cfg, args.split)[: args.episodes]
    scenarios = own_scenarios(cfg) if args.matrix == "own" else test_scenarios(cfg)

    rows = []
    for sc in scenarios:
        kw = env_kwargs_from_cfg(cfg["environment"], layout_type=sc["layout_type"],
                                 order_lambda=sc["order_lambda"], num_robots=sc["num_robots"],
                                 order_mode=sc["order_mode"])
        tag = {k: v for k, v in sc.items()}
        if not args.no_baseline:
            for ep in run_baseline_episodes(kw, seeds):
                rows.append({"method": "baseline", "variant": "baseline", "train_seed": -1, "run": "baseline",
                             **tag, **ep})
        for run_dir in args.runs:
            info = run_info(run_dir)
            try:
                eps = evaluate_run(run_dir, args.checkpoint, kw, seeds, cfg)
            except ValueError as err:  # e.g. a MAPPO critic padded for fewer robots
                print(f"(!) {info['run']} on {sc['scenario']}: {err}")
                continue
            rows.extend({**info, **tag, **ep} for ep in eps)
        df = pd.DataFrame([r for r in rows if r["scenario"] == sc["scenario"]])
        print(f"--> {sc['scenario']:42s} lambda={sc['order_lambda']:.3f} | " + "  ".join(
            f"{m}={g['throughput'].mean():.1f}" for m, g in df.groupby("variant", sort=False)), flush=True)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"--> Saved {len(rows)} episode rows to {args.output}")


if __name__ == "__main__":
    main()
