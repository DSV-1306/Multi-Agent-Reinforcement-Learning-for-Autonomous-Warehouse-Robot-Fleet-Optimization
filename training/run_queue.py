"""
Phase 5 experiment queue: every (variant, seed) in `experiments` of the YAML, run as separate processes,
`experiments.workers` at a time. Finished runs (summary.json present) are skipped, so the queue can be
stopped and restarted. Each run logs to <experiments.runs_dir>/<variant>_s<seed>.log.

    python training/run_queue.py                 # everything
    python training/run_queue.py --only ippo mappo --dry-run
"""

import argparse
import os
import subprocess
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.config_utils import load_config

SCRIPTS = {"ippo": "training/train_ippo.py", "mappo": "training/train_mappo.py", "hier": "training/train_hier.py"}
POLL_SECONDS = 5


def build_jobs(cfg: Dict, only: List[str] = None) -> List[Dict]:
    ex = cfg["experiments"]
    jobs = []
    for seed in ex["seeds"]:  # seed-major order: every variant gets seed 1 before any gets seed 2
        for name, v in ex["variants"].items():
            if only and name not in only:
                continue
            steps = ex["hier_timesteps"] if v["method"] == "hier" else ex["rl_timesteps"]
            run_name = f"{name}_s{seed}"
            cmd = [sys.executable, "-u", SCRIPTS[v["method"]], "--seed", str(seed), "--timesteps", str(steps),
                   "--run-name", run_name, "--variant", name, "--set", f"training.runs_dir={ex['runs_dir']}"]
            for o in v["overrides"]:
                cmd += ["--set", o]
            jobs.append({"name": run_name, "cmd": cmd, "dir": os.path.join(ex["runs_dir"], run_name)})
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--only", nargs="*", default=None, help="Variant names to run")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    workers = args.workers or cfg["experiments"]["workers"]
    todo = [j for j in build_jobs(cfg, args.only) if not os.path.exists(os.path.join(j["dir"], "summary.json"))]
    print(f"--> {len(todo)} job(s) to run with {workers} worker(s)", flush=True)
    if args.dry_run:
        for j in todo:
            print(" ", " ".join(j["cmd"]))
        return
    os.makedirs(cfg["experiments"]["runs_dir"], exist_ok=True)
    running = []
    start = time.time()
    while todo or running:
        while todo and len(running) < workers:
            j = todo.pop(0)
            log = open(os.path.join(cfg["experiments"]["runs_dir"], f"{j['name']}.log"), "w")
            running.append((j, subprocess.Popen(j["cmd"], stdout=log, stderr=subprocess.STDOUT), log, time.time()))
            print(f"[{(time.time() - start) / 60:6.1f} min] start {j['name']}", flush=True)
        for item in list(running):
            j, proc, log, t0 = item
            if proc.poll() is not None:
                log.close()
                running.remove(item)
                status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
                print(f"[{(time.time() - start) / 60:6.1f} min] {status} {j['name']} "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        time.sleep(POLL_SECONDS)
    print(f"--> queue finished in {(time.time() - start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
