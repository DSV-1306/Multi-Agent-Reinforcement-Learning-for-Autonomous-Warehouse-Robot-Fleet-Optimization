"""
Means and 95% confidence intervals from the long CSV written by evaluation/scenario_matrix.py (Phase 5).

Unit of replication:
  * RL variants: the TRAINING SEED. Each seed's score is its mean over the evaluation episodes; the CI is a
    Student-t interval over seeds (n = 3 here, so t_{0.975,2} = 4.30 and intervals are wide on purpose).
  * Baseline (deterministic planner, no training): the EVALUATION EPISODE; t-interval over episodes.
  * "diff vs baseline": for each training seed, the mean over evaluation seeds of (RL - baseline) on the
    SAME evaluation seed (paired), then a t-interval over training seeds. A CI that excludes 0 is the
    evidence for a difference.

    python evaluation/stats.py --input results/eval/p5_own_dev.csv --output results/eval/p5_own_dev_summary.csv
"""

import argparse
import sys
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats as st

CONFIDENCE = 0.95
METRICS = ["throughput", "robot_collisions", "collisions", "mean_latency"]


def mean_ci(values: np.ndarray) -> Dict[str, float]:
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n == 0:
        return {"mean": float("nan"), "ci": float("nan"), "n": 0}
    if n == 1:
        return {"mean": float(v[0]), "ci": float("nan"), "n": 1}
    half = st.t.ppf(0.5 + CONFIDENCE / 2, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return {"mean": float(v.mean()), "ci": float(half), "n": n}


def summarize(df: pd.DataFrame, metrics: List[str] = METRICS) -> pd.DataFrame:
    rows = []
    base = df[df["variant"] == "baseline"].set_index(["scenario", "seed"])
    groups = [(sc, g) for sc, g in df.groupby("scenario", sort=False)] + [("ALL", df)]
    for scenario, g in groups:
        for variant, gv in g.groupby("variant", sort=False):
            for m in metrics:
                if variant == "baseline":
                    # per evaluation seed (averaged over scenarios when scenario == ALL)
                    vals = gv.groupby("seed")[m].mean().to_numpy()
                    r = {**mean_ci(vals), "unit": "eval episodes"}
                    diff = {"mean": 0.0, "ci": 0.0, "n": r["n"]}
                else:
                    vals = gv.groupby("train_seed")[m].mean().to_numpy()
                    r = {**mean_ci(vals), "unit": "training seeds"}
                    paired = gv.join(base[[m]].rename(columns={m: "base"}), on=["scenario", "seed"])
                    d = paired.assign(d=paired[m] - paired["base"]).groupby("train_seed")["d"].mean().to_numpy()
                    diff = mean_ci(d)
                rows.append({"scenario": scenario, "variant": variant, "metric": m, "mean": r["mean"],
                             "ci95": r["ci"], "n": r["n"], "unit": r["unit"],
                             "diff_vs_baseline": diff["mean"], "diff_ci95": diff["ci"]})
    return pd.DataFrame(rows)


def fmt(mean: float, ci: float) -> str:
    if np.isnan(mean):
        return "n/a"
    return f"{mean:.1f}" if np.isnan(ci) else f"{mean:.1f} ± {ci:.1f}"


def markdown_table(summary: pd.DataFrame, metric: str, with_diff: bool = True) -> str:
    s = summary[summary["metric"] == metric]
    variants = list(dict.fromkeys(s["variant"]))
    scenarios = list(dict.fromkeys(s["scenario"]))
    head = "| scenario | " + " | ".join(variants) + " |\n|---|" + "---|" * len(variants)
    lines = [head]
    for sc in scenarios:
        cells = []
        for v in variants:
            r = s[(s["scenario"] == sc) & (s["variant"] == v)]
            if r.empty:
                cells.append("n/a")
                continue
            r = r.iloc[0]
            cell = fmt(r["mean"], r["ci95"])
            if with_diff and v != "baseline" and not np.isnan(r["diff_vs_baseline"]):
                cell += f" (Δ {r['diff_vs_baseline']:+.1f} ± {r['diff_ci95']:.1f})"
            cells.append(cell)
        lines.append(f"| {sc} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="One or more scenario_matrix CSVs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", nargs="*", default=METRICS)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252
    df = pd.concat([pd.read_csv(p) for p in args.input], ignore_index=True)
    summary = summarize(df, args.metrics)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    summary.to_csv(args.output, index=False)
    md = [f"## {m} (mean ± 95% CI; Δ = paired difference vs baseline)\n\n{markdown_table(summary, m)}\n"
          for m in args.metrics]
    md_path = os.path.splitext(args.output)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    print(f"--> Saved {args.output} and {md_path}")


if __name__ == "__main__":
    main()
