"""
Result figures from evaluation/stats.py summaries (Phase 5).

    python evaluation/plot_results.py --summary results/eval/p5_own_dev_summary.csv \
        --variants baseline ippo mappo hier --metrics throughput robot_collisions --output results/plots/p5_own_dev.png

One panel per metric (each with its own y-axis); bars = mean, whiskers = 95% CI (see stats.py for the
unit of replication). Colour follows the variant in a fixed order; the baseline is grey.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BASELINE_COLOR = "#8a8984"
TEXT, GRID, SURFACE = "#0b0b0b", "#e4e3df", "#fcfcfb"
TITLES = {"throughput": "Orders delivered / episode", "robot_collisions": "Robot-robot collisions / episode",
          "collisions": "All collisions / episode", "mean_latency": "Mean order latency (steps)"}
BAR_GROUP_WIDTH = 0.8
LEGEND_MAX_COLS = 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--variants", nargs="+", required=True, help="Order = colour order")
    parser.add_argument("--metrics", nargs="+", default=["throughput", "robot_collisions"])
    parser.add_argument("--scenarios", nargs="*", default=None, help="Default: all, ALL last")
    parser.add_argument("--title", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    s = pd.read_csv(args.summary)
    scenarios = args.scenarios or [x for x in dict.fromkeys(s["scenario"]) if x != "ALL"] + ["ALL"]
    colors = {}
    k = 0
    for v in args.variants:
        if v == "baseline":
            colors[v] = BASELINE_COLOR
        else:
            colors[v] = PALETTE[k]
            k += 1

    fig, axes = plt.subplots(len(args.metrics), 1, figsize=(max(8, 1.3 * len(scenarios) + 3), 3.6 * len(args.metrics)),
                             facecolor=SURFACE, squeeze=False)
    width = BAR_GROUP_WIDTH / len(args.variants)
    x = np.arange(len(scenarios))
    for ax, metric in zip(axes[:, 0], args.metrics):
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for i, v in enumerate(args.variants):
            sub = s[(s["metric"] == metric) & (s["variant"] == v)].set_index("scenario")
            means = [sub["mean"].get(sc, np.nan) for sc in scenarios]
            cis = [sub["ci95"].get(sc, np.nan) for sc in scenarios]
            ax.bar(x - BAR_GROUP_WIDTH / 2 + (i + 0.5) * width, means, width * 0.92, color=colors[v], label=v,
                   yerr=np.nan_to_num(cis), capsize=2, error_kw={"elinewidth": 1, "ecolor": TEXT})
        ax.set_xticks(x)
        ax.set_xticklabels([sc.replace("mapf:", "").replace("/", "\n") for sc in scenarios], fontsize=8, color=TEXT)
        ax.set_title(TITLES.get(metric, metric) + " (mean, 95% CI)", loc="left", fontsize=10, color=TEXT)
        ax.set_ylim(bottom=0)  # all metrics are non-negative; t-intervals can extend below 0
    handles, labels = axes[0, 0].get_legend_handles_labels()
    ncol = min(len(labels), LEGEND_MAX_COLS)
    legend_rows = -(-len(labels) // ncol)
    top = (0.90 if args.title else 0.94) - 0.05 * (legend_rows - 1)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, top + 0.05 * legend_rows),
               ncol=ncol, frameon=False)
    if args.title:
        fig.suptitle(args.title, y=0.995, fontsize=11, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, top))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=130, facecolor=SURFACE)
    print(f"--> Saved {args.output}")


if __name__ == "__main__":
    main()
