"""
Plot learning curves (dev-seed evaluations) from one or more training runs.

    python evaluation/plot_learning_curves.py --runs results/runs/ippo_seed1 results/runs/mappo_seed1 \
        --baseline-csv results/eval/p1_base_noop_true.csv --output results/plots/p1_curves.png

Panels (each its own y-axis): dev throughput (band = +/-1 std over dev episodes), dev robot-robot
collisions, policy entropy, critic explained variance. Colour is fixed per method, so several seeds
of one method share a colour. The baseline (if given) is a grey dashed reference line.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Fixed categorical order; colour follows the variant (all seeds of one variant share it), never its rank
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BASELINE_COLOR = "#52514e"
MAX_RUNS_WITH_BANDS = 3  # +/-1 std bands only for small figures; more runs make them unreadable
TEXT = "#0b0b0b"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

PANELS = [
    ("dev_throughput_mean", "dev_throughput_std", "Orders delivered / episode (dev seeds)", "throughput_mean"),
    ("dev_robot_collisions_mean", None, "Robot-robot collisions / episode (dev)", "robot_collisions_mean"),
    ("entropy", None, "Policy entropy (nats)", None),
    ("explained_variance", None, "Critic explained variance", None),
]


def run_label(run_dir: str):
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            c = json.load(f)
        variant = c.get("variant", c["method"])
        return variant, f"{variant} s{c['seed']}"
    name = os.path.basename(os.path.normpath(run_dir))
    return name.split("_")[0], name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--baseline-csv", default=None, help="Summary CSV from evaluation/evaluate.py")
    parser.add_argument("--output", default="results/plots/learning_curves.png")
    args = parser.parse_args()

    baseline = None
    if args.baseline_csv:
        b = pd.read_csv(args.baseline_csv)
        baseline = b[b["method"] == "baseline"].iloc[0]

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
                         "xtick.color": TEXT, "ytick.color": TEXT})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor=SURFACE)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    variants = list(dict.fromkeys(run_label(r)[0] for r in args.runs))
    if len(variants) > len(PALETTE):
        raise ValueError(f"at most {len(PALETTE)} variants per figure; split into several plots")
    for run_dir in args.runs:
        df = pd.read_csv(os.path.join(run_dir, "learning_curve.csv"))
        variant, label = run_label(run_dir)
        color = PALETTE[variants.index(variant)]
        x = df["env_steps"] / 1e6
        for ax, (col, std_col, title, _) in zip(axes.flat, PANELS):
            if col not in df:
                continue
            y = df[col]
            ax.plot(x, y, color=color, linewidth=2, label=label, marker="o", markersize=3)
            if std_col and std_col in df and len(args.runs) <= MAX_RUNS_WITH_BANDS:
                ax.fill_between(x, y - df[std_col], y + df[std_col], color=color, alpha=0.15, linewidth=0)
            last = y.dropna()
            if len(last):
                ax.annotate(label, (x[last.index[-1]], last.iloc[-1]), xytext=(4, 0),
                            textcoords="offset points", fontsize=8, color=TEXT, va="center")

    for ax, (_, _, title, base_key) in zip(axes.flat, PANELS):
        if baseline is not None and base_key:
            ax.axhline(baseline[base_key], color=BASELINE_COLOR, linestyle="--", linewidth=1.5,
                       label="Baseline (A*) on dev")
        ax.set_title(title, loc="left", color=TEXT, fontsize=10)
        ax.set_xlabel("Environment steps (millions)")
    # One legend entry per variant (seeds share a colour and are direct-labelled on the lines)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=PALETTE[i], linewidth=2, label=v) for i, v in enumerate(variants)]
    if baseline is not None:
        handles.append(Line2D([], [], color=BASELINE_COLOR, linestyle="--", linewidth=1.5, label="Baseline (A*) on dev"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=130, facecolor=SURFACE)
    print(f"--> Saved {args.output}")


if __name__ == "__main__":
    main()
