"""
Publication-grade Plotting and Comparative Analytics Dashboard.
Generates comparative bar charts, error intervals, and metric dashboards.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_benchmark_dashboard(
    csv_path: str = "results/evaluation_results.csv",
    output_dir: str = "results/plots",
):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

    methods = df["Method"].tolist()
    palette = {m: colors[i % len(colors)] for i, m in enumerate(methods)}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Multi-Agent Warehouse Robot Fleet: Comparative Benchmark", fontsize=16, fontweight="bold", y=0.98)

    # 1. Fleet Throughput (Higher is better)
    ax1 = axes[0, 0]
    bars1 = ax1.bar(
        df["Method"],
        df["avg_throughput"],
        yerr=df.get("std_throughput", 0),
        capsize=5,
        color=[palette[m] for m in methods],
        alpha=0.85,
    )
    ax1.set_title("Fleet Throughput (Orders Delivered / Episode)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Completed Orders", fontsize=11)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval / 2.0, f"{yval:.1f}", ha="center", va="center", color="white", fontweight="bold", fontsize=12)

    # 2. Average Order Completion Time / Latency (Lower is better)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(
        df["Method"],
        df["avg_completion_time"],
        yerr=df.get("std_completion_time", 0),
        capsize=5,
        color=[palette[m] for m in methods],
        alpha=0.85,
    )
    ax2.set_title("Average Order Latency (Steps to Deliver)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Simulation Steps", fontsize=11)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval / 2.0, f"{yval:.1f}", ha="center", va="center", color="white", fontweight="bold", fontsize=12)

    # 3. Collision Count (Lower is better)
    ax3 = axes[1, 0]
    bars3 = ax3.bar(
        df["Method"],
        df["avg_collisions"],
        yerr=df.get("std_collisions", 0),
        capsize=5,
        color=[palette[m] for m in methods],
        alpha=0.85,
    )
    ax3.set_title("Fleet Collisions per Episode", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Total Collisions", fontsize=11)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, yval / 2.0 if yval > 5 else yval + 1, f"{yval:.1f}", ha="center", va="bottom" if yval <= 5 else "center", color="white" if yval > 5 else "black", fontweight="bold", fontsize=12)

    # 4. Total Travel Distance / Energy (Steps taken)
    ax4 = axes[1, 1]
    bars4 = ax4.bar(
        df["Method"],
        df["avg_travel_distance"],
        yerr=df.get("std_travel_distance", 0),
        capsize=5,
        color=[palette[m] for m in methods],
        alpha=0.85,
    )
    ax4.set_title("Fleet Total Travel Distance (Grid Steps)", fontsize=13, fontweight="bold")
    ax4.set_ylabel("Sum of Robot Steps", fontsize=11)
    for bar in bars4:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2.0, yval / 2.0, f"{int(yval)}", ha="center", va="center", color="white", fontweight="bold", fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    dashboard_path = os.path.join(output_dir, "comparative_dashboard.png")
    plt.savefig(dashboard_path, dpi=200)
    plt.close()
    print(f"--> [Plotter] Saved comparative dashboard to: {dashboard_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="results/evaluation_results.csv")
    parser.add_argument("--outdir", type=str, default="results/plots")
    args = parser.parse_args()

    plot_benchmark_dashboard(csv_path=args.csv, output_dir=args.outdir)
