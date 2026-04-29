# ============================================
# Manufacturing Conversion Cost Reduction Project
# Pareto Analysis + SPC (X-bar & R) + Fishbone Diagram
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os
import warnings
import sys

# -----------------------------
# SAFE CONFIG
# -----------------------------
warnings.filterwarnings('ignore')
np.random.seed(42)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Output folder
OUT = os.path.join(os.getcwd(), "outputs")
os.makedirs(OUT, exist_ok=True)

# Colors
BLUE   = "#2E5FA3"
RED    = "#C0392B"
GREEN  = "#27AE60"
ORANGE = "#E67E22"
GREY   = "#95A5A6"

# ============================================
# 1. DATA GENERATION
# ============================================
def generate_data(n=100):
    t = pd.date_range("2024-01-01", periods=n, freq="1h")
    cycle_time = 120 + np.random.normal(0, 7, n)

    spikes = np.random.choice(n, size=5, replace=False)
    cycle_time[spikes] += np.random.uniform(20, 40, 5)

    return pd.DataFrame({
        "timestamp": t,
        "cycle_time": cycle_time
    })

# ============================================
# 2. PARETO ANALYSIS
# ============================================
def plot_pareto():
    categories = [
        "Rework / Rejections", "Machine Downtime", "Material Waste",
        "Excess Cycle Time", "Scrap Loss", "Energy Overuse",
        "Labour Inefficiency", "Setup Errors",
    ]

    losses = np.array([310, 275, 205, 170, 135, 90, 65, 38])

    idx = np.argsort(losses)[::-1]
    cats = [categories[i] for i in idx]
    vals = losses[idx]
    cum  = np.cumsum(vals) / vals.sum() * 100

    fig, ax1 = plt.subplots(figsize=(12, 6))

    bar_colors = [GREEN if c <= 80 else GREY for c in cum]
    bars = ax1.bar(range(len(cats)), vals, color=bar_colors)

    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 4,
                 str(v), ha='center', fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(range(len(cats)), cum, color=BLUE, marker='o')
    ax2.axhline(80, color=RED, linestyle='--')

    ax1.set_xticks(range(len(cats)))
    ax1.set_xticklabels(cats, rotation=25, ha='right')
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Cumulative %")
    ax1.set_title("Pareto Analysis")

    vital_patch = mpatches.Patch(color=GREEN, label='Vital Few')
    many_patch  = mpatches.Patch(color=GREY, label='Useful Many')
    ax1.legend(handles=[vital_patch, many_patch])

    path = os.path.join(OUT, "pareto.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close()

    return path

# ============================================
# 3. SPC (X-bar & R)
# ============================================
def xbar_r_constants(n):
    A2 = {2:1.880,3:1.023,4:0.729,5:0.577,6:0.483}
    D3 = {2:0,3:0,4:0,5:0,6:0}
    D4 = {2:3.267,3:2.575,4:2.282,5:2.115,6:2.004}

    if n not in A2:
        raise ValueError("Subgroup size must be between 2 and 6")

    return A2[n], D3[n], D4[n]

def plot_spc(df, subgroup=5):
    vals = df["cycle_time"].values

    if len(vals) < subgroup:
        raise ValueError("Not enough data for SPC")

    n_groups = len(vals) // subgroup
    groups = vals[:n_groups * subgroup].reshape(n_groups, subgroup)

    xbars = groups.mean(axis=1)
    ranges = groups.max(axis=1) - groups.min(axis=1)

    Xbar = xbars.mean()
    Rbar = ranges.mean()

    A2, D3, D4 = xbar_r_constants(subgroup)

    UCL_x = Xbar + A2 * Rbar
    LCL_x = Xbar - A2 * Rbar
    UCL_r = D4 * Rbar
    LCL_r = D3 * Rbar

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    x = np.arange(n_groups)

    # X-bar chart
    out_x = (xbars > UCL_x) | (xbars < LCL_x)

    ax1.plot(x, xbars, marker='o', color=BLUE)
    ax1.axhline(Xbar, color=GREEN)
    ax1.axhline(UCL_x, color=RED, linestyle='--')
    ax1.axhline(LCL_x, color=ORANGE, linestyle='--')
    ax1.scatter(x[out_x], xbars[out_x], color=RED)

    ax1.set_title("SPC X-bar Chart")
    ax1.set_ylabel("Mean")

    # R chart
    out_r = ranges > UCL_r

    ax2.bar(x, ranges, color=BLUE, alpha=0.6)
    ax2.axhline(Rbar, color=GREEN)
    ax2.axhline(UCL_r, color=RED, linestyle='--')
    ax2.scatter(x[out_r], ranges[out_r], color=RED)

    ax2.set_title("SPC R Chart")
    ax2.set_xlabel("Subgroup")
    ax2.set_ylabel("Range")

    path = os.path.join(OUT, "spc_chart.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close()

    return path, {
        "Xbar": Xbar,
        "UCL_x": UCL_x,
        "LCL_x": LCL_x,
        "Rbar": Rbar,
        "UCL_r": UCL_r,
        "out_x": int(out_x.sum())
    }

# ============================================
# 4. FISHBONE DIAGRAM
# ============================================
def plot_fishbone():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    # Main spine
    ax.annotate("", xy=(14, 4), xytext=(2, 4),
                arrowprops=dict(arrowstyle='->', lw=3))

    # Problem box
    box = FancyBboxPatch((14, 3), 2, 2,
                         boxstyle="round", facecolor=RED)
    ax.add_patch(box)
    ax.text(15, 4, "HIGH COST",
            ha='center', va='center', color='white')

    # Causes
    causes = ["Man", "Machine", "Method", "Material"]

    y_positions = [6, 5, 3, 2]

    for i, cause in enumerate(causes):
        ax.plot([4, 8], [y_positions[i], 4], color='black')
        ax.text(3.5, y_positions[i], cause)

    path = os.path.join(OUT, "fishbone.png")
    fig.savefig(path, dpi=150)
    plt.close()

    return path

# ============================================
# MAIN
# ============================================
def main():
    print("Generating charts...")

    df = generate_data(100)

    plot_pareto()
    _, spc = plot_spc(df)
    plot_fishbone()

    print("\nSPC Summary:")
    print(f"Mean: {spc['Xbar']:.2f}")
    print(f"UCL: {spc['UCL_x']:.2f}")
    print(f"LCL: {spc['LCL_x']:.2f}")
    print(f"Out-of-control points: {spc['out_x']}")

    print("\nDone! Check 'outputs' folder.")

if __name__ == "__main__":
    main()
