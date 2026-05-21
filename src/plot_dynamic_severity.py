"""
Plot dynamic severity visualizations for the paper.

Usage:
    python src/plot_dynamic_severity.py --input ./results/computed_severity.csv

Outputs:
    - results/figures/dynamic_vs_literature_scatter.png
    - results/figures/severity_distribution_by_crop.png
    - results/figures/per_class_severity_comparison.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})


def plot_scatter_literature_vs_computed(df, out_dir):
    """Figure: scatter plot of literature vs. computed severity."""
    # Aggregate by class
    agg = df.groupby("true_class").agg({
        "computed_severity": "mean",
        "literature_severity": "first"
    }).reset_index()

    # Filter to diseased classes only
    diseased = agg[agg["literature_severity"] > 0].copy()

    fig, ax = plt.subplots(figsize=(6, 5))

    x = diseased["literature_severity"]
    y = diseased["computed_severity"]
    labels = [l.replace("___", " ").replace("_", " ") for l in diseased["true_class"]]

    # Scatter
    ax.scatter(x, y, s=80, c="#2d6cdf", alpha=0.7, edgecolors="white", linewidth=0.5)

    # Diagonal reference line (perfect agreement)
    lims = [0, max(x.max(), y.max()) * 1.1]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1, label="Perfect agreement")

    # Annotate points
    for i, txt in enumerate(labels):
        ax.annotate(txt, (x.iloc[i], y.iloc[i]), fontsize=7, 
                   xytext=(5, 5), textcoords="offset points", alpha=0.8)

    ax.set_xlabel("Literature Severity (fixed)")
    ax.set_ylabel("Computed Severity (dynamic)")
    ax.set_title("Dynamic vs. Literature Severity by Disease Class")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.2)

    # Correlation
    if len(diseased) > 2:
        corr = diseased["computed_severity"].corr(diseased["literature_severity"])
        ax.text(0.05, 0.95, f"Pearson r = {corr:.3f}", transform=ax.transAxes,
               fontsize=9, verticalalignment="top",
               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    out_path = out_dir / "dynamic_vs_literature_scatter.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_distribution_by_crop(df, out_dir):
    """Figure: box plots of computed severity distribution by crop."""

    # Map disease to crop
    DISEASE_TO_CROP = {
        "Corn___Common_Rust": "Maize", "Corn___Leaf_Blight": "Maize",
        "Corn___Healthy": "Maize", "Potato___Early_Blight": "Potatoes",
        "Potato___Late_Blight": "Potatoes", "Potato___Healthy": "Potatoes",
        "Rice___Brown_Spot": "Rice", "Rice___Hispa": "Rice",
        "Rice___Leaf_Blast": "Rice", "Rice___Healthy": "Rice",
        "Wheat___Brown_Rust": "Wheat", "Wheat___Yellow_Rust": "Wheat",
        "Wheat___Healthy": "Wheat", "Invalid": "Maize",
    }

    df["crop"] = df["true_class"].map(DISEASE_TO_CROP)

    # Only diseased (severity > 0)
    diseased = df[df["literature_severity"] > 0].copy()

    fig, ax = plt.subplots(figsize=(7, 4.5))

    crops = ["Maize", "Potatoes", "Rice", "Wheat"]
    data_by_crop = [diseased[diseased["crop"] == c]["computed_severity"].values 
                    for c in crops]

    bp = ax.boxplot(data_by_crop, labels=crops, patch_artist=True,
                    notch=False, sym="+")

    colors = ["#e8f4ea", "#e4f3ec", "#dce8ff", "#fff3e0"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#333")
        patch.set_linewidth(1.2)

    for whisker in bp["whiskers"]:
        whisker.set(color="#555", linewidth=1)
    for cap in bp["caps"]:
        cap.set(color="#555", linewidth=1)
    for median in bp["medians"]:
        median.set(color="#c0392b", linewidth=2)

    ax.set_ylabel("Computed Severity")
    ax.set_xlabel("Crop Type")
    ax.set_title("Distribution of Dynamic Severity Scores by Crop")
    ax.grid(True, axis="y", alpha=0.2)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    out_path = out_dir / "severity_distribution_by_crop.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_per_class_comparison(df, out_dir):
    """Figure: grouped bar chart — literature vs. computed per class."""

    agg = df.groupby("true_class").agg({
        "computed_severity": ["mean", "std"],
        "literature_severity": "first"
    }).reset_index()

    agg.columns = ["class", "computed_mean", "computed_std", "literature"]

    # Sort by literature severity descending
    agg = agg.sort_values("literature", ascending=False)

    # Filter to diseased
    diseased = agg[agg["literature"] > 0].copy()

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(diseased))
    width = 0.35

    labels = [l.replace("___", " ").replace("_", " ") for l in diseased["class"]]

    bars1 = ax.bar(x - width/2, diseased["literature"], width, 
                   label="Literature (fixed)", color="#95a5a6", alpha=0.8,
                   edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, diseased["computed_mean"], width,
                   yerr=diseased["computed_std"],
                   label="Computed (dynamic)", color="#2d6cdf", alpha=0.8,
                   edgecolor="white", linewidth=0.5,
                   capsize=3, error_kw={"elinewidth": 1, "alpha": 0.6})

    ax.set_ylabel("Severity Score")
    ax.set_xlabel("Disease Class")
    ax.set_title("Literature vs. Dynamic Severity by Disease Class")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.2)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    out_path = out_dir / "per_class_severity_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def print_summary_stats(df):
    """Print key numbers for the paper text."""
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS FOR PAPER")
    print("=" * 60)

    agg = df.groupby("true_class").agg({
        "computed_severity": ["mean", "std", "min", "max", "count"],
        "literature_severity": "first",
        "area_ratio": "mean"
    }).reset_index()

    diseased = agg[agg["literature_severity"] > 0]

    print(f"\nTotal images processed: {len(df)}")
    print(f"Diseased classes: {len(diseased)}")
    print(f"Mean computed severity (diseased): {diseased['computed_severity']['mean'].mean():.3f}")
    print(f"Mean literature severity (diseased): {diseased['literature_severity']['first'].mean():.3f}")
    print(f"Mean absolute difference: {(diseased['computed_severity']['mean'] - diseased['literature_severity']['first']).abs().mean():.3f}")

    if len(diseased) > 2:
        corr = diseased['computed_severity']['mean'].corr(diseased['literature_severity']['first'])
        print(f"Pearson correlation: {corr:.3f}")

    # Per-crop stats
    DISEASE_TO_CROP = {
        "Corn___Common_Rust": "Maize", "Corn___Leaf_Blight": "Maize",
        "Corn___Healthy": "Maize", "Potato___Early_Blight": "Potatoes",
        "Potato___Late_Blight": "Potatoes", "Potato___Healthy": "Potatoes",
        "Rice___Brown_Spot": "Rice", "Rice___Hispa": "Rice",
        "Rice___Leaf_Blast": "Rice", "Rice___Healthy": "Rice",
        "Wheat___Brown_Rust": "Wheat", "Wheat___Yellow_Rust": "Wheat",
        "Wheat___Healthy": "Wheat", "Invalid": "Maize",
    }
    df["crop"] = df["true_class"].map(DISEASE_TO_CROP)
    crop_stats = df[df["literature_severity"] > 0].groupby("crop")["computed_severity"].mean()

    print(f"\nPer-crop mean computed severity:")
    for crop, sev in crop_stats.items():
        print(f"  {crop}: {sev:.3f}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./results/computed_severity.csv",
                       help="Path to computed_severity.csv")
    parser.add_argument("--out_dir", default="./results/figures",
                       help="Output directory for figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} records from {args.input}")

    print("\nGenerating figures...")
    plot_scatter_literature_vs_computed(df, out_dir)
    plot_distribution_by_crop(df, out_dir)
    plot_per_class_comparison(df, out_dir)

    print_summary_stats(df)

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()