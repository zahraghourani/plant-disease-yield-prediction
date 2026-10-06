"""
make_faster_rcnn_ap_figure.py

Redraws the Faster RCNN per-class AP@0.5 chart from your real CSV
(results/faster_rcnn/faster_rcnn_per_class_ap.csv), with:
  * the number of validation annotations (n) shown next to each class
  * classes with too few validation annotations (n < 5) drawn hatched in
    grey with an "n/a" label, instead of as misleading zero-height bars.

n values are the validation-set annotation counts printed by
check_val_class_counts.py for the seed-42 group-aware split.

Run from the project root:
    python make_faster_rcnn_ap_figure.py
Writes:  paper_figures_final/faster_rcnn_per_class_ap.png
         results/figures/faster_rcnn_per_class_ap.png
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "results/faster_rcnn/faster_rcnn_per_class_ap.csv"
OUTS = ["paper_figures_final/faster_rcnn_per_class_ap.png",
        "results/figures/faster_rcnn_per_class_ap.png"]
MIN_N = 5

VAL_N = {
    "Rice___Leaf_Blast": 99, "Wheat___Brown_Rust": 70, "Potato___Healthy": 54,
    "Corn___Healthy": 14, "Corn___Leaf_Blight": 22, "Potato___Late_Blight": 25,
    "Invalid": 55, "Potato___Early_Blight": 3, "Rice___Hispa": 52,
    "Rice___Healthy": 47, "Rice___Brown_Spot": 136, "Wheat___Healthy": 0,
    "Corn___Common_Rust": 14, "Wheat___Yellow_Rust": 28,
}


def pretty(name):
    return name.replace("___", " ").replace("_", " ")


df = pd.read_csv(CSV)
missing = [c for c in df["class_name"] if c not in VAL_N]
if missing:
    raise SystemExit(f"No validation count for: {missing}")
df["n"] = df["class_name"].map(VAL_N)
df = df.sort_values("ap_50").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(9, 6))
for i, r in df.iterrows():
    ok = r["n"] >= MIN_N
    ax.barh(i, r["ap_50"] if ok else 0.0, color="#2d6cdf" if ok else "#bbbbbb",
            edgecolor="white")
    if ok:
        ax.text(r["ap_50"] + 0.01, i, f"{r['ap_50']:.2f}", va="center", fontsize=9)
    else:
        ax.barh(i, 1.0, color="none", edgecolor="#888888", hatch="///", linewidth=0.8)
        ax.text(0.02, i, f"not evaluable (n={int(r['n'])})", va="center", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
ax.set_yticks(range(len(df)))
ax.set_yticklabels([f"{pretty(c)} (n={n})" for c, n in zip(df["class_name"], df["n"])], fontsize=9)
ax.set_xlim(0, 1.08)
ax.set_xlabel("AP@0.5")
ax.set_title("Faster RCNN per-class AP@0.5 (held-out validation set)")
ax.grid(axis="x", linestyle="--", alpha=0.35)
fig.tight_layout()
for out in OUTS:
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("wrote", out)