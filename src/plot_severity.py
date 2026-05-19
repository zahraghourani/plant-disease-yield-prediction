"""
plot_severity.py
----------------
Reads severity_comparison.csv and produces:
  1. severity_comparison.png  — clean grouped bar chart
  2. Prints LaTeX table code

HOW TO RUN (from project root):
    python src/plot_severity.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT       = Path(__file__).resolve().parents[1]
INPUT_CSV  = ROOT / "results" / "severity_comparison.csv"
OUTPUT_PNG = ROOT / "results" / "severity_comparison.png"

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
df["literature_severity"] = pd.to_numeric(df["literature_severity"], errors="coerce")

# ── Short display names ───────────────────────────────────────────────────────
NAME_MAP = {
    "Corn___Common_Rust":    "Corn\nCommon Rust",
    "Corn___Healthy":        "Corn\nHealthy",
    "Corn___Leaf_Blight":    "Corn\nLeaf Blight",
    "Invalid":               "Invalid",
    "Potato___Early_Blight": "Potato\nEarly Blight",
    "Potato___Healthy":      "Potato\nHealthy",
    "Potato___Late_Blight":  "Potato\nLate Blight",
    "Rice___Brown_Spot":     "Rice\nBrown Spot",
    "Rice___Healthy":        "Rice\nHealthy",
    "Rice___Hispa":          "Rice\nHispa",
    "Rice___Leaf_Blast":     "Rice\nLeaf Blast",
    "Wheat___Brown_Rust":    "Wheat\nBrown Rust",
    "Wheat___Healthy":       "Wheat\nHealthy",
    "Wheat___Yellow_Rust":   "Wheat\nYellow Rust",
}
df["short_name"] = df["disease_class"].map(NAME_MAP).fillna(df["disease_class"])

# ── Remove Invalid row from disease comparison (not a disease class) ──────────
df_plot = df[df["disease_class"] != "Invalid"].reset_index(drop=True)

n = len(df_plot)
x = np.arange(n)
width = 0.35

# ── Clip error bars: never go below 0 or above 1 ─────────────────────────────
means  = df_plot["computed_mean"].values
stds   = df_plot["computed_std"].values
lits   = df_plot["literature_severity"].values

# Lower error: don't go below 0
err_low  = np.minimum(stds, means)
# Upper error: don't go above 1
err_high = np.minimum(stds, 1.0 - means)
err_high = np.maximum(err_high, 0)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 5.5))

bars1 = ax.bar(
    x - width/2, lits, width,
    label="Literature (fixed)",
    color="#3a7abf", alpha=0.9, zorder=3
)
bars2 = ax.bar(
    x + width/2, means, width,
    label="Computed (dynamic)",
    color="#e07b39", alpha=0.9, zorder=3,
    yerr=[err_low, err_high],
    capsize=5,
    error_kw={
        "elinewidth": 1.5,
        "ecolor": "#555555",
        "capthick": 1.5,
        "zorder": 4,
    }
)

# ── Styling ───────────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(df_plot["short_name"], fontsize=8.5)
ax.set_ylabel("Severity Score", fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_xlim(-0.6, n - 0.4)
ax.set_title(
    "Disease Severity: Literature (Fixed) vs Computed (Dynamic)\n"
    r"severity$_{\rm dynamic}$ = (lesion area / image area) $\times$ CNN confidence",
    fontsize=11
)
ax.legend(fontsize=10, loc="upper right")
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Value labels on bars
for bar in bars1:
    h = bar.get_height()
    if h > 0:
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.01,
            f"{h:.2f}", ha="center", va="bottom", fontsize=7, color="#3a7abf"
        )
for bar in bars2:
    h = bar.get_height()
    if h > 0:
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.01,
            f"{h:.2f}", ha="center", va="bottom", fontsize=7, color="#e07b39"
        )

fig.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
print(f"[SAVED] {OUTPUT_PNG}")
plt.close()

# ── Print LaTeX table ─────────────────────────────────────────────────────────
print("\n" + "="*70)
print("LATEX TABLE — copy into paper:")
print("="*70)

latex = r"""\begin{table}[h!]
\centering
\caption{Dynamic severity scores computed from Faster RCNN bounding box
  lesion areas and EfficientNetV2S confidence ($n = 15$ images per class),
  compared with fixed literature values. Std = standard deviation across
  images. High variance reflects inconsistent lesion detection by the
  COCO-pretrained Faster RCNN, especially for small-lesion classes.}
\label{tab:dynamic_severity}
\setlength{\tabcolsep}{6pt}
{\small
\begin{tabular}{lccc}
\toprule
\textbf{Disease Class} & \textbf{Literature} &
\textbf{Computed Mean} & \textbf{Computed Std}\\
\midrule
"""

for _, row in df[df["disease_class"] != "Invalid"].iterrows():
    cls  = row["disease_class"].replace("___", " ").replace("_", " ")
    lit  = f"{row['literature_severity']:.2f}" if pd.notna(row["literature_severity"]) else "N/A"
    mean = f"{row['computed_mean']:.4f}"
    std  = f"{row['computed_std']:.4f}"
    latex += f"{cls} & {lit} & {mean} & {std} \\\\\n"

latex += r"""\bottomrule
\end{tabular}
}
\end{table}"""

print(latex)

# ── Key observations ──────────────────────────────────────────────────────────
print("\n" + "="*70)
print("KEY NUMBERS FOR PAPER TEXT:")
print("="*70)
for _, row in df_plot.iterrows():
    diff = abs(row["computed_mean"] - row["literature_severity"])
    print(f"  {row['disease_class']:30s}  lit={row['literature_severity']:.2f}  "
          f"comp={row['computed_mean']:.4f}  diff={diff:.4f}  std={row['computed_std']:.4f}")
print("="*70)