"""
generate_figures_3_and_6.py

Regenerates:
  - Figure 3: bar chart of all 38 models' accuracy, sorted descending,
    colour-coded by architecture family
  - Figure 6: accuracy vs. CPU inference time scatter plot

Uses the corrected, leak-free results in ALL_38_MODELS_COMBINED.csv.

Run with:
    python src/generate_figures_3_and_6.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import os

COMBINED_CSV = 'ALL_38_MODELS_COMBINED.csv'
OUTPUT_DIR = './figures'


def get_family(model_name):
    """Map a model name to its architecture family for colour-coding."""
    if model_name.startswith('ConvNeXt'):
        return 'ConvNeXt'
    if model_name.startswith('EfficientNetV2'):
        return 'EfficientNetV2'
    if model_name.startswith('EfficientNet'):
        return 'EfficientNet'
    if model_name.startswith('DenseNet'):
        return 'DenseNet'
    if model_name.startswith('ResNet'):
        return 'ResNet'
    if model_name.startswith('Inception'):
        return 'Inception'
    if model_name.startswith('VGG'):
        return 'VGG'
    if model_name.startswith('MobileNet'):
        return 'MobileNet'
    if model_name.startswith('NASNet'):
        return 'NASNet'
    return 'Other'


FAMILY_COLORS = {
    'ConvNeXt': '#1f77b4',
    'EfficientNet': '#ff7f0e',
    'EfficientNetV2': '#2ca02c',
    'DenseNet': '#d62728',
    'ResNet': '#9467bd',
    'Inception': '#8c564b',
    'VGG': '#e377c2',
    'MobileNet': '#7f7f7f',
    'NASNet': '#bcbd22',
    'Other': '#17becf',
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(COMBINED_CSV)
    df['family'] = df['model_name'].apply(get_family)
    df_sorted = df.sort_values('accuracy', ascending=False).reset_index(drop=True)

    # ── FIGURE 3: bar chart, all 38 models, sorted, colour-coded ──────────
    fig, ax = plt.subplots(figsize=(16, 6))
    colors = [FAMILY_COLORS[f] for f in df_sorted['family']]
    ax.bar(range(len(df_sorted)), df_sorted['accuracy'], color=colors)
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(df_sorted['model_name'], rotation=90, fontsize=8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Test-set accuracy of all 38 CNN architectures, sorted descending')
    ax.set_ylim(0.85, 1.0)

    # Legend for families actually present
    present_families = df_sorted['family'].unique()
    handles = [mpatches.Patch(color=FAMILY_COLORS[f], label=f) for f in present_families]
    ax.legend(handles=handles, loc='upper right', ncol=2, fontsize=8)

    plt.tight_layout()
    fig3_path = os.path.join(OUTPUT_DIR, 'all_models_accuracy.png')
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved Figure 3 to {fig3_path}")

    # ── FIGURE 6: accuracy vs. inference time ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    for family in present_families:
        sub = df_sorted[df_sorted['family'] == family]
        ax.scatter(sub['inference_time_ms'], sub['accuracy'],
                   color=FAMILY_COLORS[family], label=family, s=60, alpha=0.8)

    # Label the top model and a couple of notable ones
    top_row = df_sorted.iloc[0]
    ax.annotate(top_row['model_name'],
                (top_row['inference_time_ms'], top_row['accuracy']),
                textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel('Inference Time (ms) — lower is better')
    ax.set_ylabel('Accuracy — higher is better')
    ax.set_title('Accuracy vs Speed Tradeoff (all 38 models)')
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig6_path = os.path.join(OUTPUT_DIR, 'accuracy_vs_speed.png')
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    print(f"Saved Figure 6 to {fig6_path}")

    # ── Console summary for the caption text ──────────────────────────────
    print("\n--- For your figure captions / text ---")
    print(f"Top model: {top_row['model_name']} "
          f"({top_row['accuracy']:.4f} acc, {top_row['model_size_mb']:.1f} MB, "
          f"{top_row['inference_time_ms']:.1f} ms)")

    # Fastest model among the top 10 by accuracy (useful for deployment framing)
    top10 = df_sorted.head(10)
    fastest_in_top10 = top10.loc[top10['inference_time_ms'].idxmin()]
    print(f"Fastest model in top 10 by accuracy: {fastest_in_top10['model_name']} "
          f"({fastest_in_top10['accuracy']:.4f} acc, {fastest_in_top10['inference_time_ms']:.1f} ms)")


if __name__ == "__main__":
    main()