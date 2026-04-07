import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

RESULTS_DIR = "./results"

# ── Load all results ──────────────────────────────────────────────
dfs = []
for person in ['zahra', 'sireen', 'tala']:
    path = os.path.join(RESULTS_DIR, f"{person}_results.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['person'] = person
        dfs.append(df)
        print(f"✓ Loaded {len(df)} models from {person}")
    else:
        print(f"✗ Missing: {path}")

all_results = pd.concat(dfs, ignore_index=True)
all_results = all_results.sort_values('accuracy', ascending=False)

# ── Print leaderboard ─────────────────────────────────────────────
print("\n" + "="*80)
print("FULL MODEL LEADERBOARD")
print("="*80)
cols = ['model_name', 'person', 'accuracy', 'f1_macro', 
        'precision_macro', 'recall_macro', 'roc_auc_ovr', 
        'mcc', 'inference_time_ms', 'model_size_mb']
print(all_results[cols].to_string(index=False))

print("\n🏆 TOP 5 MODELS:")
print(all_results[cols].head())

# ── Save leaderboard ──────────────────────────────────────────────
all_results.to_csv(os.path.join(RESULTS_DIR, "all_models_leaderboard.csv"), index=False)
print(f"\n✓ Leaderboard saved to {RESULTS_DIR}/all_models_leaderboard.csv")

# ── Plot 1: Accuracy comparison ───────────────────────────────────
plt.figure(figsize=(20, 7))
colors = {'zahra': '#2196F3', 'sireen': '#4CAF50', 'tala': '#FF5722'}
bar_colors = [colors[p] for p in all_results['person']]
plt.bar(all_results['model_name'], all_results['accuracy'], color=bar_colors)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Accuracy')
plt.title('All Models - Accuracy Comparison')
plt.axhline(y=all_results['accuracy'].mean(), color='red', linestyle='--', label='Mean')
plt.legend()
# Add legend for persons
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=p.capitalize()) for p, c in colors.items()]
plt.legend(handles=legend_elements)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "figures/all_models_accuracy.png"), dpi=300)
plt.close()
print("✓ Accuracy plot saved")

# ── Plot 2: Accuracy vs Inference Time (efficiency) ───────────────
plt.figure(figsize=(12, 8))
for person, group in all_results.groupby('person'):
    plt.scatter(group['inference_time_ms'], group['accuracy'], 
                label=person.capitalize(), s=100, color=colors[person])
    for _, row in group.iterrows():
        plt.annotate(row['model_name'], 
                    (row['inference_time_ms'], row['accuracy']),
                    fontsize=7, ha='left')
plt.xlabel('Inference Time (ms) — Lower is better')
plt.ylabel('Accuracy — Higher is better')
plt.title('Accuracy vs Speed Tradeoff')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "figures/accuracy_vs_speed.png"), dpi=300)
plt.close()
print("✓ Accuracy vs Speed plot saved")

# ── Plot 3: Heatmap of all metrics ───────────────────────────────
metric_cols = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'mcc']
heatmap_data = all_results.set_index('model_name')[metric_cols]
plt.figure(figsize=(10, max(8, len(heatmap_data) * 0.4)))
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGn', 
            linewidths=0.5, vmin=0.7, vmax=1.0)
plt.title('All Models - Metrics Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "figures/metrics_heatmap.png"), dpi=300)
plt.close()
print("✓ Heatmap saved")

# ── Per-person summary ────────────────────────────────────────────
print("\n" + "="*80)
print("PER-PERSON BEST MODEL")
print("="*80)
for person, group in all_results.groupby('person'):
    best = group.iloc[0]
    print(f"\n{person.capitalize()}'s best: {best['model_name']}")
    print(f"  Accuracy:  {best['accuracy']:.4f}")
    print(f"  F1-Macro:  {best['f1_macro']:.4f}")
    print(f"  MCC:       {best['mcc']:.4f}")
    print(f"  Inference: {best['inference_time_ms']:.2f} ms")