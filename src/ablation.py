import pandas as pd
import matplotlib.pyplot as plt

# Your ablation data
experiments = ['Climate only', 'Climate + crop type', 
               'Climate + disease severity', 'Full model']
r2 = [-0.1151, 0.9699, 0.9606, 0.9743]
rmse = [95171.64, 15643.07, 17882.89, 14453.27]
colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# R² plot
bars1 = axes[0].bar(experiments, r2, color=colors, edgecolor='black')
axes[0].set_ylabel('R² Score')
axes[0].set_title('R² by Feature Set')
axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
for bar, val in zip(bars1, r2):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., 
                height + 0.02 if height >= 0 else height - 0.08,
                f'{val:.3f}', ha='center', 
                va='bottom' if height >= 0 else 'top', fontweight='bold')
axes[0].set_ylim(-0.3, 1.1)

# RMSE plot
bars2 = axes[1].bar(experiments, rmse, color=colors, edgecolor='black')
axes[1].set_ylabel('RMSE (hg/ha)')
axes[1].set_title('RMSE by Feature Set')
for bar, val in zip(bars2, rmse):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1000,
                f'{val:,.0f}', ha='center', va='bottom', fontweight='bold')

plt.suptitle('Ablation Study: XGBoost Yield Prediction', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/ablation_study_yield_prediction.png', dpi=150, bbox_inches='tight')
plt.show()