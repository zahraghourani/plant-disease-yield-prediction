"""
compute_stats_table4.py

Computes Wilson 95% confidence intervals for accuracy and McNemar's
test p-values (vs. the top-accuracy model) for all 38 CNN models,
producing the data needed for the paper's Table 4.

Run with:
    python src/compute_stats_table4.py
"""
import pandas as pd
import numpy as np
from scipy import stats

PRED_CSV = 'test_predictions_all_38.csv'
COMBINED_CSV = 'ALL_38_MODELS_COMBINED.csv'
OUTPUT_CSV = 'table4_full_38_model_stats.csv'


def wilson_ci(successes, n, z=1.96):
    """Wilson score interval for a binomial proportion (95% CI by default)."""
    phat = successes / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt((phat * (1 - phat) / n) + (z**2 / (4 * n**2)))) / denom
    return center - margin, center + margin


def mcnemar_test(correct_a, correct_b):
    """
    McNemar's test (with continuity correction) comparing two models'
    per-image correctness on the SAME test set.

    correct_a, correct_b: boolean arrays, same length, same order.
    Returns the p-value.
    """
    # b = model A right, model B wrong; c = model A wrong, model B right
    b = np.sum(correct_a & ~correct_b)
    c = np.sum(~correct_a & correct_b)

    n_discordant = b + c
    if n_discordant == 0:
        return 1.0  # models agree on every image, nothing to test

    # Continuity-corrected McNemar's chi-square statistic
    chi2_stat = (abs(b - c) - 1) ** 2 / n_discordant
    p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
    return p_value


def main():
    pred_df = pd.read_csv(PRED_CSV)
    combined_df = pd.read_csv(COMBINED_CSV)

    model_cols = [c for c in pred_df.columns if c not in ('filename', 'y_true')]
    y_true = pred_df['y_true'].values
    n_test = len(y_true)

    # Precompute correctness (boolean array) per model
    correctness = {
        model: (pred_df[model].values == y_true)
        for model in model_cols
    }

    # Identify the top model by accuracy (from the combined CSV)
    top_model = combined_df.loc[combined_df['accuracy'].idxmax(), 'model_name']
    print(f"Top model (reference for McNemar's test): {top_model}")
    top_correct = correctness[top_model]

    rows = []
    for model in model_cols:
        row = combined_df[combined_df['model_name'] == model].iloc[0]
        n_correct = correctness[model].sum()
        ci_low, ci_high = wilson_ci(n_correct, n_test)

        if model == top_model:
            p_value = np.nan  # no self-comparison, matches original Table 4 style ("–")
        else:
            p_value = mcnemar_test(top_correct, correctness[model])

        rows.append({
            'model_name': model,
            'person': row.get('person', ''),
            'accuracy': round(row['accuracy'], 4),
            'precision_macro': round(row['precision_macro'], 4),
            'recall_macro': round(row['recall_macro'], 4),
            'f1_macro': round(row['f1_macro'], 4),
            'f1_weighted': round(row['f1_weighted'], 4),
            'mcc': round(row['mcc'], 4),
            'roc_auc_ovr': round(row['roc_auc_ovr'], 4) if not pd.isna(row['roc_auc_ovr']) else None,
            'ci_95_low': round(ci_low, 3),
            'ci_95_high': round(ci_high, 3),
            'p_value_vs_top': round(p_value, 3) if not pd.isna(p_value) else None,
            'significant_p<0.05': (p_value < 0.05) if not pd.isna(p_value) else False,
            'inference_time_ms': row.get('inference_time_ms', None),
            'model_size_mb': row.get('model_size_mb', None),
        })

    result_df = pd.DataFrame(rows).sort_values('accuracy', ascending=False).reset_index(drop=True)
    result_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nSaved full stats table to {OUTPUT_CSV}\n")
    print("=" * 100)
    print(f"TOP 10 MODELS (n_test = {n_test})")
    print("=" * 100)
    display_cols = ['model_name', 'accuracy', 'f1_macro', 'mcc', 'ci_95_low', 'ci_95_high', 'p_value_vs_top']
    print(result_df[display_cols].head(10).to_string(index=False))

    print("\n" + "=" * 100)
    print("STATISTICAL SIGNIFICANCE SUMMARY")
    print("=" * 100)
    n_indistinguishable = (~result_df['significant_p<0.05']).sum() - 1  # exclude top model itself
    print(f"Models statistically indistinguishable from {top_model} (p >= 0.05): {n_indistinguishable}")
    print(f"Models significantly weaker (p < 0.05): {result_df['significant_p<0.05'].sum()}")

    weakest = result_df.iloc[-1]
    print(f"\nWeakest model: {weakest['model_name']} "
          f"(acc={weakest['accuracy']:.4f}, p={weakest['p_value_vs_top']})")


if __name__ == "__main__":
    main()