"""
check_predictions_consistency.py

Sanity check before building the Wilson CI / McNemar's test script:
confirms test_predictions_all_38.csv and ALL_38_MODELS_COMBINED.csv
agree on accuracy for every model, and that the predictions file has
no missing values.

Run with:
    python src/check_predictions_consistency.py
"""
import pandas as pd
import numpy as np

PRED_CSV = 'test_predictions_all_38.csv'
COMBINED_CSV = 'ALL_38_MODELS_COMBINED.csv'

pred_df = pd.read_csv(PRED_CSV)
combined_df = pd.read_csv(COMBINED_CSV)

model_cols = [c for c in pred_df.columns if c not in ('filename', 'y_true')]
print(f"Predictions file: {len(pred_df)} test images, {len(model_cols)} models")
print(f"Combined metrics file: {len(combined_df)} models")

# Check for missing values
n_missing = pred_df[model_cols + ['y_true']].isna().sum().sum()
print(f"Missing values in predictions file: {n_missing}")

# Recompute accuracy from predictions and compare to combined CSV
print("\nCross-checking accuracy (predictions file vs combined CSV):")
mismatches = []
for model in model_cols:
    acc_from_preds = (pred_df[model] == pred_df['y_true']).mean()
    row = combined_df[combined_df['model_name'] == model]
    if row.empty:
        print(f"  {model}: NOT FOUND in combined CSV")
        mismatches.append(model)
        continue
    acc_from_combined = row['accuracy'].values[0]
    diff = abs(acc_from_preds - acc_from_combined)
    status = "OK" if diff < 0.0005 else "MISMATCH"
    if status == "MISMATCH":
        mismatches.append(model)
    print(f"  {model}: preds={acc_from_preds:.4f}  combined={acc_from_combined:.4f}  [{status}]")

print(f"\n{'ALL CONSISTENT' if not mismatches else f'{len(mismatches)} MISMATCHES FOUND'}")
if mismatches:
    print("Do not proceed to stats until these are resolved:", mismatches)