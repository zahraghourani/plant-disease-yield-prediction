import pandas as pd
import os

csv_file = 'my_models_zahra_format.csv'  # your master CSV

new_row = {
    'model_name': 'EfficientNetB4',
    'accuracy': 0.8201,
    'precision_macro': 0.8703,
    'recall_macro': 0.8579,
    'f1_macro': 0.8374,
    'mcc': 0.8111,
    'roc_auc_ovr': 0.0,
    'person': 'sireen',
    # Add any other columns your CSV has with None or empty string
}

# Load existing CSV if it exists, otherwise create new
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
    # Add missing columns if new row has extra fields
    for col in new_row:
        if col not in df.columns:
            df[col] = None
    # Ensure new row has all columns from df
    for col in df.columns:
        if col not in new_row:
            new_row[col] = None
    # Append
    if new_row['model_name'] not in df['model_name'].values:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_file, index=False)
        print(f'Added {new_row["model_name"]} to {csv_file}')
    else:
        print(f'{new_row["model_name"]} already exists in CSV.')
else:
    pd.DataFrame([new_row]).to_csv(csv_file, index=False)
    print(f'Created {csv_file} with {new_row["model_name"]}')