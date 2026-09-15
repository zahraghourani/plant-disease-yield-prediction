"""
generate_test_predictions.py

Reloads each of the 38 trained models and predicts on the SAME held-out
test set, saving per-image predictions to a single CSV. This is required
for McNemar's test, which needs to know — for each test image — whether
each pair of models got it right or wrong, not just aggregate accuracy.

Run with:
    python src/generate_test_predictions.py
"""
import sys
sys.path.append('src')
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from data_preprocessing import create_data_generators
from model_factory import MODEL_CONFIG, get_model
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = r'C:\Users\HPZ4-03-Adm01\plant-disease-yield-prediction\data\Crop___DIsease'
CHECKPOINT_DIR = './checkpoints'
OUTPUT_CSV = 'test_predictions_all_38.csv'
BATCH_SIZE = 16
NUM_CLASSES = 15

# ── Find checkpoints (same logic as evaluate_all_38.py) ──────────────
all_h5 = os.listdir(CHECKPOINT_DIR)
final_ckpts = {f[6:].replace('.weights.h5', '').replace('.h5', ''): f
               for f in all_h5 if f.startswith('final_') and f.endswith('.h5')}
best_ckpts = {f[5:].replace('.weights.h5', '').replace('.h5', ''): f
              for f in all_h5 if f.startswith('best_') and f.endswith('.h5')}

checkpoints = {}
for name in set(list(final_ckpts.keys()) + list(best_ckpts.keys())):
    if name in final_ckpts:
        checkpoints[name] = (final_ckpts[name], 'final')
    else:
        checkpoints[name] = (best_ckpts[name], 'best')

print(f"Found {len(checkpoints)} models to predict")

# ── Resume support ────────────────────────────────────────────────────
if os.path.exists(OUTPUT_CSV):
    df_existing = pd.read_csv(OUTPUT_CSV)
    done_models = [c for c in df_existing.columns if c not in ('filename', 'y_true')]
    print(f"Resuming — {len(done_models)} models already predicted: {done_models}")
else:
    df_existing = None
    done_models = []

results = {}
y_true_saved = None
filenames_saved = None

for i, (model_name, (fname, ckpt_type)) in enumerate(sorted(checkpoints.items()), 1):
    if model_name in done_models:
        print(f"  [{i}/{len(checkpoints)}] Skipping {model_name} (already predicted)")
        continue

    if model_name not in MODEL_CONFIG:
        print(f"  [{i}/{len(checkpoints)}] Skipping {model_name} — not in MODEL_CONFIG")
        continue

    print(f"\n[{i}/{len(checkpoints)}] Predicting {model_name} ({ckpt_type})...")

    preprocess_func = MODEL_CONFIG[model_name]['preprocess']
    input_size = MODEL_CONFIG[model_name]['input_size']

    try:
        _, _, test_gen, class_names, _ = create_data_generators(
            DATA_DIR, preprocess_func, input_size, BATCH_SIZE, validation_split=0.15
        )

        model_path = os.path.join(CHECKPOINT_DIR, fname)
        model, _, _ = get_model(model_name, num_classes=NUM_CLASSES)
        model.load_weights(model_path)

        test_gen.reset()
        y_pred_proba = model.predict(test_gen, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)

        if y_true_saved is None:
            y_true_saved = test_gen.classes
            filenames_saved = test_gen.filenames

        results[model_name] = y_pred
        print(f"  Done. Accuracy check: {(y_pred == y_true_saved).mean():.4f}")

        del model
        tf.keras.backend.clear_session()

    except Exception as e:
        print(f"  ERROR on {model_name}: {e}")
        continue

    # Save progress after every model
    if results:
        out_df = pd.DataFrame({'filename': filenames_saved, 'y_true': y_true_saved})
        for m, preds in results.items():
            out_df[m] = preds
        if df_existing is not None:
            for col in df_existing.columns:
                if col not in out_df.columns:
                    out_df[col] = df_existing[col]
        out_df.to_csv(OUTPUT_CSV, index=False)
        print(f"  Saved progress to {OUTPUT_CSV}")

print(f"\nAll done. Final predictions saved to {OUTPUT_CSV}")