# evaluate_all_38.py
import sys
sys.path.append('src')
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from data_preprocessing import create_data_generators
from model_factory import MODEL_CONFIG, get_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# ConvNeXt fix
try:
    from keras.applications.convnext import LayerScale
    custom_objects = {'LayerScale': LayerScale}
except:
    custom_objects = {}

# ── CONFIG ──────────────────────────────────────────────────────────
DATA_DIR    = r'C:\Users\HPZ4-03-Adm01\plant-disease-yield-prediction\data\Crop___DIsease'
CHECKPOINT_DIR = './checkpoints'
OUTPUT_CSV  = 'ALL_38_MODELS_COMBINED.csv'
BATCH_SIZE  = 16
NUM_CLASSES = 15

# ── Find all final_ checkpoints ─────────────────────────────────────
# Prefer final_ over best_ (final = after fine-tuning)
all_h5 = os.listdir(CHECKPOINT_DIR)
final_ckpts = {f[6:].replace('.weights.h5', '').replace('.h5', ''): f for f in all_h5 if f.startswith('final_') and f.endswith('.h5')}
best_ckpts  = {f[5:].replace('.weights.h5', '').replace('.h5', ''):  f for f in all_h5 if f.startswith('best_')  and f.endswith('.h5')}

# Merge: use final_ if available, else fall back to best_
checkpoints = {}
for name in set(list(final_ckpts.keys()) + list(best_ckpts.keys())):
    if name in final_ckpts:
        checkpoints[name] = (final_ckpts[name], 'final')
    else:
        checkpoints[name] = (best_ckpts[name], 'best')

print(f"Found {len(checkpoints)} models to evaluate")
print(sorted(checkpoints.keys()))

# ── Load existing progress so you can resume if interrupted ─────────
if os.path.exists(OUTPUT_CSV):
    df_existing = pd.read_csv(OUTPUT_CSV)
    done = set(df_existing['model_name'].tolist())
    results = df_existing.to_dict('records')
    print(f"\nResuming — {len(done)} already done: {done}")
else:
    done = set()
    results = []

# ── Evaluate ────────────────────────────────────────────────────────
for model_name, (fname, ckpt_type) in sorted(checkpoints.items()):
    if model_name in done:
        print(f"  Skipping {model_name} (already evaluated)")
        continue

    if model_name not in MODEL_CONFIG:
        print(f"  Skipping {model_name} — not in MODEL_CONFIG")
        continue

    print(f"\n[{len(results)+1}/{len(checkpoints)}] Evaluating {model_name} ({ckpt_type})...")

    preprocess_func = MODEL_CONFIG[model_name]['preprocess']
    input_size      = MODEL_CONFIG[model_name]['input_size']
    person          = MODEL_CONFIG[model_name]['owner'].lower()

    try:
        _, val_gen, class_names, _ = create_data_generators(
            DATA_DIR, preprocess_func, input_size, BATCH_SIZE, validation_split=0.15
        )

        model_path = os.path.join(CHECKPOINT_DIR, fname)
        model, _, _ = get_model(model_name, num_classes=NUM_CLASSES)
        model.load_weights(model_path)

        # Inference time
        import time
        val_gen.reset()
        dummy = np.zeros((1, *input_size, 3), dtype=np.float32)
        model.predict(dummy, verbose=0)  # warmup
        t0 = time.time()
        val_gen.reset()
        y_pred_proba = model.predict(val_gen, verbose=0)
        inference_ms = (time.time() - t0) / len(y_pred_proba) * 1000

        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = val_gen.classes

        acc                               = accuracy_score(y_true, y_pred)
        prec_mac, rec_mac, f1_mac, _      = precision_recall_fscore_support(y_true, y_pred, average='macro')
        prec_wei, rec_wei, f1_wei, _      = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        mcc                               = matthews_corrcoef(y_true, y_pred)

        y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
        try:
            roc_auc = roc_auc_score(y_true_bin, y_pred_proba, multi_class='ovr', average='macro')
        except:
            roc_auc = float('nan')

        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

        row = {
            'model_name':        model_name,
            'person':            person,
            'accuracy':          round(acc, 6),
            'f1_macro':          round(f1_mac, 6),
            'precision_macro':   round(prec_mac, 6),
            'recall_macro':      round(rec_mac, 6),
            'f1_weighted':       round(f1_wei, 6),
            'precision_weighted':round(prec_wei, 6),
            'recall_weighted':   round(rec_wei, 6),
            'mcc':               round(mcc, 6),
            'roc_auc_ovr':       round(roc_auc, 6) if not np.isnan(roc_auc) else None,
            'inference_time_ms': round(inference_ms, 3),
            'model_size_mb':     round(model_size_mb, 2),
            'checkpoint_type':   ckpt_type,
        }
        results.append(row)
        print(f"  ✓ Accuracy: {acc:.4f} | F1: {f1_mac:.4f} | MCC: {mcc:.4f}")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        continue

    # Save after every model so a crash doesn't lose progress
    pd.DataFrame(results).sort_values('accuracy', ascending=False).to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved to {OUTPUT_CSV}")

# ── Final leaderboard ────────────────────────────────────────────────
df = pd.DataFrame(results).sort_values('accuracy', ascending=False)
df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "="*70)
print("TOP 10 MODELS ACROSS ALL 38")
print("="*70)
print(df[['model_name','person','accuracy','f1_macro','mcc','inference_time_ms']].head(10).to_string(index=False))

print("\nPer-person best:")
for person, grp in df.groupby('person'):
    best = grp.iloc[0]
    print(f"  {person.capitalize():8s} → {best['model_name']:25s} acc={best['accuracy']:.4f}")