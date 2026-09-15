import sys
sys.path.append('src')
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from data_preprocessing import create_data_generators
from model_factory import MODEL_CONFIG
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, matthews_corrcoef
)
from sklearn.preprocessing import label_binarize

# Custom objects for ConvNeXt
try:
    from keras.applications.convnext import LayerScale
    custom_objects = {'LayerScale': LayerScale}
except:
    custom_objects = {}

# ----------------------------------------------------------------------
# Configuration
DATA_DIR = r'C:\Users\HPZ4-03-Adm01\plant-disease-yield-prediction\data\Crop___DIsease'
CHECKPOINT_DIR = './checkpoints'
BATCH_SIZE = 16
NUM_CLASSES = 15

# Get all checkpoint files (best_ and final_)
all_checkpoints = [f for f in os.listdir(CHECKPOINT_DIR)
                   if (f.startswith('best_') or f.startswith('final_')) and f.endswith('.h5')]

# ----------------------------------------------------------------------
# Helper to get model info (params, input size)
def get_model_info_from_checkpoint(model_path, model_name):
    try:
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        # Input size from MODEL_CONFIG
        input_size = MODEL_CONFIG[model_name]['input_size']
        # Model size in MB
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        return total_params, trainable_params, input_size, model_size_mb
    except:
        return None, None, None, None

# ----------------------------------------------------------------------
results = []

for fname in all_checkpoints:
    if fname.startswith('best_'):
        model_name = fname[5:-3]
        ckpt_type = 'best'
    else:
        model_name = fname[6:-3]
        ckpt_type = 'final'

    print(f'\nEvaluating {model_name} ({ckpt_type}) ...')
    if model_name not in MODEL_CONFIG:
        print(f'  Skipping {model_name} (not in MODEL_CONFIG)')
        continue

    preprocess_func = MODEL_CONFIG[model_name]['preprocess']
    input_size = MODEL_CONFIG[model_name]['input_size']

    try:
        # Create validation generator
        _, val_gen, class_names, _ = create_data_generators(
            DATA_DIR, preprocess_func, input_size,
            BATCH_SIZE, validation_split=0.15
        )

        # Load model
        model_path = os.path.join(CHECKPOINT_DIR, fname)
        model = load_model(model_path, custom_objects=custom_objects, compile=False)

        # Predict
        val_gen.reset()
        y_pred_proba = model.predict(val_gen, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = val_gen.classes

        # ---- Metrics ----
        acc = accuracy_score(y_true, y_pred)

        # Macro and weighted
        prec_mac, rec_mac, f1_mac, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        prec_wei, rec_wei, f1_wei, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )

        # Per-class metrics
        prec_per_class, rec_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(NUM_CLASSES)
        )
        # Convert to lists for CSV
        prec_per_class = list(prec_per_class)
        rec_per_class = list(rec_per_class)
        f1_per_class = list(f1_per_class)

        # ROC AUC
        y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
        try:
            roc_auc_ovr = roc_auc_score(y_true_bin, y_pred_proba, multi_class='ovr', average='macro')
            roc_auc_ovo = roc_auc_score(y_true_bin, y_pred_proba, multi_class='ovo', average='macro')
        except:
            roc_auc_ovr = roc_auc_ovo = np.nan

        mcc = matthews_corrcoef(y_true, y_pred)

        # Model info
        total_params, trainable_params, inp_size, model_size_mb = get_model_info_from_checkpoint(model_path, model_name)
        if inp_size is None:
            inp_size = input_size

        # ---- Build row similar to Zahra's ----
        row = {
            'model_name': model_name,
            'accuracy': acc,
            'precision_macro': prec_mac,
            'recall_macro': rec_mac,
            'f1_macro': f1_mac,
            'precision_weighted': prec_wei,
            'recall_weighted': rec_wei,
            'f1_weighted': f1_wei,
            'precision_per_class': str(prec_per_class),
            'recall_per_class': str(rec_per_class),
            'f1_per_class': str(f1_per_class),
            'roc_auc_ovr': roc_auc_ovr,
            'roc_auc_ovo': roc_auc_ovo,
            'mcc': mcc,
            'model_size_mb': model_size_mb,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'input_size': inp_size,
            # Training-time fields (not available from checkpoint)
            'training_time_sec': None,
            'epochs_trained': None,
            'final_val_accuracy': acc,   # approximate
            'best_val_accuracy': acc,
            'final_val_loss': None,
            'inference_time_ms': None,
            'person': 'sireen' if 'sireen' in model_name.lower() else 'tala'  # rough assignment
        }
        results.append(row)
        print(f'  Accuracy: {acc:.4f}, F1_macro: {f1_mac:.4f}')

    except Exception as e:
        print(f'  Error evaluating {model_name}: {e}')

# Save results
if results:
    df = pd.DataFrame(results)
    df.to_csv('my_models_zahra_format.csv', index=False)
    print('\n✅ Saved to my_models_zahra_format.csv')
else:
    print('No models evaluated.')