import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from data_preprocessing import create_data_generators
from model_factory import MODEL_CONFIG
import yaml
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, matthews_corrcoef, accuracy_score
from sklearn.preprocessing import label_binarize

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

DATA_DIR = r"C:\Users\HPZ4-03-Adm01\plant-disease-yield-prediction\data\Crop___DIsease"
CHECKPOINT_DIR = "./checkpoints"
BATCH_SIZE = 16
NUM_CLASSES = config['num_classes']

# Get list of best model files
model_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('best_') and f.endswith('.h5')]

results = []

for fname in sorted(model_files):
    model_name = fname.replace('best_', '').replace('.h5', '')
    print(f"\nEvaluating {model_name}...")

    # Get model config
    if model_name not in MODEL_CONFIG:
        print(f"  Warning: {model_name} not in MODEL_CONFIG, skipping")
        continue
    preprocess_func = MODEL_CONFIG[model_name]['preprocess']
    input_size = MODEL_CONFIG[model_name]['input_size']

    # Create validation generator (same as training)
    from data_preprocessing import create_data_generators
    _, val_gen, class_names, _ = create_data_generators(
        DATA_DIR,
        preprocess_func=preprocess_func,
        input_size=input_size,
        batch_size=BATCH_SIZE,
        validation_split=0.15
    )

    # Load model
    model_path = os.path.join(CHECKPOINT_DIR, fname)
    model = load_model(model_path)

    # Get predictions
    val_gen.reset()
    y_pred_proba = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = val_gen.classes

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    # ROC AUC (one-vs-rest)
    y_true_bin = label_binarize(y_true, classes=range(NUM_CLASSES))
    try:
        roc_auc_ovr = roc_auc_score(y_true_bin, y_pred_proba, multi_class='ovr', average='macro')
    except:
        roc_auc_ovr = np.nan

    mcc = matthews_corrcoef(y_true, y_pred)

    results.append({
        'model_name': model_name,
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'roc_auc_ovr': roc_auc_ovr,
        'mcc': mcc
    })

    print(f"  Accuracy: {accuracy:.4f}, F1 macro: {f1_macro:.4f}")

# Save and display
df = pd.DataFrame(results)
df = df.sort_values('accuracy', ascending=False)
df.to_csv('all_models_results.csv', index=False)
print("\n" + "="*60)
print("All models results saved to all_models_results.csv")
print("="*60)
print(df[['model_name', 'accuracy', 'f1_macro', 'roc_auc_ovr', 'mcc']].to_string())