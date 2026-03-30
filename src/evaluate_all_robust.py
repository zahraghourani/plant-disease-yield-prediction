import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from data_preprocessing import create_data_generators
from model_factory import MODEL_CONFIG, SIREEN_MODELS
import yaml
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, matthews_corrcoef, accuracy_score
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

DATA_DIR = r"C:\Users\HPZ4-03-Adm01\plant-disease-yield-prediction\data\Crop___DIsease"
CHECKPOINT_DIR = "./checkpoints"
BATCH_SIZE = 16
NUM_CLASSES = config['num_classes']

# Use the list of models assigned to Sireen
model_list = SIREEN_MODELS

results = []
output_file = 'all_models_results.csv'

for model_name in model_list:
    print(f"\nEvaluating {model_name}...")
    
    # Build checkpoint file path
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f'best_{model_name}.h5')
    if not os.path.exists(checkpoint_file):
        print(f"  Warning: checkpoint file {checkpoint_file} not found. Skipping.")
        continue

    # Get model config
    if model_name not in MODEL_CONFIG:
        print(f"  Warning: {model_name} not in MODEL_CONFIG, skipping")
        continue
    preprocess_func = MODEL_CONFIG[model_name]['preprocess']
    input_size = MODEL_CONFIG[model_name]['input_size']

    # Create validation generator
    try:
        _, val_gen, class_names, _ = create_data_generators(
            DATA_DIR,
            preprocess_func=preprocess_func,
            input_size=input_size,
            batch_size=BATCH_SIZE,
            validation_split=0.15
        )
    except Exception as e:
        print(f"  Error creating data generator: {e}")
        continue

    # Load model
    try:
        model = load_model(checkpoint_file, compile=False)
        # Recompile with metrics we need for predictions? Not needed.
    except Exception as e:
        print(f"  Error loading model: {e}")
        continue

    # Get predictions
    try:
        val_gen.reset()
        y_pred_proba = model.predict(val_gen, verbose=1)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = val_gen.classes
    except Exception as e:
        print(f"  Error during prediction: {e}")
        continue

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

    # Save intermediate results to CSV after each model
    df_interim = pd.DataFrame(results)
    df_interim.to_csv(output_file, index=False)

# Final sort and display
if results:
    df = pd.DataFrame(results)
    df = df.sort_values('accuracy', ascending=False)
    df.to_csv(output_file, index=False)
    print("\n" + "="*60)
    print("All models results saved to", output_file)
    print("="*60)
    print(df[['model_name', 'accuracy', 'f1_macro', 'roc_auc_ovr', 'mcc']].to_string())
else:
    print("No models were successfully evaluated.")