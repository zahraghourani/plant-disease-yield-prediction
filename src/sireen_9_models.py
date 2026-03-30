import sys
sys.path.append('src')  # Ensure we can import from the src folder

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from data_preprocessing import create_data_generators
from model_factory import MODEL_CONFIG
from sklearn.metrics import accuracy_score

# Paths
DATA_DIR = r'C:\Users\HPZ4-03-Adm01\plant-disease-yield-prediction\data\Crop___DIsease'
CHECKPOINT_DIR = './checkpoints'
BATCH_SIZE = 16

# List of models with saved checkpoints
models = [
    'VGG19', 'ResNet50', 'ResNet50V2',
    'DenseNet121', 'DenseNet169', 'DenseNet201',
    'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5'
]

results = []

for model_name in models:
    print(f'\nEvaluating {model_name}...')
    checkpoint = os.path.join(CHECKPOINT_DIR, f'best_{model_name}.h5')
    if not os.path.exists(checkpoint):
        print(f'  Warning: checkpoint file {checkpoint} not found. Skipping.')
        continue

    # Get model configuration
    if model_name not in MODEL_CONFIG:
        print(f'  Warning: {model_name} not in MODEL_CONFIG. Skipping.')
        continue
    config = MODEL_CONFIG[model_name]
    preprocess_func = config['preprocess']
    input_size = config['input_size']

    # Create validation generator
    try:
        _, val_gen, _, _ = create_data_generators(
            DATA_DIR,
            preprocess_func=preprocess_func,
            input_size=input_size,
            batch_size=BATCH_SIZE,
            validation_split=0.15
        )
    except Exception as e:
        print(f'  Error creating data generator: {e}')
        continue

    # Load model (without compilation to avoid metric issues)
    try:
        model = load_model(checkpoint, compile=False)
    except Exception as e:
        print(f'  Error loading model: {e}')
        continue

    # Get predictions
    try:
        val_gen.reset()
        y_pred_proba = model.predict(val_gen, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = val_gen.classes
    except Exception as e:
        print(f'  Error during prediction: {e}')
        continue

    # Compute accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f'  Accuracy: {acc:.4f}')
    results.append({'model_name': model_name, 'accuracy': acc})

# Save results
if results:
    df = pd.DataFrame(results)
    df = df.sort_values('accuracy', ascending=False)
    output_file = 'sireen_9_models.csv'
    df.to_csv(output_file, index=False)
    print(f'\n✅ Results saved to {output_file}')
else:
    print('\n⚠️ No models were successfully evaluated.')