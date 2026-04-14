import sys
sys.path.append('src')
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from data_preprocessing import create_data_generators
from model_factory import MODEL_CONFIG
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef

# Import custom layers for ConvNeXt models
try:
    from keras.applications.convnext import LayerScale
    custom_objects = {'LayerScale': LayerScale}
except:
    custom_objects = {}

DATA_DIR = r'C:\Users\HPZ4-03-Adm01\plant-disease-yield-prediction\data\Crop___DIsease'
CHECKPOINT_DIR = './checkpoints'
BATCH_SIZE = 16

# List of Tala's models (in order)
tala_models = [
    'ResNet101', 'ResNet101V2', 'InceptionV3', 'InceptionResNetV2',
    'NASNetMobile', 'NASNetLarge', 'EfficientNetB6', 'EfficientNetB7',
    'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3',
    'ConvNeXtBase', 'ConvNeXtXLarge'
]

results = []

for model_name in tala_models:
    checkpoint = os.path.join(CHECKPOINT_DIR, f'final_{model_name}.h5')
    if not os.path.exists(checkpoint):
        print(f'Checkpoint for {model_name} not found – skipping (not yet trained).')
        continue
    print(f'Evaluating {model_name}...')
    if model_name not in MODEL_CONFIG:
        print(f'  Skipping {model_name} (not in MODEL_CONFIG)')
        continue
    preprocess_func = MODEL_CONFIG[model_name]['preprocess']
    input_size = MODEL_CONFIG[model_name]['input_size']
    _, val_gen, _, _ = create_data_generators(
        DATA_DIR, preprocess_func, input_size, BATCH_SIZE, validation_split=0.15
    )
    model = load_model(checkpoint, custom_objects=custom_objects, compile=False)
    val_gen.reset()
    y_pred_proba = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = val_gen.classes
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)
    results.append({
        'model_name': model_name,
        'accuracy': acc,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'mcc': mcc
    })
    print(f'  Accuracy: {acc:.4f}')

if results:
    df = pd.DataFrame(results)
    df.to_csv('tala_models_results.csv', index=False)
    print('\nSaved to tala_models_results.csv')
else:
    print('No Tala models found. Have you trained any?')