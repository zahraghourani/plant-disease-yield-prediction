import sys
sys.path.append('src')
from tensorflow.keras.models import load_model
import os

models = ['EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5']

for model_name in models:
    ckpt = f'checkpoints/best_{model_name}.h5'
    if not os.path.exists(ckpt):
        print(f'{model_name}: checkpoint not found')
        continue
    try:
        model = load_model(ckpt, compile=False)
        print(f'{model_name}: checkpoint loads successfully')
    except Exception as e:
        print(f'{model_name}: error loading checkpoint - {e}')