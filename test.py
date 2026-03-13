import os
import sys

# Fix for Windows - set CUDA paths BEFORE importing tensorflow
conda_prefix = r'C:\Users\zahra.elghourani\.conda\envs\plant_disease'
os.environ['PATH'] = os.path.join(conda_prefix, 'Library', 'bin') + ';' + os.environ.get('PATH', '')
os.environ['LD_LIBRARY_PATH'] = os.path.join(conda_prefix, 'Library', 'lib')
os.environ['CUDA_PATH'] = conda_prefix

import tensorflow as tf
sys.path.append('./src')
from utils import setup_gpu
from model_factory import get_model

print("Testing setup...")

# Test GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPU found: {len(gpus)} device(s)")
    for gpu in gpus:
        print(f"  - {gpu}")
else:
    print("✗ No GPU found, using CPU")

# Test model creation
print("\nCreating MobileNetV2...")
model, preprocess, input_size = get_model('MobileNetV2', num_classes=15)
print(f"✓ Model created with input size: {input_size}")
print(f"✓ Total params: {model.count_params():,}")

print("\n✅ All tests passed! Ready to train.")