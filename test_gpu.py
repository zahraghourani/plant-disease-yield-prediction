import os
import sys

# Explicitly set paths for this session
conda_env_path = r'C:\Users\zahra.elghourani\.conda\envs\plant_disease'
os.environ['PATH'] = os.path.join(conda_env_path, 'Library', 'bin') + ';' + os.environ.get('PATH', '')
os.environ['LD_LIBRARY_PATH'] = os.path.join(conda_env_path, 'Library', 'lib')

import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("CUDA built with:", tf.test.is_built_with_cuda())
print("GPU devices:", tf.config.list_physical_devices('GPU'))

# Try to force GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"\n✅ GPU is working! Found {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"\n❌ GPU error: {e}")
else:
    print("\n❌ No GPU found")