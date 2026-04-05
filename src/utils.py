"""
Utility functions for the project
"""
import os
import yaml
import json
import time
import logging
from datetime import datetime
import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, matthews_corrcoef)
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
def setup_logging(log_dir='./results/logs'):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# GPU Setup
# In utils.py, modify setup_gpu():
def setup_gpu():
    """Configure GPU settings"""
    # Load config first
    config = load_config()
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                # tf.config.experimental.set_memory_growth(gpu, True)
                # tf.config.experimental.set_memory_growth(gpu, False)
                tf.config.experimental.set_virtual_device_configuration(
                    gpu, 
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7000)]  # 7GB limit
                )
            print(f"✓ GPU memory growth enabled for {len(gpus)} GPU(s)")
            
            # Only enable mixed precision if config says so
            if config.get('mixed_precision', False):
                from tensorflow.keras.mixed_precision import Policy, set_global_policy
                policy = Policy('mixed_float16')
                set_global_policy(policy)
                print("✓ Mixed precision enabled")
            else:
                print("✓ Mixed precision disabled (using float32)")
            
            return True
        except RuntimeError as e:
            print(f"✗ GPU setup error: {e}")
            return False
    else:
        print("✗ No GPU found, using CPU")
        return False

# Load config
def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Comprehensive metrics calculation
def calculate_metrics(y_true, y_pred, y_pred_proba=None, num_classes=15):
    """
    Calculate all required metrics for the Dr.
    Returns dict with all metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Matthews Correlation Coefficient (important for imbalanced data)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    # Per-class metrics
    metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    
    # ROC-AUC (One-vs-Rest)
    if y_pred_proba is not None:
        try:
            from sklearn.preprocessing import label_binarize
            y_true_bin = label_binarize(y_true, classes=range(num_classes))
            metrics['roc_auc_ovr'] = roc_auc_score(y_true_bin, y_pred_proba, multi_class='ovr', average='macro')
            metrics['roc_auc_ovo'] = roc_auc_score(y_true_bin, y_pred_proba, multi_class='ovo', average='macro')
        except:
            metrics['roc_auc_ovr'] = 0.0
            metrics['roc_auc_ovo'] = 0.0
    
    return metrics

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {save_path}")
    plt.close()

# Plot training history
def plot_training_history(history, model_name, save_dir='./results/figures'):
    """Plot training curves"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train')
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train')
    axes[0, 1].plot(history.history['val_loss'], label='Validation')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Train')
        axes[1, 0].plot(history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Train')
        axes[1, 1].plot(history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history saved to {save_path}")
    plt.close()

# Save results to JSON
def save_results(results, filename):
    """Save results dictionary to JSON file (handles numpy types)"""
    import json
    import numpy as np

    def convert(obj):
        import tensorflow as tf
        if isinstance(obj, tf.Tensor):
            return obj.numpy().tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    # Recursively convert
    converted = []
    for res in results:
        converted.append({k: convert(v) for k, v in res.items()})

    with open(filename, 'w') as f:
        json.dump(converted, f, indent=4)
    print(f"✓ Results saved to {filename}")

# Timer decorator
class Timer:
    """Context manager for timing operations"""
    def __init__(self, name="Operation"):
        self.name = name
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print(f"⏱ {self.name} took {self.interval:.2f} seconds")

# Get model info
def get_model_info(model):
    """Get model size and parameter count"""
    params = model.count_params()
    size_mb = (params * 4) / (1024 ** 2)  # Assuming float32
    return {
        'total_params': params,
        'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
        'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]),
        'size_mb': size_mb
    }