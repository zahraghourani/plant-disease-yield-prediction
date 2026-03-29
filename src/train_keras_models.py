"""
Main training script for all Keras models
Run with: python src/train_keras_models.py --person zahra
"""
import os
import sys
import argparse
import json
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint, 
                                       ReduceLROnPlateau, TensorBoard)
import tensorflow as tf

import yaml
from tensorflow.keras import mixed_precision
# Uncomment to disable mixed precision if NaN persists:
# mixed_precision.set_global_policy('float32')
# with open('config.yaml', 'r') as f:
#     config = yaml.safe_load(f)
# NUM_CLASSES = config['num_classes']

# Replace lines 27-28 with:
from utils import load_config
config = load_config()
NUM_CLASSES = config['num_classes']

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

print("GPU:", tf.config.list_physical_devices('GPU'))
# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_factory import get_model, unfreeze_layers
from data_preprocessing import create_data_generators, get_dataset_info
from utils import (setup_gpu, calculate_metrics, plot_confusion_matrix, 
                  plot_training_history, save_results, Timer, get_model_info)

# Configuration
DATA_DIR = r"C:\Users\zahra.elghourani\Desktop\Zahra\Plant Disease Detection and Crop Yield Prediction\data\Crop___DIsease"
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.0001

# Model assignments
MODELS_ZAHRA = [
    'Xception', 'VGG16', 'ResNet152', 
    'ResNet152V2', 'MobileNet', 'MobileNetV2',
    'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetV2B0', 
    'EfficientNetV2S', 'EfficientNetV2M', 'ConvNeXtLarge'
]

MODELS_SIREEN = [
    'VGG19', 'ResNet50', 'ResNet50V2', 'DenseNet121',
    'DenseNet169', 'DenseNet201', 'EfficientNetB3', 'EfficientNetB4',
    'EfficientNetB5', 'EfficientNetV2L', 'ConvNeXtTiny', 'ConvNeXtSmall'
]

MODELS_TALA = [
    'ResNet101', 'ResNet101V2', 'InceptionV3',
    'InceptionResNetV2', 'NASNetMobile', 'NASNetLarge', 'EfficientNetB6', 'EfficientNetB7',
    'EfficientNetV2B1', 'EfficientNetV2B2', 'EfficientNetV2B3', 'ConvNeXtBase', 'ConvNeXtXLarge'
]

def train_model(model_name, person_name):
    """Train a single model"""
    print(f"\n{'='*70}")
    print(f"Training {model_name} for {person_name}")
    print(f"{'='*70}")
    
    # Create directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(f"{RESULTS_DIR}/figures", exist_ok=True)
    
    # Get model
    model, preprocess_func, input_size = get_model(
        model_name, 
        num_classes=NUM_CLASSES, 
        learning_rate=LEARNING_RATE
    )

        # Recompile with only accuracy to avoid JSON serialization issues with custom metrics
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model info
    model_info = get_model_info(model)
    print(f"\nModel Info:")
    print(f"  Total params: {model_info['total_params']:,}")
    print(f"  Trainable params: {model_info['trainable_params']:,}")
    print(f"  Non-trainable params: {model_info['non_trainable_params']:,}")
    print(f"  Model size: {model_info['size_mb']:.2f} MB")
    
    # Create data generators
    train_gen, val_gen, class_names, class_weights = create_data_generators(
        DATA_DIR,
        preprocess_func=preprocess_func,
        input_size=input_size,
        batch_size=BATCH_SIZE,
        validation_split=0.15
    )
    
    # Callbacks
    # callbacks = [
    #     EarlyStopping(
    #         monitor='val_loss',
    #         patience=10,
    #         restore_best_weights=True,
    #         verbose=1
    #     ),
    #     ModelCheckpoint(
    #         os.path.join(CHECKPOINT_DIR, f'best_{model_name}.h5'),
    #         monitor='val_accuracy',
    #         save_best_only=True,
    #         verbose=1
    #     ),
    #     ReduceLROnPlateau(
    #         monitor='val_loss',
    #         factor=0.5,
    #         patience=5,
    #         min_lr=1e-7,
    #         verbose=1
    #     )
    # ]
    callbacks = [] 
    # Training Phase 1: Frozen base
    print(f"\n{'='*70}")
    print("Phase 1: Training with frozen base layers")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS // 2,  # Train half epochs frozen
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Training Phase 2: Fine-tuning (optional, for larger models)
    if model_name not in ['MobileNet', 'MobileNetV2', 'NASNetMobile']:
        print(f"\n{'='*70}")
        print("Phase 2: Fine-tuning top layers")
        print(f"{'='*70}")
        
        model = unfreeze_layers(model, num_layers_to_unfreeze=10, learning_rate=1e-5)
        
        history_fine = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            initial_epoch=len(history.history['loss']),
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Combine histories
        for key in history.history.keys():
            history.history[key].extend(history_fine.history[key])
    
    training_time = time.time() - start_time
    
    # Evaluate
    print(f"\n{'='*70}")
    print("Final Evaluation")
    print(f"{'='*70}")
    
    val_gen.reset()
    y_pred_proba = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = val_gen.classes
    
    # Calculate all metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba, num_classes=NUM_CLASSES)
    metrics['model_name'] = model_name
    metrics['person'] = person_name
    metrics['training_time_sec'] = training_time
    metrics['epochs_trained'] = len(history.history['loss'])
    metrics['final_val_accuracy'] = history.history['val_accuracy'][-1]
    metrics['best_val_accuracy'] = max(history.history['val_accuracy'])
    metrics['final_val_loss'] = history.history['val_loss'][-1]
    metrics['model_size_mb'] = model_info['size_mb']
    metrics['total_params'] = model_info['total_params']
    metrics['trainable_params'] = model_info['trainable_params']
    metrics['input_size'] = input_size
    
    # Inference time benchmark
    dummy_input = np.random.rand(1, *input_size, 3).astype(np.float32)
    start = time.time()
    for _ in range(100):
        model.predict(dummy_input, verbose=0)
    avg_inference_time = (time.time() - start) / 100 * 1000  # ms
    metrics['inference_time_ms'] = avg_inference_time
    
    # Print results
    print(f"\nResults for {model_name}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"  F1-Score (macro): {metrics['f1_macro']:.4f}")
    print(f"  ROC-AUC (OvR): {metrics.get('roc_auc_ovr', 0):.4f}")
    print(f"  MCC: {metrics['mcc']:.4f}")
    print(f"  Training time: {training_time:.2f} sec")
    print(f"  Inference time: {avg_inference_time:.2f} ms")
    
    # Save plots
    plot_training_history(history, model_name, save_dir=f"{RESULTS_DIR}/figures")
    plot_confusion_matrix(y_true, y_pred, class_names, 
                         save_path=f"{RESULTS_DIR}/figures/{model_name}_confusion_matrix.png")
    
    # Save model summary
    with open(f"{RESULTS_DIR}/{model_name}_summary.txt", 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # Clear memory
    del model
    tf.keras.backend.clear_session()
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Train Keras models for plant disease detection')
    parser.add_argument('--person', type=str, required=True, 
                       choices=['zahra', 'sireen', 'tala'],
                       help='Which person is running the script')
    parser.add_argument('--model', type=str, default=None,
                       help='Train specific model only (optional)')
    args = parser.parse_args()
    
    # Setup GPU
    setup_gpu()
    
    # Get model list
    if args.person == 'zahra':
        model_list = MODELS_ZAHRA
    elif args.person == 'sireen':
        model_list = MODELS_SIREEN
    else:
        model_list = MODELS_TALA
    
    # Train specific model or all
    if args.model:
        if args.model in model_list:
            model_list = [args.model]
        else:
            print(f"Error: {args.model} not in {args.person}'s list")
            return
    
    print(f"\n{'='*70}")
    print(f"Starting training for {args.person}")
    print(f"Models to train: {len(model_list)}")
    print(f"Models: {model_list}")
    print(f"{'='*70}")
    
    # Get dataset info
    get_dataset_info(DATA_DIR)
    
    # Train all models
    all_results = []
    for i, model_name in enumerate(model_list, 1):
        try:
            print(f"\n\nModel {i}/{len(model_list)}")
            result = train_model(model_name, args.person)
            all_results.append(result)
            
            # Save intermediate results
            df = pd.DataFrame(all_results)
            df.to_csv(f"{RESULTS_DIR}/{args.person}_results.csv", index=False)
            
        except Exception as e:
            print(f"\n✗ Error training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Log error
            with open(f"{RESULTS_DIR}/{args.person}_errors.log", 'a') as f:
                f.write(f"{model_name}: {str(e)}\n")
            continue
    
    # Final summary
    if all_results:
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE - FINAL SUMMARY")
        print(f"{'='*70}")
        
        df_final = pd.DataFrame(all_results)
        df_final = df_final.sort_values('accuracy', ascending=False)
        
        print(f"\nTop 5 Models by Accuracy:")
        print(df_final[['model_name', 'accuracy', 'f1_macro', 'training_time_sec']].head().to_string())
        
        # Save final results
        df_final.to_csv(f"{RESULTS_DIR}/{args.person}_final_results.csv", index=False)
        save_results(all_results, f"{args.person}_final_results.json")
        
        # Create comparison plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        x_pos = range(len(df_final))
        plt.bar(x_pos, df_final['accuracy'])
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title(f'Model Comparison - {args.person}')
        plt.xticks(x_pos, df_final['model_name'], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/figures/{args.person}_comparison.png", dpi=300)
        plt.close()
        
        print(f"\n✓ All results saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    main()