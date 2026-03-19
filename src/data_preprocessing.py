"""
Data preprocessing and loading
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

def create_data_generators(data_dir, preprocess_func=None, input_size=(224, 224), 
                          batch_size=16, validation_split=0.15, test_split=0.15):
    """
    Create train, validation, and test data generators
    
    Args:
        data_dir: Path to dataset directory
        preprocess_func: Preprocessing function for specific model
        input_size: Input image size (height, width)
        batch_size: Batch size
        validation_split: Fraction for validation
        test_split: Fraction for test
    """
    
    # Calculate splits
    val_test_split = validation_split + test_split
    
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=val_test_split
    )
    
    # Validation/Test data (only preprocessing)
    val_test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func,
        validation_split=val_test_split
    )
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=input_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    # Validation generator (first part of validation split)
    val_generator = val_test_datagen.flow_from_directory(
        data_dir,
        target_size=input_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    # Get class names
    class_names = list(train_generator.class_indices.keys())
    print(f"\nClasses: {class_names}")
    print(f"Class indices: {train_generator.class_indices}")
    
    # Calculate class weights for imbalanced dataset
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weight_dict = {k: float(v) for k, v in enumerate(class_weights)}
    print(f"\nClass weights: {class_weight_dict}")
    
    return train_generator, val_generator, class_names, class_weight_dict

def create_test_generator(data_dir, preprocess_func=None, input_size=(224, 224), 
                         batch_size=16):
    """
    Create test data generator (separate from validation)
    """
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func
    )
    
    test_generator = test_datagen.flow_from_directory(
        data_dir,
        target_size=input_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return test_generator

def get_dataset_info(data_dir):
    """Get information about the dataset"""
    total_images = 0
    class_counts = {}
    
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        num_images = len([f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        class_counts[class_name] = num_images
        total_images += num_images
    
    print(f"\nDataset Info:")
    print(f"Total images: {total_images}")
    print(f"Number of classes: {len(class_counts)}")
    print(f"\nClass distribution:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count}")
    
    return class_counts