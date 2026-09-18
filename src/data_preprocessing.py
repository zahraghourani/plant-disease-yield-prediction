"""
Data preprocessing and loading — GROUP-AWARE SPLIT (fixes duplicate leakage)
"""
import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def _get_base_id(filepath):
    fname = os.path.basename(filepath)
    fname = re.sub(r'\.(jpg|jpeg|png)$', '', fname, flags=re.IGNORECASE)
    fname = re.sub(r'\(\d+\)$', '', fname)
    fname = re.sub(r'_?(flipLR|flipTB|\d{2,3}deg)', '', fname, flags=re.IGNORECASE)
    fname = re.sub(r'_new\w*', '', fname, flags=re.IGNORECASE)
    fname = re.sub(r'\s*copy\s*\d*$', '', fname, flags=re.IGNORECASE)
    return fname.strip()


def _build_file_dataframe(data_dir):
    records = []
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(class_dir, fname)
                records.append({
                    'filepath': filepath,
                    'label': class_name,
                    'base_id': f"{class_name}::{_get_base_id(filepath)}"
                })
    return pd.DataFrame(records)


def get_stratified_splits(data_dir, val_split=0.15, test_split=0.15,
                           seed=42, cache_path='./splits.csv'):
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        train_df = df[df['split'] == 'train'].reset_index(drop=True)
        val_df = df[df['split'] == 'val'].reset_index(drop=True)
        test_df = df[df['split'] == 'test'].reset_index(drop=True)
        print(f"Loaded cached split from {cache_path} "
              f"(train={len(train_df)}, val={len(val_df)}, test={len(test_df)})")
        return train_df, val_df, test_df

    print(f"No cached split found at {cache_path} — building a new "
          f"GROUP-AWARE split (seed={seed})")
    full_df = _build_file_dataframe(data_dir)

    group_labels = full_df.groupby('base_id')['label'].first().reset_index()

    trainval_groups, test_groups = train_test_split(
        group_labels, test_size=test_split,
        stratify=group_labels['label'], random_state=seed
    )
    relative_val_size = val_split / (1 - test_split)
    train_groups, val_groups = train_test_split(
        trainval_groups, test_size=relative_val_size,
        stratify=trainval_groups['label'], random_state=seed
    )

    train_ids = set(train_groups['base_id'])
    val_ids = set(val_groups['base_id'])
    test_ids = set(test_groups['base_id'])

    train_df = full_df[full_df['base_id'].isin(train_ids)].copy()
    val_df = full_df[full_df['base_id'].isin(val_ids)].copy()
    test_df = full_df[full_df['base_id'].isin(test_ids)].copy()

    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'

    pd.concat([train_df, val_df, test_df]).to_csv(cache_path, index=False)
    print(f"Saved new GROUP-AWARE split to {cache_path} "
          f"(train={len(train_df)}, val={len(val_df)}, test={len(test_df)})")

    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def create_data_generators(data_dir, preprocess_func=None, input_size=(224, 224),
                            batch_size=16, validation_split=0.15, test_split=0.15,
                            seed=42, split_cache='./splits.csv'):
    train_df, val_df, test_df = get_stratified_splits(
        data_dir, val_split=validation_split, test_split=test_split,
        seed=seed, cache_path=split_cache
    )

    all_classes = sorted(pd.concat([train_df, val_df, test_df])['label'].unique())

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func,
        rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
        horizontal_flip=True, vertical_flip=True, zoom_range=0.2,
        shear_range=0.2, brightness_range=[0.8, 1.2], fill_mode='nearest'
    )
    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_func)

    train_generator = train_datagen.flow_from_dataframe(
        train_df, x_col='filepath', y_col='label', classes=all_classes,
        target_size=input_size, batch_size=batch_size,
        class_mode='sparse', shuffle=True, seed=seed
    )
    val_generator = val_test_datagen.flow_from_dataframe(
        val_df, x_col='filepath', y_col='label', classes=all_classes,
        target_size=input_size, batch_size=batch_size,
        class_mode='sparse', shuffle=False
    )
    test_generator = val_test_datagen.flow_from_dataframe(
        test_df, x_col='filepath', y_col='label', classes=all_classes,
        target_size=input_size, batch_size=batch_size,
        class_mode='sparse', shuffle=False
    )

    class_names = list(train_generator.class_indices.keys())
    print(f"\nClasses: {class_names}")
    print(f"Class indices: {train_generator.class_indices}")

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weight_dict = {k: float(v) for k, v in enumerate(class_weights)}
    print(f"\nClass weights: {class_weight_dict}")

    return train_generator, val_generator, test_generator, class_names, class_weight_dict


def get_dataset_info(data_dir):
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
    print(f"\nDataset Info:\nTotal images: {total_images}\nNumber of classes: {len(class_counts)}\n\nClass distribution:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count}")
    return class_counts