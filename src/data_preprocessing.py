"""
Data preprocessing and loading — FIXED VERSION
Now produces a real train/val/test split (previously val and test were the
same 30% chunk, meaning "test" metrics were computed on data used for
checkpoint selection / early stopping).

The split is computed once, cached to disk (split_cache), and reused by every
model so all 38 architectures are trained and evaluated on IDENTICAL
partitions — this also matches what the paper already claims
("stratified split, seed=42").
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def _build_file_dataframe(data_dir):
    """Walk data_dir and build a dataframe of (filepath, label) for every image."""
    records = []
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                records.append({
                    'filepath': os.path.join(class_dir, fname),
                    'label': class_name
                })
    return pd.DataFrame(records)


def get_stratified_splits(data_dir, val_split=0.15, test_split=0.15,
                           seed=42, cache_path='./splits.csv'):
    """
    Build (or load cached) stratified train/val/test split.

    IMPORTANT: this is cached to `cache_path` so that once generated, every
    subsequent call (for every model, by any of the three people training on
    this machine) reuses the EXACT same partitions. Do not delete/regenerate
    splits.csv mid-way through the 38-model run, or different models will be
    evaluated on different test sets and results won't be comparable.
    """
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        train_df = df[df['split'] == 'train'].reset_index(drop=True)
        val_df = df[df['split'] == 'val'].reset_index(drop=True)
        test_df = df[df['split'] == 'test'].reset_index(drop=True)
        print(f"Loaded cached split from {cache_path} "
              f"(train={len(train_df)}, val={len(val_df)}, test={len(test_df)})")
        return train_df, val_df, test_df

    print(f"No cached split found at {cache_path} — building a new one (seed={seed})")
    full_df = _build_file_dataframe(data_dir)

    # First carve off the held-out test set — this is NEVER touched again
    # after this point until final evaluation.
    trainval_df, test_df = train_test_split(
        full_df, test_size=test_split, stratify=full_df['label'], random_state=seed
    )

    # Split the remainder into train/val. val_split was expressed as a
    # fraction of the ORIGINAL total, so rescale it relative to trainval_df.
    relative_val_size = val_split / (1 - test_split)
    train_df, val_df = train_test_split(
        trainval_df, test_size=relative_val_size,
        stratify=trainval_df['label'], random_state=seed
    )

    train_df = train_df.copy(); train_df['split'] = 'train'
    val_df = val_df.copy(); val_df['split'] = 'val'
    test_df = test_df.copy(); test_df['split'] = 'test'

    pd.concat([train_df, val_df, test_df]).to_csv(cache_path, index=False)
    print(f"Saved new split to {cache_path} "
          f"(train={len(train_df)}, val={len(val_df)}, test={len(test_df)})")

    return (train_df.reset_index(drop=True),
            val_df.reset_index(drop=True),
            test_df.reset_index(drop=True))


def create_data_generators(data_dir, preprocess_func=None, input_size=(224, 224),
                            batch_size=16, validation_split=0.15, test_split=0.15,
                            seed=42, split_cache='./splits.csv'):
    """
    Create train, validation, AND a true held-out test generator.

    Returns:
        train_generator, val_generator, test_generator, class_names, class_weight_dict
    """
    train_df, val_df, test_df = get_stratified_splits(
        data_dir, val_split=validation_split, test_split=test_split,
        seed=seed, cache_path=split_cache
    )

    # Explicit sorted class list so class_indices is IDENTICAL across
    # train/val/test generators, even if a class happens to have zero
    # samples in one split by chance.
    all_classes = sorted(pd.concat([train_df, val_df, test_df])['label'].unique())

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
        fill_mode='nearest'
    )

    # Validation and test data: NO augmentation, only preprocessing.
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
    """Get information about the dataset (unchanged)."""
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