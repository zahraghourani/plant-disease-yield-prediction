"""
external_validation_plantvillage.py

Item 21 — External dataset validation (scoped).

IMPORTANT SCOPE NOTE: Only 3 of our 15 classes have a PlantVillage
counterpart in the downloaded subset (Pepper/Potato/Tomato only, not the
full PlantVillage dataset with Corn/Rice/Wheat): Potato Early Blight,
Potato Late Blight, and Potato Healthy. This script evaluates ONLY those
three classes. This is a genuine but LIMITED generalization check, not
full external validation across all crops — state this explicitly in
the paper (Section 4.x / Limitations), do not imply broader coverage.

Evaluation uses the model's FULL 15-class softmax head (realistic
deployment scenario: the model doesn't know in advance that only Potato
classes will appear), so a misclassification into any of the other 12
classes counts as wrong, not just confusion among the 3 Potato classes.

Run with:
    python src/external_validation_plantvillage.py
"""
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from model_factory import get_model

# ── CONFIG ──────────────────────────────────────────────────────────────
MODEL_NAME = "ConvNeXtLarge"   # our best model; change to "EfficientNetV2S" to test the deployed model instead
CHECKPOINT = ROOT / "checkpoints" / f"final_{MODEL_NAME}.weights.h5"
NUM_CLASSES = 15

PLANTVILLAGE_DIR = Path(r"C:\Users\HPZ4-03-Adm01\plant-disease-yield-prediction\data\PlantVillage")

# Our model's class order (must match training class_indices exactly)
OUR_CLASSES = [
    'Corn___Common_Rust', 'Corn___Gray_Leaf_Spot', 'Corn___Healthy', 'Corn___Leaf_Blight',
    'Invalid', 'Potato___Early_Blight', 'Potato___Healthy', 'Potato___Late_Blight',
    'Rice___Brown_Spot', 'Rice___Healthy', 'Rice___Hispa', 'Rice___Leaf_Blast',
    'Wheat___Brown_Rust', 'Wheat___Healthy', 'Wheat___Yellow_Rust',
]

# Map PlantVillage folder names -> our class index
PV_TO_OUR_CLASS = {
    "Potato___Early_blight": "Potato___Early_Blight",
    "Potato___healthy":      "Potato___Healthy",
    "Potato___Late_blight":  "Potato___Late_Blight",
}

OUTPUT_CSV = "plantvillage_external_validation_results.csv"


def load_image(path, input_size, preprocess_func):
    img = tf.io.read_file(str(path))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, input_size)
    img = img.numpy().astype(np.float32)
    if preprocess_func is not None:
        img = preprocess_func(img)
    return img


def main():
    print(f"Loading {MODEL_NAME}...")
    model, preprocess_func, input_size = get_model(MODEL_NAME, num_classes=NUM_CLASSES, base_weights=None)
    model.load_weights(str(CHECKPOINT))

    class_to_idx = {c: i for i, c in enumerate(OUR_CLASSES)}

    rows = []
    for pv_folder, our_class in PV_TO_OUR_CLASS.items():
        folder_path = PLANTVILLAGE_DIR / pv_folder
        if not folder_path.exists():
            print(f"  MISSING folder: {folder_path}")
            continue

        true_idx = class_to_idx[our_class]
        image_files = [f for f in folder_path.iterdir()
                       if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        print(f"\n{pv_folder} -> {our_class} ({len(image_files)} images)")

        batch_images = []
        for f in image_files:
            batch_images.append(load_image(f, input_size, preprocess_func))

        batch_images = np.stack(batch_images, axis=0)
        probs = model.predict(batch_images, batch_size=32, verbose=0)
        preds = np.argmax(probs, axis=1)

        for f, pred_idx in zip(image_files, preds):
            rows.append({
                "filename": f.name,
                "plantvillage_folder": pv_folder,
                "true_class": our_class,
                "predicted_class": OUR_CLASSES[pred_idx],
                "correct": pred_idx == true_idx,
            })

        acc = (preds == true_idx).mean()
        print(f"  Accuracy: {acc:.4f} ({(preds == true_idx).sum()}/{len(preds)})")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 70)
    print(f"OVERALL PlantVillage external validation ({MODEL_NAME}, Potato classes only)")
    print("=" * 70)
    print(f"Total images evaluated: {len(df)}")
    print(f"Overall accuracy: {df['correct'].mean():.4f}")
    print("\nPer-class accuracy:")
    print(df.groupby('true_class')['correct'].mean().to_string())

    print("\nMisclassification breakdown (where model predicted something other than the true Potato class):")
    wrong = df[~df['correct']]
    if len(wrong) > 0:
        print(wrong['predicted_class'].value_counts().to_string())
    else:
        print("  None — perfect accuracy on this subset.")

    print(f"\nSaved detailed results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()