"""
compute_severity.py
-------------------
Proof-of-concept: dynamic disease severity from:
  - Faster R-CNN (PyTorch, COCO pretrained)  ->  lesion bounding boxes
  - EfficientNetV2S (TensorFlow/Keras .h5)   ->  class + confidence

Formula:
    severity = (sum of lesion box areas / image area) * CNN_confidence

Outputs saved to results/:
    severity_scores_per_image.csv   -- one row per image
    severity_comparison.csv         -- computed vs literature values

HOW TO RUN (from project root):
    venv39\Scripts\activate
    python src/compute_severity.py
"""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from PIL import Image
import tensorflow as tf

# =============================================================================
#  PATHS
# =============================================================================
ROOT = Path(__file__).resolve().parents[1]

# Your EfficientNetV2S weights
CLASSIFIER_WEIGHTS = ROOT / "checkpoints" / "final_EfficientNetV2S.weights.h5"

# detection_data/images — flat folder, filenames start with class name
IMAGES_DIR = ROOT / "detection_data" / "images"

# Output folder
OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Images per class to process (keep small for speed)
N_PER_CLASS = 15

# Input size used during training
IMG_SIZE = (224, 224)

# =============================================================================
#  CLASS NAMES — 14 classes (your model was trained with 15 but the last
#  output was shape 15; we keep 14 matching your actual folders and add
#  a dummy to match the saved weight shape)
# =============================================================================
CLASS_NAMES = [
    "Corn___Common_Rust",
    "Corn___Healthy",
    "Corn___Leaf_Blight",
    "Invalid",
    "Potato___Early_Blight",
    "Potato___Healthy",
    "Potato___Late_Blight",
    "Rice___Brown_Spot",
    "Rice___Healthy",
    "Rice___Hispa",
    "Rice___Leaf_Blast",
    "Wheat___Brown_Rust",
    "Wheat___Healthy",
    "Wheat___Yellow_Rust",
    "Unknown",          # 15th slot to match saved weight shape (256->15)
]

# Known disease class prefixes in the flat image folder
# (extracted from filename before first non-class underscore segment)
KNOWN_CLASSES = [
    "Corn___Common_Rust",
    "Corn___Healthy",
    "Corn___Leaf_Blight",
    "Invalid",
    "Potato___Early_Blight",
    "Potato___Healthy",
    "Potato___Late_Blight",
    "Rice___Brown_Spot",
    "Rice___Healthy",
    "Rice___Hispa",
    "Rice___Leaf_Blast",
    "Wheat___Brown_Rust",
    "Wheat___Healthy",
    "Wheat___Yellow_Rust",
]

# Literature severity scores
LITERATURE_SEVERITY = {
    "Corn___Common_Rust":    0.35,
    "Corn___Leaf_Blight":    0.40,
    "Corn___Healthy":        0.00,
    "Potato___Early_Blight": 0.20,
    "Potato___Late_Blight":  0.45,
    "Potato___Healthy":      0.00,
    "Rice___Brown_Spot":     0.25,
    "Rice___Healthy":        0.00,
    "Rice___Hispa":          0.30,
    "Rice___Leaf_Blast":     0.50,
    "Wheat___Brown_Rust":    0.35,
    "Wheat___Healthy":       0.00,
    "Wheat___Yellow_Rust":   0.40,
    "Invalid":               0.00,
}

# =============================================================================
#  HELPER: extract true class from filename prefix
# =============================================================================
def extract_true_class(filename: str) -> str:
    """
    Filenames look like:
        Corn___Common_Rust_RS_Rust 2743.JPG
        Potato___Early_Blight_image (915).JPG
        Invalid_image (867).jpg
    We match against KNOWN_CLASSES by checking which known class
    the filename starts with (longest match wins).
    """
    stem = filename  # full filename
    best = "Unknown"
    best_len = 0
    for cls in KNOWN_CLASSES:
        if stem.startswith(cls) and len(cls) > best_len:
            best = cls
            best_len = len(cls)
    return best


# =============================================================================
#  STEP 1: Build EfficientNetV2S and load weights
# =============================================================================
def build_classifier(weights_path: Path) -> tf.keras.Model:
    print(f"[INFO] Building EfficientNetV2S (15-class output to match saved weights)...")
    print(f"[INFO] Weights: {weights_path}")

    base = tf.keras.applications.EfficientNetV2S(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 3),
        pooling="avg",
    )
    base.trainable = False

    inputs  = tf.keras.Input(shape=(224, 224, 3))
    x       = base(inputs, training=False)
    x       = tf.keras.layers.BatchNormalization()(x)
    x       = tf.keras.layers.Dense(512, activation="relu")(x)
    x       = tf.keras.layers.Dropout(0.5)(x)
    x       = tf.keras.layers.Dense(256, activation="relu")(x)
    x       = tf.keras.layers.Dropout(0.3)(x)
    # 15 outputs to match the saved weight shape (256, 15)
    outputs = tf.keras.layers.Dense(15, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    # Load weights — by_name matches layers by name, skips shape mismatches
    model.load_weights(str(weights_path), by_name=True, skip_mismatch=True)
    print("[INFO] Classifier ready.")
    return model


# =============================================================================
#  STEP 2: Load Faster R-CNN
# =============================================================================
def build_detector() -> torch.nn.Module:
    print("[INFO] Loading Faster R-CNN (COCO pretrained)...")
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model   = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    print("[INFO] Detector ready.")
    return model


# =============================================================================
#  STEP 3: Compute severity for one image
# =============================================================================
def compute_one(
    image_path: str,
    detector: torch.nn.Module,
    classifier: tf.keras.Model,
    true_class: str,
    conf_threshold: float = 0.4,
) -> dict:

    pil_img    = Image.open(image_path).convert("RGB")
    image_area = pil_img.width * pil_img.height

    # -- Faster R-CNN: bounding boxes -----------------------------------------
    img_tensor = TF.to_tensor(pil_img)
    with torch.no_grad():
        dets = detector([img_tensor])[0]

    boxes = dets["boxes"][dets["scores"] >= conf_threshold]
    total_lesion_area = sum(
        (b[2] - b[0]).item() * (b[3] - b[1]).item() for b in boxes
    )
    n_boxes = len(boxes)

    # -- EfficientNetV2S: class + confidence ----------------------------------
    arr   = np.array(pil_img.resize(IMG_SIZE), dtype=np.float32)
    arr   = tf.keras.applications.efficientnet_v2.preprocess_input(arr)
    probs = classifier.predict(arr[None], verbose=0)[0]  # shape (15,)

    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    pred_class = (
        CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else "Unknown"
    )

    # -- Severity formula ------------------------------------------------------
    severity = (total_lesion_area / image_area) * confidence

    return {
        "image":             os.path.basename(image_path),
        "true_class":        true_class,
        "predicted_class":   pred_class,
        "confidence":        round(confidence, 4),
        "n_boxes":           n_boxes,
        "total_lesion_area": round(total_lesion_area, 2),
        "image_area":        image_area,
        "lesion_ratio":      round(total_lesion_area / image_area, 4),
        "severity_score":    round(severity, 4),
    }


# =============================================================================
#  STEP 4: Loop over flat images folder, sample N_PER_CLASS per class
# =============================================================================
def run_flat_folder(detector, classifier):
    # Group images by true class extracted from filename
    from collections import defaultdict
    class_images = defaultdict(list)

    all_images = (
        list(IMAGES_DIR.glob("*.jpg")) +
        list(IMAGES_DIR.glob("*.JPG")) +
        list(IMAGES_DIR.glob("*.jpeg")) +
        list(IMAGES_DIR.glob("*.png"))
    )

    if not all_images:
        raise FileNotFoundError(
            f"No images found in {IMAGES_DIR}.\n"
            f"Check that IMAGES_DIR is correct."
        )

    for img_path in all_images:
        cls = extract_true_class(img_path.name)
        class_images[cls].append(img_path)

    print(f"[INFO] Found {len(all_images)} images across "
          f"{len(class_images)} classes in flat folder.")
    for cls, imgs in sorted(class_images.items()):
        print(f"       {cls}: {len(imgs)} images")

    rows = []
    for cls in sorted(class_images.keys()):
        imgs = class_images[cls][:N_PER_CLASS]
        print(f"\n[INFO] Processing: {cls} ({len(imgs)} images)")
        for p in imgs:
            try:
                rows.append(compute_one(
                    str(p), detector, classifier, true_class=cls
                ))
            except Exception as e:
                print(f"  [WARN] {p.name}: {e}")

    return pd.DataFrame(rows)


# =============================================================================
#  STEP 5: Comparison table
# =============================================================================
def make_comparison(df: pd.DataFrame) -> pd.DataFrame:
    avg = (
        df.groupby("true_class")
        .agg(
            n_images=("severity_score", "count"),
            computed_mean=("severity_score", "mean"),
            computed_std=("severity_score", "std"),
            avg_lesion_ratio=("lesion_ratio", "mean"),
            avg_confidence=("confidence", "mean"),
        )
        .reset_index()
        .rename(columns={"true_class": "disease_class"})
    )
    avg["literature_severity"] = avg["disease_class"].map(
        LITERATURE_SEVERITY
    ).fillna("N/A")
    avg["computed_mean"]     = avg["computed_mean"].round(4)
    avg["computed_std"]      = avg["computed_std"].round(4)
    avg["avg_lesion_ratio"]  = avg["avg_lesion_ratio"].round(4)
    avg["avg_confidence"]    = avg["avg_confidence"].round(4)

    return avg[[
        "disease_class", "n_images",
        "literature_severity",
        "computed_mean", "computed_std",
        "avg_lesion_ratio", "avg_confidence",
    ]]


# =============================================================================
#  MAIN
# =============================================================================
def main():
    # -- Validate paths -------------------------------------------------------
    if not CLASSIFIER_WEIGHTS.exists():
        ckpt_dir = ROOT / "checkpoints"
        available = list(ckpt_dir.glob("*.h5")) if ckpt_dir.exists() else []
        raise FileNotFoundError(
            f"\n[ERROR] Weights not found:\n  {CLASSIFIER_WEIGHTS}\n\n"
            f"Available .h5 files:\n" +
            ("\n".join(f"  {p.name}" for p in available) or "  (none)")
        )

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Images folder not found:\n  {IMAGES_DIR}"
        )

    # -- Load models ----------------------------------------------------------
    classifier = build_classifier(CLASSIFIER_WEIGHTS)
    detector   = build_detector()

    # -- Run ------------------------------------------------------------------
    df = run_flat_folder(detector, classifier)

    if df.empty:
        print("[ERROR] No results. Check image paths.")
        return

    # -- Save -----------------------------------------------------------------
    out1 = OUTPUT_DIR / "severity_scores_per_image.csv"
    df.to_csv(out1, index=False)
    print(f"\n[SAVED] {out1}")

    comp = make_comparison(df)
    out2 = OUTPUT_DIR / "severity_comparison.csv"
    comp.to_csv(out2, index=False)
    print(f"[SAVED] {out2}")

    # -- Print ----------------------------------------------------------------
    print("\n" + "="*75)
    print("  SEVERITY: Computed (Dynamic) vs Literature (Fixed)")
    print("="*75)
    print(comp.to_string(index=False))
    print("="*75)
    print(f"\n[DONE] Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()