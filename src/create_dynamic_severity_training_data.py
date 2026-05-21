"""
Create synthetic dynamic severity training data for XGBoost.

Since the FAO dataset has no images, we approximate the dynamic severity
distribution by running Faster R-CNN + EfficientNet on a sample of
your detection dataset images, then merging statistics back into FAO.

Usage:
    python src/create_dynamic_severity_training_data.py

This produces: data/yield_data/yield_df_dynamic_severity.csv
which replaces the fixed CROP_AVG_SEVERITY with learned distributions.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "yield_data" / "yield_df.csv"
DETECTION_IMAGE_DIR = ROOT / "data" / "Crop___DIsease"
RCNN_SEVERITY = ROOT / "checkpoints" / "faster_rcnn_with_severity.pth"
EFFICIENTNET_CHECKPOINT = ROOT / "checkpoints" / "final_EfficientNetV2S.weights.h5"

NUM_CLASSES = 15
DETECTION_THRESHOLD = 0.4

# Disease → crop mapping
DISEASE_TO_CROP = {
    "Corn___Common_Rust": "Maize", "Corn___Leaf_Blight": "Maize",
    "Corn___Healthy": "Maize", "Potato___Early_Blight": "Potatoes",
    "Potato___Late_Blight": "Potatoes", "Potato___Healthy": "Potatoes",
    "Rice___Brown_Spot": "Rice, paddy", "Rice___Hispa": "Rice, paddy",
    "Rice___Leaf_Blast": "Rice, paddy", "Rice___Healthy": "Rice, paddy",
    "Wheat___Brown_Rust": "Wheat", "Wheat___Yellow_Rust": "Wheat",
    "Wheat___Healthy": "Wheat", "Invalid": "Maize", "Unknown": "Maize",
}


def load_severity_rcnn():
    """Load the trained severity-aware Faster R-CNN."""
    sys.path.insert(0, str(ROOT / "src"))
    import importlib
    train_module = importlib.import_module("train_faster_rcnn_torchvision")
    SeverityAwareFasterRCNN = train_module.SeverityAwareFasterRCNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeverityAwareFasterRCNN(num_classes=NUM_CLASSES)
    ckpt = torch.load(str(RCNN_SEVERITY), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    return model, device


def load_efficientnet():
    """Load EfficientNetV2S for classification confidence."""
    sys.path.insert(0, str(ROOT / "src"))
    from model_factory import get_model

    model, preprocess, input_size = get_model(
        "EfficientNetV2S", num_classes=15, base_weights=None
    )
    model.load_weights(EFFICIENTNET_CHECKPOINT)
    return model, preprocess, input_size


def compute_dynamic_severity_for_image(image_path, rcnn_model, effnet_model, 
                                       preprocess, input_size, device):
    """
    Run both models on one image and return dynamic severity.

    severity = (total lesion area / image area) * cnn_confidence
    """
    # ── Load image ──────────────────────────────────────────────────────────
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    image_area = img_w * img_h

    # ── Faster R-CNN: detect lesions ────────────────────────────────────────
    tensor = T.ToTensor()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = rcnn_model(tensor)

    out = outputs[0]
    boxes = out["boxes"].cpu().numpy()
    scores = out["scores"].cpu().numpy()

    # Filter by threshold and compute lesion area
    total_lesion_area = 0.0
    n_boxes = 0
    for box, score in zip(boxes, scores):
        if score >= DETECTION_THRESHOLD:
            x1, y1, x2, y2 = box
            total_lesion_area += (x2 - x1) * (y2 - y1)
            n_boxes += 1

    lesion_ratio = total_lesion_area / image_area if image_area > 0 else 0.0

    # ── EfficientNetV2S: classify + get confidence ──────────────────────────
    resized = img.resize(input_size)
    arr = np.array(resized, dtype=np.float32)
    if preprocess is not None:
        arr = preprocess(arr)

    preds = effnet_model.predict(np.expand_dims(arr, 0), verbose=0)[0]
    class_idx = int(np.argmax(preds))
    cnn_conf = float(preds[class_idx])

    # ── Dynamic severity ────────────────────────────────────────────────────
    severity = float(min(max(lesion_ratio * cnn_conf, 0.0), 1.0))

    # Get disease name from folder structure
    disease = image_path.parent.name
    crop = DISEASE_TO_CROP.get(disease, "Maize")

    return {
        "disease": disease,
        "crop": crop,
        "severity": severity,
        "lesion_ratio": lesion_ratio,
        "cnn_confidence": cnn_conf,
        "n_boxes": n_boxes,
        "image_area": image_area,
    }


def sample_severity_by_crop(n_per_class=50):
    """
    Sample n images per disease class, compute dynamic severity,
    and return statistics by crop.
    """
    print("Loading models...")
    rcnn_model, device = load_severity_rcnn()
    effnet_model, preprocess, input_size = load_efficientnet()

    results = []

    # Walk through detection dataset folders
    for disease_dir in sorted(DETECTION_IMAGE_DIR.iterdir()):
        if not disease_dir.is_dir():
            continue

        disease = disease_dir.name
        crop = DISEASE_TO_CROP.get(disease, "Maize")

        # Get image files
        images = list(disease_dir.glob("*.jpg")) + list(disease_dir.glob("*.png"))
        if len(images) == 0:
            continue

        # Sample n_per_class (or all if fewer)
        n_sample = min(n_per_class, len(images))
        sampled = np.random.choice(images, size=n_sample, replace=False)

        print(f"  Processing {disease}: {n_sample} images...")

        for img_path in sampled:
            try:
                result = compute_dynamic_severity_for_image(
                    img_path, rcnn_model, effnet_model, 
                    preprocess, input_size, device
                )
                results.append(result)
            except Exception as e:
                print(f"    Error on {img_path}: {e}")
                continue

    return pd.DataFrame(results)


def merge_into_fao(severity_stats):
    """
    Merge computed severity statistics into FAO yield data.

    For each crop, we use the MEAN of sampled dynamic severities
    as the training value (better than fixed literature).
    """
    # Load FAO data
    df = pd.read_csv(DATA_DIR)
    df = df[df["Item"].isin(["Maize", "Wheat", "Rice, paddy", "Potatoes"])].copy()

    # Compute per-crop mean severity from dynamic sampling
    crop_severity = severity_stats.groupby("crop")["severity"].mean().to_dict()

    print("\nDynamic severity statistics by crop:")
    for crop, sev in sorted(crop_severity.items()):
        print(f"  {crop}: {sev:.4f}")

    # Also show distribution stats
    print("\nFull distribution stats:")
    print(severity_stats.groupby("crop")["severity"].describe())

    # Replace literature values with dynamic means
    df["disease_severity"] = df["Item"].map(crop_severity).fillna(0.0)

    # Save
    out_path = ROOT / "data" / "yield_data" / "yield_df_dynamic_severity.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    return df, crop_severity


def main():
    print("=" * 60)
    print("Creating dynamic severity training data for XGBoost")
    print("=" * 60)

    # Step 1: Sample images and compute dynamic severity
    severity_stats = sample_severity_by_crop(n_per_class=50)

    # Step 2: Merge into FAO data
    df, crop_severity = merge_into_fao(severity_stats)

    # Step 3: Summary
    print("\n" + "=" * 60)
    print("Comparison: Literature vs. Dynamic severity")
    print("=" * 60)

    literature = {
        "Maize": (0.35 + 0.40 + 0.00) / 3,
        "Wheat": (0.35 + 0.40 + 0.00) / 3,
        "Rice, paddy": (0.25 + 0.30 + 0.50 + 0.00) / 4,
        "Potatoes": (0.20 + 0.45 + 0.00) / 3,
    }

    print(f"{'Crop':<15} {'Literature':>12} {'Dynamic':>12} {'Diff':>12}")
    print("-" * 55)
    for crop in sorted(literature.keys()):
        lit = literature[crop]
        dyn = crop_severity.get(crop, 0.0)
        diff = dyn - lit
        print(f"{crop:<15} {lit:>12.4f} {dyn:>12.4f} {diff:>+12.4f}")

    print("\nDone. Use yield_df_dynamic_severity.csv for XGBoost training.")


if __name__ == "__main__":
    main()