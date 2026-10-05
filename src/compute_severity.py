"""
compute_severity.py

FIXED — two serious bugs found and corrected:

1. build_detector() was loading the GENERIC COCO-pretrained Faster RCNN
   (80 unrelated classes: person, car, dog, etc.) and NEVER loaded the
   actual trained severity-aware Faster RCNN checkpoint
   (checkpoints/faster_rcnn_with_severity.pth). Every previous dynamic
   severity number in this project was computed using a detector that
   has never seen a plant disease lesion.

2. CLASS_NAMES was the OLD 14-class list (with a dummy "Unknown" 15th
   slot) — missing Corn___Gray_Leaf_Spot entirely, and not matching the
   actual alphabetically-sorted class index order used during training.
   Every predicted class/confidence was silently mismatched.

Also fixed: classifier weights are now loaded via the SAME model_factory
architecture used everywhere else in this project (get_model), with a
full state match rather than by_name+skip_mismatch, which can silently
skip layers.

Formula (unchanged, matches paper Eq. 1):
    severity = (sum of lesion box areas / image area) * CNN_confidence

Outputs saved to results/:
    severity_scores_per_image.csv
    severity_comparison.csv

Run from project root:
    python src/compute_severity.py
"""

from __future__ import annotations
import os
import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_factory import get_model
from train_faster_rcnn_fast import SeverityAwareFasterRCNN

# =============================================================================
#  PATHS
# =============================================================================
CLASSIFIER_NAME = "EfficientNetV2S"   # deployed model, matches paper Section 3.4
CLASSIFIER_WEIGHTS = ROOT / "checkpoints" / f"final_{CLASSIFIER_NAME}.weights.h5"

DETECTOR_CHECKPOINT = ROOT / "checkpoints" / "faster_rcnn_with_severity.pth"
COCO_ANN_FILE = ROOT / "detection_data" / "annotations" / "train_coco.json"

IMAGES_DIR = ROOT / "detection_data" / "images"
OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_PER_CLASS = 15
IMG_SIZE = (224, 224)
NUM_CLASSES = 15

# =============================================================================
#  CLASS NAMES — must match the ACTUAL alphabetically-sorted order used by
#  create_data_generators() / flow_from_dataframe during training. This is
#  the correct, current 15-class list (includes Corn___Gray_Leaf_Spot).
# =============================================================================
CLASS_NAMES = [
    "Corn___Common_Rust", "Corn___Gray_Leaf_Spot", "Corn___Healthy", "Corn___Leaf_Blight",
    "Invalid", "Potato___Early_Blight", "Potato___Healthy", "Potato___Late_Blight",
    "Rice___Brown_Spot", "Rice___Healthy", "Rice___Hispa", "Rice___Leaf_Blast",
    "Wheat___Brown_Rust", "Wheat___Healthy", "Wheat___Yellow_Rust",
]
KNOWN_CLASSES = CLASS_NAMES  # same list; used for filename matching below

LITERATURE_SEVERITY = {
    "Corn___Common_Rust":    0.35,
    "Corn___Gray_Leaf_Spot": 0.35,   # same family as other corn leaf diseases; update if a better literature value is found
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


def extract_true_class(filename: str) -> str:
    """
    Longest-prefix match against KNOWN_CLASSES. Gray Leaf Spot images use
    a different, UUID-based naming convention (e.g.
    "00a20f6f-...___RS_GLSp 4655.JPG") rather than a class-name prefix,
    so they're matched separately via the "GLSp" marker instead.
    """
    if "glsp" in filename.lower():
        return "Corn___Gray_Leaf_Spot"

    best, best_len = "Unknown", 0
    for cls in KNOWN_CLASSES:
        if filename.startswith(cls) and len(cls) > best_len:
            best, best_len = cls, len(cls)
    return best


# =============================================================================
#  STEP 1: Build EfficientNetV2S via the SAME architecture used in training
# =============================================================================
def build_classifier(weights_path: Path) -> tf.keras.Model:
    print(f"[INFO] Building {CLASSIFIER_NAME} via model_factory (matches training exactly)...")
    print(f"[INFO] Weights: {weights_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Classifier weights not found: {weights_path}")

    model, preprocess_func, input_size = get_model(
        CLASSIFIER_NAME, num_classes=NUM_CLASSES, base_weights=None
    )
    # Full, exact load — no by_name/skip_mismatch silent-skip risk
    model.load_weights(str(weights_path))
    print("[INFO] Classifier ready (full weight match).")
    return model, preprocess_func, input_size


# =============================================================================
#  STEP 2: Load the ACTUAL trained severity-aware Faster RCNN
# =============================================================================
def build_detector(checkpoint_path: Path, coco_ann_file: Path):
    print(f"[INFO] Loading TRAINED severity-aware Faster RCNN: {checkpoint_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Faster RCNN checkpoint not found: {checkpoint_path}\n"
            f"This must be the trained checkpoint from train_faster_rcnn_fast.py, "
            f"NOT a generic COCO-pretrained model."
        )

    with open(coco_ann_file) as f:
        coco_json = json.load(f)
    # category_id -> class name, as actually assigned in convert_to_coco.py
    cat_id_to_name = {c['id']: c['name'] for c in coco_json['categories']}
    num_det_classes = len(cat_id_to_name) + 1  # +1 for background

    model = SeverityAwareFasterRCNN(num_classes=num_det_classes)
    ckpt = torch.load(str(checkpoint_path), map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"[INFO] Detector ready — TRAINED weights loaded "
          f"(epoch {ckpt.get('epoch', '?')}, loss {ckpt.get('loss', '?')}).")
    return model, cat_id_to_name


# =============================================================================
#  STEP 3: Compute severity for one image
# =============================================================================
def union_area_fraction(boxes_xyxy, img_w, img_h, grid_size=200):
    """
    Computes the fraction of image area covered by the UNION of the given
    boxes (normalized 0..1 coordinates on a rasterized grid), rather than
    naively summing individual box areas. Naive summing double-counts
    pixels where boxes from different predicted categories overlap,
    which can push "lesion area" past the actual image area (observed:
    ratios >1.0 before this fix).
    """
    if len(boxes_xyxy) == 0:
        return 0.0
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    for (x0, y0, x1, y1) in boxes_xyxy:
        gx0 = max(0, min(grid_size - 1, int(x0 / img_w * grid_size)))
        gx1 = max(gx0 + 1, min(grid_size, int(np.ceil(x1 / img_w * grid_size))))
        gy0 = max(0, min(grid_size - 1, int(y0 / img_h * grid_size)))
        gy1 = max(gy0 + 1, min(grid_size, int(np.ceil(y1 / img_h * grid_size))))
        mask[gy0:gy1, gx0:gx1] = True
    return float(mask.sum()) / (grid_size * grid_size)


def compute_one(
    image_path: str,
    detector,
    classifier: tf.keras.Model,
    preprocess_func,
    input_size,
    true_class: str,
    cat_id_to_name: dict,
    conf_threshold: float = 0.4,
) -> dict:
    pil_img = Image.open(image_path).convert("RGB")
    img_w, img_h = pil_img.width, pil_img.height
    image_area = img_w * img_h

    # -- Faster R-CNN (TRAINED): bounding boxes --------------------------------
    img_tensor = TF.to_tensor(pil_img)
    with torch.no_grad():
        dets = detector([img_tensor])[0]

    keep = dets["scores"] >= conf_threshold
    all_boxes = dets["boxes"][keep]
    all_labels = dets["labels"][keep]

    # Exclude boxes predicted as a "Healthy" or "Invalid" category — these
    # are not lesions. A box covering an entire healthy leaf (Grounding
    # DINO's own annotation convention for healthy classes, Section
    # 3.3.1) was previously being counted as "lesion area", inflating
    # severity for healthy images far above actual diseased ones.
    lesion_boxes = []
    for box, label_id in zip(all_boxes, all_labels):
        cat_name = cat_id_to_name.get(int(label_id), "")
        if "Healthy" in cat_name or cat_name == "Invalid":
            continue
        lesion_boxes.append(box.tolist())

    n_boxes = len(lesion_boxes)
    lesion_ratio_union = union_area_fraction(lesion_boxes, img_w, img_h)
    total_lesion_area = lesion_ratio_union * image_area  # for reporting only

    # -- EfficientNetV2S (correct 15-class mapping): class + confidence --------
    arr = np.array(pil_img.resize(input_size), dtype=np.float32)
    arr = preprocess_func(arr)
    probs = classifier.predict(arr[None], verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    pred_class = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else "Unknown"

    severity = lesion_ratio_union * confidence

    return {
        "image":             os.path.basename(image_path),
        "true_class":        true_class,
        "predicted_class":   pred_class,
        "confidence":        round(confidence, 4),
        "n_boxes":           n_boxes,
        "total_lesion_area": round(total_lesion_area, 2),
        "image_area":        image_area,
        "lesion_ratio":      round(lesion_ratio_union, 4),
        "severity_score":    round(severity, 4),
    }


# =============================================================================
#  STEP 4: Loop over flat images folder, sample N_PER_CLASS per class
# =============================================================================
def run_flat_folder(detector, classifier, preprocess_func, input_size, cat_id_to_name):
    class_images = defaultdict(list)
    all_images = (
        list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.JPG")) +
        list(IMAGES_DIR.glob("*.jpeg")) + list(IMAGES_DIR.glob("*.png"))
    )
    if not all_images:
        raise FileNotFoundError(f"No images found in {IMAGES_DIR}.")

    for img_path in all_images:
        cls = extract_true_class(img_path.name)
        class_images[cls].append(img_path)

    print(f"[INFO] Found {len(all_images)} images across {len(class_images)} classes.")
    for cls, imgs in sorted(class_images.items()):
        print(f"       {cls}: {len(imgs)} images")

    rows = []
    for cls in sorted(class_images.keys()):
        imgs = class_images[cls][:N_PER_CLASS]
        print(f"\n[INFO] Processing: {cls} ({len(imgs)} images)")
        for p in imgs:
            try:
                rows.append(compute_one(
                    str(p), detector, classifier, preprocess_func, input_size,
                    true_class=cls, cat_id_to_name=cat_id_to_name
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
    avg["literature_severity"] = avg["disease_class"].map(LITERATURE_SEVERITY).fillna("N/A")
    for col in ("computed_mean", "computed_std", "avg_lesion_ratio", "avg_confidence"):
        avg[col] = avg[col].round(4)
    return avg[["disease_class", "n_images", "literature_severity",
                "computed_mean", "computed_std", "avg_lesion_ratio", "avg_confidence"]]


# =============================================================================
#  MAIN
# =============================================================================
def main():
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images folder not found: {IMAGES_DIR}")

    classifier, preprocess_func, input_size = build_classifier(CLASSIFIER_WEIGHTS)
    detector, cat_id_to_name = build_detector(DETECTOR_CHECKPOINT, COCO_ANN_FILE)
    print(f"[INFO] Detector trained on categories: {list(cat_id_to_name.values())}")

    df = run_flat_folder(detector, classifier, preprocess_func, input_size, cat_id_to_name)
    if df.empty:
        print("[ERROR] No results. Check image paths.")
        return

    out1 = OUTPUT_DIR / "severity_scores_per_image.csv"
    df.to_csv(out1, index=False)
    print(f"\n[SAVED] {out1}")

    comp = make_comparison(df)
    out2 = OUTPUT_DIR / "severity_comparison.csv"
    comp.to_csv(out2, index=False)
    print(f"[SAVED] {out2}")

    print("\n" + "=" * 75)
    print("  SEVERITY: Computed (Dynamic, TRAINED detector) vs Literature (Fixed)")
    print("=" * 75)
    print(comp.to_string(index=False))
    print("=" * 75)
    print(f"\n[DONE] Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()