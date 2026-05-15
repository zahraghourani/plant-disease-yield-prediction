"""
Run from project root:
    python src/compute_roc_auc.py

Loads every checkpoint in /checkpoints/, runs predict() on the test set,
computes ROC-AUC OVR and OVO, and saves results to results/roc_auc_results.csv
"""
import os, sys, time, csv
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_factory import get_model   # same function used in training

DATA_DIR   = ROOT / "data" / "Crop___Disease"
CKPT_DIR   = ROOT / "checkpoints"
OUTPUT_CSV = ROOT / "results" / "roc_auc_results.csv"

NUM_CLASSES = 15
IMG_SIZE    = (224, 224)   # overridden per model below
BATCH_SIZE  = 32
SEED        = 42

# ── Load test images ──────────────────────────────────────────────────────────
def load_test_dataset(class_names, input_size):
    """Returns (X_test, y_test) as numpy arrays."""
    all_paths, all_labels = [], []
    for idx, cls in enumerate(class_names):
        folder = DATA_DIR / cls
        if not folder.exists():
            continue
        for img_path in folder.glob("*.jpg"):
            all_paths.append(str(img_path))
            all_labels.append(idx)
        for img_path in folder.glob("*.JPG"):
            all_paths.append(str(img_path))
            all_labels.append(idx)
        for img_path in folder.glob("*.png"):
            all_paths.append(str(img_path))
            all_labels.append(idx)

    # Use same 85/15 split as training — fix seed so test set is identical
    _, test_paths, _, test_labels = train_test_split(
        all_paths, all_labels, test_size=0.15, random_state=SEED,
        stratify=all_labels
    )

    images = []
    for p in test_paths:
        img = tf.io.read_file(p)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, input_size)
        images.append(img.numpy())

    return np.array(images, dtype=np.float32), np.array(test_labels)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    class_names = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
    print(f"Classes ({len(class_names)}): {class_names}")

    results = []
    checkpoints = sorted(CKPT_DIR.glob("final_*.weights.h5"))
    print(f"\nFound {len(checkpoints)} checkpoints\n")

    for ckpt in checkpoints:
        # Extract model name from filename e.g. final_ConvNeXtXLarge.weights.h5
        model_name = ckpt.stem.replace("final_", "")
        print(f"[{model_name}] Loading...")

        try:
            model, preprocess_fn, input_size = get_model(
                model_name,
                num_classes=NUM_CLASSES,
                base_weights=None,   # don't load ImageNet weights
            )
            model.load_weights(str(ckpt))
        except Exception as e:
            print(f"  ERROR loading {model_name}: {e}")
            continue

        # Load test data at correct resolution
        print(f"  Loading test images at {input_size}...")
        X_test, y_test = load_test_dataset(class_names, list(input_size))

        # Apply preprocessing
        if preprocess_fn is not None:
            X_test = preprocess_fn(X_test)

        # Predict probabilities
        print(f"  Predicting on {len(X_test)} test images...")
        t0 = time.time()
        y_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
        elapsed = time.time() - t0

        # Compute ROC-AUC
        y_bin = label_binarize(y_test, classes=list(range(NUM_CLASSES)))
        try:
            auc_ovr = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
            auc_ovo = roc_auc_score(y_bin, y_prob, multi_class="ovo", average="macro")
        except Exception as e:
            print(f"  ROC-AUC error: {e}")
            auc_ovr, auc_ovo = None, None

        print(f"  ROC-AUC OVR={auc_ovr:.4f}  OVO={auc_ovo:.4f}  ({elapsed:.1f}s)")
        results.append({
            "model_name": model_name,
            "roc_auc_ovr": round(auc_ovr, 6) if auc_ovr else "",
            "roc_auc_ovo": round(auc_ovo, 6) if auc_ovo else "",
        })

        # Free GPU memory between models
        tf.keras.backend.clear_session()

    # Save
    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model_name","roc_auc_ovr","roc_auc_ovo"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone. Saved to {OUTPUT_CSV}")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()