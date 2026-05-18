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

# Models were trained with 15 output classes (14 disease/healthy + 1 Invalid)
NUM_CLASSES = 15
BATCH_SIZE  = 32
SEED        = 42


# ── Load test images ──────────────────────────────────────────────────────────
def load_test_dataset(class_names, input_size):
    """Returns (X_test, y_test) as numpy arrays using same split as training."""
    all_paths, all_labels = [], []
    for idx, cls in enumerate(class_names):
        folder = DATA_DIR / cls
        if not folder.exists():
            continue
        for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]:
            for img_path in folder.glob(ext):
                all_paths.append(str(img_path))
                all_labels.append(idx)

    # Same 85/15 split + same seed as training → identical test set
    _, test_paths, _, test_labels = train_test_split(
        all_paths, all_labels,
        test_size=0.15, random_state=SEED, stratify=all_labels
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
        model_name = ckpt.name.replace("final_", "").replace(".weights.h5", "")
        print(f"[{model_name}] Loading...")

        # ── Load model with 15 classes (matches training) ──────────────────
        try:
            model, preprocess_fn, input_size = get_model(
                model_name,
                num_classes=NUM_CLASSES,
                base_weights=None,
            )
            model.load_weights(str(ckpt))
        except Exception as e:
            print(f"  ERROR loading {model_name}: {e}")
            continue

        # ── Load test images ────────────────────────────────────────────────
        print(f"  Loading test images at {input_size}...")
        X_test, y_test = load_test_dataset(class_names, list(input_size))

        if preprocess_fn is not None:
            X_test = preprocess_fn(X_test)

        # ── Predict ─────────────────────────────────────────────────────────
        print(f"  Predicting on {len(X_test)} test images...")
        t0 = time.time()
        y_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
        elapsed = time.time() - t0

        # y_prob has shape (n, 15) — model output classes
        # y_test has values 0..13 (14 folders) — binarize against 15 cols
        y_bin = label_binarize(y_test, classes=list(range(NUM_CLASSES)))

        # If test set is missing the 15th class (Invalid), pad with zeros
        if y_bin.shape[1] < NUM_CLASSES:
            pad = np.zeros((y_bin.shape[0], NUM_CLASSES - y_bin.shape[1]))
            y_bin = np.hstack([y_bin, pad])

        # ── ROC-AUC ─────────────────────────────────────────────────────────
        # Compute ROC-AUC — only on classes present in test set
        unique_classes = sorted(np.unique(y_test).tolist())
        y_bin  = label_binarize(y_test, classes=list(range(NUM_CLASSES)))
        y_bin_f   = y_bin[:, unique_classes]
        y_prob_f  = y_prob[:, unique_classes]

        try:
            auc_ovr = roc_auc_score(y_bin_f, y_prob_f, average="macro")
            auc_ovo = roc_auc_score(y_bin_f, y_prob_f, average="macro")
            print(f"  ROC-AUC OVR={auc_ovr:.4f}  OVO={auc_ovo:.4f}  ({elapsed:.1f}s)")
            results.append({
                "model_name":  model_name,
                "roc_auc_ovr": round(auc_ovr, 6),
                "roc_auc_ovo": round(auc_ovo, 6),
            })
        except Exception as e:
            print(f"  ROC-AUC error: {e}  ({elapsed:.1f}s)")
            results.append({
                "model_name":  model_name,
                "roc_auc_ovr": "",
                "roc_auc_ovo": "",
            })

        # Save after every model so progress is not lost if it crashes
        OUTPUT_CSV.parent.mkdir(exist_ok=True)
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["model_name", "roc_auc_ovr", "roc_auc_ovo"]
            )
            writer.writeheader()
            writer.writerows(results)

        tf.keras.backend.clear_session()

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\nDone. Saved to {OUTPUT_CSV}")
    df = pd.DataFrame(results)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()