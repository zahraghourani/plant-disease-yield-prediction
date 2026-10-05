"""
ROC curves and calibration analysis for ConvNeXtLarge (featured model).

Creates:
  - Per-class ROC curves
  - Per-class AUC CSV
  - Reliability diagram
  - Expected Calibration Error (ECE) CSV

IMPORTANT: this version uses the SAME corrected, group-aware, leak-free
test split (splits.csv, via create_data_generators) as evaluate_all_38.py
and generate_test_predictions.py — NOT an independent validation_split,
which would be a different, unverified population and could reintroduce
the duplicate-leakage bug that was already found and fixed.

Run from the project root:
    python src/cnn_calibration_roc.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "Crop___DIsease"
CHECKPOINT = ROOT / "checkpoints" / "final_ConvNeXtLarge.weights.h5"
OUTPUT_DIR = ROOT / "results" / "cnn_calibration_roc_convnext_large"
FIGURES_DIR = ROOT / "results" / "figures"
MODEL_NAME = "ConvNeXtLarge"
N_BINS = 10
BATCH_SIZE = 16


def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU visible to TensorFlow; running on CPU.", flush=True)
        return
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"TensorFlow GPU enabled: {gpus}", flush=True)


def build_model_and_test_set():
    sys.path.insert(0, str(ROOT / "src"))
    from model_factory import get_model
    from data_preprocessing import create_data_generators

    model, preprocess_func, input_size = get_model(
        MODEL_NAME, num_classes=15, base_weights=None,
    )
    model.load_weights(str(CHECKPOINT))

    # Reuse the SAME corrected, group-aware split as the rest of the paper.
    _, _, test_gen, class_names, _ = create_data_generators(
        str(DATA_DIR), preprocess_func, input_size, BATCH_SIZE,
        validation_split=0.15,
    )
    return model, test_gen, class_names


def compute_ece(confidences, correct, n_bins):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []
    for idx in range(n_bins):
        lower, upper = bins[idx], bins[idx + 1]
        mask = (confidences >= lower) & (confidences < upper if idx < n_bins - 1 else confidences <= upper)
        count = int(mask.sum())
        if count:
            avg_confidence = float(confidences[mask].mean())
            accuracy = float(correct[mask].mean())
            gap = abs(accuracy - avg_confidence)
            ece += (count / len(confidences)) * gap
        else:
            avg_confidence, accuracy, gap = 0.0, 0.0, 0.0
        rows.append({"bin_lower": lower, "bin_upper": upper, "count": count,
                      "accuracy": accuracy, "avg_confidence": avg_confidence, "abs_gap": gap})
    return float(ece), rows


def save_roc_plot(y_true, y_prob, class_names, output_dir):
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    auc_rows = []
    plt.figure(figsize=(11, 8))
    for idx, class_name in enumerate(class_names):
        if y_true_bin[:, idx].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_prob[:, idx])
        roc_auc = auc(fpr, tpr)
        auc_rows.append({"class_name": class_name, "auc": float(roc_auc)})
        plt.plot(fpr, tpr, linewidth=1.8, label=f"{class_name} ({roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{MODEL_NAME} Per-Class ROC Curves (held-out test set)")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / f"{MODEL_NAME.lower()}_per_class_roc.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / f"{MODEL_NAME}_per_class_roc.png", dpi=300, bbox_inches="tight")
    plt.close()

    with (output_dir / f"{MODEL_NAME.lower()}_per_class_auc.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_name", "auc"])
        writer.writeheader()
        writer.writerows(auc_rows)
    return auc_rows


def save_reliability_plot(bin_rows, ece, output_dir):
    centers = [(r["bin_lower"] + r["bin_upper"]) / 2 for r in bin_rows]
    accuracies = [r["accuracy"] for r in bin_rows]
    confidences = [r["avg_confidence"] for r in bin_rows]
    counts = [r["count"] for r in bin_rows]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.bar(centers, accuracies, width=0.09, color="#2d6cdf", alpha=0.75,
           edgecolor="white", label="Bin accuracy")
    ax.plot(centers, confidences, color="#157a58", marker="o", label="Mean confidence")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title(f"{MODEL_NAME} Reliability Diagram (ECE={ece:.4f}, held-out test set)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    for x, y, count in zip(centers, accuracies, counts):
        if count:
            ax.text(x, min(y + 0.035, 0.98), str(count), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"{MODEL_NAME.lower()}_reliability_diagram.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{MODEL_NAME}_reliability_diagram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    configure_gpu()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model, test_gen, class_names = build_model_and_test_set()

    test_gen.reset()
    y_true = test_gen.classes
    y_prob_all = model.predict(test_gen, verbose=1)
    y_prob = y_prob_all[:, :len(class_names)]
    y_prob = y_prob / np.clip(y_prob.sum(axis=1, keepdims=True), 1e-12, None)
    y_pred = np.argmax(y_prob, axis=1)

    print(f"Test set size: {len(y_true)} (should match n=4792 used elsewhere)")

    auc_rows = save_roc_plot(y_true, y_prob, class_names, OUTPUT_DIR)

    confidences = np.max(y_prob, axis=1)
    correct = (y_pred == y_true).astype(np.float32)
    ece, bin_rows = compute_ece(confidences, correct, N_BINS)

    save_reliability_plot(bin_rows, ece, OUTPUT_DIR)

    with (OUTPUT_DIR / f"{MODEL_NAME.lower()}_ece.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ece", "n_bins", "n_samples", "accuracy"])
        writer.writeheader()
        writer.writerow({"ece": ece, "n_bins": N_BINS, "n_samples": len(y_true),
                          "accuracy": float(correct.mean())})

    print(f"\nSaved ROC and calibration outputs to {OUTPUT_DIR}")
    print(f"Test accuracy (sanity check, should match ~0.9610): {correct.mean():.4f}")
    print(f"ECE ({N_BINS} bins): {ece:.4f}")
    print("Per-class AUC:")
    for row in auc_rows:
        print(f"  {row['class_name']:<25} {row['auc']:.4f}")


if __name__ == "__main__":
    main()