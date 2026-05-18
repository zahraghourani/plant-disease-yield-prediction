"""
ROC curves and calibration analysis for the best CNN model: ConvNeXtXLarge.

Creates:
  - Per-class ROC curves
  - Per-class AUC CSV
  - Reliability diagram
  - Expected Calibration Error (ECE) CSV

Run from Git Bash with GPU enabled:
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate tf-gpu
    export PATH="$CONDA_PREFIX/Library/bin:$PATH"
    python src/cnn_calibration_roc.py
"""
from __future__ import annotations

import argparse
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "Crop___DIsease"
CHECKPOINT = ROOT / "checkpoints" / "final_ConvNeXtXLarge.weights.h5"
OUTPUT_DIR = ROOT / "results" / "cnn_calibration_roc"
FIGURES_DIR = ROOT / "results" / "figures"
MODEL_NAME = "ConvNeXtXLarge"


def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU visible to TensorFlow; running on CPU.", flush=True)
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"TensorFlow GPU enabled: {gpus}", flush=True)
    except RuntimeError as exc:
        print(f"GPU memory configuration skipped: {exc}", flush=True)


def build_model():
    sys.path.insert(0, str(ROOT / "src"))
    from model_factory import get_model

    model, preprocess_func, input_size = get_model(
        MODEL_NAME,
        num_classes=15,
        base_weights=None,
    )
    model.load_weights(CHECKPOINT)
    return model, preprocess_func, input_size


def create_validation_generator(preprocess_func, input_size, batch_size: int):
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_func,
        validation_split=0.30,
    )
    return datagen.flow_from_directory(
        DATA_DIR,
        target_size=input_size,
        batch_size=batch_size,
        class_mode="sparse",
        subset="validation",
        shuffle=False,
        seed=42,
    )


def compute_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    rows = []

    for idx in range(n_bins):
        lower = bins[idx]
        upper = bins[idx + 1]
        if idx == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)

        count = int(mask.sum())
        if count:
            avg_confidence = float(confidences[mask].mean())
            accuracy = float(correct[mask].mean())
            gap = abs(accuracy - avg_confidence)
            ece += (count / len(confidences)) * gap
        else:
            avg_confidence = 0.0
            accuracy = 0.0
            gap = 0.0

        rows.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "accuracy": accuracy,
                "avg_confidence": avg_confidence,
                "abs_gap": gap,
            }
        )

    return float(ece), rows


def save_roc_plot(y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str], output_dir: Path):
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
    plt.title("ConvNeXtXLarge Per-Class ROC Curves")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "convnext_xlarge_per_class_roc.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "ConvNeXtXLarge_per_class_roc.png", dpi=300, bbox_inches="tight")
    plt.close()

    with (output_dir / "convnext_xlarge_per_class_auc.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_name", "auc"])
        writer.writeheader()
        writer.writerows(auc_rows)

    return auc_rows


def save_reliability_plot(bin_rows: list[dict], ece: float, output_dir: Path):
    centers = [(row["bin_lower"] + row["bin_upper"]) / 2 for row in bin_rows]
    accuracies = [row["accuracy"] for row in bin_rows]
    confidences = [row["avg_confidence"] for row in bin_rows]
    counts = [row["count"] for row in bin_rows]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.bar(
        centers,
        accuracies,
        width=0.09,
        color="#2d6cdf",
        alpha=0.75,
        edgecolor="white",
        label="Bin accuracy",
    )
    ax.plot(centers, confidences, color="#157a58", marker="o", label="Mean confidence")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"ConvNeXtXLarge Reliability Diagram (ECE={ece:.4f})")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")

    for x, y, count in zip(centers, accuracies, counts):
        if count:
            ax.text(x, min(y + 0.035, 0.98), str(count), ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "convnext_xlarge_reliability_diagram.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "ConvNeXtXLarge_reliability_diagram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run(batch_size: int, n_bins: int) -> None:
    configure_gpu()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model, preprocess_func, input_size = build_model()
    val_gen = create_validation_generator(preprocess_func, input_size, batch_size)
    class_names = list(val_gen.class_indices.keys())

    y_true = val_gen.classes
    y_prob_all = model.predict(val_gen, verbose=1)
    y_prob = y_prob_all[:, : len(class_names)]
    y_prob = y_prob / np.clip(y_prob.sum(axis=1, keepdims=True), 1e-12, None)
    y_pred = np.argmax(y_prob, axis=1)

    auc_rows = save_roc_plot(y_true, y_prob, class_names, OUTPUT_DIR)

    confidences = np.max(y_prob, axis=1)
    correct = (y_pred == y_true).astype(np.float32)
    ece, bin_rows = compute_ece(confidences, correct, n_bins)

    save_reliability_plot(bin_rows, ece, OUTPUT_DIR)

    with (OUTPUT_DIR / "convnext_xlarge_ece.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ece", "n_bins", "n_samples", "accuracy"])
        writer.writeheader()
        writer.writerow(
            {
                "ece": ece,
                "n_bins": n_bins,
                "n_samples": len(y_true),
                "accuracy": float(correct.mean()),
            }
        )

    with (OUTPUT_DIR / "convnext_xlarge_calibration_bins.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["bin_lower", "bin_upper", "count", "accuracy", "avg_confidence", "abs_gap"],
        )
        writer.writeheader()
        writer.writerows(bin_rows)

    print(f"Saved ROC and calibration outputs to {OUTPUT_DIR}")
    print(f"Validation accuracy: {correct.mean():.4f}")
    print(f"ECE ({n_bins} bins): {ece:.4f}")
    print("Per-class AUC:")
    for row in auc_rows:
        print(f"  {row['class_name']:<25} {row['auc']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ROC and ECE plots for ConvNeXtXLarge.")
    parser.add_argument("--batch-size", type=int, default=8, help="Validation prediction batch size.")
    parser.add_argument("--bins", type=int, default=10, help="Number of bins for ECE.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(batch_size=args.batch_size, n_bins=args.bins)


if __name__ == "__main__":
    main()
