"""
Grad-CAM explainability for the best CNN model.

Generates heatmaps for 3 correct and 3 incorrect ConvNeXtXLarge predictions.

Run from the project root:
    python src/explainability.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import Model


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "Crop___DIsease"
CHECKPOINT = ROOT / "checkpoints" / "final_ConvNeXtLarge.weights.h5"
OUTPUT_DIR = ROOT / "results" / "gradcam_convnext_large"
MODEL_NAME = "ConvNeXtLarge"
NUM_CLASSES = 15


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


@dataclass
class PredictionCase:
    image_path: Path
    true_index: int
    true_class: str
    predicted_index: int
    predicted_class: str
    confidence: float
    is_correct: bool


def load_class_names(data_dir: Path) -> list[str]:
    class_names = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
    while len(class_names) < NUM_CLASSES:
        class_names.append("Unknown")
    return class_names


def iter_image_paths(data_dir: Path, class_names: list[str]):
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    per_class = []
    for class_index, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            continue
        paths = [
            image_path
            for image_path in sorted(class_dir.iterdir())
            if image_path.suffix.lower() in image_exts
        ]
        per_class.append((class_index, class_name, paths))

    max_len = max((len(paths) for _, _, paths in per_class), default=0)
    for offset in range(max_len):
        for class_index, class_name, paths in per_class:
            if offset < len(paths):
                yield paths[offset], class_index, class_name


def load_image_array(image_path: Path, input_size: tuple[int, int], preprocess_func=None):
    image = Image.open(image_path).convert("RGB")
    resized = image.resize(input_size)
    array = np.asarray(resized, dtype=np.float32)
    if preprocess_func is not None:
        array = preprocess_func(array)
    return image, np.expand_dims(array, axis=0)


def build_model():
    sys.path.insert(0, str(ROOT / "src"))
    from model_factory import get_model

    model, preprocess_func, input_size = get_model(
        MODEL_NAME,
        num_classes=NUM_CLASSES,
        base_weights=None,
    )
    model.load_weights(CHECKPOINT)
    return model, preprocess_func, input_size


def find_prediction_cases(
    model,
    preprocess_func,
    input_size: tuple[int, int],
    class_names: list[str],
    max_images: int,
    needed_each: int,
    batch_size: int,
) -> list[PredictionCase]:
    correct: list[PredictionCase] = []
    incorrect: list[PredictionCase] = []

    pending_images = []
    pending_meta = []
    scanned = 0

    def flush_batch() -> bool:
        nonlocal pending_images, pending_meta
        if not pending_images:
            return False

        probs_batch = model.predict(np.stack(pending_images, axis=0), verbose=0)
        for probs, (image_path, true_index, true_class) in zip(probs_batch, pending_meta):
            predicted_index = int(np.argmax(probs))
            predicted_class = class_names[predicted_index]
            confidence = float(probs[predicted_index])
            is_correct = predicted_index == true_index

            case = PredictionCase(
                image_path=image_path,
                true_index=true_index,
                true_class=true_class,
                predicted_index=predicted_index,
                predicted_class=predicted_class,
                confidence=confidence,
                is_correct=is_correct,
            )

            if is_correct and len(correct) < needed_each:
                correct.append(case)
                print(f"Correct {len(correct)}/{needed_each}: {image_path.name} -> {predicted_class}", flush=True)
            elif not is_correct and len(incorrect) < needed_each:
                incorrect.append(case)
                print(
                    f"Incorrect {len(incorrect)}/{needed_each}: "
                    f"{image_path.name} true={true_class} pred={predicted_class}",
                    flush=True,
                )

            if len(correct) >= needed_each and len(incorrect) >= needed_each:
                pending_images = []
                pending_meta = []
                return True

        pending_images = []
        pending_meta = []
        return False

    for image_path, true_index, true_class in iter_image_paths(DATA_DIR, class_names):
        scanned += 1
        if scanned > max_images:
            break
        _, batch = load_image_array(image_path, input_size, preprocess_func)
        pending_images.append(batch[0])
        pending_meta.append((image_path, true_index, true_class))

        if len(pending_images) >= batch_size and flush_batch():
            break

    if pending_images:
        flush_batch()

    return correct + incorrect


def make_gradcam_heatmap(model, image_batch: np.ndarray, class_index: int) -> np.ndarray:
    target_layer = None
    for layer in reversed(model.layers):
        output_shape = getattr(layer, "output_shape", None)
        if isinstance(output_shape, tuple) and len(output_shape) == 4:
            target_layer = layer
            break
    if target_layer is None:
        raise ValueError("Could not find a 4D convolutional feature layer for Grad-CAM.")

    grad_model = Model(model.inputs, [target_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch, training=False)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.nn.relu(heatmap)

    max_value = tf.reduce_max(heatmap)
    if float(max_value) > 0:
        heatmap = heatmap / max_value
    return heatmap.numpy()


def overlay_heatmap(original: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    heatmap_image = Image.fromarray(np.uint8(255 * heatmap)).resize(original.size, Image.BILINEAR)
    colored = plt.get_cmap("jet")(np.asarray(heatmap_image) / 255.0)[..., :3]
    colored = Image.fromarray(np.uint8(colored * 255))
    return Image.blend(original.convert("RGB"), colored, alpha)


def save_case_figure(
    case: PredictionCase,
    original: Image.Image,
    heatmap: np.ndarray,
    overlay: Image.Image,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(
        f"{'Correct' if case.is_correct else 'Incorrect'} | "
        f"True: {case.true_class.replace('___', ' ')} | "
        f"Pred: {case.predicted_class.replace('___', ' ')} "
        f"({case.confidence * 100:.1f}%)",
        fontsize=11,
    )

    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for axis in axes:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_summary(cases: list[PredictionCase], output_dir: Path) -> None:
    summary_path = output_dir / "gradcam_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "true_class",
                "predicted_class",
                "confidence",
                "is_correct",
                "gradcam_file",
            ],
        )
        writer.writeheader()
        for i, case in enumerate(cases, start=1):
            status = "correct" if case.is_correct else "incorrect"
            writer.writerow(
                {
                    "image": str(case.image_path),
                    "true_class": case.true_class,
                    "predicted_class": case.predicted_class,
                    "confidence": f"{case.confidence:.6f}",
                    "is_correct": case.is_correct,
                    "gradcam_file": f"{i:02d}_{status}_{case.true_class}_pred_{case.predicted_class}.png",
                }
            )


def generate_gradcams(
    max_images: int,
    needed_each: int,
    output_dir: Path,
    batch_size: int,
) -> list[PredictionCase]:
    configure_gpu()
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob("*.png"):
        old_file.unlink()
    summary_file = output_dir / "gradcam_summary.csv"
    if summary_file.exists():
        summary_file.unlink()

    class_names = load_class_names(DATA_DIR)
    model, preprocess_func, input_size = build_model()

    print(f"Loaded {MODEL_NAME} with checkpoint: {CHECKPOINT}")
    print(f"Classes: {class_names}")
    print(f"Searching for {needed_each} correct and {needed_each} incorrect predictions...")

    cases = find_prediction_cases(
        model=model,
        preprocess_func=preprocess_func,
        input_size=input_size,
        class_names=class_names,
        max_images=max_images,
        needed_each=needed_each,
        batch_size=batch_size,
    )

    if len([case for case in cases if case.is_correct]) < needed_each:
        print("Warning: fewer correct cases found than requested.")
    if len([case for case in cases if not case.is_correct]) < needed_each:
        print("Warning: fewer incorrect cases found than requested.")

    for i, case in enumerate(cases, start=1):
        original, batch = load_image_array(case.image_path, input_size, preprocess_func)
        heatmap = make_gradcam_heatmap(model, batch, case.predicted_index)
        overlay = overlay_heatmap(original, heatmap)
        status = "correct" if case.is_correct else "incorrect"
        output_path = output_dir / (
            f"{i:02d}_{status}_{case.true_class}_pred_{case.predicted_class}.png"
        )
        save_case_figure(case, original, heatmap, overlay, output_path)
        print(f"Saved {output_path}")

    save_summary(cases, output_dir)
    print(f"Summary saved to {output_dir / 'gradcam_summary.csv'}")
    return cases


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM heatmaps for ConvNeXtXLarge.")
    parser.add_argument("--max-images", type=int, default=1500, help="Maximum images to scan.")
    parser.add_argument("--needed-each", type=int, default=3, help="Correct and incorrect cases to save.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--batch-size", type=int, default=2, help="Prediction scan batch size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_gradcams(
        max_images=args.max_images,
        needed_each=args.needed_each,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
