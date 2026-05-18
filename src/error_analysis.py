"""
Error analysis for ConvNeXtXLarge plant disease classification.

Creates:
  - confusion_error_pairs.csv
  - misclassified_examples.csv
  - 3 example failure images
  - error_analysis.md with a short report-ready discussion

Run from Git Bash with GPU enabled:
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate tf-gpu
    export PATH="$CONDA_PREFIX/Library/bin:$PATH"
    python src/error_analysis.py --max-images 1200 --examples 3 --batch-size 1
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageStat


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "error_analysis_convnext_xlarge"
CONFUSION_MATRIX_PATH = ROOT / "results" / "figures" / "ConvNeXtXLarge_confusion_matrix.png"


@dataclass
class PredictionRecord:
    image_path: Path
    true_class: str
    predicted_class: str
    confidence: float
    brightness: float
    contrast: float

    @property
    def is_correct(self) -> bool:
        return self.true_class == self.predicted_class


def load_explainability_helpers():
    sys.path.insert(0, str(ROOT / "src"))
    from explainability import (
        DATA_DIR,
        build_model,
        configure_gpu,
        iter_image_paths,
        load_class_names,
        load_image_array,
    )

    return DATA_DIR, build_model, configure_gpu, iter_image_paths, load_class_names, load_image_array


def image_quality_stats(image_path: Path) -> tuple[float, float]:
    image = Image.open(image_path).convert("L").resize((224, 224))
    stat = ImageStat.Stat(image)
    brightness = float(stat.mean[0])
    contrast = float(stat.stddev[0])
    return brightness, contrast


def collect_predictions(max_images: int, batch_size: int) -> list[PredictionRecord]:
    (
        data_dir,
        build_model,
        configure_gpu,
        iter_image_paths,
        load_class_names,
        load_image_array,
    ) = load_explainability_helpers()

    configure_gpu()
    class_names = load_class_names(data_dir)
    model, preprocess_func, input_size = build_model()

    records: list[PredictionRecord] = []
    pending_images = []
    pending_meta = []

    def flush_batch() -> None:
        nonlocal pending_images, pending_meta
        if not pending_images:
            return
        probs_batch = model.predict(np.stack(pending_images, axis=0), verbose=0)
        for probs, (image_path, true_class) in zip(probs_batch, pending_meta):
            pred_idx = int(np.argmax(probs))
            pred_class = class_names[pred_idx]
            brightness, contrast = image_quality_stats(image_path)
            records.append(
                PredictionRecord(
                    image_path=image_path,
                    true_class=true_class,
                    predicted_class=pred_class,
                    confidence=float(probs[pred_idx]),
                    brightness=brightness,
                    contrast=contrast,
                )
            )
        pending_images = []
        pending_meta = []

    for count, (image_path, _, true_class) in enumerate(
        iter_image_paths(data_dir, class_names),
        start=1,
    ):
        if count > max_images:
            break
        _, batch = load_image_array(image_path, input_size, preprocess_func)
        pending_images.append(batch[0])
        pending_meta.append((image_path, true_class))
        if len(pending_images) >= batch_size:
            flush_batch()

    flush_batch()
    return records


def rank_confusions(records: list[PredictionRecord]) -> list[tuple[tuple[str, str], int]]:
    counter = Counter(
        (record.true_class, record.predicted_class)
        for record in records
        if not record.is_correct
    )
    return counter.most_common()


def choose_examples(records: list[PredictionRecord], examples: int) -> list[PredictionRecord]:
    by_pair: dict[tuple[str, str], list[PredictionRecord]] = defaultdict(list)
    for record in records:
        if not record.is_correct:
            by_pair[(record.true_class, record.predicted_class)].append(record)

    selected: list[PredictionRecord] = []
    for pair, pair_records in sorted(by_pair.items(), key=lambda item: len(item[1]), reverse=True):
        pair_records = sorted(pair_records, key=lambda record: record.confidence, reverse=True)
        selected.append(pair_records[0])
        if len(selected) >= examples:
            return selected

    # If one confusion pair dominates, fill remaining slots from that pair.
    if selected and len(selected) < examples:
        dominant_pair = (selected[0].true_class, selected[0].predicted_class)
        extras = [
            record
            for record in sorted(by_pair[dominant_pair], key=lambda item: item.confidence, reverse=True)
            if record not in selected
        ]
        selected.extend(extras[: examples - len(selected)])

    return selected[:examples]


def save_example_figure(record: PredictionRecord, output_path: Path) -> None:
    image = Image.open(record.image_path).convert("RGB")
    enhanced = ImageEnhance.Contrast(image).enhance(1.4)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(
        f"True: {record.true_class.replace('___', ' ')} | "
        f"Pred: {record.predicted_class.replace('___', ' ')} "
        f"({record.confidence * 100:.1f}%)",
        fontsize=10,
    )
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[1].imshow(enhanced)
    axes[1].set_title("Contrast enhanced")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_csv_outputs(
    records: list[PredictionRecord],
    confusion_pairs: list[tuple[tuple[str, str], int]],
    examples: list[PredictionRecord],
    output_dir: Path,
) -> None:
    with (output_dir / "confusion_error_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["true_class", "predicted_class", "count"])
        writer.writeheader()
        for (true_class, predicted_class), count in confusion_pairs:
            writer.writerow(
                {
                    "true_class": true_class,
                    "predicted_class": predicted_class,
                    "count": count,
                }
            )

    with (output_dir / "misclassified_examples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "true_class",
                "predicted_class",
                "confidence",
                "brightness",
                "contrast",
                "example_file",
            ],
        )
        writer.writeheader()
        for idx, record in enumerate(examples, start=1):
            writer.writerow(
                {
                    "image": str(record.image_path),
                    "true_class": record.true_class,
                    "predicted_class": record.predicted_class,
                    "confidence": f"{record.confidence:.6f}",
                    "brightness": f"{record.brightness:.2f}",
                    "contrast": f"{record.contrast:.2f}",
                    "example_file": f"failure_example_{idx}.png",
                }
            )


def write_discussion(
    records: list[PredictionRecord],
    confusion_pairs: list[tuple[tuple[str, str], int]],
    examples: list[PredictionRecord],
    output_dir: Path,
) -> None:
    total = len(records)
    incorrect = len([record for record in records if not record.is_correct])
    accuracy = 1 - incorrect / total if total else 0
    top_lines = "\n".join(
        f"- `{true}` confused as `{pred}`: {count} examples"
        for (true, pred), count in confusion_pairs[:5]
    )
    example_lines = "\n".join(
        f"- `failure_example_{idx}.png`: true `{record.true_class}`, "
        f"predicted `{record.predicted_class}` with {record.confidence * 100:.1f}% confidence. "
        f"Brightness={record.brightness:.1f}, contrast={record.contrast:.1f}."
        for idx, record in enumerate(examples, start=1)
    )

    discussion = f"""# Error Analysis: ConvNeXtXLarge

This section analyzes failure cases for the best CNN model, ConvNeXtXLarge. The saved confusion matrix is available at:

`{CONFUSION_MATRIX_PATH.relative_to(ROOT)}`

The automated scan evaluated {total} images and found {incorrect} incorrect predictions, giving an approximate scanned accuracy of {accuracy * 100:.2f}%.

## Most Common Confusion Cases

{top_lines if top_lines else "- No incorrect predictions found in the scanned subset."}

## Example Failure Images

{example_lines if example_lines else "- No failure examples were available."}

## Short Discussion

The dominant errors in this scan are within visually similar rice categories, especially healthy rice leaves predicted as `Rice___Hispa`. This suggests that the model sometimes responds to local texture, leaf edges, lighting variation, or small blemishes as if they were disease symptoms. These are plausible failure modes for leaf-disease classification because many classes share similar green leaf backgrounds while the disease cues may be small, sparse, or affected by illumination.

The selected failure examples show cases where the model is confident despite being incorrect. This is important for deployment: high softmax confidence should not be interpreted as a guarantee of correctness. In practice, the interface should show confidence and allow a human user to override the detected crop/disease when the visual evidence is ambiguous. Additional training data with more lighting conditions, healthy leaves, and visually similar rice disease categories would likely reduce these confusions.
"""
    (output_dir / "error_analysis.md").write_text(discussion, encoding="utf-8")


def run(max_images: int, examples_count: int, batch_size: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob("failure_example_*.png"):
        old_file.unlink()

    records = collect_predictions(max_images=max_images, batch_size=batch_size)
    confusion_pairs = rank_confusions(records)
    examples = choose_examples(records, examples_count)

    for idx, record in enumerate(examples, start=1):
        save_example_figure(record, output_dir / f"failure_example_{idx}.png")

    save_csv_outputs(records, confusion_pairs, examples, output_dir)
    write_discussion(records, confusion_pairs, examples, output_dir)

    print(f"Saved error analysis to {output_dir}")
    print("Top confusion pairs:")
    for (true_class, predicted_class), count in confusion_pairs[:5]:
        print(f"  {true_class} -> {predicted_class}: {count}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ConvNeXtXLarge error analysis examples.")
    parser.add_argument("--max-images", type=int, default=1200, help="Maximum images to scan.")
    parser.add_argument("--examples", type=int, default=3, help="Number of failure examples to save.")
    parser.add_argument("--batch-size", type=int, default=1, help="Prediction batch size.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        max_images=args.max_images,
        examples_count=args.examples,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
