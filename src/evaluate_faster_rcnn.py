import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import csv
import torch
import torchvision
from pathlib import Path
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader
from train_faster_rcnn_torchvision import CocoDetection, get_transform


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "faster_rcnn"
FIGURES_DIR = ROOT / "results" / "figures"


def evaluate(model, data_loader, device, coco_gt):
    model.eval()
    results = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                img_id = targets[i]["image_id"].item()
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()

                for box, score, label in zip(boxes, scores, labels):
                    xmin, ymin, xmax, ymax = box
                    results.append(
                        {
                            "image_id": img_id,
                            "category_id": int(label),
                            "bbox": [
                                float(xmin),
                                float(ymin),
                                float(xmax - xmin),
                                float(ymax - ymin),
                            ],
                            "score": float(score),
                        }
                    )

    if not results:
        print("No predictions found.")
        return None

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


def summarize_per_class_ap(coco_eval, coco_gt):
    """Return per-class AP@0.5 and AP@0.5:0.95 from a completed COCOeval."""
    precision = coco_eval.eval["precision"]  # [IoU, recall, class, area, max detections]
    cat_ids = coco_eval.params.catIds
    iou_50_index = int(np.where(np.isclose(coco_eval.params.iouThrs, 0.5))[0][0])

    rows = []
    for class_index, cat_id in enumerate(cat_ids):
        category = coco_gt.loadCats([cat_id])[0]
        precision_all = precision[:, :, class_index, 0, -1]
        precision_50 = precision[iou_50_index, :, class_index, 0, -1]
        valid_all = precision_all[precision_all > -1]
        valid_50 = precision_50[precision_50 > -1]

        rows.append(
            {
                "category_id": cat_id,
                "class_name": category["name"],
                "ap_50_95": float(np.mean(valid_all)) if valid_all.size else 0.0,
                "ap_50": float(np.mean(valid_50)) if valid_50.size else 0.0,
            }
        )

    return sorted(rows, key=lambda row: row["ap_50"], reverse=True)


def save_per_class_ap_outputs(per_class_df, coco_eval):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with (RESULTS_DIR / "faster_rcnn_per_class_ap.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category_id", "class_name", "ap_50_95", "ap_50"])
        writer.writeheader()
        writer.writerows(per_class_df)

    with (RESULTS_DIR / "faster_rcnn_coco_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mAP_50_95", "mAP_50", "mAP_75"])
        writer.writeheader()
        writer.writerow(
            {
                "mAP_50_95": coco_eval.stats[0],
                "mAP_50": coco_eval.stats[1],
                "mAP_75": coco_eval.stats[2],
            }
        )

    plot_df = sorted(per_class_df, key=lambda row: row["ap_50"])
    fig, ax = plt.subplots(figsize=(10, max(5, 0.42 * len(plot_df))))
    ap_values = [row["ap_50"] for row in plot_df]
    class_names = [row["class_name"] for row in plot_df]
    bars = ax.barh(class_names, ap_values, color="#2d6cdf")
    ax.set_xlim(0, 1)
    ax.set_xlabel("AP@0.5")
    ax.set_title("Faster R-CNN Per-Class AP@0.5")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, ap_values):
        ax.text(
            min(value + 0.015, 0.98),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "faster_rcnn_per_class_ap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved per-class AP CSV to {RESULTS_DIR / 'faster_rcnn_per_class_ap.csv'}")
    print(f"Saved clean AP figure to {FIGURES_DIR / 'faster_rcnn_per_class_ap.png'}")


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    data_dir = "data/Crop___DIsease"
    ann_file = "detection_data/annotations/train_coco.json"
    dataset = CocoDetection(data_dir, ann_file, transforms=get_transform())
    coco_gt = dataset.coco
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    num_classes = len(coco_gt.getCatIds()) + 1
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features,
        num_classes,
    )
    model.load_state_dict(torch.load(ROOT / "faster_rcnn_model.pth", map_location=device))
    model.to(device)

    coco_eval = evaluate(model, loader, device, coco_gt)
    if coco_eval is None:
        return

    print(f"\nCOCO mAP@0.5:0.95 = {coco_eval.stats[0]:.4f}")
    print(f"COCO mAP@0.5      = {coco_eval.stats[1]:.4f}")

    per_class_df = summarize_per_class_ap(coco_eval, coco_gt)
    print("\nPer-class AP@0.5:")
    for row in per_class_df:
        print(f"{row['class_name']:<25} AP@0.5={row['ap_50']:.4f} AP@0.5:0.95={row['ap_50_95']:.4f}")
    save_per_class_ap_outputs(per_class_df, coco_eval)


if __name__ == "__main__":
    main()
