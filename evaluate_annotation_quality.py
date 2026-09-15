"""
evaluate_annotation_quality.py

Quantitative validation of Grounding DINO auto-annotations against the
manually-annotated ground truth (Reviewer 2 #6, Reviewer 5 #4, Reviewer 6 #4,
new reviewer #3 — all asked for an actual IoU metric, not just "50 samples
reviewed, looked reasonable").

For every manually-annotated image, this script:
  1. Loads the manual (ground truth) boxes from detection_data/annotations/*.xml
  2. Re-runs Grounding DINO on that same image with the same class prompt
     used in auto_annotate.py
  3. Matches predicted boxes to ground-truth boxes (greedy, by IoU) and
     computes per-image and per-class IoU
  4. Reports mean IoU overall and per class, plus simple precision/recall
     at a chosen IoU threshold

Run with:
    python evaluate_annotation_quality.py
"""

import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
grounding_dino_path = os.path.join(project_root, 'GroundingDINO')
sys.path.insert(0, grounding_dino_path)

import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from groundingdino.util.inference import load_model, load_image, predict

# ── CONFIG ──────────────────────────────────────────────────────────────────
MANUAL_ANNOTATIONS_DIR = "./detection_data/annotations"
MANUAL_IMAGES_DIR = "./detection_data/images"
WEIGHTS_PATH = "./weights/groundingdino_swint_ogc.pth"
CONFIG_PATH = "./weights/GroundingDINO_SwinT_OGC.py"

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
IOU_MATCH_THRESHOLD = 0.5   # for precision/recall @ this IoU
OUTPUT_CSV = "annotation_quality_results.csv"

# Must match the naming used in auto_annotate.py's DISEASE_PROMPTS exactly,
# since evaluation matches ground-truth <name> tags against these keys.
DISEASE_PROMPTS = {
    "Corn___Common_Rust":    "rust spots on corn leaf",
    "Corn___Leaf_Blight":    "blight lesion on corn leaf",
    "Corn___Gray_Leaf_Spot": "gray leaf spot lesions on corn leaf",
    "Corn___Healthy":        "healthy green corn leaf",
    "Potato___Early_Blight": "early blight spots on potato leaf",
    "Potato___Late_Blight":  "late blight on potato leaf",
    "Potato___Healthy":      "healthy green potato leaf",
    "Rice___Brown_Spot":     "brown spots on rice leaf",
    "Rice___Healthy":        "healthy green rice leaf",
    "Rice___Hispa":          "hispa damage on rice leaf",
    "Rice___Leaf_Blast":     "leaf blast on rice leaf",
    "Wheat___Brown_Rust":    "brown rust on wheat leaf",
    "Wheat___Healthy":       "healthy green wheat leaf",
    "Wheat___Yellow_Rust":   "yellow rust on wheat leaf",
}


def parse_voc_xml(xml_path):
    """Parse a Pascal VOC XML file, returning (class_name, [boxes])."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    class_name = None
    for obj in root.findall("object"):
        name = obj.find("name").text
        class_name = name  # all objects in our files share the folder's class
        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])
    return class_name, boxes


def compute_iou(box_a, box_b):
    """IoU between two [xmin, ymin, xmax, ymax] boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def run_dino_on_image(model, image_path, prompt, img_w, img_h):
    """Run Grounding DINO on a single image, return list of [xmin,ymin,xmax,ymax] boxes."""
    image_source, image = load_image(image_path)
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    boxes_xyxy = []
    for box in boxes:
        cx, cy, bw, bh = box
        x1 = max(0, (cx - bw / 2) * img_w)
        y1 = max(0, (cy - bh / 2) * img_h)
        x2 = min(img_w, (cx + bw / 2) * img_w)
        y2 = min(img_h, (cy + bh / 2) * img_h)
        boxes_xyxy.append([x1, y1, x2, y2])
    return boxes_xyxy


def match_boxes_greedy(gt_boxes, pred_boxes):
    """
    Greedy one-to-one matching between ground-truth and predicted boxes,
    by descending IoU. Returns list of (gt_idx, pred_idx, iou) matches,
    plus counts of unmatched GT (false negatives) and unmatched pred
    (false positives).
    """
    if not gt_boxes or not pred_boxes:
        return [], len(gt_boxes), len(pred_boxes)

    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            iou_matrix[i, j] = compute_iou(gt, pred)

    matches = []
    matched_gt = set()
    matched_pred = set()

    # Flatten and sort all pairs by IoU descending, greedily assign
    pairs = [(iou_matrix[i, j], i, j)
             for i in range(len(gt_boxes)) for j in range(len(pred_boxes))]
    pairs.sort(reverse=True)

    for iou, i, j in pairs:
        if i in matched_gt or j in matched_pred:
            continue
        if iou <= 0:
            break
        matches.append((i, j, iou))
        matched_gt.add(i)
        matched_pred.add(j)

    fn = len(gt_boxes) - len(matched_gt)
    fp = len(pred_boxes) - len(matched_pred)
    return matches, fn, fp


def main():
    print("Loading Grounding DINO model...")
    model = load_model(CONFIG_PATH, WEIGHTS_PATH)
    print("Model loaded.\n")

    xml_files = sorted(f for f in os.listdir(MANUAL_ANNOTATIONS_DIR) if f.endswith(".xml"))
    print(f"Found {len(xml_files)} manually-annotated XML files.\n")

    per_image_rows = []

    for i, xml_name in enumerate(xml_files, 1):
        xml_path = os.path.join(MANUAL_ANNOTATIONS_DIR, xml_name)
        base_name = os.path.splitext(xml_name)[0]

        # find matching image (try common extensions)
        image_path = None
        for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
            candidate = os.path.join(MANUAL_IMAGES_DIR, base_name + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break

        if image_path is None:
            print(f"[{i}/{len(xml_files)}] SKIP — no matching image for {xml_name}")
            continue

        class_name, gt_boxes = parse_voc_xml(xml_path)

        if class_name not in DISEASE_PROMPTS:
            print(f"[{i}/{len(xml_files)}] SKIP — no prompt for class {class_name}")
            continue

        prompt = DISEASE_PROMPTS[class_name]

        try:
            from PIL import Image
            with Image.open(image_path) as im:
                img_w, img_h = im.size

            pred_boxes = run_dino_on_image(model, image_path, prompt, img_w, img_h)
            matches, fn, fp = match_boxes_greedy(gt_boxes, pred_boxes)

            mean_iou = np.mean([m[2] for m in matches]) if matches else 0.0
            tp_at_thresh = sum(1 for m in matches if m[2] >= IOU_MATCH_THRESHOLD)

            per_image_rows.append({
                "filename": base_name,
                "class": class_name,
                "num_gt_boxes": len(gt_boxes),
                "num_pred_boxes": len(pred_boxes),
                "num_matches": len(matches),
                "mean_iou": mean_iou,
                "tp_at_iou_thresh": tp_at_thresh,
                "fn": fn,
                "fp": fp,
            })

            print(f"[{i}/{len(xml_files)}] {class_name} | {base_name} | "
                  f"GT={len(gt_boxes)} Pred={len(pred_boxes)} "
                  f"MeanIoU={mean_iou:.3f}")

        except Exception as e:
            print(f"[{i}/{len(xml_files)}] ERROR on {base_name}: {e}")
            continue

    if not per_image_rows:
        print("\nNo results collected — check paths and class name matching.")
        return

    df = pd.DataFrame(per_image_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nPer-image results saved to {OUTPUT_CSV}")

    # ── Aggregate summary ────────────────────────────────────────────────
    total_tp = df["tp_at_iou_thresh"].sum()
    total_fn = df["fn"].sum()
    total_fp = df["fp"].sum()

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    print("\n" + "=" * 70)
    print("OVERALL ANNOTATION QUALITY SUMMARY")
    print("=" * 70)
    print(f"Images evaluated:        {len(df)}")
    print(f"Mean IoU (matched pairs): {df['mean_iou'].mean():.4f}")
    print(f"Precision @ IoU>={IOU_MATCH_THRESHOLD}: {precision:.4f}")
    print(f"Recall @ IoU>={IOU_MATCH_THRESHOLD}:    {recall:.4f}")

    print("\nPer-class mean IoU:")
    per_class = df.groupby("class")["mean_iou"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    print(per_class.to_string())

    per_class.to_csv("annotation_quality_per_class.csv")
    print(f"\nPer-class summary saved to annotation_quality_per_class.csv")


if __name__ == "__main__":
    main()