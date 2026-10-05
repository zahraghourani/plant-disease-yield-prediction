"""
diagnose_zero_detections.py

Checks whether Corn_Common_Rust, Potato_Early_Blight, and
Potato_Late_Blight scoring exactly 0.0000 severity is a genuine
detection failure or just the 0.4 confidence threshold being too
strict for these classes — by printing the RAW top detection scores
(no threshold applied) for a few sample images per class.

Run with:
    python src/diagnose_zero_detections.py
"""
import sys
import json
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train_faster_rcnn_fast import SeverityAwareFasterRCNN

CHECKPOINT = ROOT / "checkpoints" / "faster_rcnn_with_severity.pth"
COCO_ANN_FILE = ROOT / "detection_data" / "annotations" / "train_coco.json"
IMAGES_DIR = ROOT / "detection_data" / "images"

CLASSES_TO_CHECK = {
    "Corn___Common_Rust":    "RS_Rust",
    "Potato___Early_Blight": "Early_Blight",
    "Potato___Late_Blight":  "Late_Blight",
}
N_SAMPLES_PER_CLASS = 3


def main():
    with open(COCO_ANN_FILE) as f:
        coco_json = json.load(f)
    cat_id_to_name = {c['id']: c['name'] for c in coco_json['categories']}
    num_det_classes = len(cat_id_to_name) + 1

    model = SeverityAwareFasterRCNN(num_classes=num_det_classes)
    ckpt = torch.load(str(CHECKPOINT), map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    for class_name, filename_hint in CLASSES_TO_CHECK.items():
        matches = [p for p in IMAGES_DIR.glob("*.jpg")] + \
                  [p for p in IMAGES_DIR.glob("*.JPG")]
        matches = [p for p in matches if class_name in p.name or filename_hint in p.name][:N_SAMPLES_PER_CLASS]

        print(f"\n=== {class_name} ===")
        if not matches:
            print("  No sample images found with this naming pattern.")
            continue

        for img_path in matches:
            pil_img = Image.open(img_path).convert("RGB")
            img_tensor = TF.to_tensor(pil_img)
            with torch.no_grad():
                dets = model([img_tensor])[0]

            scores = dets["scores"]
            labels = dets["labels"]

            if len(scores) == 0:
                print(f"  {img_path.name}: NO detections at all (even unthresholded)")
                continue

            # Show top 5 raw detections, regardless of threshold
            top_k = min(5, len(scores))
            print(f"  {img_path.name}: top {top_k} raw detections (threshold-free):")
            for i in range(top_k):
                cat_name = cat_id_to_name.get(int(labels[i]), f"id={int(labels[i])}")
                print(f"    score={scores[i].item():.4f}  category={cat_name}")


if __name__ == "__main__":
    main()