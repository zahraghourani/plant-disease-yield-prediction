"""
generate_iou_comparison_figure.py

Produces one PNG with two side-by-side panels:
  (a) a healthy-class image where manual and Grounding DINO boxes nearly
      perfectly overlap (green = manual, red = Grounding DINO)
  (b) a multi-lesion diseased image where Grounding DINO predicts a single
      box against many manual ground-truth boxes, visually demonstrating
      the "systematic undercounting" failure mode described in
      Section 3.3.1

Run with:
    python generate_iou_comparison_figure.py
"""

import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
grounding_dino_path = os.path.join(project_root, 'GroundingDINO')
sys.path.insert(0, grounding_dino_path)

import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
from groundingdino.util.inference import load_model, load_image, predict

# ── CONFIG ──────────────────────────────────────────────────────────────────
MANUAL_ANNOTATIONS_DIR = "./detection_data/annotations"
MANUAL_IMAGES_DIR = "./detection_data/images"
WEIGHTS_PATH = "./weights/groundingdino_swint_ogc.pth"
CONFIG_PATH = "./weights/GroundingDINO_SwinT_OGC.py"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
OUTPUT_PATH = "./figures/iou_comparison_healthy_vs_multilesion.png"

# Same-crop pairing (both Potato) to avoid the odd aspect-ratio crop we
# hit with the Wheat Brown Rust image — these two come from the same
# photo family so rendering style is consistent.
# - good agreement: GT=1, Pred=2, IoU=0.872
# - undercounting: GT=18, Pred=1, IoU=0.027
GOOD_AGREEMENT_XML = "Potato___Late_Blight_image (770).xml"
UNDERCOUNT_XML = "Potato___Early_Blight_image (993).xml"

DISEASE_PROMPTS = {
    "Potato___Late_Blight":  "late blight on potato leaf",
    "Potato___Early_Blight": "early blight spots on potato leaf",
}


def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes, class_name, filename = [], None, None
    filename = root.find("filename").text
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        bnd = obj.find("bndbox")
        boxes.append([
            float(bnd.find("xmin").text), float(bnd.find("ymin").text),
            float(bnd.find("xmax").text), float(bnd.find("ymax").text),
        ])
    return class_name, filename, boxes


def find_image(filename):
    for ext_dir_name in (filename,):
        p = os.path.join(MANUAL_IMAGES_DIR, filename)
        if os.path.exists(p):
            return p
    return None


def run_dino(model, image_path, prompt, img_w, img_h):
    image_source, image = load_image(image_path)
    boxes, logits, phrases = predict(
        model=model, image=image, caption=prompt,
        box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
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


def draw_panel(ax, image_path, manual_boxes, pred_boxes, title):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    for box in manual_boxes:
        x1, y1, x2, y2 = box
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    edgecolor='lime', facecolor='none', linewidth=2))
    for box in pred_boxes:
        x1, y1, x2, y2 = box
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                    edgecolor='red', facecolor='none', linewidth=2,
                                    linestyle='--'))
    ax.set_title(title, fontsize=11)
    ax.axis('off')


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Loading Grounding DINO model...")
    model = load_model(CONFIG_PATH, WEIGHTS_PATH)
    print("Model loaded.\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, xml_name, panel_label in [
        (axes[0], POTATO_XML, "(a) Potato Early Blight"),
        (axes[1], WHEAT_XML, "(b) Wheat Yellow Rust"),
    ]:
        xml_path = os.path.join(MANUAL_ANNOTATIONS_DIR, xml_name)
        class_name, filename, manual_boxes = parse_voc_xml(xml_path)
        image_path = find_image(filename)

        if image_path is None:
            print(f"MISSING IMAGE for {xml_name} -> {filename}")
            continue

        img = cv2.imread(image_path)
        img_h, img_w = img.shape[:2]
        prompt = DISEASE_PROMPTS[class_name]
        pred_boxes = run_dino(model, image_path, prompt, img_w, img_h)

        title = f"{panel_label}\nManual: {len(manual_boxes)} boxes | Grounding DINO: {len(pred_boxes)} boxes"
        draw_panel(ax, image_path, manual_boxes, pred_boxes, title)

    # Shared legend
    handles = [
        plt.Line2D([0], [0], color='lime', lw=2, label='Manual (ground truth)'),
        plt.Line2D([0], [0], color='red', lw=2, linestyle='--', label='Grounding DINO (predicted)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight')
    print(f"\nSaved comparison figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()