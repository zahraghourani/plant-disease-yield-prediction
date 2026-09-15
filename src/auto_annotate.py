import sys
import os

# Add GroundingDINO to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
grounding_dino_path = os.path.join(project_root, 'GroundingDINO')
sys.path.insert(0, grounding_dino_path)

# Now these imports will work
from groundingdino.util.inference import load_model, load_image, predict

import os
import torch
from groundingdino.util.inference import load_model, load_image, predict
import xml.etree.ElementTree as ET
from xml.dom import minidom
from PIL import Image
import cv2
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

IMAGES_DIR      = "./data/Crop___Disease"           # Your 29K image dataset
ANNOTATIONS_DIR = "./detection_data/auto_annotations_full"  # NEW folder for full run
WEIGHTS_PATH    = "./weights/groundingdino_swint_ogc.pth"
CONFIG_PATH     = "./weights/GroundingDINO_SwinT_OGC.py"

BOX_THRESHOLD   = 0.35   # Slightly higher to reduce false positives
TEXT_THRESHOLD  = 0.25

# Minimum image dimensions to include (skip tiny thumbnails)
MIN_IMG_SIZE = 100  # pixels

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
    # EXCLUDE "Invalid" — these have no disease to detect
}

# ═══════════════════════════════════════════════════════════════════════════════
# XML SAVE (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def save_xml(image_path, boxes, labels, w, h, save_path):
    """Save Pascal VOC format XML. Skip if no valid boxes."""
    if not boxes:
        return False  # Signal: nothing to save
    
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "images"
    ET.SubElement(root, "filename").text = os.path.basename(image_path)
    ET.SubElement(root, "path").text = str(image_path)
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = "3"
    
    for box, label in zip(boxes, labels):
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = label
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(box[0]))
        ET.SubElement(bndbox, "ymin").text = str(int(box[1]))
        ET.SubElement(bndbox, "xmax").text = str(int(box[2]))
        ET.SubElement(bndbox, "ymax").text = str(int(box[3]))
    
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
    with open(save_path, "w") as f:
        f.write(xml_str)
    return True

# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def is_valid_image(path):
    """Check image is readable, not corrupted, and meets minimum size."""
    try:
        # Method 1: PIL
        with Image.open(path) as img:
            w, h = img.size
            if w < MIN_IMG_SIZE or h < MIN_IMG_SIZE:
                return False, "too_small"
            if img.mode not in ('RGB', 'RGBA', 'L'):
                return False, f"mode_{img.mode}"
        
        # Method 2: OpenCV verification
        cv_img = cv2.imread(path)
        if cv_img is None:
            return False, "cv2_none"
        if cv_img.shape[0] < MIN_IMG_SIZE or cv_img.shape[1] < MIN_IMG_SIZE:
            return False, "cv2_too_small"
            
        return True, "ok"
    except Exception as e:
        return False, f"exception: {e}"

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ANNOTATION LOOP — ALL VALID IMAGES
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    
    print("Loading Grounding DINO model...")
    model = load_model(CONFIG_PATH, WEIGHTS_PATH)
    print("Model loaded!\n")

    # ══ COLLECT ALL IMAGES ═════════════════════════════════════════════════════
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG', '.bmp', '.tiff'}
    image_files = []
    
    for class_folder in sorted(os.listdir(IMAGES_DIR)):
        class_path = os.path.join(IMAGES_DIR, class_folder)
        if not os.path.isdir(class_path):
            continue
        
        # SKIP "Invalid" class — no disease to detect
        if class_folder == "Invalid" or "invalid" in class_folder.lower():
            print(f"  Skipping class: {class_folder} (no disease annotations needed)")
            continue
            
        # Skip if no prompt defined
        if class_folder not in DISEASE_PROMPTS:
            print(f"  ⚠️  No prompt for class: {class_folder} — skipping")
            continue
        
        for fname in sorted(os.listdir(class_path)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in image_extensions:
                continue
            
            full_path = os.path.join(class_path, fname)
            image_files.append((full_path, class_folder, fname))
    
    total_images = len(image_files)
    print(f"{'='*60}")
    print(f"Found {total_images} valid images across {len(set(c for _, c, _ in image_files))} classes")
    print(f"Annotations will save to: {ANNOTATIONS_DIR}")
    print(f"{'='*60}\n")

    # ══ ANNOTATE ALL ═══════════════════════════════════════════════════════════
    stats = {
        'processed': 0,
        'annotated_with_boxes': 0,
        'no_detection': 0,      # DINO found nothing — SKIP (don't create fake box)
        'invalid_image': 0,
        'error': 0,
        'skipped_existing': 0,
    }

    for i, (image_path, class_name, fname) in enumerate(image_files, 1):
        xml_name  = os.path.splitext(fname)[0] + ".xml"
        save_path = os.path.join(ANNOTATIONS_DIR, xml_name)

        # Skip if already annotated (but track it)
        if os.path.exists(save_path):
            stats['skipped_existing'] += 1
            if i % 500 == 0:
                print(f"[{i}/{total_images}] SKIP (exists): {fname}")
            continue

        # Validate image
        valid, reason = is_valid_image(image_path)
        if not valid:
            stats['invalid_image'] += 1
            print(f"[{i}/{total_images}] INVALID ({reason}): {fname}")
            continue

        print(f"[{i}/{total_images}] {class_name} | {fname}")

        prompt = DISEASE_PROMPTS[class_name]

        try:
            image_source, image = load_image(image_path)
            h, w = image_source.shape[:2]

            boxes, logits, phrases = predict(
                model=model,
                image=image,
                caption=prompt,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD
            )

            # ══ CRITICAL FIX: Only save if DINO actually found something ═══════
            if len(boxes) == 0:
                print(f"  ⚠️  NO DETECTION — skipping (no fake full-image box)")
                stats['no_detection'] += 1
                continue  # DON'T create [0,0,w,h] fake annotation!

            # Convert normalized boxes to pixel coordinates
            boxes_xyxy = []
            for box in boxes:
                cx, cy, bw, bh = box
                x1 = max(0, int((cx - bw/2) * w))
                y1 = max(0, int((cy - bh/2) * h))
                x2 = min(w, int((cx + bw/2) * w))
                y2 = min(h, int((cy + bh/2) * h))
                
                # Filter tiny boxes (< 1% of image area = likely noise)
                box_area = (x2 - x1) * (y2 - y1)
                img_area = w * h
                if box_area < 0.01 * img_area:
                    print(f"  🗑️  Filtered tiny box: {box_area/img_area:.3%} of image")
                    continue
                    
                boxes_xyxy.append([x1, y1, x2, y2])

            if not boxes_xyxy:
                print(f"  ⚠️  All boxes filtered — skipping")
                stats['no_detection'] += 1
                continue

            labels = [class_name] * len(boxes_xyxy)
            saved = save_xml(image_path, boxes_xyxy, labels, w, h, save_path)
            
            if saved:
                print(f"  ✓ {len(boxes_xyxy)} boxes saved")
                stats['annotated_with_boxes'] += 1
            else:
                stats['no_detection'] += 1

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            stats['error'] += 1
            continue
        
        stats['processed'] += 1

        # Progress summary every 100 images
        if i % 100 == 0:
            print(f"\n--- Progress [{i}/{total_images}] ---")
            print(f"  Annotated: {stats['annotated_with_boxes']} | "
                  f"No detection: {stats['no_detection']} | "
                  f"Invalid: {stats['invalid_image']} | "
                  f"Errors: {stats['error']}\n")

    # ══ FINAL SUMMARY ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("ANNOTATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total images found:      {total_images}")
    print(f"Skipped (existing):      {stats['skipped_existing']}")
    print(f"Invalid images:          {stats['invalid_image']}")
    print(f"Processed:               {stats['processed']}")
    print(f"  ├─ Annotated (boxes):  {stats['annotated_with_boxes']}")
    print(f"  ├─ No detection:       {stats['no_detection']}")
    print(f"  └─ Errors:             {stats['error']}")
    print(f"\nFinal annotated dataset: {stats['annotated_with_boxes']} images with boxes")
    print(f"Saved to: {ANNOTATIONS_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()