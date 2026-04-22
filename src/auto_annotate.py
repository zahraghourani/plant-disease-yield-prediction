import os
import torch
from groundingdino.util.inference import load_model, load_image, predict
import xml.etree.ElementTree as ET
from xml.dom import minidom

# ── CONFIG ──────────────────────────────────────────────────────────
IMAGES_DIR      = "./data/Crop___Disease"
ANNOTATIONS_DIR = "./detection_data/auto_annotations"
WEIGHTS_PATH    = "./weights/groundingdino_swint_ogc.pth"
CONFIG_PATH     = "./weights/GroundingDINO_SwinT_OGC.py"
BOX_THRESHOLD   = 0.30
TEXT_THRESHOLD  = 0.25

DISEASE_PROMPTS = {
    "Corn___Common_Rust":    "rust spots on corn leaf",
    "Corn___Leaf_Blight":    "blight lesion on corn leaf",
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
    "Invalid":               "plant leaf",
}

def save_xml(image_path, boxes, labels, w, h, save_path):
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

def main():
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    print("Loading Grounding DINO model...")
    model = load_model(CONFIG_PATH, WEIGHTS_PATH)
    print("Model loaded!\n")

    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    # Collect all images from all class subfolders
    image_files = []
    for class_folder in os.listdir(IMAGES_DIR):
        class_path = os.path.join(IMAGES_DIR, class_folder)
        if os.path.isdir(class_path):
            for fname in os.listdir(class_path):
                if any(fname.endswith(ext) for ext in image_extensions):
                    image_files.append((os.path.join(class_path, fname), class_folder, fname))

    print(f"Found {len(image_files)} images across all classes\n")

    annotated = 0
    skipped   = 0

    for i, (image_path, class_name, fname) in enumerate(image_files):
        xml_name  = os.path.splitext(fname)[0] + ".xml"
        save_path = os.path.join(ANNOTATIONS_DIR, xml_name)

        if os.path.exists(save_path):
            skipped += 1
            continue

        print(f"[{i+1}/{len(image_files)}] {class_name} | {fname}")
        prompt = DISEASE_PROMPTS.get(class_name, "disease on plant leaf")

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

            if len(boxes) == 0:
                boxes_xyxy = [[0, 0, w, h]]
            else:
                boxes_xyxy = []
                for box in boxes:
                    cx, cy, bw, bh = box
                    x1 = max(0, int((cx - bw/2) * w))
                    y1 = max(0, int((cy - bh/2) * h))
                    x2 = min(w, int((cx + bw/2) * w))
                    y2 = min(h, int((cy + bh/2) * h))
                    boxes_xyxy.append([x1, y1, x2, y2])

            labels = [class_name] * len(boxes_xyxy)
            save_xml(image_path, boxes_xyxy, labels, w, h, save_path)
            print(f"  ✓ {len(boxes_xyxy)} boxes saved")
            annotated += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\n{'='*50}")
    print(f"Done! Annotated: {annotated} | Skipped: {skipped}")
    print(f"Saved to: {ANNOTATIONS_DIR}")

if __name__ == "__main__":
    main()