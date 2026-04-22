import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ANNOTATIONS_DIR = "./detection_data/auto_annotations"
IMAGES_DIR      = "./data/Crop___Disease"
OUTPUT_JSON     = "./detection_data/annotations/train_coco.json"

os.makedirs("./detection_data/annotations", exist_ok=True)

categories = {}
images     = []
annotations = []
ann_id     = 1
image_id   = 1

xml_files = list(Path(ANNOTATIONS_DIR).glob("*.xml"))
print(f"Found {len(xml_files)} XML files")

for xml_file in xml_files:
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.findtext("filename")
    size     = root.find("size")
    w        = int(size.findtext("width"))
    h        = int(size.findtext("height"))

    # Find the actual image path including subfolder
    matches = list(Path(IMAGES_DIR).rglob(filename))
    if matches:
        # Store relative path like "Corn___Common_Rust/filename.jpg"
        rel_path = matches[0].relative_to(IMAGES_DIR)
        file_name_with_path = str(rel_path)
    else:
        file_name_with_path = filename

    images.append({
        "id":        image_id,
        "file_name": file_name_with_path,  # ← now includes subfolder
        "width":     w,
        "height":    h
    })

    for obj in root.findall("object"):
        label = obj.findtext("name")
        if label not in categories:
            categories[label] = len(categories) + 1

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.findtext("xmin"))
        ymin = float(bndbox.findtext("ymin"))
        xmax = float(bndbox.findtext("xmax"))
        ymax = float(bndbox.findtext("ymax"))
        bw   = xmax - xmin
        bh   = ymax - ymin

        annotations.append({
            "id":          ann_id,
            "image_id":    image_id,
            "category_id": categories[label],
            "bbox":        [xmin, ymin, bw, bh],
            "area":        bw * bh,
            "iscrowd":     0
        })
        ann_id += 1

    image_id += 1

coco = {
    "images":      images,
    "annotations": annotations,
    "categories":  [{"id": v, "name": k} for k, v in categories.items()]
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(coco, f, indent=2)

print(f"✓ Saved to {OUTPUT_JSON}")
print(f"  Images:      {len(images)}")
print(f"  Annotations: {len(annotations)}")
print(f"  Categories:  {list(categories.keys())}")