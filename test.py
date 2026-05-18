import cv2
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path

images_dir = Path("detection_data/images")
xml_dir    = Path("detection_data/auto_annotations")

# blast XMLs = Rice Leaf Blast images
# Find any Rice Leaf Blast image
img_path = None
for f in images_dir.rglob("*"):
    if "Leaf_Blast" in f.name and f.suffix.upper() in [".JPG",".JPEG",".PNG"]:
        img_path = f
        break

# Use first blast xml
xml_path = next(xml_dir.rglob("blast_0.xml"))

print("Image:", img_path)
print("XML:  ", xml_path)

original = cv2.imread(str(img_path))
if original is None:
    raise ValueError(f"Cannot load image: {img_path}")

annotated = original.copy()
tree = ET.parse(str(xml_path))
boxes_found = 0
for obj in tree.findall('object'):
    bb = obj.find('bndbox')
    x1 = int(float(bb.find('xmin').text))
    y1 = int(float(bb.find('ymin').text))
    x2 = int(float(bb.find('xmax').text))
    y2 = int(float(bb.find('ymax').text))
    H, W = annotated.shape[:2]
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(W,x2), min(H,y2)
    print(f"  Box: ({x1},{y1}) -> ({x2},{y2})  image size: {W}x{H}")
    if x2 > x1 and y2 > y1:
        cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,0,220), 4)
        cv2.putText(annotated, "Rice Leaf Blast", (x1, max(y1-10,20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,220), 2)
        boxes_found += 1

print(f"Boxes drawn: {boxes_found}")

h = 420
def resize_h(img, h):
    w = int(img.shape[1] * h / img.shape[0])
    return cv2.resize(img, (w, h))

orig_r = resize_h(original,  h)
ann_r  = resize_h(annotated, h)

def add_label(img, text):
    out = cv2.copyMakeBorder(img, 44, 10, 10, 10,
                              cv2.BORDER_CONSTANT, value=(255,255,255))
    cv2.putText(out, text, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30,30,30), 2)
    return out

left  = add_label(orig_r, "(a) Original image")
right = add_label(ann_r,  "(b) Grounding DINO annotation")
divider = np.ones((left.shape[0], 8, 3), dtype=np.uint8) * 180
collage = np.hstack([left, divider, right])

Path("figures").mkdir(exist_ok=True)
cv2.imwrite("figures/annotation_collage.png", collage)
print("Saved: figures/annotation_collage.png")