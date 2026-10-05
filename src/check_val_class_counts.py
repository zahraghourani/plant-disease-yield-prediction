"""
check_val_class_counts.py

Quick check: how many ground-truth annotations for Potato Early Blight
and Wheat Healthy actually ended up in the held-out val split, to
understand whether their AP@0.5=0.0 is a small-sample artifact or a
genuine detection failure.

Run with:
    python src/check_val_class_counts.py
"""
import json
from collections import Counter
from train_faster_rcnn_fast import group_aware_split, SEED

ANN_FILE = 'detection_data/annotations/train_coco.json'

with open(ANN_FILE) as f:
    coco = json.load(f)

train_ids, val_ids = group_aware_split(ANN_FILE, train_frac=0.8, seed=SEED)

cats = {c['id']: c['name'] for c in coco['categories']}

val_ann_counts = Counter()
train_ann_counts = Counter()
for ann in coco['annotations']:
    if ann['image_id'] in val_ids:
        val_ann_counts[ann['category_id']] += 1
    elif ann['image_id'] in train_ids:
        train_ann_counts[ann['category_id']] += 1

print(f"\n{'Class':<25} {'Train anns':<12} {'Val anns':<10}")
print("-" * 50)
for cid, name in cats.items():
    print(f"{name:<25} {train_ann_counts.get(cid, 0):<12} {val_ann_counts.get(cid, 0):<10}")