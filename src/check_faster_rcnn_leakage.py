"""
check_faster_rcnn_leakage.py

Checks whether the 2,256 Grounding-DINO-annotated detection images
(train_coco.json) contain augmented-duplicate siblings (e.g.
"RS_Rust 2743.JPG" / "RS_Rust 2743_flipLR.JPG") — the same issue found
and fixed in the CNN classification dataset. train_faster_rcnn_fast.py
currently splits these 2,256 images with a pure random permutation
(no grouping, no fixed seed), so if duplicate siblings exist within
this pool, a random split will very likely scatter them across
train/val, leaking near-identical images across the split.

Run with:
    python src/check_faster_rcnn_leakage.py
"""
import json
import re
from collections import defaultdict

COCO_JSON = './detection_data/annotations/train_coco.json'


def get_base_id(file_name):
    """Same duplicate-grouping logic as data_preprocessing.py / 
    check_duplicate_leakage.py, applied to the COCO 'file_name' field
    (which includes the class subfolder, e.g. 'Corn___Common_Rust/RS_Rust 2743.JPG')."""
    parts = file_name.replace('\\', '/').split('/')
    class_name = parts[0] if len(parts) > 1 else 'unknown'
    fname = parts[-1]

    fname_noext = re.sub(r'\.(jpg|jpeg|png)$', '', fname, flags=re.IGNORECASE)

    # Same "image (N)" special case as the CNN-side fix
    if re.match(r'^image\s*\(\d+\)$', fname_noext, flags=re.IGNORECASE):
        return f"{class_name}::{fname_noext.strip()}"

    stripped = fname_noext
    stripped = re.sub(r'\(\d+\)$', '', stripped)
    stripped = re.sub(r'_?(flipLR|flipTB|\d{2,3}deg)', '', stripped, flags=re.IGNORECASE)
    stripped = re.sub(r'_new\w*', '', stripped, flags=re.IGNORECASE)
    stripped = re.sub(r'\s*copy\s*\d*$', '', stripped, flags=re.IGNORECASE)
    return f"{class_name}::{stripped.strip()}"


def main():
    with open(COCO_JSON) as f:
        coco = json.load(f)

    images = coco['images']
    print(f"Total annotated images in train_coco.json: {len(images)}")

    groups = defaultdict(list)
    for img in images:
        base_id = get_base_id(img['file_name'])
        groups[base_id].append(img['file_name'])

    print(f"Unique base photos: {len(groups)}")
    print(f"Average copies per base photo: {len(images) / len(groups):.2f}\n")

    duplicate_groups = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"Base photos with 2+ copies WITHIN this 2,256-image annotated pool: {len(duplicate_groups)}")
    n_images_in_dup_groups = sum(len(v) for v in duplicate_groups.values())
    print(f"Images affected (sitting in a duplicate group): {n_images_in_dup_groups} "
          f"out of {len(images)} ({100 * n_images_in_dup_groups / len(images):.1f}%)")

    print(f"\nCURRENT SPLIT LOGIC in train_faster_rcnn_fast.py uses a random "
          f"permutation with NO grouping and NO fixed seed. Any base photo "
          f"with 2+ copies in this pool has a high chance of being split "
          f"across train/val on every run.")

    if duplicate_groups:
        print("\nExample duplicate groups found in the annotated pool:")
        for base_id, files in list(duplicate_groups.items())[:5]:
            print(f"\n  base_id: {base_id}")
            for f in files:
                print(f"    {f}")
    else:
        print("\nNo duplicate groups found — the annotated pool appears to be "
              "made of genuinely distinct source photos. Random splitting is safe.")


if __name__ == "__main__":
    main()