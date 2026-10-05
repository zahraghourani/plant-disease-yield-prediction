"""
regenerate_annotation_figures.py

Regenerates BOTH broken annotation figures from scratch:
  - figures/annotation_collage.png        (single before/after example;
                                           old version had its panel label
                                           clipped at the image edge)
  - figures/annotation_collage_multi.png  (multi-class grid; old version had
                                           "???" glyphs baked into the labels)

Output filenames match the originals, so these are drop-in replacements --
no LaTeX changes needed. All labels are plain ASCII, which avoids the
missing-glyph "???" problem.

Run from the project root:
    python src/regenerate_annotation_figures.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "GroundingDINO"))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from groundingdino.util.inference import load_model, load_image, predict

WEIGHTS_PATH = ROOT / "weights" / "groundingdino_swint_ogc.pth"
CONFIG_PATH = ROOT / "weights" / "GroundingDINO_SwinT_OGC.py"
DATA_DIR = ROOT / "data" / "Crop___DIsease"
OUT_DIR = ROOT / "figures"

BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# (class folder, prompt used by auto_annotate.py, display name)
EXAMPLES = [
    ("Rice___Leaf_Blast",   "leaf blast on rice leaf",       "Rice Leaf Blast"),
    ("Wheat___Yellow_Rust", "yellow rust on wheat leaf",     "Wheat Yellow Rust"),
    ("Corn___Common_Rust",  "rust spots on corn leaf",       "Corn Common Rust"),
]


def first_image(class_folder):
    folder = DATA_DIR / class_folder
    files = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.JPG")))
    return files[0] if files else None


def annotate(model, image_path, prompt):
    """Return (original RGB image, annotated RGB image, number of boxes)."""
    _, image = load_image(str(image_path))
    boxes, _, _ = predict(
        model=model, image=image, caption=prompt,
        box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
    )
    img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    out = img.copy()
    thickness = max(2, int(min(h, w) / 120))
    for cx, cy, bw, bh in boxes:
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), thickness)
    return img, out, len(boxes)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading Grounding DINO...")
    model = load_model(str(CONFIG_PATH), str(WEIGHTS_PATH))

    results = []
    for class_folder, prompt, display in EXAMPLES:
        path = first_image(class_folder)
        if path is None:
            print(f"  No images found for {class_folder}, skipping")
            continue
        print(f"  {display}: {path.name}")
        orig, ann, n = annotate(model, path, prompt)
        results.append((display, orig, ann, n))

    if not results:
        print("No examples could be generated.")
        return

    # ---- Single before/after example (first class) -------------------------
    display, orig, ann, n = results[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(orig)
    axes[0].set_title("(a) Original image", fontsize=13)
    axes[1].imshow(ann)
    axes[1].set_title("(b) Grounding DINO annotation", fontsize=13)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "annotation_collage.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_DIR / 'annotation_collage.png'}")

    # ---- Multi-class grid ---------------------------------------------------
    rows = len(results)
    fig, axes = plt.subplots(rows, 2, figsize=(9, 4.2 * rows))
    if rows == 1:
        axes = [axes]
    for (display, orig, ann, n), (ax_o, ax_a) in zip(results, axes):
        ax_o.imshow(orig)
        ax_o.set_title(f"{display}: original", fontsize=12)
        ax_a.imshow(ann)
        ax_a.set_title(f"{display}: Grounding DINO ({n} box{'es' if n != 1 else ''})",
                       fontsize=12)
        ax_o.axis("off")
        ax_a.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "annotation_collage_multi.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_DIR / 'annotation_collage_multi.png'}")


if __name__ == "__main__":
    main()