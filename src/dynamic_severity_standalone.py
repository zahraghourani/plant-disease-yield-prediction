"""
dynamic_severity_standalone.py

Compute dynamic severity from existing Faster R-CNN annotations (Pascal VOC XML).
No model loading needed - works with your existing Grounding DINO annotations.

Usage:
    python src/dynamic_severity_standalone.py ^
        --detections_dir ./detection_data/auto_annotations^
        --images_dir ./data/Crop___Disease ^
        --output ./results/computed_severity.csv ^
        --aggregate
"""

import os
import sys
import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import cv2

# ── Constants ──
LITERATURE_SEVERITY = {
    "Rice___Leaf_Blast": 0.50,
    "Potato___Late_Blight": 0.45,
    "Corn___Leaf_Blight": 0.40,
    "Wheat___Yellow_Rust": 0.40,
    "Corn___Common_Rust": 0.35,
    "Wheat___Brown_Rust": 0.35,
    "Rice___Hispa": 0.30,
    "Rice___Brown_Spot": 0.25,
    "Potato___Early_Blight": 0.20,
    "Corn___Healthy": 0.00,
    "Potato___Healthy": 0.00,
    "Rice___Healthy": 0.00,
    "Wheat___Healthy": 0.00,
    "Invalid": 0.00,
}


def parse_pascal_voc(xml_path):
    """Parse Pascal VOC XML and return list of boxes: [(x1,y1,x2,y2,class_name), ...]"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        x1 = int(bbox.find("xmin").text)
        y1 = int(bbox.find("ymin").text)
        x2 = int(bbox.find("xmax").text)
        y2 = int(bbox.find("ymax").text)
        boxes.append((x1, y1, x2, y2, name))
    
    return boxes


def compute_leaf_area(image_path, method="contour"):
    """Estimate leaf area using green pixel segmentation."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Green range for leaf pixels
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    leaf_pixels = np.count_nonzero(mask)
    total_pixels = mask.size
    
    return {
        "leaf_pixels": int(leaf_pixels),
        "total_pixels": int(total_pixels),
        "leaf_ratio": round(leaf_pixels / total_pixels, 4) if total_pixels > 0 else 0
    }


def compute_dynamic_severity(boxes, leaf_info, cnn_confidence=1.0):
    """
    Compute severity = (lesion_area / leaf_area) × CNN_confidence
    
    boxes: list of (x1, y1, x2, y2, class_name)
    leaf_info: dict from compute_leaf_area()
    cnn_confidence: float [0,1]
    """
    if not boxes or leaf_info is None:
        return 0.0, {
            "n_lesions": 0,
            "lesion_area": 0,
            "leaf_area": 0,
            "area_ratio": 0,
            "cnn_confidence": cnn_confidence,
            "severity_score": 0.0
        }
    # Sum lesion areas (exclude healthy classes)
    lesion_area = 0
    n_lesions = 0
    for x1, y1, x2, y2, cls in boxes:
        if "Healthy" in cls or "Invalid" in cls:
            continue
        area = (x2 - x1) * (y2 - y1)
        lesion_area += area
        n_lesions += 1
    
    leaf_area = leaf_info["leaf_pixels"]
    
    if leaf_area == 0:
        return 0.0, {
            "n_lesions": n_lesions,
            "lesion_area": int(lesion_area),
            "leaf_area": 0,
            "area_ratio": 0,
            "cnn_confidence": cnn_confidence,
            "severity_score": 0.0
        }
    # Compute ratio
    area_ratio = lesion_area / leaf_area
    area_ratio = min(area_ratio, 1.0)  # Clamp
    
    # Apply CNN confidence
    severity = area_ratio * cnn_confidence
    severity = min(severity, 1.0)
    
    details = {
        "n_lesions": n_lesions,
        "lesion_area": int(lesion_area),
        "leaf_area": int(leaf_area),
        "area_ratio": round(area_ratio, 4),
        "cnn_confidence": round(cnn_confidence, 4),
        "severity_score": round(severity, 4)
    }
    
    return severity, details


def process_all(detections_dir, images_dir, output_csv, max_samples=None):
    """Process all images and compute severity."""
    
    # Find all annotation XML files
    xml_files = list(Path(detections_dir).glob("*.xml"))
    if max_samples:
        xml_files = xml_files[:max_samples]
    
    print(f"Found {len(xml_files)} annotation files")
    
    results = []
    
    for i, xml_path in enumerate(xml_files, 1):
        if i % 100 == 0:
            print(f"  [{i}/{len(xml_files)}] {xml_path.name}")
        
        # Parse boxes
        boxes = parse_pascal_voc(str(xml_path))
        
        # Find corresponding image
        img_name = xml_path.stem
        img_path = None
        
        # Search in all subdirectories of images_dir
        for subdir in Path(images_dir).iterdir():
            if not subdir.is_dir():
                continue
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                candidate = subdir / (img_name + ext)
                if candidate.exists():
                    img_path = str(candidate)
                    break
            if img_path:
                break
        
        if not img_path:
            # Try direct in images_dir
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                candidate = Path(images_dir) / (img_name + ext)
                if candidate.exists():
                    img_path = str(candidate)
                    break
        
        if not img_path:
            print(f"  Warning: Image not found for {xml_path.name}")
            continue
        
        # Determine true class from subdirectory name
        true_class = "Unknown"
        for part in Path(img_path).parts:
            if "___" in part:
                true_class = part
                break
        
        # Compute leaf area
        leaf_info = compute_leaf_area(img_path, method="contour")
        if leaf_info is None:
            continue
        
        # Default CNN confidence (since we don't have CNN results yet)
        cnn_conf = 1.0
        
        # Compute dynamic severity
        severity, details = compute_dynamic_severity(boxes, leaf_info, cnn_conf)
        
        # Literature severity
        lit_severity = LITERATURE_SEVERITY.get(true_class, 0.0)
        
        results.append({
            "filename": xml_path.name.replace(".xml", ""),
            "true_class": true_class,
            "n_detections": len(boxes),
            "n_lesions": details["n_lesions"],
            "lesion_area": details["lesion_area"],
            "leaf_area": details["leaf_area"],
            "leaf_ratio": leaf_info["leaf_ratio"],
            "area_ratio": details["area_ratio"],
            "cnn_confidence": details["cnn_confidence"],
            "computed_severity": severity,
            "literature_severity": lit_severity,
            "difference": round(severity - lit_severity, 4)
        })
    
    # Save results
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    print(f"\nSaved {len(results)} results to {output_csv}")
    return results


def aggregate_results(results_csv):
    """Aggregate by class and produce comparison table."""
    import pandas as pd
    
    df = pd.read_csv(results_csv)
    
    # Group by class
    agg = df.groupby("true_class").agg({
        "computed_severity": ["mean", "std", "min", "max", "count"],
        "literature_severity": "first",
        "area_ratio": "mean",
        "n_lesions": "mean"
    }).reset_index()
    
    # Flatten columns
    agg.columns = ["class", "computed_mean", "computed_std", "computed_min",
                   "computed_max", "n_samples", "literature", "avg_area_ratio",
                   "avg_n_lesions"]
    
    # Round
    for col in agg.columns:
        if col != "class" and col != "n_samples":
            agg[col] = agg[col].round(4)
    
    agg["difference"] = (agg["computed_mean"] - agg["literature"]).round(4)
    
    # Save
    out_path = results_csv.replace(".csv", "_by_class.csv")
    agg.to_csv(out_path, index=False)
    
    # Print LaTeX table
    print("\n" + "="*60)
    print("LATEX TABLE FOR PAPER")
    print("="*60)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Dynamic severity scores computed from Faster R-CNN lesion area, compared with literature-based fixed severity values.}")
    print(r"\label{tab:dynamic_severity}")
    print(r"\small")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(r"\textbf{Disease Class} & \textbf{Literature} & \textbf{Computed} & \textbf{Std} & \textbf{Min} & \textbf{Max} & \textbf{N} \\")
    print(r"\midrule")
    
    for _, row in agg.iterrows():
        if row["literature"] > 0:  # Only diseased classes
            cls = row["class"].replace("___", " ").replace("_", " ")
            print(f"{cls:25s} & {row['literature']:.2f} & {row['computed_mean']:.3f} & "
                  f"{row['computed_std']:.3f} & {row['computed_min']:.3f} & "
                  f"{row['computed_max']:.3f} & {int(row['n_samples'])} \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("="*60)
    
    # Print summary stats
    print("\nSUMMARY STATISTICS")
    print("-" * 40)
    diseased = agg[agg["literature"] > 0]
    print(f"Diseased classes: {len(diseased)}")
    print(f"Mean computed severity: {diseased['computed_mean'].mean():.3f}")
    print(f"Mean literature severity: {diseased['literature'].mean():.3f}")
    print(f"Mean absolute difference: {diseased['difference'].abs().mean():.3f}")
    if len(diseased) > 2:
        print(f"Correlation: {diseased['computed_mean'].corr(diseased['literature']):.3f}")
    
    return agg


def main():
    parser = argparse.ArgumentParser(description="Compute dynamic severity from existing annotations")
    parser.add_argument("--detections_dir", required=True, help="Directory with Pascal VOC XML annotations")
    parser.add_argument("--images_dir", required=True, help="Directory with original images")
    parser.add_argument("--output", default="./results/computed_severity.csv", help="Output CSV")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit for testing")
    parser.add_argument("--aggregate", action="store_true", help="Produce aggregated table")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DYNAMIC SEVERITY COMPUTATION (Standalone)")
    print("=" * 60)
    
    results = process_all(args.detections_dir, args.images_dir, 
                         args.output, args.max_samples)
    
    if args.aggregate:
        aggregate_results(args.output)
    
    print("\nDone!")


if __name__ == "__main__":
    main()