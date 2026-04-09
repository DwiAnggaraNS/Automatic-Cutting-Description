#!/usr/bin/env python3
"""
Expert Classifier Crop Extractor
Transforms a YOLO segmentation or detection dataset into an Image Classification dataset
by cropping instances out of the full images based on bounding boxes/polygons.

Crucial Feature: Prevents Data Leakage. 
All crops generated from an image in the 'train' folder stay strictly within the 'train' folder.
Crops strictly inherit the split of the original full image.

Output Structure:
dataset_classifier/
├── train/
│   ├── limestone/
│   │   ├── A001_crop_001.jpg
...
"""

import os
import cv2
import yaml
from pathlib import Path
from tqdm import tqdm

def parse_yolo_line(line, width, height):
    """
    Parses a YOLO format line.
    Return (class_id, x_min, y_min, x_max, y_max)
    """
    parts = list(map(float, line.strip().split()))
    class_id = int(parts[0])

    if len(parts) == 5:
        # Bounding Box format: class_id cx cy w h
        cx, cy, w, h = parts[1:5]
        x_min = max(0.0, (cx - w / 2) * width)
        y_min = max(0.0, (cy - h / 2) * height)
        x_max = min(float(width), (cx + w / 2) * width)
        y_max = min(float(height), (cy + h / 2) * height)
    elif len(parts) > 5:
        # Segmentation Polygon format: class_id x1 y1 x2 y2 ...
        polygon_points = parts[1:]
        x_coords = polygon_points[0::2]
        y_coords = polygon_points[1::2]
        
        x_min = max(0.0, min(x_coords) * width)
        x_max = min(float(width), max(x_coords) * width)
        y_min = max(0.0, min(y_coords) * height)
        y_max = min(float(height), max(y_coords) * height)
    else:
        return None

    return class_id, int(x_min), int(y_min), int(x_max), int(y_max)

def crop_split(split: str, src_path: Path, dst_path: Path, class_names: dict, min_crop_size: int = 10):
    """
    Process a single split (train/val/test).
    Calculates the crops and exports them directly into the respective classification folders.
    """
    images_dir = src_path / split / "images"
    labels_dir = src_path / split / "labels"
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"⚠️ Split '{split}' missing images or labels. Skipping.")
        return

    # Create destination folders per class
    for class_name in class_names.values():
        (dst_path / split / class_name).mkdir(parents=True, exist_ok=True)

    image_files = [f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in ('.jpg', '.png', '.jpeg')]
    
    total_crops = 0

    for img_path in tqdm(image_files, desc=f"Cropping {split} subset"):
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        if not label_path.exists():
            continue
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        height, width = img.shape[:2]
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for idx, line in enumerate(lines):
            parsed = parse_yolo_line(line, width, height)
            if not parsed:
                continue
                
            class_id, x1, y1, x2, y2 = parsed
            
            # Verify coordinates are valid and crop has some area
            if x2 <= x1 or y2 <= y1:
                continue
                
            crop_w = x2 - x1
            crop_h = y2 - y1
            if crop_w < min_crop_size or crop_h < min_crop_size:
                continue
                
            # Class dict lookup with fallback safety
            class_name = class_names.get(class_id, f"unknown_class_{class_id}")
            if class_id not in class_names:
                # If unknown class occurs, lazily initialize folder
                (dst_path / split / class_name).mkdir(parents=True, exist_ok=True)

            crop = img[y1:y2, x1:x2]
            
            # Save crop format: img_stem_crop_index.jpg
            crop_filename = f"{img_path.stem}_crop_{idx+1:03d}.jpg"
            crop_save_path = dst_path / split / class_name / crop_filename
            
            cv2.imwrite(str(crop_save_path), crop)
            total_crops += 1

    print(f"✅ Finished '{split}'. Extracted {total_crops} crops.")

def extract_crops(src_dir: str, dst_dir: str, min_size: int = 10):
    src_path = Path(src_dir).resolve()
    dst_path = Path(dst_dir).resolve()

    if not src_path.exists():
        print(f"❌ Error: Source YOLO directory '{src_path}' does not exist.")
        return

    yaml_path = src_path / "data.yaml"
    if not yaml_path.exists():
        print(f"❌ Error: 'data.yaml' not found in '{src_path}'. Cannot determine class mappings.")
        return

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    
    class_list = data.get("names", [])
    if isinstance(class_list, dict):
        class_names = class_list
    else:
        # Array of strings to dict mapping
        class_names = {idx: name for idx, name in enumerate(class_list)}
        
    print(f"Found {len(class_names)} classes.")
    
    print("\nStarting extraction. (Images smaller than min_size threshold are ignored.)")
    for split in ["train", "val", "test"]:
        crop_split(split, src_path, dst_path, class_names, min_size)

    print(f"\n🎉 Extraction Complete! Data safely structured at '{dst_path}'")

def main():
    print("==================================================")
    print(" Expert Classifier Crop Extractor (Anti-Leakage)  ")
    print("==================================================")
    print("This will slice multi-class full-sized images into an ImageFolder")
    print("structure strictly maintaining the original train/val/test splits.")
    
    src = input("Enter source MULTI-CLASS YOLO dataset path: ").strip()
    dst = input("Enter destination (new) ImageClassifier dataset path: ").strip()
    min_size_input = input("Enter minimum crop dimensions (e.g. 10): ").strip()
    
    min_size = 10
    if min_size_input.isdigit():
        min_size = int(min_size_input)
        
    if src and dst:
        extract_crops(src, dst, min_size)
    else:
        print("Paths cannot be empty.")

if __name__ == "__main__":
    main()
