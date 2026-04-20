#!/usr/bin/env python3
"""
Single-Class YOLO Dataset Converter
Transforms a multi-class YOLO segmentation or detection dataset into a 
single-class dataset (where all objects are labeled as 'rock' at index 0).

This is designed for Stage 1 of the Dual Model pipeline (Segmentor).
"""

import os
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm

def process_labels(src_labels_dir: Path, dst_labels_dir: Path):
    """
    Reads all text files in the source label directory, changes their 
    class index to 0, and saves them to the destination directory.
    """
    dst_labels_dir.mkdir(parents=True, exist_ok=True)
    
    label_files = list(src_labels_dir.glob("*.txt"))
    if not label_files:
        return

    for label_file in tqdm(label_files, desc=f"Converting labels in {src_labels_dir.parent.name}", leave=False):
        dst_file = dst_labels_dir / label_file.name
        
        with open(label_file, "r") as f_in, open(dst_file, "w") as f_out:
            for line in f_in:
                parts = line.strip().split()
                if not parts:
                    continue
                # Force class ID to 0, keep the rest of the coordinates identical
                parts[0] = "0"
                f_out.write(" ".join(parts) + "\n")

def convert_dataset(src_dir: str, dst_dir: str):
    src_path = Path(src_dir).resolve()
    dst_path = Path(dst_dir).resolve()

    if not src_path.exists():
        print(f"❌ Error: Source directory '{src_path}' does not exist.")
        return

    if dst_path.exists():
        print(f"⚠️ Warning: Destination directory '{dst_path}' already exists.")
        proceed = input("Do you want to overwrite/merge with it? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Aborted.")
            return

    # Create destination base folder
    dst_path.mkdir(parents=True, exist_ok=True)

    # 1. Update data.yaml
    src_yaml = src_path / "data.yaml"
    dst_yaml = dst_path / "data.yaml"
    
    if src_yaml.exists():
        with open(src_yaml, "r") as f:
            data = yaml.safe_load(f)
            
        print(f"Original dataset had {data.get('nc', 'unknown')} classes.")
        
        # Override classes setup
        data['nc'] = 1
        data['names'] = ['rock']
        
        # Ensure path variables in yaml point relative or to train/val directly
        data.pop('path', None) # Let it use relative paths easily
        
        with open(dst_yaml, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        print("✅ Created new data.yaml with 1 class ('rock').")
    else:
        print(f"⚠️ No data.yaml found in {src_path}. Creating a stub data.yaml...")
        with open(dst_yaml, "w") as f:
            yaml.dump({
                "nc": 1,
                "names": ["rock"],
                "train": "train/images",
                "val": "val/images",
                "test": "test/images"
            }, f, default_flow_style=False, sort_keys=False)

    # 2. Process Splits
    splits = ["train", "val", "test"]
    for split in splits:
        src_split = src_path / split
        if not src_split.exists():
            continue
            
        print(f"\nProcessing '{split}' split...")
        dst_split = dst_path / split
        dst_split.mkdir(parents=True, exist_ok=True)

        # Copy Images
        src_images = src_split / "images"
        dst_images = dst_split / "images"
        if src_images.exists():
            if not dst_images.exists():
                print(f"Copying images for {split}...")
                shutil.copytree(src_images, dst_images)
            else:
                print(f"Images folder for {split} already exists. Skipping copy.")
        
        # Convert Labels
        src_labels = src_split / "labels"
        dst_labels = dst_split / "labels"
        if src_labels.exists():
            process_labels(src_labels, dst_labels)

    print(f"\n🎉 Successfully created single-class dataset at '{dst_path}'")

def main():
    print("==================================================")
    print(" Single-Class YOLO Dataset Converter (Rock Finder)")
    print("==================================================")
    print("This will copy images and convert all labels to Class 0 ('rock').")
    
    src = input("Enter source YOLO dataset path: ").strip()
    dst = input("Enter destination (new) YOLO dataset path: ").strip()
    
    if src and dst:
        convert_dataset(src, dst)
    else:
        print("Paths cannot be empty.")

if __name__ == "__main__":
    main()
