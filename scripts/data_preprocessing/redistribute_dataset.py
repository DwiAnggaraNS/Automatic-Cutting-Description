import os
import json
import shutil
import random
import argparse
from collections import defaultdict

def create_coco_structure(data_template, images, annotations):
    """Factory to create a new COCO JSON structure."""
    return {
        "licenses": data_template.get("licenses", []),
        "info": data_template.get("info", {}),
        "categories": data_template.get("categories", []),
        "images": images,
        "annotations": annotations
    }

def iterative_stratification(images, annotations, categories, ratios):
    """
    Splits images into train/val/test using a Greedy Multi-Label Stratification approach.
    This ensures images with rare classes are distributed fairly.
    """
    train_r, val_r, test_r = ratios
    
    # Map image id to its annotations and distinct categories
    img_to_anns = defaultdict(list)
    img_to_cats = defaultdict(set)
    cat_counts = defaultdict(int)
    
    for ann in annotations:
        img_to_anns[ann["image_id"]].append(ann)
        img_to_cats[ann["image_id"]].add(ann["category_id"])
        cat_counts[ann["category_id"]] += 1
        
    # Calculate target counts for each split per category
    targets = {
        "train": {cat: count * train_r for cat, count in cat_counts.items()},
        "val": {cat: count * val_r for cat, count in cat_counts.items()},
        "test": {cat: count * test_r for cat, count in cat_counts.items()}
    }
    
    current = {
        "train": defaultdict(int),
        "val": defaultdict(int),
        "test": defaultdict(int)
    }
    
    # Sort images: prioritize images with the rarest categories to be assigned first
    # We score an image based on the rarity of its categories (sum of 1/total_count)
    def img_rarity_score(img):
        return sum(1.0 / cat_counts[c] for c in img_to_cats[img["id"]])
        
    sorted_images = sorted(images, key=img_rarity_score, reverse=True)
    
    splits = {"train": [], "val": [], "test": []}
    
    for img in sorted_images:
        cats_in_img = img_to_cats[img["id"]]
        
        # Calculate assignment cost for each split
        # We want to assign to the split that has the highest deficit for these categories
        best_split = "train"
        max_deficit = -float('inf')
        
        for split in ["train", "val", "test"]:
            split_deficit = 0
            for cat in cats_in_img:
                # Difference between target and current proportion
                deficit = targets[split][cat] - current[split][cat]
                split_deficit += deficit
            
            if split_deficit > max_deficit:
                max_deficit = split_deficit
                best_split = split
                
        # Assign image to best split
        splits[best_split].append(img)
        
        # Update current counts
        for cat in cats_in_img:
            # We count occurrences based on instances, though could be boolean based
            instances_in_img = sum(1 for a in img_to_anns[img["id"]] if a["category_id"] == cat)
            current[best_split][cat] += instances_in_img

    return splits, img_to_anns

def redistribute_dataset(input_dir, output_dir, ratios=(0.8, 0.1, 0.1)):
    """
    Reads a unified COCO dataset, applies iterative stratification, 
    and outputs split COCO datasets.
    """
    json_path = os.path.join(input_dir, "annotations", "instances_default.json")
    img_dir = os.path.join(input_dir, "images")
    
    if not os.path.exists(json_path) or not os.path.exists(img_dir):
        print(f"❌ Error: Expected unified COCO dataset in {input_dir}")
        return False
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    
    print("\n⏳ Applying Iterative Multi-Label Stratification...")
    splits, img_to_anns = iterative_stratification(images, annotations, categories, ratios)
    
    for split_name in ["train", "val", "test"]:
        split_images = splits[split_name]
        
        if not split_images:
            print(f"⚠️ Warning: Split '{split_name}' has 0 images. Adjust your ratios.")
            continue
            
        # Collect annotations for this split
        split_annotations = []
        for img in split_images:
            split_annotations.extend(img_to_anns[img["id"]])
            
        # Prepare output directories
        split_dir = os.path.join(output_dir, split_name)
        out_img_dir = os.path.join(split_dir, "images")
        out_ann_dir = os.path.join(split_dir, "annotations")
        
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_ann_dir, exist_ok=True)
        
        # Save JSON
        split_data = create_coco_structure(data, split_images, split_annotations)
        with open(os.path.join(out_ann_dir, f"instances_default.json"), 'w') as f:
            json.dump(split_data, f, indent=4)
            
        # Copy Images
        for img in split_images:
            src = os.path.join(img_dir, img["file_name"])
            dst = os.path.join(out_img_dir, img["file_name"])
            if os.path.exists(src):
                shutil.copy(src, dst)
                
        print(f"✅ Created '{split_name}': {len(split_images)} images, {len(split_annotations)} annotations.")
        
    print(f"\n🏁 Finished redistribution! Stored at: {output_dir}")
    return True

def main():
    print("====================================================")
    print("   COCO Dataset Multi-Label Stratified Splitter    ")
    print("====================================================")
    print("This script uses Iterative Stratification to split")
    print("datasets while preserving minority class distribution.")
    
    input_path = input("\nEnter the DIRECTORY path of the UNIFIED COCO dataset: ").strip()
    output_path = input("Enter the OUTPUT parent path for the split dataset: ").strip()
    
    try:
        train_input = input("Enter Train ratio (default 0.8): ").strip()
        val_input = input("Enter Validation ratio (default 0.1): ").strip()
        test_input = input("Enter Test ratio (default 0.1): ").strip()

        train_r = float(train_input) if train_input else 0.8
        val_r = float(val_input) if val_input else 0.1
        test_r = float(test_input) if test_input else 0.1
    except ValueError:
        print("❌ Error: Invalid ratios. Using defaults (0.8, 0.1, 0.1).")
        train_r, val_r, test_r = 0.8, 0.1, 0.1
    
    if abs((train_r + val_r + test_r) - 1.0) > 1e-5:
        print("❌ Error: Ratios must sum up to 1.0")
        return
        
    redistribute_dataset(input_path, output_path, (train_r, val_r, test_r))

if __name__ == "__main__":
    main()
