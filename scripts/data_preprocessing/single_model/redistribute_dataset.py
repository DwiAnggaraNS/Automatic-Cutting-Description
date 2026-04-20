import os
import json
import shutil
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any

def create_coco_structure(data_template: Dict[str, Any], images: List[Dict[str, Any]], annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Factory to create a new COCO JSON structure based on a template.
    """
    return {
        "licenses": data_template.get("licenses", []),
        "info": data_template.get("info", {}),
        "categories": data_template.get("categories", []),
        "images": images,
        "annotations": annotations
    }

def iterative_stratification(
    images: List[Dict[str, Any]], 
    annotations: List[Dict[str, Any]], 
    categories: List[Dict[str, Any]], 
    ratios: Tuple[float, float, float]
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[int, List[Dict[str, Any]]]]:
    """
    Splits images into train/val/test using a Greedy Multi-Label Stratification approach.
    Uses a normalized deficit ratio to distribute images fairly regardless of the target split size.
    """
    train_ratio, val_ratio, test_ratio = ratios
    
    # Map image ID to its annotations and distinct category IDs
    image_to_annotations = defaultdict(list)
    image_to_categories = defaultdict(set)
    category_counts = defaultdict(int)
    
    for ann in annotations:
        image_to_annotations[ann["image_id"]].append(ann)
        image_to_categories[ann["image_id"]].add(ann["category_id"])
        category_counts[ann["category_id"]] += 1
        
    # Calculate target instance counts for each split per category
    targets = {
        "train": {cat: count * train_ratio for cat, count in category_counts.items()},
        "val":   {cat: count * val_ratio for cat, count in category_counts.items()},
        "test":  {cat: count * test_ratio for cat, count in category_counts.items()}
    }
    
    current_counts = {
        "train": defaultdict(int),
        "val":   defaultdict(int),
        "test":  defaultdict(int)
    }
    
    # Sort images by rarity. Images with the rarest combination of categories are assigned first
    def compute_image_rarity_score(img: Dict[str, Any]) -> float:
        return sum(1.0 / category_counts[cat] for cat in image_to_categories[img["id"]])
        
    sorted_images = sorted(images, key=compute_image_rarity_score, reverse=True)
    
    splits = {"train": [], "val": [], "test": []}
    available_splits = ["train", "val", "test"]
    
    for img in sorted_images:
        cats_in_image = image_to_categories[img["id"]]
        
        best_split = "train"
        max_score = -float('inf')
        
        # Evaluate assignment cost for each split
        # We want to assign the image to the split that has the highest proportional deficit
        for split in available_splits:
            split_score = 0.0
            
            for cat in cats_in_image:
                instances_in_img = sum(1 for a in image_to_annotations[img["id"]] if a["category_id"] == cat)
                target = targets[split][cat]
                
                if target > 0:
                    # Normalized Deficit: How much of the target is STILL missing relative to the target size?
                    # This prevents the larger partition (e.g., 'train') from greedy absorbing all minority classes natively.
                    future_count = current_counts[split][cat] + instances_in_img
                    deficit_ratio = (target - future_count) / target
                    split_score += deficit_ratio
                else:
                    # Penalize assignment if the algorithm assigns a category with target 0
                    split_score -= (current_counts[split][cat] + instances_in_img)
            
            if split_score > max_score:
                max_score = split_score
                best_split = split
                
        # Assign image to the chosen split
        splits[best_split].append(img)
        
        # Update the tracked current counts
        for cat in cats_in_image:
            instances_in_img = sum(1 for a in image_to_annotations[img["id"]] if a["category_id"] == cat)
            current_counts[best_split][cat] += instances_in_img

    return splits, image_to_annotations

def redistribute_dataset(input_dir: str, output_dir: str, ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> bool:
    """
    Reads a unified COCO dataset, applies normalized iterative stratification, 
    and outputs split COCO datasets structure via dedicated folders.
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
    
    print("\n⏳ Applying Iterative Multi-Label Stratification (Normalized Ratio)...")
    splits, image_to_annotations = iterative_stratification(images, annotations, categories, ratios)
    
    for split_name in ["train", "val", "test"]:
        split_images = splits[split_name]
        
        if not split_images:
            print(f"⚠️ Warning: Split '{split_name}' has 0 images. Adjust your ratios.")
            continue
            
        # Collect annotations for the images in this split
        split_annotations = []
        for img in split_images:
            split_annotations.extend(image_to_annotations[img["id"]])
            
        # Prepare output directories
        split_dir = os.path.join(output_dir, split_name)
        out_img_dir = os.path.join(split_dir, "images")
        out_ann_dir = os.path.join(split_dir, "annotations")
        
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_ann_dir, exist_ok=True)
        
        # Save the new subset JSON
        split_data = create_coco_structure(data, split_images, split_annotations)
        with open(os.path.join(out_ann_dir, "instances_default.json"), 'w') as f:
            json.dump(split_data, f, indent=4)
            
        # Copy images over to the required split directory
        for img in split_images:
            src = os.path.join(img_dir, img["file_name"])
            dst = os.path.join(out_img_dir, img["file_name"])
            if os.path.exists(src):
                shutil.copy(src, dst)
                
        print(f"✅ Created '{split_name}': {len(split_images):>5} images, {len(split_annotations):>6} annotations.")
        
    print(f"\n🏁 Finished dataset redistribution! Stored sequentially at:\n   ➡️ {output_dir}")
    return True

def main():
    print("====================================================")
    print("   COCO Dataset Multi-Label Stratified Splitter     ")
    print("====================================================")
    print("This script uses Normalized Iterative Stratification")
    print("to fairly split datasets while preserving class distribution.")
    
    input_path = input("\nEnter the DIRECTORY path of the UNIFIED COCO dataset: ").strip()
    output_path = input("Enter the OUTPUT parent path for the split dataset: ").strip()
    
    try:
        train_input = input("Enter Train ratio (default 0.8): ").strip()
        val_input = input("Enter Validation ratio (default 0.1): ").strip()
        test_input = input("Enter Test ratio (default 0.1): ").strip()

        train_ratio = float(train_input) if train_input else 0.8
        val_ratio = float(val_input) if val_input else 0.1
        test_ratio = float(test_input) if test_input else 0.1
    except ValueError:
        print("❌ Error: Invalid ratio formatting. Using default rules (0.8, 0.1, 0.1).")
        train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
    
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
        print("❌ Error: Total cumulative ratios must exactly sum up to 1.0 (e.g., 0.8 + 0.1 + 0.1).")
        return
        
    redistribute_dataset(input_path, output_path, (train_ratio, val_ratio, test_ratio))

if __name__ == "__main__":
    main()
