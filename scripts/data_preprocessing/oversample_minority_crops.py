#!/usr/bin/env python3
"""
Classifier Minority Class Oversampling script.

Applies heavy augmentation to minority classes strictly in the 'train' split 
to balance the class distribution for image classification models.

Assumes a directory structure where each class has its own folder containing crops:
dataset_classifier/
└── train/
    ├── limestone/
    │   ├── A001_crop_001.jpg
    ├── sandstone/
    ...
"""

import os
import random
from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def get_heavy_augmentation():
    """
    Returns a torchvision transform pipeline for heavy augmentation.
    """
    return transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), 
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.GaussianBlur(kernel_size=3)
    ])

def balance_train_split(train_dir: Path, target_count: int = None):
    """
    Balances the classes in the train directory by copying and augmenting
    existing images until each minority class reaches the target_count.
    """
    if not train_dir.exists() or not train_dir.is_dir():
        print(f"❌ Error: '{train_dir}' is not a valid directory.")
        return

    # Find class directories
    class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        print("⚠️ No class directories found in the provided path.")
        return

    print("Analyzing current distribution...")
    class_counts = {}
    for c in class_dirs:
        # Count only standard image files
        images = list(c.glob("*.jpg")) + list(c.glob("*.png")) + list(c.glob("*.jpeg"))
        class_counts[c.name] = len(images)

    for cls_name, count in class_counts.items():
        print(f"  - {cls_name}: {count} images")

    # Determine target count based on the majority class if not specified
    if target_count is None:
        target_count = max(class_counts.values()) if class_counts else 0
        print(f"\nAuto-selected target count to match the majority class: {target_count}")
    else:
        print(f"\nUsing manual target count: {target_count}")

    # Initialize augmentation transforms
    aug_transform = get_heavy_augmentation()

    # Process each minority class
    for cls_dir in class_dirs:
        count = class_counts[cls_dir.name]
        
        if count == 0:
            print(f"⚠️ Skipping {cls_dir.name} because it has 0 images to augment from.")
            continue
            
        if count >= target_count:
            print(f"✅ {cls_dir.name} is already balanced or exceeds target ({count} >= {target_count}).")
            continue

        needed = target_count - count
        print(f"🚀 Oversampling {cls_dir.name}: applying augmentation to generate {needed} new images...")
        
        original_images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg"))
        
        for i in tqdm(range(needed), desc=f"Augmenting {cls_dir.name}"):
            src_img_path = random.choice(original_images)
            
            try:
                img = Image.open(src_img_path).convert("RGB")
                # Apply transformations
                aug_img = aug_transform(img)
                
                # Save the new augmented crop
                # Naming: original_name_aug_0001.jpg
                save_name = f"{src_img_path.stem}_aug_{i:04d}{src_img_path.suffix}"
                save_path = cls_dir / save_name
                
                aug_img.save(save_path, quality=95)
            except Exception as e:
                print(f"Failed to process {src_img_path.name}: {e}")

    print("\n🎉 Oversampling and heavy augmentation complete!")

def main():
    print("==================================================")
    print(" Classifier Minority Class Oversampling Tool      ")
    print("==================================================")
    print("This script will duplicate and strongly augment image crops to balance the dataset.")
    print("WARNING: Ensure you point this ONLY to the 'train' fold to prevent data leakage!\n")
    
    src = input("Enter path to the 'train' folder (e.g., datasets/classifier_crops/train): ").strip()
    
    if not src:
        print("Path cannot be empty.")
        return
        
    train_dir = Path(src).resolve()
    
    target_input = input("Enter target number of images per class (press Enter to auto-balance to max): ").strip()
    target_count = int(target_input) if target_input.isdigit() else None
    
    balance_train_split(train_dir, target_count)

if __name__ == "__main__":
    main()
