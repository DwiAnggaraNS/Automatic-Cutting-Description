"""
Redistribute COCO Dataset Script
================================
This script redistributes images and annotations across train, val, and test splits
based on specified criteria. It also allows forcing specific images to the train set.

Features:
- Merge all existing annotations from train/val/test folders
- Redistribute based on specified split ratios
- Force specific images to train dataset by name
- Move images and update COCO annotations accordingly
"""

import os
import json
import shutil
import random
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict


class COCODatasetRedistributor:
    def __init__(self, base_dir: str):
        """
        Initialize the redistributor with the base dataset directory.
        
        Args:
            base_dir: Path to the dataset root directory containing train/val/test folders
        """
        self.base_dir = Path(base_dir)
        self.train_dir = self.base_dir / "train"
        self.val_dir = self.base_dir / "val"
        self.test_dir = self.base_dir / "test"
        
        # Merged data containers
        self.all_images: Dict[int, dict] = {}
        self.all_annotations: List[dict] = []
        self.categories: List[dict] = []
        self.licenses: List[dict] = []
        self.info: dict = {}
        
        # Mapping from filename to image info
        self.filename_to_image: Dict[str, dict] = {}
        # Mapping from image_id to annotations
        self.image_id_to_annotations: Dict[int, List[dict]] = defaultdict(list)
        
    def load_coco_json(self, json_path: Path) -> Optional[dict]:
        """Load a COCO JSON annotation file."""
        if not json_path.exists():
            print(f"Warning: {json_path} does not exist")
            return None
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not data:
                    return None
                return data
        except json.JSONDecodeError:
            print(f"Warning: {json_path} is empty or invalid JSON")
            return None
    
    def merge_all_annotations(self, use_simplified: bool = True):
        """
        Merge annotations from all splits (train, val, test).
        
        Args:
            use_simplified: If True, use simplified annotations if available
        """
        suffix = "_simplified" if use_simplified else ""
        
        annotation_files = [
            (self.train_dir / f"train_annotations{suffix}.json", self.train_dir / "images"),
            (self.val_dir / f"val_annotations{suffix}.json", self.val_dir / "images"),
            (self.test_dir / f"test_annotations{suffix}.json", self.test_dir / "images"),
        ]
        
        # Fallback to non-simplified if simplified doesn't exist
        if use_simplified:
            fallback_files = [
                (self.train_dir / "train_annotations.json", self.train_dir / "images"),
                (self.val_dir / "val_annotations.json", self.val_dir / "images"),
                (self.test_dir / "test_annotations.json", self.test_dir / "images"),
            ]
            annotation_files = [
                (ann if ann[0].exists() else fallback) 
                for ann, fallback in zip(annotation_files, fallback_files)
            ]
        
        new_image_id = 1
        new_annotation_id = 1
        old_to_new_image_id: Dict[int, int] = {}
        
        for ann_path, img_dir in annotation_files:
            data = self.load_coco_json(ann_path)
            if data is None:
                # Try to load images directly from the folder
                if img_dir.exists():
                    for img_file in img_dir.iterdir():
                        if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
                            if img_file.name not in self.filename_to_image:
                                # Create a minimal image entry
                                img_entry = {
                                    "id": new_image_id,
                                    "file_name": img_file.name,
                                    "width": 0,
                                    "height": 0,
                                    "source_dir": str(img_dir)
                                }
                                self.all_images[new_image_id] = img_entry
                                self.filename_to_image[img_file.name] = img_entry
                                new_image_id += 1
                continue
            
            # Store categories, licenses, info from first valid file
            if not self.categories and "categories" in data:
                self.categories = data.get("categories", [])
            if not self.licenses and "licenses" in data:
                self.licenses = data.get("licenses", [])
            if not self.info and "info" in data:
                self.info = data.get("info", {})
            
            # Process images
            for img in data.get("images", []):
                old_id = img["id"]
                filename = img["file_name"]
                
                if filename not in self.filename_to_image:
                    # Assign new ID
                    new_id = new_image_id
                    new_image_id += 1
                    
                    new_img = img.copy()
                    new_img["id"] = new_id
                    new_img["source_dir"] = str(img_dir)
                    
                    self.all_images[new_id] = new_img
                    self.filename_to_image[filename] = new_img
                    old_to_new_image_id[old_id] = new_id
                else:
                    # Image already exists, map old ID to existing new ID
                    old_to_new_image_id[old_id] = self.filename_to_image[filename]["id"]
            
            # Process annotations
            for ann in data.get("annotations", []):
                old_img_id = ann["image_id"]
                if old_img_id in old_to_new_image_id:
                    new_ann = ann.copy()
                    new_ann["id"] = new_annotation_id
                    new_ann["image_id"] = old_to_new_image_id[old_img_id]
                    new_annotation_id += 1
                    
                    self.all_annotations.append(new_ann)
                    self.image_id_to_annotations[new_ann["image_id"]].append(new_ann)
        
        print(f"Merged {len(self.all_images)} images and {len(self.all_annotations)} annotations")
        print(f"Categories: {[c['name'] for c in self.categories]}")
    
    def get_category_statistics(self) -> Dict[int, Dict]:
        """
        Get statistics for each category.
        
        Returns:
            Dictionary mapping category_id to stats (name, instance_count, image_count, image_filenames)
        """
        category_stats: Dict[int, Dict] = {}
        
        # Initialize stats for each category
        for cat in self.categories:
            category_stats[cat["id"]] = {
                "name": cat["name"],
                "instance_count": 0,
                "image_ids": set(),
                "image_filenames": set()
            }
        
        # Count instances and images per category
        for ann in self.all_annotations:
            cat_id = ann["category_id"]
            img_id = ann["image_id"]
            
            if cat_id in category_stats:
                category_stats[cat_id]["instance_count"] += 1
                category_stats[cat_id]["image_ids"].add(img_id)
        
        # Map image IDs to filenames
        for cat_id, stats in category_stats.items():
            for img_id in stats["image_ids"]:
                if img_id in self.all_images:
                    stats["image_filenames"].add(self.all_images[img_id]["file_name"])
        
        return category_stats
    
    def get_images_with_rare_categories(self, min_instances_threshold: int = 20) -> List[str]:
        """
        Get filenames of images containing categories with fewer than threshold instances.
        
        Args:
            min_instances_threshold: Categories with fewer instances than this will have
                                    all their images forced to train
        
        Returns:
            List of image filenames that should be forced to train
        """
        category_stats = self.get_category_statistics()
        
        rare_category_images: Set[str] = set()
        
        print(f"\nCategory instance counts (threshold: {min_instances_threshold}):")
        for cat_id, stats in category_stats.items():
            is_rare = stats["instance_count"] < min_instances_threshold
            status = "-> FORCE TO TRAIN" if is_rare else ""
            print(f"  {stats['name']}: {stats['instance_count']} instances in {len(stats['image_filenames'])} images {status}")
            
            if is_rare:
                rare_category_images.update(stats["image_filenames"])
        
        print(f"\nTotal images with rare categories: {len(rare_category_images)}")
        
        return list(rare_category_images)
    
    def get_split_category_distribution(self, filenames: List[str]) -> Dict[int, Dict]:
        """
        Get category distribution for a specific split (list of filenames).
        
        Args:
            filenames: List of image filenames in the split
        
        Returns:
            Dictionary mapping category_id to stats (name, instance_count, image_count)
        """
        # Get image IDs for this split
        image_ids = set()
        for filename in filenames:
            if filename in self.filename_to_image:
                image_ids.add(self.filename_to_image[filename]["id"])
        
        # Initialize stats for each category
        category_dist: Dict[int, Dict] = {}
        for cat in self.categories:
            category_dist[cat["id"]] = {
                "name": cat["name"],
                "instance_count": 0,
                "image_count": 0,
                "image_ids": set()
            }
        
        # Count instances per category for this split
        for ann in self.all_annotations:
            if ann["image_id"] in image_ids:
                cat_id = ann["category_id"]
                if cat_id in category_dist:
                    category_dist[cat_id]["instance_count"] += 1
                    category_dist[cat_id]["image_ids"].add(ann["image_id"])
        
        # Count images per category
        for cat_id in category_dist:
            category_dist[cat_id]["image_count"] = len(category_dist[cat_id]["image_ids"])
            del category_dist[cat_id]["image_ids"]  # Clean up
        
        return category_dist
    
    def print_distribution_table(
        self,
        train_files: List[str],
        val_files: List[str],
        test_files: List[str],
        min_train_instances: int = 0
    ):
        """
        Print a table showing category distribution across all splits.
        
        Args:
            train_files: Filenames in train split
            val_files: Filenames in val split
            test_files: Filenames in test split
            min_train_instances: Highlight if train has fewer instances than this
        """
        train_dist = self.get_split_category_distribution(train_files)
        val_dist = self.get_split_category_distribution(val_files)
        test_dist = self.get_split_category_distribution(test_files)
        
        print("\n" + "=" * 100)
        print("CATEGORY DISTRIBUTION ACROSS SPLITS")
        print("=" * 100)
        print(f"{'Category':<30} | {'TRAIN':^20} | {'VAL':^20} | {'TEST':^20} | {'TOTAL':^10}")
        print(f"{'':<30} | {'Inst':>8} {'Img':>8}   | {'Inst':>8} {'Img':>8}   | {'Inst':>8} {'Img':>8}   | {'Inst':>10}")
        print("-" * 100)
        
        warnings = []
        
        for cat in self.categories:
            cat_id = cat["id"]
            cat_name = cat["name"][:28]  # Truncate long names
            
            train_inst = train_dist[cat_id]["instance_count"]
            train_img = train_dist[cat_id]["image_count"]
            val_inst = val_dist[cat_id]["instance_count"]
            val_img = val_dist[cat_id]["image_count"]
            test_inst = test_dist[cat_id]["instance_count"]
            test_img = test_dist[cat_id]["image_count"]
            total_inst = train_inst + val_inst + test_inst
            
            # Check if train has minimum instances
            warning = ""
            if min_train_instances > 0 and train_inst < min_train_instances:
                warning = " ⚠️ LOW"
                warnings.append(f"{cat['name']}: only {train_inst} instances in TRAIN (min: {min_train_instances})")
            
            print(f"{cat_name:<30} | {train_inst:>8} {train_img:>8}   | {val_inst:>8} {val_img:>8}   | {test_inst:>8} {test_img:>8}   | {total_inst:>10}{warning}")
        
        print("-" * 100)
        
        # Print totals
        total_train_inst = sum(d["instance_count"] for d in train_dist.values())
        total_train_img = len(train_files)
        total_val_inst = sum(d["instance_count"] for d in val_dist.values())
        total_val_img = len(val_files)
        total_test_inst = sum(d["instance_count"] for d in test_dist.values())
        total_test_img = len(test_files)
        grand_total = total_train_inst + total_val_inst + total_test_inst
        
        print(f"{'TOTAL':<30} | {total_train_inst:>8} {total_train_img:>8}   | {total_val_inst:>8} {total_val_img:>8}   | {total_test_inst:>8} {total_test_img:>8}   | {grand_total:>10}")
        print("=" * 100)
        
        # Print warnings
        if warnings:
            print("\n⚠️  WARNINGS - Low instance count in TRAIN:")
            for w in warnings:
                print(f"   - {w}")
        else:
            print("\n✅ All categories have sufficient instances in TRAIN.")
    
    def redistribute(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        force_train_images: List[str] = None,
        random_seed: int = 42,
        stratify_by_category: bool = False,
        min_instances_threshold: int = 20
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Redistribute images into train, val, test splits.
        
        Args:
            train_ratio: Fraction of images for training
            val_ratio: Fraction of images for validation
            test_ratio: Fraction of images for testing
            force_train_images: List of image filenames that must go to train
            random_seed: Random seed for reproducibility
            stratify_by_category: If True, try to maintain category distribution
            min_instances_threshold: Categories with fewer instances than this
                                    will have all their images forced to train
            
        Returns:
            Tuple of (train_filenames, val_filenames, test_filenames)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, \
            "Ratios must sum to 1.0"
        
        random.seed(random_seed)
        
        force_train_images = force_train_images or []
        force_train_set = set(force_train_images)
        
        # Add images with rare categories to forced train set
        if min_instances_threshold > 0:
            rare_category_images = self.get_images_with_rare_categories(min_instances_threshold)
            force_train_set.update(rare_category_images)
        
        # Separate forced train images from the rest
        all_filenames = list(self.filename_to_image.keys())
        forced_train = [f for f in all_filenames if f in force_train_set]
        remaining = [f for f in all_filenames if f not in force_train_set]
        
        # Shuffle remaining images
        random.shuffle(remaining)
        
        # Calculate split sizes (accounting for forced train images)
        total = len(all_filenames)
        remaining_count = len(remaining)
        
        # Adjust ratios for remaining images
        target_train = int(total * train_ratio) - len(forced_train)
        target_val = int(total * val_ratio)
        target_test = total - target_train - target_val - len(forced_train)
        
        # Ensure we don't have negative numbers
        target_train = max(0, target_train)
        
        # Split remaining images
        train_from_remaining = remaining[:target_train]
        val_split = remaining[target_train:target_train + target_val]
        test_split = remaining[target_train + target_val:]
        
        # Combine forced train with randomly selected train
        train_split = forced_train + train_from_remaining
        
        print(f"\nRedistribution summary:")
        print(f"  Total images: {total}")
        print(f"  Forced to train (explicit + rare categories): {len(forced_train)}")
        print(f"  Train: {len(train_split)} ({len(train_split)/total*100:.1f}%)")
        print(f"  Val: {len(val_split)} ({len(val_split)/total*100:.1f}%)")
        print(f"  Test: {len(test_split)} ({len(test_split)/total*100:.1f}%)")
        
        # Show category distribution across splits
        self.print_distribution_table(
            train_split, val_split, test_split,
            min_train_instances=min_instances_threshold
        )
        
        return train_split, val_split, test_split
    
    def create_coco_json(self, filenames: List[str]) -> dict:
        """Create a COCO format JSON for a subset of images."""
        images = []
        annotations = []
        image_ids = set()
        
        for filename in filenames:
            if filename in self.filename_to_image:
                img = self.filename_to_image[filename].copy()
                # Remove source_dir from output
                img.pop("source_dir", None)
                images.append(img)
                image_ids.add(img["id"])
        
        # Get annotations for these images
        ann_id = 1
        for img in images:
            for ann in self.image_id_to_annotations.get(img["id"], []):
                new_ann = ann.copy()
                new_ann["id"] = ann_id
                annotations.append(new_ann)
                ann_id += 1
        
        return {
            "licenses": self.licenses,
            "info": self.info,
            "categories": self.categories,
            "images": images,
            "annotations": annotations
        }
    
    def save_split(
        self,
        filenames: List[str],
        output_dir: Path,
        split_name: str,
        copy_images: bool = True
    ):
        """
        Save a split to disk (images and annotations).
        
        Args:
            filenames: List of image filenames for this split
            output_dir: Directory to save to
            split_name: Name of the split (train, val, test)
            copy_images: If True, copy image files
        """
        output_dir = Path(output_dir)
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Save COCO JSON
        coco_data = self.create_coco_json(filenames)
        
        ann_path = output_dir / f"{split_name}_annotations.json"
        with open(ann_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2)
        print(f"Saved {len(coco_data['annotations'])} annotations to {ann_path}")
        
        # Also save simplified version (same content in this case)
        ann_simplified_path = output_dir / f"{split_name}_annotations_simplified.json"
        with open(ann_simplified_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2)
        
        # Copy images
        if copy_images:
            copied = 0
            for filename in filenames:
                if filename in self.filename_to_image:
                    source_dir = self.filename_to_image[filename].get("source_dir")
                    if source_dir:
                        src_path = Path(source_dir) / filename
                        dst_path = images_dir / filename
                        
                        if src_path.exists() and src_path != dst_path:
                            shutil.copy2(src_path, dst_path)
                            copied += 1
                        elif not src_path.exists():
                            print(f"Warning: Source image not found: {src_path}")
            print(f"Copied {copied} images to {images_dir}")
    
    def run_redistribution(
        self,
        output_base_dir: str = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        force_train_images: List[str] = None,
        random_seed: int = 42,
        use_simplified: bool = True,
        backup_existing: bool = True,
        min_instances_threshold: int = 20,
        stratify_by_category: bool = False
    ):
        """
        Run the full redistribution pipeline.
        
        Args:
            output_base_dir: Output directory (default: same as base_dir)
            train_ratio: Fraction for training
            val_ratio: Fraction for validation  
            test_ratio: Fraction for testing
            force_train_images: Images that must be in train set
            random_seed: Random seed for reproducibility
            use_simplified: Use simplified annotations if available
            backup_existing: Create backup of existing data
            min_instances_threshold: Categories with fewer instances than this
                                    will have all their images forced to train
        """
        output_base = Path(output_base_dir) if output_base_dir else self.base_dir
        
        # Step 1: Merge all existing annotations
        print("=" * 60)
        print("Step 1: Merging all annotations")
        print("=" * 60)
        self.merge_all_annotations(use_simplified=use_simplified)
        
        # Step 2: Redistribute
        print("\n" + "=" * 60)
        print("Step 2: Redistributing images")
        print("=" * 60)
        train_files, val_files, test_files = self.redistribute(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            force_train_images=force_train_images,
            random_seed=random_seed,
            min_instances_threshold=min_instances_threshold,
            stratify_by_category=stratify_by_category
        )
        
        # Step 3: Copy all images to temp directory first (to avoid losing source images)
        print("\n" + "=" * 60)
        print("Step 3: Collecting all source images")
        print("=" * 60)
        
        temp_images_dir = self.base_dir / "_temp_images"
        temp_images_dir.mkdir(exist_ok=True)
        
        # Copy all source images to temp directory
        collected = 0
        for filename, img_info in self.filename_to_image.items():
            source_dir = img_info.get("source_dir")
            if source_dir:
                src_path = Path(source_dir) / filename
                dst_path = temp_images_dir / filename
                
                if src_path.exists() and not dst_path.exists():
                    shutil.copy2(src_path, dst_path)
                    collected += 1
                elif dst_path.exists():
                    collected += 1  # Already collected
        
        print(f"Collected {collected} images to temp directory")
        
        # Update source_dir to point to temp directory
        for filename in self.filename_to_image:
            self.filename_to_image[filename]["source_dir"] = str(temp_images_dir)
        
        # Step 4: Backup existing if needed
        if backup_existing and output_base == self.base_dir:
            print("\n" + "=" * 60)
            print("Step 4: Creating backup")
            print("=" * 60)
            backup_dir = self.base_dir / "backup"
            backup_dir.mkdir(exist_ok=True)
            
            for folder in ["train", "val", "test"]:
                src = self.base_dir / folder
                dst = backup_dir / folder
                if src.exists() and not dst.exists():
                    shutil.copytree(src, dst)
                    print(f"Backed up {folder} to {dst}")
        
        # Step 5: Clear output directories
        print("\n" + "=" * 60)
        print("Step 5: Preparing output directories")
        print("=" * 60)
        
        output_train = output_base / "train"
        output_val = output_base / "val"
        output_test = output_base / "test"
        
        for d in [output_train, output_val, output_test]:
            if d.exists():
                # Clear images folder
                img_dir = d / "images"
                if img_dir.exists():
                    for f in img_dir.iterdir():
                        f.unlink()
            d.mkdir(parents=True, exist_ok=True)
            (d / "images").mkdir(exist_ok=True)
        
        # Step 6: Save splits
        print("\n" + "=" * 60)
        print("Step 6: Saving splits")
        print("=" * 60)
        
        self.save_split(train_files, output_train, "train")
        self.save_split(val_files, output_val, "val")
        self.save_split(test_files, output_test, "test")
        
        # Step 7: Cleanup temp directory
        print("\n" + "=" * 60)
        print("Step 7: Cleaning up")
        print("=" * 60)
        
        if temp_images_dir.exists():
            shutil.rmtree(temp_images_dir)
            print(f"Removed temp directory: {temp_images_dir}")
        
        print("\n" + "=" * 60)
        print("Redistribution complete!")
        print("=" * 60)
        
        return train_files, val_files, test_files


def main():
    """Main function to run the redistribution."""
    
    # ============================================================
    # CONFIGURATION - Modify these values as needed
    # ============================================================
    
    # Path to the dataset root directory
    BASE_DIR = ""
    
    # Split ratios (must sum to 1.0)
    TRAIN_RATIO = 0.80   # 70% for training
    VAL_RATIO = 0.10    # 15% for validation
    TEST_RATIO = 0.10   # 15% for testing
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    # Images that MUST be in the training set (specify by filename)
    # Add filenames here (with extension, e.g., "image1.png")
    FORCE_TRAIN_IMAGES = [
        # Example:
        "0705_02.png",
        "0705_03.png",
        "0705_04.png",
        "0705_05.png",
        "0705_06.png",
        "0705_07.png",
        "0705_08.png",
        "0705_09.png",
        "0705_10.png",
        "0705_11.png",
        "0705_12.png",
        "0705_13.png",
        "0705_14.png",
        "0705_15.png",
        "0705_16.png",

        "0805_01.png",
        "0805_02.png",
        "0805_03.png",
        "0805_04.png",
        "0805_05.png",
        "0805_06.png",
        "0805_07.png",
        "0805_08.png",
        "0805_09.png",
        "0805_10.png",
        "0805_11.png",
        "0805_12.png",
        "0805_13.png",
        "0805_14.png",
        "0805_15.png",

        "LIMESTONE_0003.png",
        "LIMESTONE_0004.png",
        "LIMESTONE_0005.png",
        "LIMESTONE_0006.png",
        "LIMESTONE_0007.png",
        "LIMESTONE_0008.png",
        "LIMESTONE_0009.png",
    ]
    
    # Whether to use simplified annotations (if available)
    USE_SIMPLIFIED = True
    
    # Whether to create a backup before redistribution
    BACKUP_EXISTING = True
    
    # Minimum instances threshold for rare categories
    # Categories with fewer instances than this will have ALL their images forced to train
    MIN_INSTANCES_THRESHOLD = 20  # Set to 0 to disable this feature
    
    # Output directory (None = same as BASE_DIR, or specify a different path)
    OUTPUT_DIR = None  # Set to a path like "/path/to/output" to save elsewhere
    
    # ============================================================
    # RUN REDISTRIBUTION
    # ============================================================
    
    print("\nCOCO Dataset Redistributor")
    print("=" * 60)
    print(f"Base directory: {BASE_DIR}")
    print(f"Train/Val/Test ratio: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print(f"Forced train images (explicit): {len(FORCE_TRAIN_IMAGES)}")
    print(f"Min instances threshold: {MIN_INSTANCES_THRESHOLD}")
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 60)
    
    # Create redistributor and run
    redistributor = COCODatasetRedistributor(BASE_DIR)
    
    train_files, val_files, test_files = redistributor.run_redistribution(
        output_base_dir=OUTPUT_DIR,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        force_train_images=FORCE_TRAIN_IMAGES,
        random_seed=RANDOM_SEED,
        use_simplified=USE_SIMPLIFIED,
        backup_existing=BACKUP_EXISTING,
        min_instances_threshold=MIN_INSTANCES_THRESHOLD,
        stratify_by_category = True
    )
    
    # Print final file lists
    print("\n" + "=" * 60)
    print("FINAL SPLIT LISTS")
    print("=" * 60)
    
    print(f"\nTRAIN ({len(train_files)} images):")
    for f in sorted(train_files):
        print(f"  - {f}")
    
    print(f"\nVAL ({len(val_files)} images):")
    for f in sorted(val_files):
        print(f"  - {f}")
    
    print(f"\nTEST ({len(test_files)} images):")
    for f in sorted(test_files):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
