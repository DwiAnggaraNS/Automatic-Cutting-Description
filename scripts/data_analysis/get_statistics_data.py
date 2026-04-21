#!/usr/bin/env python3
"""
Script to collect statistics for each class in the dataset.
Supports both YOLO and COCO instance segmentation formats.
Analyzes train, test, and val splits.
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import yaml

def load_class_names_yolo(data_yaml_path):
    """Load class names from YOLO data.yaml"""
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    names = data.get('names', {})
    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}
    return names

def load_class_names_coco(json_path):
    """Load class names from COCO annotations json"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return {cat['id']: cat['name'] for cat in data.get('categories', [])}

def count_class_instances_yolo(labels_dir):
    """Count instances of each class in YOLO label files"""
    class_counts = defaultdict(int)
    file_count = 0
    
    if not os.path.exists(labels_dir):
        return class_counts, 0
    
    for label_file in Path(labels_dir).glob('*.txt'):
        file_count += 1
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1
        except Exception as e:
            print(f"Error reading {label_file}: {e}")
    
    return class_counts, file_count

def count_class_instances_coco(json_path):
    """Count instances of each class in COCO annotation file"""
    class_counts = defaultdict(int)
    
    if not os.path.exists(json_path):
        return class_counts, 0
        
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        for ann in data.get('annotations', []):
            class_counts[ann['category_id']] += 1
            
        file_count = len(data.get('images', []))
        return class_counts, file_count
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return class_counts, 0

def create_simplified_mapping():
    """
    Create mapping from detailed classes to simplified categories.
    """
    return {
        0: 'Silt',
        1: 'Sandstone',
        2: 'Limestone',
        3: 'Coal',
        4: 'Shalestone',
        5: 'Quartz',
        6: 'Cement'
    }

def main():
    print("========================================")
    print("   Dataset Statistics Analyzer          ")
    print("========================================")
    
    dataset_path_input = input("Enter the absolute path to your dataset folder (e.g., /path/to/dataset): ").strip()
    base_dir = Path(dataset_path_input)
    
    if not base_dir.exists() or not base_dir.is_dir():
        print(f"[Error] The path '{base_dir}' does not exist or is not a directory.")
        return

    dataset_format = input("Enter the dataset format (YOLO or COCO): ").strip().upper()
    if dataset_format not in ["YOLO", "COCO"]:
        print("[Error] Invalid format. Supported formats are YOLO and COCO.")
        return

    class_names = {}
    splits = ['train', 'test', 'val']
    all_splits_stats = {}
    total_instances_all = defaultdict(int)
    total_files_all = 0
    
    if dataset_format == "YOLO":
        data_yaml_path = base_dir / 'data.yaml'
        if not data_yaml_path.exists():
            print(f"[Error] data.yaml not found in {base_dir}. Ensure it is a valid YOLO dataset.")
            return
            
        class_names = load_class_names_yolo(data_yaml_path)
        
        for split in splits:
            labels_dir = base_dir / split / 'labels'
            if labels_dir.exists():
                class_counts, file_count = count_class_instances_yolo(labels_dir)
                all_splits_stats[split] = (class_counts, file_count)
                
    elif dataset_format == "COCO":
        for split in splits:
            json_path = base_dir / split / 'annotations' / 'instances_default.json'
            if not json_path.exists():
                json_path = base_dir / split / '_annotations.coco.json'
                
            if json_path.exists():
                if not class_names:
                    class_names = load_class_names_coco(json_path)
                class_counts, file_count = count_class_instances_coco(json_path)
                all_splits_stats[split] = (class_counts, file_count)
                
        if not class_names:
            print(f"[Error] No COCO annotations found in train/test/val subdirectories.")
            return

    simplified_mapping = create_simplified_mapping()
    
    print("=" * 80)
    print("DATASET STATISTICS - CLASS DISTRIBUTION")
    print("=" * 80)
    print(f"\nFormat: {dataset_format}")
    print(f"Total Classes: {len(class_names)}\n")
    
    for split in splits:
        if split not in all_splits_stats:
            continue
            
        class_counts, file_count = all_splits_stats[split]
        total_files_all += file_count
        
        print(f"\n{split.upper()} SPLIT")
        print("-" * 80)
        print(f"Number of labeled images: {file_count}")
        
        total_instances = sum(class_counts.values())
        print(f"Total instances: {total_instances}\n")
        
        print(f"{'Class ID':<10} {'Class Name':<40} {'Count':<10} {'Percentage':<10}")
        print("-" * 80)
        
        for class_id, class_name in class_names.items():
            count = class_counts.get(class_id, 0)
            percentage = (count / total_instances * 100) if total_instances > 0 else 0
            print(f"{class_id:<10} {class_name:<40} {count:<10} {percentage:>6.2f}%")
            total_instances_all[class_id] += count
    
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS (ALL SPLITS COMBINED)")
    print("=" * 80)
    print(f"Total labeled images: {total_files_all}\n")
    
    total_all_instances = sum(total_instances_all.values())
    print(f"Total instances: {total_all_instances}\n")
    
    print(f"{'Class ID':<10} {'Class Name':<40} {'Count':<10} {'Percentage':<10}")
    print("-" * 80)
    
    for class_id, class_name in class_names.items():
        count = total_instances_all.get(class_id, 0)
        percentage = (count / total_all_instances * 100) if total_all_instances > 0 else 0
        print(f"{class_id:<10} {class_name:<40} {count:<10} {percentage:>6.2f}%")
    
    print("\n" + "=" * 80)
    print("SIMPLIFIED STATISTICS (GROUPED CLASSES)")
    print("=" * 80)
    print("Grouping similar classes for simplified analysis:\n")
    
    simplified_counts = defaultdict(int)
    for class_id, count in total_instances_all.items():
        simplified_category = simplified_mapping.get(class_id, class_names.get(class_id, f"Unknown_{class_id}"))
        simplified_counts[simplified_category] += count
    
    total_simplified_instances = sum(simplified_counts.values())
    print(f"Total instances: {total_simplified_instances}\n")
    
    print(f"{'Simplified Category':<40} {'Count':<10} {'Percentage':<10}")
    print("-" * 80)
    
    for category in sorted(simplified_counts.keys(), key=lambda x: simplified_counts[x], reverse=True):
        count = simplified_counts[category]
        percentage = (count / total_simplified_instances * 100) if total_simplified_instances > 0 else 0
        print(f"{category:<40} {count:<10} {percentage:>6.2f}%")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
