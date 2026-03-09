#!/usr/bin/env python3
"""
Script to collect statistics for each class in the YOLO dataset.
Analyzes train, test, and val splits.
"""

import os
from pathlib import Path
from collections import defaultdict
import yaml

def load_class_names(data_yaml_path):
    """Load class names from data.yaml"""
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

def count_class_instances(labels_dir, num_classes):
    """Count instances of each class in label files"""
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
                            if 0 <= class_id < num_classes:
                                class_counts[class_id] += 1
        except Exception as e:
            print(f"Error reading {label_file}: {e}")
    
    return class_counts, file_count

def create_simplified_mapping():
    """
    Create mapping from detailed classes to simplified categories.
    Combines similar classes (e.g., Silt + Loose Silt, Limestone + Loose Limestone)
    """
    return {
        # Silt categories
        0: 'Siltstone',
        1: 'Sandstone',
        2: 'Sandstone',
        3: 'Limestone',
        4: 'Mix Sand and Silt',
        5: 'Siltstone',
        6: 'Limestone',
        7: 'Coal',
    }

def main():
    base_dir = Path(__file__).parent
    data_yaml_path = base_dir / 'data.yaml'
    
    # Load class names and number of classes
    class_names = load_class_names(data_yaml_path)
    num_classes = len(class_names)
    
    # Create simplified mapping
    simplified_mapping = create_simplified_mapping()
    
    print("=" * 80)
    print("DATASET STATISTICS - CLASS DISTRIBUTION")
    print("=" * 80)
    print(f"\nTotal Classes: {num_classes}\n")
    
    # prefix = "backup/"
    prefix = ""
    splits = [f'{prefix}train', f'{prefix}test', f'{prefix}val']
    all_splits_stats = {}
    total_instances_all = defaultdict(int)
    total_files_all = 0
    
    # Analyze each split
    for split in splits:
        labels_dir = base_dir / split / 'labels'
        class_counts, file_count = count_class_instances(labels_dir, num_classes)
        all_splits_stats[split] = (class_counts, file_count)
        total_files_all += file_count
        
        # Print split statistics
        print(f"\n{split.upper()} SPLIT")
        print("-" * 80)
        print(f"Number of labeled files: {file_count}")
        
        total_instances = sum(class_counts.values())
        print(f"Total instances: {total_instances}\n")
        
        # Print per-class statistics
        print(f"{'Class ID':<10} {'Class Name':<40} {'Count':<10} {'Percentage':<10}")
        print("-" * 80)
        
        for class_id in range(num_classes):
            count = class_counts[class_id]
            percentage = (count / total_instances * 100) if total_instances > 0 else 0
            class_name = class_names[class_id] if class_id < len(class_names) else f"Unknown_{class_id}"
            print(f"{class_id:<10} {class_name:<40} {count:<10} {percentage:>6.2f}%")
            total_instances_all[class_id] += count
    
    # Print overall statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS (ALL SPLITS COMBINED)")
    print("=" * 80)
    print(f"Total labeled files: {total_files_all}\n")
    
    total_all_instances = sum(total_instances_all.values())
    print(f"Total instances: {total_all_instances}\n")
    
    print(f"{'Class ID':<10} {'Class Name':<40} {'Count':<10} {'Percentage':<10}")
    print("-" * 80)
    
    for class_id in range(num_classes):
        count = total_instances_all[class_id]
        percentage = (count / total_all_instances * 100) if total_all_instances > 0 else 0
        class_name = class_names[class_id] if class_id < len(class_names) else f"Unknown_{class_id}"
        print(f"{class_id:<10} {class_name:<40} {count:<10} {percentage:>6.2f}%")
    
    # Print simplified statistics
    print("\n" + "=" * 80)
    print("SIMPLIFIED STATISTICS (GROUPED CLASSES)")
    print("=" * 80)
    print("Grouping similar classes for simplified analysis:\n")
    
    simplified_counts = defaultdict(int)
    for class_id, count in total_instances_all.items():
        simplified_category = simplified_mapping[class_id]
        simplified_counts[simplified_category] += count
    
    total_simplified_instances = sum(simplified_counts.values())
    print(f"Total instances: {total_simplified_instances}\n")
    
    print(f"{'Simplified Category':<40} {'Count':<10} {'Percentage':<10}")
    print("-" * 80)
    
    # Sort by count descending for better readability
    for category in sorted(simplified_counts.keys(), key=lambda x: simplified_counts[x], reverse=True):
        count = simplified_counts[category]
        percentage = (count / total_simplified_instances * 100) if total_simplified_instances > 0 else 0
        print(f"{category:<40} {count:<10} {percentage:>6.2f}%")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
