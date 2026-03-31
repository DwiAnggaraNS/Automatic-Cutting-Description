import os
import shutil
import json
import argparse

def merge_coco_datasets(input_folders, output_folder):
    """
    Merges multiple CVAT COCO dataset folders into a single output folder.
    It combines images and updates image/annotation IDs in the JSON files to prevent collisions.
    """
    os.makedirs(output_folder, exist_ok=True)
    images_out_dir = os.path.join(output_folder, 'images')
    os.makedirs(images_out_dir, exist_ok=True)
    
    merged_json = {
        "images": [],
        "annotations": [],
        "categories": [],
        "info": {},
        "licenses": []
    }
    
    image_id_offset = 0
    annotation_id_offset = 0
    categories_set = False
    
    for folder in input_folders:
        print(f"Processing folder: {folder}")
        
        # Check if paths exist
        images_in_dir = os.path.join(folder, 'images', 'default')
        json_path = os.path.join(folder, 'annotations', 'instances_default.json')
        
        if not os.path.exists(images_in_dir) or not os.path.exists(json_path):
            print(f"Warning: Expected COCO structure not found in {folder}. Skipping.")
            continue
            
        # 1. Read JSON
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Add basic info and categories from the first valid dataset
        if not categories_set:
            merged_json["categories"] = data.get("categories", [])
            merged_json["info"] = data.get("info", {})
            merged_json["licenses"] = data.get("licenses", [])
            categories_set = True
            
        max_image_id = 0
        max_annotation_id = 0
        
        # Mapping old IDs to new IDs to avoid conflicts
        image_id_mapping = {}
        
        # 2. Process Images
        for img in data.get("images", []):
            old_id = img["id"]
            new_id = old_id + image_id_offset
            image_id_mapping[old_id] = new_id
            
            img["id"] = new_id
            merged_json["images"].append(img)
            max_image_id = max(max_image_id, new_id)
            
            # Copy image file
            src_img = os.path.join(images_in_dir, img["file_name"])
            dst_img = os.path.join(images_out_dir, img["file_name"])
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
            else:
                print(f"Image {img['file_name']} not found in {images_in_dir}.")
                
        # 3. Process Annotations
        for ann in data.get("annotations", []):
            old_ann_id = ann["id"]
            new_ann_id = old_ann_id + annotation_id_offset
            max_annotation_id = max(max_annotation_id, new_ann_id)
            
            ann["id"] = new_ann_id
            ann["image_id"] = image_id_mapping.get(ann["image_id"], ann["image_id"])
            merged_json["annotations"].append(ann)
            
        image_id_offset = max_image_id + 1
        annotation_id_offset = max_annotation_id + 1
        print(f"Successfully merged {folder}.")

    # Save merged JSON
    out_annotations_dir = os.path.join(output_folder, 'annotations')
    os.makedirs(out_annotations_dir, exist_ok=True)
    out_json_path = os.path.join(out_annotations_dir, 'instances_default.json')
    
    with open(out_json_path, 'w') as f:
        json.dump(merged_json, f, indent=4)
        
    print(f"All datasets merged successfully into: {output_folder}")

def main():
    try:
        num_folders = int(input("Enter the number of dataset folders to merge: "))
    except ValueError:
        print("Invalid input for number of folders. Please enter an integer.")
        return

    if num_folders < 1:
        print("Number of folders must be at least 1.")
        return

    input_folders = []
    for i in range(num_folders):
        path = input(f"Enter path for dataset folder {i + 1}: ").strip()
        if not os.path.isdir(path):
            print(f"Error: The path '{path}' does not exist or is not a directory.")
            return
        input_folders.append(path)

    output_folder = input("Enter path for the output merged dataset folder: ").strip()

    print("\n--- Summary ---")
    print(f"Input folders to merge: {len(input_folders)}")
    for folder in input_folders:
        print(f"- {folder}")
    print(f"Output folder: {output_folder}")
    print("-----------------\n")

    merge_coco_datasets(input_folders, output_folder)

if __name__ == "__main__":
    main()
