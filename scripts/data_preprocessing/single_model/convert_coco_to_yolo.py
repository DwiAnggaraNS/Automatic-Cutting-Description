import json
import os
import shutil
import yaml

def convert_to_yolo(input_images_path, input_json_path, output_images_path, output_labels_path):
    """Converts a specific COCO split into YOLO segmentation format."""
    
    if not os.path.exists(input_json_path) or not os.path.exists(input_images_path):
        print(f"⚠️ Missing images or JSON in {input_images_path}. Skipping.")
        return False
        
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_labels_path, exist_ok=True)

    # Valid image files mapping
    img_id_to_data = {img['id']: img for img in data.get('images', [])}
    
    # Process each valid image in the JSON
    for img_id, img in img_id_to_data.items():
        filename = img['file_name']
        source = os.path.join(input_images_path, filename)
        destination = os.path.join(output_images_path, filename)
        
        if not os.path.exists(source):
            continue
            
        if not os.path.exists(destination):
            shutil.copy(source, destination)
            
        img_w = img['width']
        img_h = img['height']
        
        # Get annotations for this image
        img_ann = [ann for ann in data.get('annotations', []) if ann['image_id'] == img_id]
        
        if img_ann:
            label_filename = f"{os.path.splitext(filename)[0]}.txt"
            with open(os.path.join(output_labels_path, label_filename), "w") as file_object:
                for ann in img_ann:
                    # Note: Our categories are 1-indexed (1: Silt, 2: Sand, etc.)
                    # YOLO requires 0-indexed categories (0: Silt, 1: Sand, etc.)
                    current_category = ann['category_id'] - 1 
                    
                    if not ann.get('segmentation') or len(ann['segmentation']) == 0:
                        continue
                        
                    polygon = ann['segmentation'][0]
                    normalized_polygon = []
                    
                    for i, coord in enumerate(polygon):
                        # Even index: X coordinate (width), Odd index: Y coordinate (height)
                        norm_coord = coord / img_w if i % 2 == 0 else coord / img_h
                        # Clamp between 0 and 1 to avoid YOLO out-of-bounds error
                        norm_coord = max(0.0, min(1.0, norm_coord))
                        normalized_polygon.append(format(norm_coord, '.6f'))
                        
                    file_object.write(f"{current_category} " + " ".join(normalized_polygon) + "\n")
                    
    return True

def create_yaml(input_json_path, output_yaml_path):
    """Creates the data.yaml file required by YOLO."""
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # Sort categories by ID to ensure correct indexing mapping
    categories = sorted(data.get('categories', []), key=lambda x: x['id'])
    names = [cat['name'] for cat in categories]
    
    yaml_data = {
        'names': names,
        'nc': len(names),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images'
    }

    with open(output_yaml_path, 'w') as file:
        yaml.dump(yaml_data, file, default_flow_style=False, sort_keys=False)

def main():
    print("=========================================")
    print("   COCO Splits to YOLO Format Converter ")
    print("=========================================")
    
    input_base = input("\nEnter the DIRECTORY path of the Split COCO Dataset (containing train/val/test folders): ").strip()
    
    if not os.path.isdir(input_base):
        print(f"❌ Error: {input_base} is not a valid directory.")
        return
        
    output_base = input("Enter the OUTPUT path for the YOLO Dataset: ").strip()
    
    splits = ["train", "val", "test"]
    processed_any = False
    
    for split in splits:
        print(f"\nProcessing '{split}' split...")
        split_in_dir = os.path.join(input_base, split)
        
        # Adjusting the path expectation based on redistribute_dataset output
        images_in = os.path.join(split_in_dir, "images")
        json_in = os.path.join(split_in_dir, "annotations", "instances_default.json")
        
        images_out = os.path.join(output_base, split, "images")
        labels_out = os.path.join(output_base, split, "labels")
        
        success = convert_to_yolo(images_in, json_in, images_out, labels_out)
        if success:
            processed_any = True

    if processed_any:
        # Generate YAML using train split as base for classes
        yaml_json_path = os.path.join(input_base, "train", "annotations", "instances_default.json")
        if os.path.exists(yaml_json_path):
            yaml_out_path = os.path.join(output_base, "data.yaml")
            create_yaml(yaml_json_path, yaml_out_path)
            print(f"\n✅ Created YOLO config file: {yaml_out_path}")
            
        print("\n🏁 Conversion to YOLO format complete!")
    else:
        print("\n❌ No valid splits found. Conversion aborted.")

if __name__ == "__main__":
    main()
