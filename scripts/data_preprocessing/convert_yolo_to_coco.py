import os
import json
import shutil
from PIL import Image

# The legacy mapping found in the old dataset
LEGACY_CLASSES = {
    0: "Silt",
    1: "Loose Sand",
    2: "Sandstone",
    3: "Limestone",
    4: "Loose Sandy and Silt",
    5: "Loose Silt",
    6: "Loose Limestone",
    7: "Coal"
}

def calculate_bbox_and_area(polygon):
    """
    Calculates bounding box [x_min, y_min, width, height] and area 
    using the Shoelace formula for a given polygon [x1, y1, x2, y2, ...].
    """
    xs = polygon[0::2]
    ys = polygon[1::2]
    
    # Bounding box
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
    
    # Shoelace formula for area
    area = 0.5 * abs(sum(xs[i] * ys[i+1] - xs[i+1] * ys[i] for i in range(-1, len(xs)-1)))
    
    return bbox, area

def convert_yolo_seg_to_coco(input_folder, output_folder):
    """
    Converts a YOLO segmentation dataset split (images & labels folders) 
    back into a CVAT-like COCO instance_default.json format.
    """
    images_dir = os.path.join(input_folder, "images")
    labels_dir = os.path.join(input_folder, "labels")
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"❌ Error: Expected 'images' and 'labels' directories inside {input_folder}")
        return False
        
    os.makedirs(output_folder, exist_ok=True)
    out_images_dir = os.path.join(output_folder, "images")
    out_annotations_dir = os.path.join(output_folder, "annotations")
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_annotations_dir, exist_ok=True)
    
    # Initialize COCO structure
    coco_data = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {"contributor": "", "date_created": "", "description": "Converted from Legacy YOLO", "url": "", "version": "", "year": ""},
        "categories": [{"id": k, "name": v} for k, v in LEGACY_CLASSES.items()],
        "images": [],
        "annotations": []
    }
    
    image_id_counter = 1
    annotation_id_counter = 1
    
    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    # Iterate through images
    for filename in os.listdir(images_dir):
        if not filename.lower().endswith(valid_extensions):
            continue
            
        img_path = os.path.join(images_dir, filename)
        base_name = os.path.splitext(filename)[0]
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        
        # Read image to get dimensions (required for COCO absolute coordinates)
        try:
            with Image.open(img_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"⚠️ Warning: Could not read image {filename}: {e}. Skipping.")
            continue
            
        # Copy image to output folder
        shutil.copy(img_path, os.path.join(out_images_dir, filename))
        
        # Add image entry to COCO
        coco_data["images"].append({
            "id": image_id_counter,
            "width": img_width,
            "height": img_height,
            "file_name": filename,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        })
        
        # Process corresponding YOLO label file if it exists
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if not parts or len(parts) < 5:
                    continue # Empty or invalid line (requires class + at least 3 points = 7 values)
                    
                class_id = int(parts[0])
                
                # Reverse normalize coordinates: multiply by width and height
                normalized_coords = [float(x) for x in parts[1:]]
                absolute_coords = []
                for i in range(len(normalized_coords)):
                    if i % 2 == 0: # X coordinate
                        absolute_coords.append(normalized_coords[i] * img_width)
                    else: # Y coordinate
                        absolute_coords.append(normalized_coords[i] * img_height)
                
                bbox, area = calculate_bbox_and_area(absolute_coords)
                
                coco_data["annotations"].append({
                    "id": annotation_id_counter,
                    "image_id": image_id_counter,
                    "category_id": class_id,
                    "segmentation": [absolute_coords],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                    "attributes": {"occluded": False}
                })
                
                annotation_id_counter += 1
                
        image_id_counter += 1
        
    # Save the aggregated annotations to JSON
    out_json_path = os.path.join(out_annotations_dir, "instances_default.json")
    with open(out_json_path, 'w') as f:
        json.dump(coco_data, f, indent=4)
        
    print(f"✅ Successfully converted YOLO dataset from {input_folder} to COCO format at {output_folder}")
    print(f"   Processed {image_id_counter - 1} images and {annotation_id_counter - 1} annotations.\n")
    return True

def main():
    print("===============================================")
    print("   Legacy YOLO to COCO Conversion Tool         ")
    print("===============================================")
    print("This tool converts old train/val/test YOLO format splits")
    print("back into CVAT-like COCO format for standard remapping.")
    
    try:
        num_folders = int(input("Enter the number of legacy yolov8 splits (e.g. train, val, test) to convert: "))
    except ValueError:
        print("❌ Invalid input for number of folders. Please enter an integer.")
        return

    if num_folders < 1:
        print("❌ Number of folders must be at least 1.")
        return

    for i in range(num_folders):
        input_path = input(f"\nEnter the absolute DIRECTORY path of legacy split {i + 1} (e.g. /dataset/train): ").strip()
        output_path = input(f"Enter the absolute OUTPUT path for converted COCO split {i + 1} (e.g. /dataset/converted_coco_train): ").strip()
        
        print(f"Processing '{input_path}'...")
        convert_yolo_seg_to_coco(input_path, output_path)

    print("🏁 Conversion process finished. You can now use remap_coco_categories.py on the new output folders.")

if __name__ == "__main__":
    main()
