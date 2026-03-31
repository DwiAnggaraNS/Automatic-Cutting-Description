import os
import argparse
from sahi.slicing import slice_coco

def slice_large_images_in_dataset(input_dir, output_dir, slice_size=960, overlap_ratio=0.2):
    """
    Slices large images (e.g., 4K) into smaller patches (e.g., 960x960) 
    and automatically adjusts the COCO polygon annotations using SAHI.
    
    Images smaller than the slice_size will remain intact.
    """
    coco_json_path = os.path.join(input_dir, "annotations", "instances_default.json")
    images_dir = os.path.join(input_dir, "images")
    
    if not os.path.exists(coco_json_path) or not os.path.exists(images_dir):
        print(f"❌ Error: Could not find COCO dataset structure in {input_dir}")
        print("Expected 'images/' folder and 'annotations/instances_default.json'")
        return False
        
    os.makedirs(output_dir, exist_ok=True)
    out_images_dir = os.path.join(output_dir, "images")
    out_annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_annotations_dir, exist_ok=True)

    print("\n⏳ Starting SAHI Slicing Process...")
    print(f"Input: {input_dir}")
    print(f"Target Slice Size: {slice_size}x{slice_size}")
    print(f"Overlap Ratio: {overlap_ratio}")
    print("Please wait, this might take a few minutes depending on dataset size...\n")

    try:
        # SAHI will slice images, adjust polygons, and save the new images/json
        coco_dict, out_json_path = slice_coco(
            coco_annotation_file_path=coco_json_path,
            image_dir=images_dir,
            output_coco_annotation_file_name="instances_default",
            output_dir=output_dir, # SAHI creates a folder named after the json name without ext
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            ignore_negative_samples=False, # Keep patches even if they have no annotations (useful for background learning)
            min_area_ratio=0.1 # Drop annotations if less than 10% of them is visible in the slice
        )
        
        # SAHI writes its output to `output_dir` (which we passed as `output_dir`)
        # It creates a json file there named `<output_coco_annotation_file_name>_sliced.json`
        # and saves all sliced images directly inside `output_dir` or an internal folder depending on exactly how it parses its own internal path generation.
        # usually it saves images straight to the `output_dir` in newer versions.
        
        # Let's cleanly organize whatever SAHI dumped directly into `output_dir` into our CVAT structure
        
        # Check both potential names for the JSON output depending on SAHI version behavior
        sahi_json_output_1 = os.path.join(output_dir, "instances_default_sliced.json")
        sahi_json_output_2 = os.path.join(output_dir, "instances_default_coco.json")
        
        target_json_path = os.path.join(out_annotations_dir, "instances_default.json")
        
        if os.path.exists(sahi_json_output_1):
            os.rename(sahi_json_output_1, target_json_path)
        elif os.path.exists(sahi_json_output_2):
            os.rename(sahi_json_output_2, target_json_path)
            
        # Move all images that were dumped in the root of output_dir into output_dir/images
        # Note: SAHI might have also created an 'instances_default_sliced' suffix folder in older logic, 
        # so we check both the root of output_dir and any potential subfolder.
        
        sahi_subfolder = os.path.join(output_dir, "instances_default_sliced")
        image_source_dir = sahi_subfolder if os.path.exists(sahi_subfolder) else output_dir
        
        for item in os.listdir(image_source_dir):
            item_path = os.path.join(image_source_dir, item)
            # Skip directories (like 'images' and 'annotations' that we created)
            if os.path.isdir(item_path):
                continue
            if item.lower().endswith(('.png', '.jpg', '.jpeg')):
                os.rename(item_path, os.path.join(out_images_dir, item))

        # Cleanup potential temp subfolder if it exists and is now empty
        if os.path.exists(sahi_subfolder) and not os.listdir(sahi_subfolder):
            os.rmdir(sahi_subfolder)

        print(f"\n✅ Successfully sliced dataset!")
        print(f"Results saved to: {output_dir}")
        return True
        
    except ImportError:
        print("❌ Error: 'sahi' library is not installed.")
        print("Please install it using: pip install sahi")
        return False
    except Exception as e:
        print(f"❌ Error during slicing: {e}")
        return False

def main():
    print("===============================================")
    print("   4K Dataset Slicer Tool (SAHI Integration)   ")
    print("===============================================")
    print("This tool slices high-resolution images (e.g. 4K)")
    print("into smaller grids to prevent YOLO detail loss.")
    
    input_path = input("\nEnter the absolute DIRECTORY path of the COCO dataset to slice: ").strip()
    
    if not os.path.exists(input_path) or not os.path.isdir(input_path):
        print(f"❌ Error: The path '{input_path}' does not exist or is not a directory.")
        return

    output_path = input("Enter the absolute OUTPUT path for sliced dataset (e.g. /dataset/C_2026_Sliced): ").strip()
    
    # Default parameters based on best practices
    slice_size = 960 # Matches our YOLO training imgsz
    overlap_ratio = 0.2 # 20% overlap to avoid cutting rocks at borders
    
    slice_large_images_in_dataset(input_path, output_path, slice_size, overlap_ratio)

if __name__ == "__main__":
    main()
