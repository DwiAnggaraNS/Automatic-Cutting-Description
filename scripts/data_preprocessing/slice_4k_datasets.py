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
        
        # SAHI dumps the output in slightly different folder structure inside output_dir 
        # (it usually creates 'instances_default_sliced/'). Let's organize it to match standard CVAT COCO.
        sahi_output_folder = os.path.join(output_dir, "instances_default_sliced")
        
        if os.path.exists(sahi_output_folder):
            # Move json
            sahi_json = os.path.join(sahi_output_folder, "instances_default_sliced.json")
            if os.path.exists(sahi_json):
                os.rename(sahi_json, os.path.join(out_annotations_dir, "instances_default.json"))
            
            # Move images
            for img_file in os.listdir(sahi_output_folder):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    os.rename(os.path.join(sahi_output_folder, img_file), os.path.join(out_images_dir, img_file))
                    
            # Cleanup SAHI temp folder
            os.rmdir(sahi_output_folder)

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
