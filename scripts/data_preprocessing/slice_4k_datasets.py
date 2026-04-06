import os
import argparse
from sahi.slicing import slice_coco

def slice_large_images_in_dataset(input_dir, output_dir, slice_size=640, overlap_ratio=0.2):
    """
    Slices large images (e.g., 4K) into smaller patches (e.g., 640x640) to avoid OOM errors during YOLO training
    and automatically adjusts the COCO polygon annotations using SAHI.
    
    Images smaller than the slice_size will remain intact.

    Args:
        input_dir    : Path to COCO dataset (must contain images/ and annotations/)
        output_dir   : Path to save sliced dataset
        slice_size   : Tile size in pixels (default: 640, must be multiple of 32)
        overlap_ratio: Overlap between tiles (default: 0.2 = 20%)

    """

    # --- Validate slice_size ---
    if slice_size % 32 != 0:
        print(f"❌ Error: slice_size={slice_size} is not a multiple of 32.")
        print("   YOLO requires image sizes to be multiples of 32 (e.g. 640, 960, 1280).")
        return False

    coco_json_path = os.path.join(input_dir, "annotations", "instances_default.json")
    images_dir = os.path.join(input_dir, "images")
    
    if not os.path.exists(coco_json_path):
        print(f"❌ Error: Could not find annotations at:\n   {coco_json_path}")
        return False
    if not os.path.exists(images_dir):
        print(f"❌ Error: Could not find images directory at:\n   {images_dir}")
        return False
        
    # --- Prepare output directories ---
    os.makedirs(output_dir, exist_ok=True)
    out_images_dir = os.path.join(output_dir, "images")
    out_annotations_dir = os.path.join(output_dir, "annotations")
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_annotations_dir, exist_ok=True)

    print("\n⏳ Starting SAHI Slicing Process...")
    print(f"  Input       : {input_dir}")
    print(f"  Output      : {output_dir}")
    print(f"  Slice Size  : {slice_size}x{slice_size}px  ← set YOLO imgsz={slice_size}")
    print(f"  Overlap     : {int(overlap_ratio * 100)}%")
    print("=" * 55)
    print("\n⏳ Starting SAHI slicing... (may take a few minutes)\n")

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
    
    input_path = input("Enter the ABSOLUTE path of the input COCO dataset directory:\n> ").strip()
    if not os.path.isdir(input_path):
        print(f"❌ Error: '{input_path}' does not exist or is not a directory.")
        return
 
    output_path = input("\nEnter the ABSOLUTE path for the sliced output dataset:\n> ").strip()
    if not output_path:
        print("❌ Error: Output path cannot be empty.")
        return
 
    slice_input = input(f"\nEnter slice size in pixels [default: 640, options: 640 / 960 / 1280]:\n> ").strip()
    try:
        slice_size = int(slice_input) if slice_input else 640
    except ValueError:
        print("⚠️  Invalid input, using default slice size: 640")
        slice_size = 640
 
    overlap_input = input(f"\nEnter overlap ratio [default: 0.2 = 20%]:\n> ").strip()
    try:
        overlap_ratio = float(overlap_input) if overlap_input else 0.2
    except ValueError:
        print("⚠️  Invalid input, using default overlap: 0.2")
        overlap_ratio = 0.2
    
    slice_large_images_in_dataset(input_path, output_path, slice_size, overlap_ratio)

if __name__ == "__main__":
    main()
