import os
import json

# Master schema for final fixed categories
# Rock (dummy) is excluded here because it will either be dropped or halt the script
MASTER_CATEGORIES = {
    "CLAS - Silt": 1,
    "CLAS - Sandstone": 2,
    "CARB - Limestone": 3,
    "ORG - Coal": 4,         
    "CLAS - Shalestone": 5,
    "MIN - Quartz": 6,
    "ART - Cement": 7        
}

# Category name mapping for loose/inconsistent classes
# Mapping old name from CVAT (left) to standard name using Prefix (rigjt)
CATEGORY_MAPPING = {
    # Mapping for standard names
    "Silt": "CLAS - Silt",
    "Sandstone": "CLAS - Sandstone",
    "Limestone": "CARB - Limestone",
    "Coal": "ORG - Coal",
    "Shalestone": "CLAS - Shalestone",
    "Quartz": "MIN - Quartz",
    "Cement": "ART - Cement",
    
    # Mapping for loose classes
    "Loose Sand": "CLAS - Sandstone",
    "Sand": "CLAS - Sandstone",
    "Loose Silt": "CLAS - Silt",
    "Loose Limestone": "CARB - Limestone"
}

# Categories that should be completely dropped from the annotations
CATEGORIES_TO_DROP = ["Loose Sandy and Silt"]

def remap_dataset_categories(folder_path):
    """
    Reads a COCO export instances_default.json, remaps the category IDs based on the 
    MASTER_CATEGORIES ontology, drops unwanted annotations, and handles the 'rock' dummy class.
    """
    json_path = os.path.join(folder_path, 'annotations', 'instances_default.json')
    
    if not os.path.exists(json_path):
        print(f"❌ Error: Could not find annotations JSON at {json_path}")
        return False
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    # 1. Map old category IDs to their string names in the current dataset
    old_id_to_name = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
    
    # 2. Check the 'rock' class condition
    rock_count = 0
    rock_category_ids = [cat_id for cat_id, name in old_id_to_name.items() if name == "rock"]
    if rock_category_ids:
        for ann in data.get("annotations", []):
            if ann["category_id"] in rock_category_ids:
                rock_count += 1
                
    if rock_count >= 10:
        print(f"🛑 HALTED: Found {rock_count} 'rock' annotations in {folder_path}.")
        print("Please review and re-annotate the 'rock' class in CVAT before exporting.")
        return False
    elif rock_count > 0:
        print(f"⚠️ Warning: Found {rock_count} 'rock' annotations (< 10). They will be deleted.")

    # 3. Build a mapping from old_id -> new_master_id
    old_id_to_new_id = {}
    for old_id, name in old_id_to_name.items():
        if name == "rock" or name in CATEGORIES_TO_DROP:
            old_id_to_new_id[old_id] = None # Will be dropped
            continue
            
        # Apply name mapping if exists
        final_name = CATEGORY_MAPPING.get(name, name)
        
        if final_name not in MASTER_CATEGORIES:
            print(f"⚠️ Warning: Found an unrecognized class '{name}' (mapped to '{final_name}'). It will be ignored/dropped.")
            old_id_to_new_id[old_id] = None
        else:
            if name in CATEGORY_MAPPING:
                print(f"🔄 Remapping: '{name}' -> '{final_name}'")
            old_id_to_new_id[old_id] = MASTER_CATEGORIES[final_name]

    # 4. Filter and update annotations
    new_annotations = []
    dropped_count = 0
    
    for ann in data.get("annotations", []):
        old_cat_id = ann["category_id"]
        new_cat_id = old_id_to_new_id.get(old_cat_id, None)
        
        if new_cat_id is None:
            dropped_count += 1
            continue # Drop this annotation
            
        ann["category_id"] = new_cat_id
        new_annotations.append(ann)
        
    data["annotations"] = new_annotations
    print(f"✅ Filtered annotations: Removed {dropped_count} items (dummy rocks, dropped classes, unrecognized).")

    # 5. Replace categories list with the master schema
    new_categories = [{"id": v, "name": k} for k, v in MASTER_CATEGORIES.items()]
    data["categories"] = new_categories
    
    # 6. Save back the JSON
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"✅ Successfully remapped classes for {folder_path}\n")
    return True

def main():
    print("========================================")
    print("   COCO Dataset Category Remapping Tool ")
    print("========================================")
    
    try:
        num_folders = int(input("Enter the number of dataset folders to remap: "))
    except ValueError:
        print("❌ Invalid input for number of folders. Please enter an integer.")
        return

    if num_folders < 1:
        print("❌ Number of folders must be at least 1.")
        return

    for i in range(num_folders):
        path = input(f"Enter path for dataset folder {i + 1}: ").strip()
        if not os.path.isdir(path):
            print(f"❌ Error: The path '{path}' does not exist or is not a directory.")
            print("Skipping this path...\n")
            continue
            
        print(f"Processing '{path}'...")
        success = remap_dataset_categories(path)
        if not success:
            print("❌ Script execution aborted for this folder due to errors/conditions.\n")
            # We abort the entire script to make sure they fix the data
            print("Please fix the issues in CVAT and try again.")
            break

    print("🏁 Remapping process finished.")

if __name__ == "__main__":
    main()
