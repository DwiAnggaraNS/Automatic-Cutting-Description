import cv2
import numpy as np
import os
import random
import glob
import json
import copy
from tqdm import tqdm

DATASET_ROOT = '.'
ANNOTATION_MODE = "simplified" # single_class, '',simplified
annotation_postfix = f"_{ANNOTATION_MODE}" if ANNOTATION_MODE != '' else ''
TRAIN_JSON = f"{DATASET_ROOT}/train/train_annotations{annotation_postfix}.json" 
TRAIN_IMG_DIR = f"{DATASET_ROOT}/train/images"

# ================= KONFIGURASI PABRIK =================
# 1. Sumber Bahan Baku
BACKGROUND_DIR = f'{DATASET_ROOT}/augment_library/bg'
LIBRARY_DIR = f'{DATASET_ROOT}/augment_library'

# 2. Lokasi Output (Barang Jadi)
OUTPUT_IMG_DIR = f'{DATASET_ROOT}/train/synthetic_images'
OUTPUT_LBL_DIR = f'{DATASET_ROOT}/train/synthetic_labels'
OUTPUT_JSON = f'{DATASET_ROOT}/train/train_annotations{annotation_postfix}_synthetic.json'

# 3. Target Produksi
NUM_IMAGES_TO_GENERATE = 100  # Total foto baru yang ingin dibuat
OBJECTS_PER_IMAGE = (5, 15)   # Random antara 5 s.d. 15 batu per foto

# 4. Kelas Target (Minoritas)
# Script akan mengambil batu dari folder ID ini di library
MINORITY_IDS = [1+1, 4+1, 5+1, 6+1] 
TARGET_CLASSES = MINORITY_IDS

# 5. Parameter Augmentasi (Anti-Overfitting)
SCALE_RANGE = (0.7, 1.3)  # Resize batu antara 70% s.d. 130%
ROTATION = True           # Putar batu acak 0-360 derajat

# ================= HELPER FUNCTIONS =================

def create_dirs():
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)

def extract_contour_from_alpha(image):
    """Extract the largest contour from the alpha channel of an RGBA image."""
    if image.shape[2] != 4:
        return None
    
    alpha = image[:, :, 3]
    # Find contours from the alpha mask
    contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Return the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def rotate_contour(contour, angle, center, new_center):
    """Rotate a contour by a given angle around a center point."""
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Translate to origin, rotate, translate to new center
    rotated = []
    for point in contour:
        px, py = point[0]
        # Translate to origin
        tx = px - center[0]
        ty = py - center[1]
        # Rotate
        rx = tx * cos_a - ty * sin_a
        ry = tx * sin_a + ty * cos_a
        # Translate to new center
        nx = rx + new_center[0]
        ny = ry + new_center[1]
        rotated.append([nx, ny])
    
    return np.array(rotated, dtype=np.float32).reshape(-1, 1, 2)

def scale_contour(contour, scale, center):
    """Scale a contour by a given factor around a center point."""
    scaled = []
    for point in contour:
        px, py = point[0]
        # Translate to origin, scale, translate back
        tx = (px - center[0]) * scale + center[0]
        ty = (py - center[1]) * scale + center[1]
        scaled.append([tx, ty])
    
    return np.array(scaled, dtype=np.float32).reshape(-1, 1, 2)

def translate_contour(contour, offset_x, offset_y):
    """Translate a contour by given offsets."""
    translated = []
    for point in contour:
        px, py = point[0]
        translated.append([px + offset_x, py + offset_y])
    
    return np.array(translated, dtype=np.float32).reshape(-1, 1, 2)

def contour_to_coco_segmentation(contour):
    """Convert OpenCV contour to COCO segmentation format (flat list of x,y pairs)."""
    if contour is None or len(contour) < 3:
        return None
    
    # Flatten the contour: [[x1,y1], [x2,y2], ...] -> [x1, y1, x2, y2, ...]
    flat_list = contour.flatten().tolist()
    return [flat_list]

def compute_bbox_from_contour(contour):
    """Compute bounding box [x, y, width, height] from a contour."""
    x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
    return [float(x), float(y), float(w), float(h)]

def compute_area_from_contour(contour):
    """Compute area from a contour."""
    return float(cv2.contourArea(contour.astype(np.int32)))

def rotate_image(image, angle):
    """Memutar gambar RGBA dengan tetap menjaga transparansi.
    Returns: (rotated_image, old_center, new_center) for contour transformation.
    """
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # Border transparent
    rotated = cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return rotated, (cX, cY), (nW // 2, nH // 2)

def overlay_transparent(background, overlay, x, y):
    """Menempelkan overlay RGBA ke background BGR."""
    bg_h, bg_w, _ = background.shape
    ol_h, ol_w, _ = overlay.shape

    # Cek jika overlay keluar batas background
    if x >= bg_w or y >= bg_h: return background
    
    # Hitung area potong (clipping)
    h, w = ol_h, ol_w
    if x + w > bg_w: w = bg_w - x
    if y + h > bg_h: h = bg_h - y
    if x < 0: w = w + x; x = 0
    if y < 0: h = h + y; y = 0
    
    if w <= 0 or h <= 0: return background

    # Ambil ROI (Region of Interest)
    overlay_crop = overlay[:h, :w]
    bg_crop = background[y:y+h, x:x+w]

    # Ekstrak Alpha Channel (0-1)
    alpha = overlay_crop[:, :, 3] / 255.0
    alpha = np.dstack([alpha, alpha, alpha])

    # Rumus Blending: (1-alpha)*BG + alpha*FG
    foreground = overlay_crop[:, :, :3]
    composite = bg_crop * (1.0 - alpha) + foreground * alpha
    
    background[y:y+h, x:x+w] = composite
    return background

def check_overlap(new_box, existing_boxes, threshold=0.1):
    """Cek apakah batu baru menumpuk batu yang sudah ada."""
    nx, ny, nw, nh = new_box
    for (ex, ey, ew, eh) in existing_boxes:
        # Hitung Intersection
        ix = max(nx, ex)
        iy = max(ny, ey)
        iw = min(nx+nw, ex+ew) - ix
        ih = min(ny+nh, ey+eh) - iy
        
        if iw > 0 and ih > 0:
            intersection = iw * ih
            area_new = nw * nh
            # Jika tertutup lebih dari 10%, anggap overlap (cari tempat lain)
            if intersection / area_new > threshold:
                return True
    return False

# ================= MAIN GENERATOR =================

def generate_data():
    create_dirs()
    
    # 0. Load existing COCO JSON untuk mendapatkan struktur dan ID terakhir
    print(f"📂 Memuat anotasi dari: {TRAIN_JSON}")
    with open(TRAIN_JSON, 'r') as f:
        coco_data = json.load(f)
    
    # Buat copy untuk output synthetic
    synthetic_coco = copy.deepcopy(coco_data)
    
    # Cari ID terakhir untuk image dan annotation
    max_image_id = max([img['id'] for img in coco_data['images']]) if coco_data['images'] else 0
    max_ann_id = max([ann['id'] for ann in coco_data['annotations']]) if coco_data['annotations'] else 0
    
    current_image_id = max_image_id
    current_ann_id = max_ann_id
    
    # 1. Load Daftar Background (Pakai Silt Class 0)
    bg_files = glob.glob(os.path.join(BACKGROUND_DIR, "*.jpg")) + glob.glob(os.path.join(BACKGROUND_DIR, "*.png"))
    if not bg_files:
        print("❌ Error: Tidak ada gambar background ditemukan!")
        return

    # 2. Load Library Batu Kecil
    library = {}
    total_assets = 0
    for cls_id in TARGET_CLASSES:
        files = glob.glob(os.path.join(LIBRARY_DIR, str(cls_id), "*.png"))
        library[cls_id] = files
        total_assets += len(files)
        print(f"   > Kelas {cls_id}: {len(files)} aset ditemukan.")
    
    if total_assets == 0:
        print("❌ Error: Library kosong! Jalankan tahap ekstraksi dulu.")
        return

    print(f"\n🚀 Memulai Produksi {NUM_IMAGES_TO_GENERATE} Gambar Sintetis...")
    
    # Track new annotations untuk synthetic
    new_images = []
    new_annotations = []

    for i in tqdm(range(NUM_IMAGES_TO_GENERATE)):
        # A. Siapkan Canvas
        bg_path = random.choice(bg_files)
        bg_img = cv2.imread(bg_path)
        h_bg, w_bg = bg_img.shape[:2]
        
        # Increment image ID
        current_image_id += 1
        filename = f"synth_{i:04d}.jpg"
        
        # B. Tentukan jumlah batu yang akan ditempel
        num_objects = random.randint(*OBJECTS_PER_IMAGE)
        labels = [] # List untuk menyimpan [class, x_center, y_center, w, h]
        placed_boxes = [] # List [x, y, w, h] pixel untuk cek overlap
        image_annotations = [] # Annotations untuk gambar ini
        
        for _ in range(num_objects):
            # C. Pilih Batu Random
            cls_id = random.choice(TARGET_CLASSES)
            if not library[cls_id]: continue
            
            asset_path = random.choice(library[cls_id])
            rock_img_original = cv2.imread(asset_path, cv2.IMREAD_UNCHANGED) # Load RGBA
            if rock_img_original is None: continue
            
            # Ekstrak contour dari alpha channel SEBELUM transformasi
            original_contour = extract_contour_from_alpha(rock_img_original)
            if original_contour is None: continue
            
            rock_img = rock_img_original.copy()
            contour = original_contour.copy().astype(np.float32)
            
            # D. Augmentasi Geometri (PENTING!)
            # 1. Rotate
            angle = 0
            # if ROTATION:
            #     angle = random.randint(0, 360)
            #     rock_img, old_center, new_center = rotate_image(rock_img, angle)
            #     # Transform contour: rotate around old center, then translate to new center
            #     contour = rotate_contour(contour, -angle, old_center, new_center)
            # Di dalam loop utama, saat rotasi:
            if ROTATION:
                angle = random.randint(0, 360)
                # Dapatkan Matrix Rotasi dari OpenCV
                (h, w) = rock_img.shape[:2]
                (cX, cY) = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
                
                # Hitung ukuran bounding box baru (agar gambar tidak terpotong)
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])
                nW = int((h * sin) + (w * cos))
                nH = int((h * cos) + (w * sin))
                
                # Sesuaikan Matrix M dengan offset baru
                M[0, 2] += (nW / 2) - cX
                M[1, 2] += (nH / 2) - cY
                
                # 1. Putar GAMBAR
                rock_img = cv2.warpAffine(rock_img, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
                
                # 2. Putar KONTUR (Pakai Matrix M yang SAMA)
                # OpenCV cv2.transform hanya menerima shape (N, 1, 2)
                # Ini menjamin polygon 100% nempel di batu tanpa geser
                contour = cv2.transform(contour, M)
            
            # 2. Resize
            scale = random.uniform(*SCALE_RANGE)
            old_h, old_w = rock_img.shape[:2]
            new_w = int(old_w * scale)
            new_h = int(old_h * scale)
            rock_img = cv2.resize(rock_img, (new_w, new_h))
            
            # Scale contour dari center
            center_before_scale = (old_w / 2, old_h / 2)
            contour = scale_contour(contour, scale, center_before_scale)
            
            # E. Cari Posisi Kosong (Coba 50 kali, kalau gagal skip)
            for attempt in range(50):
                r_h, r_w = rock_img.shape[:2]
                
                # Random X, Y (Pastikan tidak keluar batas)
                if w_bg - r_w <= 0 or h_bg - r_h <= 0: break 
                
                pos_x = random.randint(0, w_bg - r_w)
                pos_y = random.randint(0, h_bg - r_h)
                
                # Cek Overlap
                current_box_pixel = (pos_x, pos_y, r_w, r_h)
                if not check_overlap(current_box_pixel, placed_boxes):
                    # F. TEMPEL!
                    # 1. Visual Paste
                    bg_img = overlay_transparent(bg_img, rock_img, pos_x, pos_y)
                    
                    # 2. Translate Contour ke posisi Final
                    final_contour = translate_contour(contour, pos_x, pos_y)
                    
                    # 3. Simplify Contour (Optimasi Size JSON)
                    # Pastikan contour dalam format yang benar untuk OpenCV
                    final_contour = final_contour.astype(np.float32)
                    epsilon = 0.005 * cv2.arcLength(final_contour, True)
                    final_contour = cv2.approxPolyDP(final_contour, epsilon, True)
                    
                    # G. Catat Label (Format YOLO Normalized)
                    x_center = (pos_x + r_w / 2) / w_bg
                    y_center = (pos_y + r_h / 2) / h_bg
                    norm_w = r_w / w_bg
                    norm_h = r_h / h_bg
                    
                    labels.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
                    placed_boxes.append(current_box_pixel)
                    
                    # H. Buat COCO Annotation
                    current_ann_id += 1
                    segmentation = contour_to_coco_segmentation(final_contour)
                    if segmentation:
                        bbox = compute_bbox_from_contour(final_contour)
                        area = compute_area_from_contour(final_contour)
                        
                        ann_entry = {
                            "id": current_ann_id,
                            "image_id": current_image_id,
                            "category_id": cls_id,
                            "segmentation": segmentation,
                            "area": area,
                            "bbox": bbox,
                            "iscrowd": 0,
                            "attributes": {
                                "occluded": False,
                                "synthetic": True
                            }
                        }
                        image_annotations.append(ann_entry)
                    
                    break # Lanjut ke batu berikutnya
        
        # I. Simpan Hasil Gambar
        cv2.imwrite(os.path.join(OUTPUT_IMG_DIR, filename), bg_img)
        
        # Simpan YOLO labels
        with open(os.path.join(OUTPUT_LBL_DIR, f"synth_{i:04d}.txt"), "w") as f:
            f.write("\n".join(labels))
        
        # J. Tambahkan image entry ke COCO
        image_entry = {
            "id": current_image_id,
            "width": w_bg,
            "height": h_bg,
            "file_name": filename,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        }
        new_images.append(image_entry)
        new_annotations.extend(image_annotations)

    # K. Update synthetic COCO data dan simpan
    synthetic_coco['images'].extend(new_images)
    synthetic_coco['annotations'].extend(new_annotations)
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(synthetic_coco, f, indent=2)
    
    print(f"\n✅ Selesai!")
    print(f"   📁 Gambar tersimpan di: {OUTPUT_IMG_DIR}")
    print(f"   📁 Label YOLO tersimpan di: {OUTPUT_LBL_DIR}")
    print(f"   📁 Anotasi COCO tersimpan di: {OUTPUT_JSON}")
    print(f"   📊 Total gambar synthetic: {len(new_images)}")
    print(f"   📊 Total anotasi baru: {len(new_annotations)}")

if __name__ == "__main__":
    generate_data()