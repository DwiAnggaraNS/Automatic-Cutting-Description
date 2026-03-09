import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import matplotlib.pyplot as plt

# ================= KONFIGURASI =================
# Path ke Dataset Training Asli
DATASET_ROOT = '.'
ANNOTATION_MODE = "simplified" # single_class, '',simplified
annotation_postfix = f"_{ANNOTATION_MODE}" if ANNOTATION_MODE != '' else ''
TRAIN_JSON = f"{DATASET_ROOT}/train/train_annotations{annotation_postfix}.json" 
TRAIN_IMG_DIR = f"{DATASET_ROOT}/train/images"

# Folder Output untuk menyimpan hasil crop
OUTPUT_LIB_DIR = f"{DATASET_ROOT}/augment_library"

# Daftar ID Kelas Minoritas yang ingin diperbanyak (Sesuai statistik Anda tadi)
# Training Statistics: (get_statistics_data.py)
#0          CLAS - Silt                              1079        39.67%
#1          CLAS - Loose Sand                        74           2.72%
#2          CLAS - Sandstone                         529         19.45%
#3          CARB - Limestone                         674         24.78%
#4          CLAS - Loose Sandy and Silt              13           0.48%
#5          CLAS - Loose Silt                        13           0.48%
#6          CARB - Loose Limestone                   33           1.21%
#7          CLAS - Coal                              305         11.21%

# +1 karena kategori dimulai dari 1 di COCO
MINORITY_IDS = [1+1, 4+1, 5+1, 6+1] 

# ===============================================

def extract_minority_objects():
    print(f"📂 Membaca anotasi dari: {TRAIN_JSON}")
    coco = COCO(TRAIN_JSON)
    
    # Buat folder output
    if not os.path.exists(OUTPUT_LIB_DIR):
        os.makedirs(OUTPUT_LIB_DIR)
        
    print(f"🎯 Target Kelas ID: {MINORITY_IDS}")
    
    total_extracted = 0
    
    # Loop per Kelas Minoritas
    for cat_id in MINORITY_IDS:
        # Buat sub-folder per kelas (biar rapi)
        class_dir = os.path.join(OUTPUT_LIB_DIR, str(cat_id))
        os.makedirs(class_dir, exist_ok=True)
        
        # Ambil semua anotasi untuk kelas ini
        ann_ids = coco.getAnnIds(catIds=[cat_id])
        anns = coco.loadAnns(ann_ids)
        
        print(f"   > Kelas {cat_id}: Ditemukan {len(anns)} objek. Mengekstrak...")
        
        for i, ann in enumerate(tqdm(anns)):
            # 1. Load Gambar Asli
            img_info = coco.loadImgs(ann['image_id'])[0]
            img_path = os.path.join(TRAIN_IMG_DIR, img_info['file_name'])
            
            if not os.path.exists(img_path):
                continue
                
            image = cv2.imread(img_path) # BGR
            
            # 2. Ambil Masker Binary
            mask = coco.annToMask(ann) # 0 = background, 1 = object
            
            # 3. Buat Gambar RGBA (4 Channel: Red, Green, Blue, Alpha)
            # Buat channel Alpha berdasarkan masker (255 = terlihat, 0 = transparan)
            h, w = image.shape[:2]
            
            # Ambil Bounding Box [x, y, w, h]
            x, y, bw, bh = map(int, ann['bbox'])
            
            # Safety check agar bbox tidak keluar gambar
            x = max(0, x); y = max(0, y)
            
            # 4. Crop Gambar & Masker sesuai Bounding Box
            # Kita crop dulu biar file kecil, baru dimasking
            crop_img = image[y:y+bh, x:x+bw]
            crop_mask = mask[y:y+bh, x:x+bw]
            
            if crop_img.size == 0: continue

            # 5. Terapkan Masking (Buat background jadi hitam murni dulu)
            # Ini membuang pixel background di dalam kotak bbox tapi di luar contour batu
            crop_img = cv2.bitwise_and(crop_img, crop_img, mask=crop_mask)
            
            # 6. Tambahkan Alpha Channel
            b, g, r = cv2.split(crop_img)
            # Alpha channel: 255 dimana mask=1, 0 dimana mask=0
            alpha = (crop_mask * 255).astype(np.uint8)
            
            rgba_crop = cv2.merge([b, g, r, alpha])
            
            # 7. Simpan sebagai PNG
            # Nama file: classID_imgID_annID.png
            filename = f"{cat_id}_{ann['image_id']}_{ann['id']}.png"
            save_path = os.path.join(class_dir, filename)
            cv2.imwrite(save_path, rgba_crop)
            
            total_extracted += 1

    print(f"\n✅ Selesai! Total {total_extracted} objek minoritas diekstrak ke: {OUTPUT_LIB_DIR}")

# Jalankan Ekstraksi
if __name__ == "__main__":
    extract_minority_objects()