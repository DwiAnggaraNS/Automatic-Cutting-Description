# Semi-Automatic Rock Annotation: SAM AutomaticMaskGenerator → CVAT

> Panduan lengkap untuk melakukan anotasi semi-otomatis menggunakan SAM di luar CVAT,
> kemudian mengimpor hasilnya ke CVAT untuk refinement manual.

---

## Daftar Isi

1. [Mengapa Semi-Otomatis?](#1-mengapa-semi-otomatis)
2. [Alur Pipeline](#2-alur-pipeline)
3. [Persiapan: SAM Weights dari Nuclio](#3-persiapan-sam-weights-dari-nuclio)
4. [Instalasi Library](#4-instalasi-library)
5. [Struktur Folder](#5-struktur-folder)
6. [Step 1 — SAM Auto Segmentation](#6-step-1--sam-auto-segmentation)
7. [Step 2 — Konversi ke CVAT XML](#7-step-2--konversi-ke-cvat-xml)
8. [Step 3 — Import ke CVAT](#8-step-3--import-ke-cvat)
9. [Step 4 — Refinement di CVAT](#9-step-4--refinement-di-cvat)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Mengapa Semi-Otomatis?

SAM di dalam CVAT membutuhkan **satu klik per objek** untuk menghasilkan segmentasi. Pada gambar batu dengan 100+ objek per frame, ini tidak efisien.

Solusinya: gunakan `SamAutomaticMaskGenerator` di luar CVAT. Mode ini menjalankan grid prompt yang rapat di seluruh gambar dalam **satu kali pass**, menghasilkan semua objek sekaligus secara otomatis. Hasilnya dikonversi ke format CVAT XML dan diimpor massal — annotator hanya perlu **mengubah label** dan **memperbaiki polygon yang salah**.

---

## 2. Alur Pipeline

```
┌─────────────────────────────────────────────────────┐
│                  GAMBAR MENTAH                      │
│              (folder ./images/)                     │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│         01_sam_auto_segmentation.ipynb              │
│                                                     │
│  • Load SAM vit_h weights (sekali)                  │
│  • Loop gambar satu per satu (memory-safe)          │
│  • Filter mask: area, pred_iou, stability_score     │
│  • Top-K: simpan N mask terbaik berdasarkan IoU     │
│  • Konversi binary mask → polygon (Douglas-Peucker) │
│  • Simpan hasil per gambar → ./sam_output/*.json    │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│            02_cvat_converter.ipynb                  │
│                                                     │
│  • Fetch frame list dari CVAT REST API              │
│    (mendapatkan frame index & nama yang tepat)      │
│  • Baca semua ./sam_output/*.json                   │
│  • Assign dummy label "rock" ke semua instance      │
│  • Build CVAT XML 1.1                               │
│  • Simpan → ./cvat_import/annotations.xml           │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│                   CVAT Import                       │
│                                                     │
│  • Upload annotations.xml ke task CVAT              │
│  • Format: CVAT 1.1                                 │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│            Refinement Manual di CVAT                │
│                                                     │
│  • Ubah label "rock" → kelas batu yang tepat        │
│  • Perbaiki polygon yang salah bentuk               │
│  • Hapus false positive                             │
│  • Tambah objek yang terlewat                       │
└─────────────────────────────────────────────────────┘
```

---

## 3. Persiapan: SAM Weights dari Nuclio

**Tidak perlu download ulang.** Saat `nuctl deploy` dijalankan, proses build Nuclio sudah mengunduh `sam_vit_h_4b8939.pth` (~2.4 GB) ke dalam Docker image layer. File ini bisa di-copy langsung dari container yang sedang berjalan.

```bash
# 1. Temukan path file weight di dalam container
sudo docker exec nuclio-nuclio-pth-facebookresearch-sam-vit-h \
    find /opt/nuclio -name "*.pth" 2>/dev/null
# Output contoh: /opt/nuclio/sam_vit_h_4b8939.pth

# 2. Buat folder tujuan dan copy file
mkdir -p ~/sam_annotation/sam_weights
sudo docker cp \
    nuclio-nuclio-pth-facebookresearch-sam-vit-h:/opt/nuclio/sam_vit_h_4b8939.pth \
    ~/sam_annotation/sam_weights/sam_vit_h_4b8939.pth
```

Jika path tidak ditemukan:
```bash
sudo docker exec nuclio-nuclio-pth-facebookresearch-sam-vit-h \
    find / -name "sam_vit_h_4b8939.pth" 2>/dev/null
```

---

## 4. Instalasi Library

```bash
conda activate <nama_env_kamu>

# SAM (Facebook Research)
pip install git+https://github.com/facebookresearch/segment-anything.git

# Dependencies (sebagian besar sudah terinstall)
pip install opencv-python-headless pycocotools tqdm requests
```

Verifikasi GPU:
```python
import torch
print(torch.cuda.is_available())   # harus True
print(torch.cuda.get_device_name(0))
```

> ⚠️ **Penting:** Sebelum menjalankan notebook, **hentikan CVAT terlebih dahulu** (`~/cvat/cvat-stop.sh`) agar Nuclio SAM container tidak memakan GPU VRAM yang sama. Jalankan kembali setelah proses SAM selesai.

---

## 5. Struktur Folder

Buat folder kerja di lokasi yang diinginkan (contoh: `~/sam_annotation/`):

```
sam_annotation/
│
├── images/                          # ← Letakkan semua gambar mentah di sini
│   ├── WIN_20260225_10_52_13_Pro.jpg
│   ├── WIN_20260225_10_52_23_Pro.jpg
│   └── ...
│
├── sam_output/                      # ← Auto-dibuat oleh Notebook 1
│   ├── WIN_20260225_10_52_13_Pro.json   # metadata mask + polygon per gambar
│   ├── WIN_20260225_10_52_23_Pro.json
│   └── ...
│
├── cvat_import/                     # ← Auto-dibuat oleh Notebook 2
│   └── annotations.xml              # siap diimport ke CVAT
│
├── sam_weights/
│   └── sam_vit_h_4b8939.pth         # ← dicopy dari Nuclio container (Step 3)
│
├── 01_sam_auto_segmentation.ipynb   # Notebook 1: Segmentasi
└── 02_cvat_converter.ipynb          # Notebook 2: Konversi ke CVAT XML
```

> **Catatan lokasi notebook:** Di repositori ini, kedua notebook berada di
> `notebooks/exploration/`. Jalankan dari sana, atau sesuaikan path `IMAGE_DIR`,
> `OUTPUT_DIR`, dan `SAM_WEIGHTS` di Cell konfigurasi.

---

## 6. Step 1 — SAM Auto Segmentation

**File:** `notebooks/exploration/01_sam_auto_segmentation.ipynb`

### Parameter Konfigurasi (Cell 2)

```python
SAM_PARAMS = dict(
    points_per_side         = 32,    # grid 32×32 = 1024 prompt points
    points_per_batch        = 16,    # batch decoding (turunkan jika OOM)
    pred_iou_thresh         = 0.90,  # threshold kualitas mask
    stability_score_thresh  = 0.96,  # threshold stabilitas batas mask
    box_nms_thresh          = 0.50,  # NMS: hapus mask yang terlalu overlap
    crop_n_layers           = 0,     # MATIKAN crop — penyebab utama OOM
    min_mask_region_area    = 3000,  # minimal ~55×55 px (sesuaikan dengan ukuran batu)
)
MIN_AREA_PX       = 3000
MAX_MASKS_PER_IMG = 150             # hard cap: simpan top-N mask terbaik
```

### Strategi Memory GPU

- **Satu gambar per iterasi** — tidak ada batching antar gambar
- `torch.cuda.empty_cache()` + `gc.collect()` dipanggil setelah **setiap gambar**
- `skip_existing=True` — gambar yang sudah diproses dilewati, aman untuk re-run
- Top-K filtering dijalankan **sebelum** konversi polygon untuk efisiensi

### Menjalankan

1. Pastikan CVAT sudah dihentikan: `~/cvat/cvat-stop.sh`
2. Buka `01_sam_auto_segmentation.ipynb`
3. **Tutup dan buka kembali notebook** setelah setiap kali edit parameter (untuk memastikan kernel membaca versi terbaru)
4. **Restart Kernel → Run All Cells**
5. Output akan tampil seperti:
   ```
   Found 103 images in './images'
   Hard cap per image : 150 masks (ranked by predicted_iou)
     WIN_20260225_10_52_13_Pro.jpg: 87 masks (3840×2160)
     WIN_20260225_10_52_23_Pro.jpg: 94 masks (3840×2160)
   ```

### Cleanup Cache (jika perlu re-proses)

Jika ada JSON lama dengan jumlah mask yang melebihi batas, jalankan **Cell 6 (Cleanup)**
sebelum pipeline — cell ini otomatis menghapus file JSON yang `n_masks > MAX_MASKS_PER_IMG`.

---

## 7. Step 2 — Konversi ke CVAT XML

**File:** `notebooks/exploration/02_cvat_converter.ipynb`

### Prasyarat: Task CVAT Sudah Dibuat

Sebelum menjalankan notebook ini, buat task di CVAT terlebih dahulu:

1. Nyalakan CVAT: `~/cvat/cvat-start.sh`
2. Buka `http://localhost:8080`
3. Klik **Create Task**
4. Isi nama task, tambahkan label **`rock`** (persis, case-sensitive)
5. Upload semua gambar dari folder `images/`
6. Klik **Submit & Open**
7. Catat **Task ID** dari URL: `http://localhost:8080/tasks/`**`3`** → ID = `3`

### Konfigurasi (Cell 1)

```python
CVAT_URL      = "http://localhost:8080"
CVAT_USERNAME = "admin"       # username CVAT kamu
CVAT_PASSWORD = "yourpassword" # password CVAT kamu
CVAT_TASK_ID  = 3             # ID task dari URL CVAT
```

### Mengapa Perlu Fetch Frame dari API CVAT?

CVAT menyimpan nama file secara internal persis seperti yang diterimanya. Jika upload
dilakukan via ZIP, nama bisa berupa `images/foto.jpg` bukan `foto.jpg`. Frame `id`
juga ditentukan oleh urutan yang CVAT tentukan, bukan urutan alfabet file kita.

Cell 2 memanggil `/api/tasks/{id}/data/meta` untuk mendapatkan:
- Frame index yang tepat (posisi dalam list respons API)
- Nama file yang tersimpan persis di CVAT

Hasilnya digunakan di Cell 3 untuk mengisi `<image id="..." name="...">` dengan
nilai yang dijamin cocok — sehingga import tidak akan menghasilkan error
`Could not match item id`.

### Menjalankan

1. Jalankan **Run All Cells**
2. Cell 2 akan mencetak daftar frame:
   ```
   Fetched 103 frames from CVAT task 3:
     [  0]  WIN_20260225_10_52_13_Pro.jpg
     [  1]  WIN_20260225_10_52_23_Pro.jpg
     ...
   ```
3. Cell 3 membangun XML dan menyimpannya ke `cvat_import/annotations.xml`
4. Cell 4 memvalidasi struktur XML

---

## 8. Step 3 — Import ke CVAT

1. Buka task yang sudah dibuat di `http://localhost:8080`
2. Klik **menu tiga titik (⋮)** di task → **Upload annotations**
3. Pilih format: **CVAT 1.1**
4. Pilih file: `cvat_import/annotations.xml`
5. Klik **OK**
6. Pantau proses import:
   ```bash
   sudo docker logs cvat_worker_import --tail 30 -f
   ```
7. Buka salah satu job → pastikan polygon terlihat menyelimuti batu-batu

---

## 9. Step 4 — Refinement di CVAT

Setelah import berhasil, semua polygon berlabel `rock`. Langkah selanjutnya:

### Tambahkan Label Kelas Final

Masuk ke **Task Settings → Labels → Add Label** untuk setiap kelas:

| Label | Kategori |
|---|---|
| `CLAS - Silt` | Klastik |
| `CLAS - Loose Sand` | Klastik |
| `CLAS - Sandstone` | Klastik |
| `CARB - Limestone` | Karbonat |
| `CLAS - Loose Sandy and Silt` | Klastik |
| `CLAS - Loose Silt` | Klastik |
| `CARB - Loose Limestone` | Karbonat |
| `CLAS - Coal` | Organik |

### Alur Refinement per Gambar

1. Klik polygon → panel kiri menampilkan dropdown label
2. Ubah `rock` → kelas batu yang sesuai
3. Untuk polygon yang salah bentuk: **double-click** → edit mode → drag vertex
4. Untuk false positive: pilih polygon → tekan `Del`
5. Untuk objek yang terlewat: gunakan tool **Polygon** atau **AI Tools → SAM**

### Tips Efisiensi (100+ Objek per Gambar)

- `N` = next object, `F` = next frame (shortcut paling sering dipakai)
- Sort objek berdasarkan **area ascending** → temukan false positive kecil lebih cepat
- Gunakan **Tag** untuk menandai gambar sebagai `reviewed` / `pending`
- Bagi pekerjaan ke beberapa annotator — CVAT mendukung multi-user pada satu task

---

## 10. Troubleshooting

| Error | Penyebab | Solusi |
|---|---|---|
| `CUDA out of memory` | GPU dipakai bersama Nuclio SAM container | Stop CVAT sebelum jalankan notebook: `~/cvat/cvat-stop.sh` |
| `CUDA out of memory` | `crop_n_layers=1` + gambar besar | Set `crop_n_layers=0` di `SAM_PARAMS` |
| Mask terlalu banyak (>150) | Kernel masih pakai fungsi lama | Tutup-buka notebook, Restart Kernel, Run All |
| `CvatImportError: Could not match item id` | Nama file di XML tidak cocok dengan CVAT | Pastikan Cell 2 di `02_cvat_converter.ipynb` berhasil fetch frame list dari API |
| `KeyError: 'idx'` | Versi CVAT lama tidak punya field `idx` | Frame index diambil dari posisi list (sudah diperbaiki: gunakan `enumerate`) |
| `401 Unauthorized` | Credentials salah | Cek `CVAT_USERNAME` dan `CVAT_PASSWORD` di Cell 1 |
| `404 Not Found` | Task ID salah | Cek URL CVAT: `http://localhost:8080/tasks/N` → N = `CVAT_TASK_ID` |
