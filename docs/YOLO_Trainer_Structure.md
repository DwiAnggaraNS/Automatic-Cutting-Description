# 📚 YOLO Trainer - Struktur Baru

## 🎯 Perubahan yang Dilakukan

### 1. **Notebook Utama: YOLO Trainer.ipynb**

#### Struktur Cell (Urutan yang Benar):

```
1. [H1] 🚀 YOLO Trainer - Enhanced with Callbacks & Metrics
   └─ Quick start guide

2. [H2] ⚙️ Konfigurasi Training (EDIT DI SINI!)
   
3. [CODE] Konfigurasi Utama
   └─ Semua parameter manual dalam SATU cell:
      - VERSION, RUNNER_NAME, HISTORY_NAME
      - TARGET_EPOCHS, BATCH_SIZE, IMG_SIZE, PATIENCE
      - Auto path detection (Colab vs Local)
      - DATASET_ROOT (edit untuk local!)

4. [H2] 📦 Import Libraries
5. [CODE] Basic imports (gc, torch, time)
6. [CODE] Install packages
7. [CODE] YOLO imports

8. [H2] 🔧 Custom Callbacks untuk YOLO Training
9. [CODE] YOLOCallbackManager class

10. [H2] 📊 Metrics Evaluator
11. [CODE] evaluate_model_metrics() - sklearn powered

12. [H2] 💾 History Saving & Management
13. [CODE] save_history(), load_history(), create_summary_from_callback()

14. [H2] 📈 Visualization & Comparison
    └─ Catatan: Gunakan YOLO Visualizer.ipynb

15. [H2] 🚀 Main Training Function
16. [CODE] run_training() function - Clean & Simple!

17. [H2] ▶️ Run Training
18. [CODE] model, callback_manager, final_metrics = run_training()
```

### 2. **Notebook Terpisah: YOLO Visualizer.ipynb**

Semua fungsi visualization & comparison dipindahkan ke sini:
- `Plotting()` - Compare 2 experiments
- `Plot_all_metrics()` - Visualize 1 experiment
- `Table_to_compare()` - Comparison table
- `Compare_multiple_experiments()` - Compare > 2 experiments
- Example usage cells

---

## 🔑 Keuntungan Struktur Baru

### ✅ **Konfigurasi Terpusat**
- Semua parameter manual dalam **1 cell** di awal
- Tidak perlu scroll jauh untuk edit VERSION/RUNNER_NAME
- Auto-detect environment (Colab/Local)

### ✅ **Cell Training yang Bersih**
- Tidak ada kode kompleks dari versi lama
- Integrasi langsung dengan callbacks & metrics
- Fokus pada training, bukan benchmarking

### ✅ **Urutan yang Benar**
```
Setup Config → Libraries → Callbacks → Metrics → History → Training
```
- Training code **setelah** semua helper functions
- History saving terintegrasi dalam training function

### ✅ **Pemisahan Concerns**
- **YOLO Trainer.ipynb**: Training only
- **YOLO Visualizer.ipynb**: Analysis & comparison
- Lebih modular dan maintainable

---

## 🚀 Cara Pakai

### Training Workflow:

1. **Buka YOLO Trainer.ipynb**

2. **Edit Cell ke-3 (Konfigurasi Utama):**
   ```python
   VERSION = "C_2026_1d80_10_10_AUG"  # Dataset Anda
   RUNNER_NAME = "YOLOv12_Medium_Experiment1"
   TARGET_EPOCHS = 150
   BATCH_SIZE = 6
   HISTORY_NAME = "history_experiment1"
   
   # ⚠️ PENTING untuk Local:
   DATASET_ROOT = f'D:/Intern-PDU/datasets/{VERSION}'  # Sesuaikan!
   ```

3. **Klik "Run All Cells"** atau run cell-by-cell

4. **Hasil:**
   - Model tersimpan: `{DATASET_ROOT}/models/{RUNNER_NAME}/`
   - History tersimpan: `training_history.txt` (dengan nama: `{HISTORY_NAME}`)

### Visualization Workflow:

1. **Buka YOLO Visualizer.ipynb**

2. **Scroll ke "Example 1: Compare Two Experiments"**

3. **Edit nama eksperimen:**
   ```python
   EXPERIMENT_1 = 'history_experiment1'
   EXPERIMENT_2 = 'history_experiment2'
   ```

4. **Run cell tersebut** untuk melihat:
   - Plot perbandingan F1-Score
   - Plot semua metrik (Precision, Recall, IoU)
   - Tabel perbandingan
   - Improvement summary

---

## 📁 File Output

### Training Results:
```
{DATASET_ROOT}/models/{RUNNER_NAME}/
├── weights/
│   ├── best.pt           # Model terbaik
│   └── last.pt           # Model epoch terakhir
├── results.csv           # Training log dari YOLO
└── confusion_matrix.png  # Confusion matrix (auto from YOLO)
```

### History File:
```
training_history.txt

Format:
# YOLO Training History Log
# Created: 2026-02-23 10:30:00

history_experiment1 = {'train_loss': [...], 'val_f1': [...], 'final_f1': 0.85, ...}

# Saved: 2026-02-23 11:45:00
history_experiment2 = {'train_loss': [...], 'val_f1': [...], 'final_f1': 0.88, ...}
```

---

## 🔧 Troubleshooting

### ❌ Error: "DATASET_ROOT not found"
**Solusi:** Edit cell konfigurasi (cell ke-3), sesuaikan path:
```python
DATASET_ROOT = f'D:/path/to/your/datasets/{VERSION}'
```

### ❌ Error: "Experiment not found in history file"
**Solusi:** Pastikan training sudah selesai dan `save_history()` dipanggil.
Check nama `HISTORY_NAME` di config dan di visualizer.

### ❌ Cell training tidak jalan
**Solusi:** Pastikan cell-cell sebelumnya (callbacks, metrics, history) sudah di-run terlebih dahulu.

### ❌ "run_training() not defined"
**Solusi:** Run cell "Main Training Function" (cell sebelum cell run training).

---

## 💡 Tips

1. **Nama Eksperimen Deskriptif:**
   ```python
   RUNNER_NAME = "YOLOv12_Medium_Baseline"
   HISTORY_NAME = "history_baseline"
   
   # Next experiment:
   RUNNER_NAME = "YOLOv12_Medium_AugmentedData"
   HISTORY_NAME = "history_augmented"
   ```

2. **Test dengan Epochs Kecil:**
   ```python
   TARGET_EPOCHS = 5  # Test dulu
   # Setelah OK, ubah ke:
   # TARGET_EPOCHS = 150
   ```

3. **Monitor GPU Usage:**
   - Training akan print VRAM available
   - Kalau OOM, turunkan BATCH_SIZE

4. **Backup History File:**
   - `training_history.txt` sangat penting
   - Backup secara berkala setelah eksperimen

---

## 📊 Comparison Best Practices

### Perbandingan yang Valid:

✅ **BENAR:**
```python
# Bandingkan dengan config yang sama, hanya ubah 1 variable
EXPERIMENT_1 = "history_baseline"           # EPOCHS=100, BS=6
EXPERIMENT_2 = "history_different_epochs"   # EPOCHS=150, BS=6
```

❌ **SALAH:**
```python
# Terlalu banyak variable berubah
EXPERIMENT_1 = "baseline"        # EPOCHS=100, BS=6, IMG=960
EXPERIMENT_2 = "experiment_new"  # EPOCHS=150, BS=4, IMG=640
# Sulit mengetahui mana yang berpengaruh!
```

### Workflow Eksperimen:

1. **Baseline:** Train dengan config default
2. **Experiment 1:** Ubah **1 parameter** (misal: epochs)
3. **Compare:** Gunakan visualizer untuk bandingkan
4. **Experiment 2:** Ubah parameter lain berdasarkan hasil
5. **Repeat**

---

## 🎓 Reference

- **YOLO Trainer Guide:** `YOLO_Trainer_Guide.md` (comprehensive)
- **YOLO Trainer Quick Ref:** `YOLO_Trainer_QuickRef.md` (cheat sheet)
- **Visualizer:** `YOLO Visualizer.ipynb` (analysis tool)

---

**Last Updated:** February 23, 2026  
**Version:** 2.1 (Restructured)
