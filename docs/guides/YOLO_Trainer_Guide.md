# 📘 YOLO Trainer - User Guide

## 📌 Overview

The **YOLO Trainer.ipynb** notebook is a complete YOLO training system featuring:
- ✅ **Auto Path Detection** (Colab/Local)
- ✅ **Integrated Callbacks** (EarlyStopping & ModelCheckpoint)
- ✅ **sklearn-powered Metrics** (Precision, Recall, F1, IoU)
- ✅ **History Management** (Auto-save & Comparison)
- ✅ **Ready for Run All** — No manual editing required!

---

## 🚀 Quick Start Guide

### 1️⃣ Setup Environment

**Lokal (Windows/Linux/Mac):**
```bash
pip install ultralytics scikit-learn pyyaml tqdm pandas matplotlib torch
```

**Google Colab:**
```python
# Already included in the first notebook cell
!pip install ultralytics scikit-learn pyyaml tqdm pandas matplotlib
```

### 2️⃣ Configure Dataset Path

The notebook **automatically detects** your environment!

**For Local**, adjust the path in **Cell 19** (Training Configuration):
```python
# If dataset is at D:/Intern-PDU/datasets/C_2026_1d80_10_10_AUG/
DATASET_ROOT = f'D:/Intern-PDU/datasets/{VERSION}'

# Or use a relative path
DATASET_ROOT = f'./datasets/{VERSION}'
```

**For Colab**, the path automatically points to Google Drive:
```python
DATASET_ROOT = f'/content/drive/MyDrive/Colab Notebooks/TA_CuttingRockDescription/datasets/{VERSION}'
```

### 3️⃣ Dataset Folder Structure

Make sure your dataset has this structure:

```
D:/Intern-PDU/datasets/C_2026_1d80_10_10_AUG/
├── data.yaml                    # Dataset configuration
├── train/
│   ├── images/                  # Training images
│   └── labels/                  # Training labels (YOLO format)
├── val/
│   ├── images/                  # Validation images
│   └── labels/                  # Validation labels
└── test/
    ├── images/                  # Test images
    └── labels/                  # Test labels (optional)
```

**Example `data.yaml` file:**
```yaml
path: D:/Intern-PDU/datasets/C_2026_1d80_10_10_AUG  # or your path
train: train/images
val: val/images
test: test/images  # optional

nc: 4  # number of classes
names: ['class1', 'class2', 'class3', 'class4']  # class names
```

### 4️⃣ Run Training

**Method 1: Run All Cells** (Recommended!)
1. Open the notebook
2. Click **Cell > Run All**
3. Wait for completion ☕

**Method 2: Run cell by cell**
1. Run cells 1–18: Setup & Helper Functions
2. Run cell 19: Training Configuration & `run_yolo()`
3. Done! ✅

---

## ⚙️ Training Configuration

Edit the configuration in **Cell 19** before training:

```python
# Dataset Version
VERSION = "C_2026_1d80_10_10_AUG"  # Your dataset folder name

# Training Hyperparameters
TARGET_EPOCHS = 150        # Number of epochs (set to 5 for a quick test)
IMG_SIZE = 960             # Input image size
BATCH_SIZE = 6             # Batch size (adjust based on GPU VRAM)
PATIENCE = 50              # Early stopping patience
RUNNER_NAME = "YOLOv12_Medium"  # Experiment name (used for history saving)
SINGLE_CLASS = False       # Set True for single-class detection

# Model Configuration
YOLO_MODEL_URL = "https://github.com/sunsmarterjie/yolov12/releases/download/seg/yolov12m-seg.pt"
YOLO_MODEL = "yolov12m-seg.pt"  # Model to download

# Testing
SMOKE_TEST = False  # Set True for a quick test (2 epochs)
```

---

## 📊 Monitoring Training

### Real-time Monitoring

Training will display:
- ✅ Epoch progress & metrics
- ✅ Early stopping status
- ✅ Model checkpoint notifications
- ✅ GPU memory usage

```
✅ Epoch 10: metrics/mAP50-95(M) improved to 0.7234
⏳ Epoch 11: metrics/mAP50-95(M) did not improve (0.7199 vs best 0.7234), patience 1/50
```

### Post-Training Evaluation

After training, evaluation runs automatically:
```
============================================================
📊 EVALUATION RESULTS (sklearn-powered)
============================================================
  Precision:     85.23%
  Recall:        82.15%
  F1-Score:      83.66%
  Avg IoU:       75.44%
  Mask mAP50:    78.92%
  Mask mAP50-95: 65.33%
============================================================
```

---

## 💾 History Management

### Auto-Save History

History is automatically saved to `training_history.txt` when training completes:

```python
# Format in the file:
# Saved: 2026-02-23 14:30:00
history_YOLOv12_Medium_C_2026_1d80_10_10_AUG = {
    'train_loss': [0.523, 0.412, 0.351, ...],
    'val_f1': [0.723, 0.789, 0.836, ...],
    'final_precision': 0.8523,
    'final_recall': 0.8215,
    'final_f1': 0.8366,
    'final_iou': 0.7544
}
```

### Load History

```python
# Load a single experiment
history_1 = load_history('history_YOLOv12_Medium_C_2026_1d80_10_10_AUG')

# Load all experiments
all_histories = load_all_histories()
```

---

## 📈 Comparison & Visualization

### Compare 2 Experiments

Run the **"Comparison & Plotting"** cell (after training):

```python
# Konfigurasi
EXPERIMENT_1 = 'history_baseline'
EXPERIMENT_2 = 'history_augmented'

# Load & Compare
history_1 = load_history(EXPERIMENT_1)
history_2 = load_history(EXPERIMENT_2)

# Plot comparison
Plotting(history_1, history_2, metric='val_f1', 
         labels=[EXPERIMENT_1, EXPERIMENT_2])

# Table comparison
Table_to_compare(history_1['final_f1'], history_2['final_f1'], 'F1-Score')
```

**Output:**
```
============================================================
📊 COMPARISON TABLE - F1-Score
============================================================
  Experiment       F1-Score
  Baseline         0.8366
  Current          0.8789
  Improvement      +0.0423 (+5.05%)
============================================================
✅ IMPROVEMENT: Model shows 5.05% increase in F1-Score
============================================================
```

### Compare Multiple Experiments

Run the **"Compare Multiple Experiments"** cell:

```python
# Load all & compare
all_histories = load_all_histories()

# Auto-generate comparison for all metrics
# Output: Bar charts & summary table
```

---

## 🎯 Evaluation Metrics

The notebook uses **sklearn** to compute metrics:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Precision** | TP / (TP + FP) | How accurate the model's predictions are |
| **Recall** | TP / (TP + FN) | How many ground-truth objects are detected |
| **F1-Score** | 2·(P·R)/(P+R) | Balance between precision & recall |
| **Avg IoU** | ∑IoU / N | Mask segmentation quality |
| **mAP50** | Mean AP @ IoU=0.5 | Detection performance (lenient) |
| **mAP50-95** | Mean AP @ IoU=0.5:0.95 | Detection performance (strict) |

---

## 🛠️ Troubleshooting

### Error: Dataset path not found

**Problem:** `FileNotFoundError: [Errno 2] No such file or directory`

**Solution:**
1. Check the path in Cell 19 (Training Configuration)
2. Make sure the path matches the actual dataset location
3. Use an absolute path to avoid errors

```python
# ✅ Correct (absolute)
DATASET_ROOT = 'D:/Intern-PDU/datasets/C_2026_1d80_10_10_AUG'

# ❌ Incorrect (path doesn't exist)
DATASET_ROOT = '/content/drive/MyDrive/...'  # When running locally
```

### Error: CUDA out of memory

**Problem:** `RuntimeError: CUDA out of memory`

**Solution:**
1. Reduce `BATCH_SIZE` in Cell 19:
```python
BATCH_SIZE = 4  # or 2 for smaller GPUs
```

2. Or reduce `IMG_SIZE`:
```python
IMG_SIZE = 640  # from 960
```

### History file not found

**Problem:** `❌ History file not found: training_history.txt`

**Solution:**
1. Run training first
2. The history file will be created automatically after training completes
3. The file is saved in the current working directory

### Warning: Experiment not found

**Problem:** `❌ Experiment 'history_xyz' not found in history file`

**Solution:**
1. Check available experiment names with:
```python
all_histories = load_all_histories()
# Prints all available experiment names
```

2. Use the correct experiment name (case-sensitive!)

---

## 💡 Best Practices

### 1. Penamaan Eksperimen

Gunakan nama deskriptif untuk `RUNNER_NAME`:

```python
# ✅ Good
RUNNER_NAME = "YOLOv12_lr0001_aug_heavy"
RUNNER_NAME = "YOLOv11_baseline_960px"

# ❌ Bad
RUNNER_NAME = "test1"
RUNNER_NAME = "model"
```

### 2. Hyperparameter Tuning

Eksperimen secara bertahap:

```python
# Experiment 1: Baseline
TARGET_EPOCHS = 150
BATCH_SIZE = 6
IMG_SIZE = 960

# Experiment 2: Lower Learning Rate
# (set in YOLO .train() parameters)

# Experiment 3: Heavy Augmentation
# (modify data.yaml augmentation settings)
```

### 3. Early Stopping

Sesuaikan `PATIENCE` dengan ukuran dataset:

```python
# Dataset kecil (<1000 images)
PATIENCE = 20

# Dataset medium (1000-5000 images)
PATIENCE = 50

# Dataset besar (>5000 images)
PATIENCE = 100
```

### 4. GPU Memory Management

```python
# Clear cache sebelum training
clear_cuda_cache()

# Monitoring GPU usage
# Training akan print: "GPU mem: 8.5GB"
```

---

## 📂 Output Files

Setelah training selesai, output tersimpan di:

```
models/YOLOv12_Medium/
├── weights/
│   ├── best.pt              # Best model checkpoint
│   └── last.pt              # Last epoch checkpoint
├── results.csv              # Training metrics per epoch
├── training_log_rich.json   # Enhanced training log
├── final_thesis_results.json # Final benchmark results
└── visual_predictions/      # Sample predictions
    ├── pred_0.jpg
    ├── pred_1.jpg
    └── pred_2.jpg
```

**Plus:**
- `training_history.txt` - History semua eksperimen (di root folder)

---

## 🔗 Integration dengan Workflow Lain

### Export untuk Detectron2 Comparison

```python
# Load YOLO results
history_yolo = load_history('history_YOLOv12_Medium_...')

# Format untuk comparison table
yolo_results = {
    'Model': 'YOLO v12',
    'F1-Score': history_yolo['final_f1'],
    'Precision': history_yolo['final_precision'],
    'Recall': history_yolo['final_recall'],
    'IoU': history_yolo['final_iou']
}
```

### Export to CSV/Excel

```python
import pandas as pd

# Load all histories
all_histories = load_all_histories()

# Convert to DataFrame
data = []
for name, hist in all_histories.items():
    data.append({
        'Experiment': name,
        'F1': hist.get('final_f1', 0),
        'Precision': hist.get('final_precision', 0),
        'Recall': hist.get('final_recall', 0),
        'IoU': hist.get('final_iou', 0)
    })

df = pd.DataFrame(data)
df.to_csv('yolo_experiments_summary.csv', index=False)
df.to_excel('yolo_experiments_summary.xlsx', index=False)
```

---

## 📞 Support & Contact

Untuk pertanyaan atau issue:
1. Check troubleshooting section di atas
2. Review cell markdown di notebook untuk detail
3. Contact: PT PDU AI Team / [Your Contact]

---

## 📝 Changelog

### Version 2.0 (2026-02-23)
- ✅ Auto path detection (Colab/Local)
- ✅ Integrated callbacks (no manual edit needed)
- ✅ sklearn-powered metrics (efficient & reliable)
- ✅ Ready for Run All
- ✅ Complete documentation

### Version 1.0 (Initial)
- Basic YOLO training
- Manual callback integration
- Custom metrics implementation

---

## 🏆 Credits

- **Ultralytics YOLO**: https://github.com/ultralytics/ultralytics
- **scikit-learn**: https://scikit-learn.org
- **Reference**: U-Net Lunar segmentation notebook (Kaggle)

---

**Happy Training! 🚀**
