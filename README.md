# Automatic Cutting Description

> AI-powered rock cutting description system using YOLO instance segmentation and SAM-assisted annotation.

---

## Overview

This project automates the classification and segmentation of rock cutting samples from drilling operations. It leverages **YOLOv12** for instance segmentation and **CVAT + SAM** for semi-automated annotation workflows.

### Key Features
- **Instance Segmentation** — Multi-class rock type detection using YOLOv12
- **Dual Model Pipeline** — Isolated training of Object Boundary Segmentors and fine-grained PyTorch/Timm Expert Classifiers
- **SAM Integration** — Semi-automated annotation via CVAT + Segment Anything Model
- **Custom Callbacks** — Early stopping and model checkpoint management
- **Minority Class Augmentation** — Synthetic data generation for imbalanced datasets
- **Comprehensive Metrics** — Precision, Recall, F1, IoU evaluation via scikit-learn

---

## 🔬 Experimental Modeling Approaches

This project implements and compares two primary architectures for rock classification and segmentation:

### 1. Single Model (Unified YOLO)
The standard setup where a single YOLOv12 model handles both the localization (segmentation masks) and the classification (rock type) simultaneously.
- **Dataset:** Standard multi-class YOLO segmentation format (all 6 rock classes).
- **Notebook:** `notebooks/training/YOLO_Trainer.ipynb`
- **Output:** End-to-end multi-class bounding box & polygon prediction.

### 2. Dual Model (Two-Stage: Segmentor + Classifier)
A novel decoupled pipeline separating "finding rocks" from "classifying rock types". By focusing YOLO strictly on boundaries, we can leverage powerful Image Classification architectures (EfficientNetV2, ConvNeXt, DaViT) exclusively on the textured rock crops, leading to potentially higher F1 scores.

#### Data Pipeline & Workflow for Dual Model:

**Step 1: Segmentor Dataset Preparation**
We convert the multi-class YOLO dataset into a generalized single-class dataset (all rock types are re-labelled as `0: 'rock'`). This focuses the Segmentor entirely on outlining boundaries regardless of texture.
```bash
python scripts/data_preprocessing/convert_to_single_class_yolo.py
```
> *Output: A clean YOLO dataset where every object is simply localized.*

**Step 2: Classifier Dataset Preparation**
We extract image *crops* directly from the original multi-class YOLO dataset's bounding boxes. This script creates a standard PyTorch `ImageFolder` structure (e.g., `/train/Limestone/`, `/train/Quartz/`) maintaining strict train/val/test splits to avoid data leakage.
```bash
python scripts/data_preprocessing/extract_classifier_crops.py
```
> *Output: Thousands of localized rock patches cleanly categorized into folders.*

**Step 3: Training & Integration (The Pipeline)**
Train the Stage-1 general YOLO segmentor on the Step 1 data, and train the Stage-2 specialized Classifiers on the Step 2 data. 
- **Notebook:** `notebooks/training/Dual_Model_Trainer.ipynb`
- **Inference Integration:** This notebook also executes the complete prediction pipeline (`src/inference_dual_model.py`) where YOLO's rough predicted polygons are smoothed via a `MaskPostProcessor`, bounding boxes are recalculated, extracted securely, and fed into the chosen PyTorch Classifier.

---

## Project Structure

```
automatic-cutting-description/
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/                        # Training & model configuration
│   └── training_config.yaml
│
├── docs/                           # Documentation
│   ├── guides/
│   │   ├── CVAT_SAM_Installation_Guide.md
│   │   └── YOLO_Trainer_Guide.md
│   ├── reports/
│   │   └── CVAT_SAM_Debugging_Report.pdf
│   └── YOLO_Trainer_Structure.md
│
├── models/                         # Saved model weights (gitignored)
│
├── notebooks/                      # Jupyter Notebooks
│   ├── training/
│   │   ├── YOLO_Trainer.ipynb           # Main training notebook
│   │   └── YOLO_Trainer_Original.ipynb  # Reference/baseline notebook
│   └── evaluation/
│       ├── Independent_Evaluator.ipynb  # Model evaluation & metrics
│       └── Interactive_Inference.ipynb  # Single image test inference & mask exploration
│   └── exploration/
│       └── YOLO_Visualizer.ipynb        # Training visualization & comparison
│
├── datasets/                       # Dataset processing and SAM Annotation
│   ├── sam-annotation/
│   │   ├── 01_sam_auto_segmentation.ipynb # SAM mask generation pipeline
│   │   └── 02_cvat_converter.ipynb        # CVAT XML converter
│
├── scripts/                        # Utility scripts
│   ├── data_preprocessing/
│   │   ├── coco_polygon_simplification.py
│   │   ├── convert_coco_to_yolo.py
│   │   ├── convert_yolo_to_coco.py
│   │   ├── convert_to_single_class_yolo.py  # Stage 1 Dataset Prep
│   │   ├── extract_classifier_crops.py      # Stage 2 Dataset Prep
│   │   ├── merge_cvat_datasets.py
│   │   ├── redistribute_dataset.py
│   │   ├── remap_coco_categories.py
│   │   └── slice_4k_datasets.py
│   ├── data_analysis/
│   │   ├── get_statistics_data.py
│   │   ├── minority_class_extractions.py
│   │   └── minority_class_generator.py
│   └── deployment/
│       ├── cvat-start.sh
│       └── cvat-stop.sh
│
└── src/                            # Core source code
    └── inference.py                # Inference pipeline
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Training

Edit `configs/training_config.yaml` or set parameters directly in the notebook:

```yaml
version: "C_2026_1d80_10_10_AUG"
runner_name: "YOLOv12m_RG_Latest"
target_epochs: 150
batch_size: 6
img_size: 960
patience: 50
model: "yolov12m-seg.pt"
```

### 3. Run Training

Open `notebooks/training/YOLO_Trainer.ipynb` and click **Run All Cells**.

---

## Data Pipeline

```
[Optional] Legacy YOLO Dataset
        ↓
scripts/data_preprocessing/convert_yolo_to_coco.py   (Convert backward to COCO)
        |
Raw Images + CVAT Annotation (Separated Tasks) + Converted COCO
        ↓
scripts/data_preprocessing/remap_coco_categories.py  (Standardize class names and IDs across datasets)
        ↓
scripts/data_preprocessing/merge_cvat_datasets.py    (Merge separated CVAT COCO datasets into one unified dataset)
        ↓
scripts/data_preprocessing/redistribute_dataset.py   (Multi-label Stratified train/val/test split on Unified COCO)
        ↓
scripts/data_preprocessing/slice_4k_datasets.py      (Slice ONLY Train & Val 4K images + polygons to 960x960 using SAHI)
        ↓
scripts/data_preprocessing/convert_coco_to_yolo.py   (Convert COCO Splits → YOLO format)
        ↓
scripts/data_analysis/get_statistics_data.py          (class balance check)
        ↓
scripts/data_analysis/minority_class_extractions.py  (extract minority classes)
        ↓
scripts/data_analysis/minority_class_generator.py    (synthetic augmentation)
        ↓
notebooks/training/Dual_Model_Trainer.ipynb          (Stage 1 Seg & Stage 2 Cls)
        ↓
        ...
        ↓
notebooks/evaluation/Interactive_Inference.ipynb     (visual & post-processing UI tests)
```

---

## Rock Classes

| ID | Class Name               | Category  |
|----|--------------------------|-----------|
| 1  | Silt                     | Clastic   |
| 2  | Sandstone                | Clastic   |
| 3  | Limestone                | Carbonate |
| 4  | Coal                     | Organic   |
| 5  | Shalestone               | Clastic   |
| 6  | Quartz                   | Mineral   |
| 7  | Cement                   | Artificial|

---

## Model

- **Architecture:** YOLOv12m-seg (instance segmentation)
- **Input Size:** 960×960
- **Task:** Multi-class instance segmentation

---

## Documentation

| Document | Description |
|----------|-------------|
| [Semi-automated annotation via CVAT + Segment Anything Model Guide](docs/guides/sam_autoannotation.md) | Environment Setup, Configuration, and Annotation Workflow |
| [YOLO Trainer Guide](docs/guides/YOLO_Trainer_Guide.md) | Training workflow & configuration |
| [CVAT + SAM Installation Guide](docs/guides/CVAT_SAM_Installation_Guide.md) | Annotation toolchain setup |
| [YOLO Trainer Structure](docs/YOLO_Trainer_Structure.md) | Notebook architecture reference |

---

## Requirements

- Python ≥ 3.9
- CUDA-capable GPU (≥ 8GB VRAM recommended)
- See `requirements.txt` for full dependency list
