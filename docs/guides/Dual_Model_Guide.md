# Experimental Approaches: Single Model vs. Dual Model

This project implements two distinct experimental approaches for the automated classification and segmentation of rock cuttings:

1. **Single Model (End-to-End YOLO)**
2. **Dual Model (Segmentor + Expert Classifier)**

---

## 1. Single Model Approach

In this traditional approach, a single instance segmentation model (YOLOv26-seg) is responsible for both localizing (segmenting boundaries) and classifying the rock types in one pass.

- **Workflow:** 
  Directly train a multi-class YOLO model using the multi-class dataset.
- **Notebook:** 
  `notebooks/training/YOLO_Trainer.ipynb`
- **Pros:** 
  Fast inference, easier to deploy, single training loop.
- **Cons:** 
  Can sometimes struggle with fine-grained classification on small patches because the model tries to balance both segmentation precision and classification accuracy simultaneously.

---

## 2. Dual Model Approach

To improve classification accuracy, this experimental approach breaks the problem into a two-stage pipeline. 

### Stage 1: Segmentor (Rock Finder)
The first model is trained purely to detect rock instances, ignoring their specific classes (treating everything as a generic "rock"). This allows the model to specialize in producing highly accurate segmentation masks.
- **Architecture:** YOLOv12m-seg (or similar)
- **Dataset:** Single-class YOLO dataset.

### Stage 2: Classifier (Expert Classifier)
The second model focuses entirely on classifying cropped images of individual rocks.
- **Architecture:** Image Classification models like YOLO-cls, EfficientNetV2, or DaViT.
- **Dataset:** Image classification dataset (extracted crops in `train/`, `val/`, `test/` folders).

---

## How to Run the Dual Model Pipeline

To successfully train and evaluate the Dual Model architecture, you must prepare the data for both stages independently before running the trainer.

### Step 1: Prepare Stage 1 Dataset (Segmentor)
Convert your existing multi-class YOLO dataset into a single-class dataset (all objects relabeled to class 0 - "rock").

```bash
python scripts/data_preprocessing/convert_to_single_class_yolo.py
```
- **Source:** Your multi-class YOLO dataset.
- **Destination:** E.g., `datasets/batch4_single_class/`
- This dataset will be used to train the YOLO Segmentor.

### Step 2: Prepare Stage 2 Dataset (Classifier)
Extract individual rock crops from the multi-class YOLO dataset using the original bounding boxes/polygons. This script organizes crops into an `ImageFolder` structure (`train/ClassA/..`, `val/ClassA/..`) while strictly preventing data leakage across splits.

```bash
python scripts/data_preprocessing/extract_classifier_crops.py
```
- **Source:** Your multi-class YOLO dataset.
- **Destination:** E.g., `datasets/classifier_crops/`
- This dataset will be used to train the classification models (YOLO-cls, EfficientNet, etc.).

### Step 3: Train the Models
Open and run all cells in the Dual Model training notebook:

- **Notebook:** `notebooks/training/Dual_Model_Trainer.ipynb`

This notebook will:
1. Train the YOLO Segmentor on the single-class dataset.
2. Train multiple Classifier architectures (YOLOv26-cls, EfficientNetV2, DaViT, etc.) on the cropped dataset.
3. Compare the evaluation metrics (Accuracy, F1-Score) across all classifiers.
4. Execute an end-to-end integration test combining the Segmentor and chosen Classifier.

### Step 4: Inference
For production or integration inference, use the `DualModelPipeline` from `src/inference_dual_model.py`. 
It natively:
1. Runs the YOLO Segmentor.
2. Passes raw outputs through a `MaskPostProcessor` to clean and simplify polygon edges.
3. Generates precise bounding boxes from the cleaned masks.
4. Crops the original image and feeds it to the chosen PyTorch/Timm/YOLO Classifier.
5. Returns composite unified results.