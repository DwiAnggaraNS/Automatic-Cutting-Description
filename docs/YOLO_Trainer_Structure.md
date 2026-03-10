# 📚 YOLO Trainer - New Structure

## 🎯 Changes Made

### 1. **Main Notebook: YOLO Trainer.ipynb**

#### Cell Structure (Correct Order):

```
1. [H1] 🚀 YOLO Trainer - Enhanced with Callbacks & Metrics
   └─ Quick start guide

2. [H2] ⚙️ Training Configuration (EDIT HERE!)
   
3. [CODE] Main Configuration
   └─ All manual parameters in ONE cell:
      - VERSION, RUNNER_NAME, HISTORY_NAME
      - TARGET_EPOCHS, BATCH_SIZE, IMG_SIZE, PATIENCE
      - Auto path detection (Colab vs Local)
      - DATASET_ROOT (edit for local!)

4. [H2] 📦 Import Libraries
5. [CODE] Basic imports (gc, torch, time)
6. [CODE] Install packages
7. [CODE] YOLO imports

8. [H2] 🔧 Custom Callbacks for YOLO Training
9. [CODE] YOLOCallbackManager class

10. [H2] 📊 Metrics Evaluator
11. [CODE] evaluate_model_metrics() - sklearn powered

12. [H2] 💾 History Saving & Management
13. [CODE] save_history(), load_history(), create_summary_from_callback()

14. [H2] 📈 Visualization & Comparison
    └─ Note: Use YOLO Visualizer.ipynb

15. [H2] 🚀 Main Training Function
16. [CODE] run_training() function - Clean & Simple!

17. [H2] ▶️ Run Training
18. [CODE] model, callback_manager, final_metrics = run_training()
```

### 2. **Separate Notebook: YOLO Visualizer.ipynb**

All visualization & comparison functions have been moved here:
- `Plotting()` - Compare 2 experiments
- `Plot_all_metrics()` - Visualize 1 experiment
- `Table_to_compare()` - Comparison table
- `Compare_multiple_experiments()` - Compare > 2 experiments
- Example usage cells

---

## 🔑 Benefits of the New Structure

### ✅ **Centralized Configuration**
- All manual parameters in **1 cell** at the top
- No need to scroll far to edit VERSION/RUNNER_NAME
- Auto-detect environment (Colab/Local)

### ✅ **Clean Training Cell**
- No complex code from the old version
- Direct integration with callbacks & metrics
- Focused on training, not benchmarking

### ✅ **Correct Order**
```
Setup Config → Libraries → Callbacks → Metrics → History → Training
```
- Training code comes **after** all helper functions
- History saving integrated within the training function

### ✅ **Separation of Concerns**
- **YOLO Trainer.ipynb**: Training only
- **YOLO Visualizer.ipynb**: Analysis & comparison
- More modular and maintainable

---

## 🚀 How to Use

### Training Workflow:

1. **Open YOLO Trainer.ipynb**

2. **Edit Cell 3 (Main Configuration):**
   ```python
   VERSION = "C_2026_1d80_10_10_AUG"  # Your dataset
   RUNNER_NAME = "YOLOv12_Medium_Experiment1"
   TARGET_EPOCHS = 150
   BATCH_SIZE = 6
   HISTORY_NAME = "history_experiment1"
   
   # ⚠️ IMPORTANT for Local:
   DATASET_ROOT = f'D:/Intern-PDU/datasets/{VERSION}'  # Adjust accordingly!
   ```

3. **Click "Run All Cells"** or run cell-by-cell

4. **Results:**
   - Model saved to: `{DATASET_ROOT}/models/{RUNNER_NAME}/`
   - History saved to: `training_history.txt` (with name: `{HISTORY_NAME}`)

### Visualization Workflow:

1. **Open YOLO Visualizer.ipynb**

2. **Scroll to "Example 1: Compare Two Experiments"**

3. **Edit experiment names:**
   ```python
   EXPERIMENT_1 = 'history_experiment1'
   EXPERIMENT_2 = 'history_experiment2'
   ```

4. **Run that cell** to view:
   - F1-Score comparison plot
   - All metrics plot (Precision, Recall, IoU)
   - Comparison table
   - Improvement summary

---

## 📁 Output Files

### Training Results:
```
{DATASET_ROOT}/models/{RUNNER_NAME}/
├── weights/
│   ├── best.pt           # Best model
│   └── last.pt           # Last epoch model
├── results.csv           # Training log from YOLO
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
**Solution:** Edit the configuration cell (cell 3) and adjust the path:
```python
DATASET_ROOT = f'D:/path/to/your/datasets/{VERSION}'
```

### ❌ Error: "Experiment not found in history file"
**Solution:** Make sure training has completed and `save_history()` was called.
Check that `HISTORY_NAME` in the config matches the name used in the visualizer.

### ❌ Training cell doesn't run
**Solution:** Make sure the preceding cells (callbacks, metrics, history) have been run first.

### ❌ "run_training() not defined"
**Solution:** Run the "Main Training Function" cell (the cell before the run training cell).

---

## 💡 Tips

1. **Use Descriptive Experiment Names:**
   ```python
   RUNNER_NAME = "YOLOv12_Medium_Baseline"
   HISTORY_NAME = "history_baseline"
   
   # Next experiment:
   RUNNER_NAME = "YOLOv12_Medium_AugmentedData"
   HISTORY_NAME = "history_augmented"
   ```

2. **Test with Small Epochs First:**
   ```python
   TARGET_EPOCHS = 5  # Test first
   # Once confirmed OK, change to:
   # TARGET_EPOCHS = 150
   ```

3. **Monitor GPU Usage:**
   - Training will print available VRAM
   - If OOM, reduce BATCH_SIZE

4. **Back Up the History File:**
   - `training_history.txt` is critical
   - Back it up periodically after each experiment

---

## 📊 Comparison Best Practices

### Valid Comparisons:

✅ **CORRECT:**
```python
# Compare with the same config, changing only 1 variable
EXPERIMENT_1 = "history_baseline"           # EPOCHS=100, BS=6
EXPERIMENT_2 = "history_different_epochs"   # EPOCHS=150, BS=6
```

❌ **INCORRECT:**
```python
# Too many variables changed at once
EXPERIMENT_1 = "baseline"        # EPOCHS=100, BS=6, IMG=960
EXPERIMENT_2 = "experiment_new"  # EPOCHS=150, BS=4, IMG=640
# Hard to tell which change had the effect!
```

### Experiment Workflow:

1. **Baseline:** Train with default config
2. **Experiment 1:** Change **1 parameter** (e.g., epochs)
3. **Compare:** Use the visualizer to compare results
4. **Experiment 2:** Adjust another parameter based on results
5. **Repeat**

---

## 🎓 Reference

- **YOLO Trainer Guide:** `YOLO_Trainer_Guide.md` (comprehensive)
- **YOLO Trainer Quick Ref:** `YOLO_Trainer_QuickRef.md` (cheat sheet)
- **Visualizer:** `YOLO Visualizer.ipynb` (analysis tool)

---

**Last Updated:** February 23, 2026  
**Version:** 2.1 (Restructured)
