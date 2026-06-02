# Color Segmentation Guide

> Guide on how to leverage the standalone color-based segmentation utility for rock instance isolation.

---

## Architecture Overview

The `color-segmentation` package provides a generalized, object-oriented pipeline utilizing OpenCV to conduct HSV thresholding and contour detection logic.

### Directory Structure
```
notebooks/color-segementation/
├── src/
│   ├── base_segmentation.py          # Core superclass with segmentation algorithm
│   ├── color_utils.py                # HSL→HSV conversion and threshold computation
│   ├── cement_segmentation.py        # Phenolphthalein-based cement detection
│   └── hydrocarbon_segmentation.py   # UV fluorescence hydrocarbon detection
├── Color_Segmentation_Demo.ipynb     # Interactive UI for live image analysis
└── Auto_Color_Thresholding.ipynb     # Diagnostic tool for threshold calibration
```

### Module Descriptions
- **`src/base_segmentation.py`**: Abstract superclass implementing the core segmentation pipeline. Handles RGB→HSV conversion, multi-range binary masking, contour-based instance detection, and area-based filtering.
- **`src/color_utils.py`**: Utility functions for color space conversion. Parses standard HSL notation from color pickers and computes optimal OpenCV HSV thresholds automatically from sample lists.
- **`src/cement_segmentation.py`**: Subclass configured for phenolphthalein indicator detection. Targets high-saturation magenta/pink hues characteristic of alkaline cement reactions.
- **`src/hydrocarbon_segmentation.py`**: Subclass calibrated for UV fluorescence detection. Targets cyan/teal hues indicative of petroleum-bearing specimens under ultraviolet illumination.

## Quick Start

### Interactive Notebook
The recommended entry point is **`Color_Segmentation_Demo.ipynb`**, which provides an interactive UI for real-time segmentation:

1. Run the notebook in Jupyter Lab or JupyterLab.
2. Upload an image (JPG or PNG) using the interactive file picker.
3. Select the analysis task from the dropdown (Cement or Hydrocarbon).
4. Click "Run Segmentation" to visualize results.

The notebook displays three side-by-side views:
- **Original Image**: Unmodified input photograph.
- **Segmented (Black Background)**: Detected regions isolated on black background.
- **Segmented Overlay**: Original image with detected regions highlighted and contours marked.

The visualization header displays the number of instances detected and the total area proportion.

### Programmatic Access
Both segmentation classes support zero-configuration execution with default calibration:

```python
import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path().absolute() / "src"))
from cement_segmentation import CementSegmentation

# Load image
img_rgb = cv2.cvtColor(cv2.imread("sample.jpg"), cv2.COLOR_BGR2RGB)

# Instantiate with default thresholds
detector = CementSegmentation()

# Run segmentation
mask, segmented, ratio, num_instances = detector.segment(img_rgb)
print(f"Detected {num_instances} cement instances covering {ratio:.2f}% of image area.")

# Display visualization (Jupyter-compatible)
detector.visualize(img_rgb)
```

### Return Values
The `segment()` method returns four values:
- **`mask`**: Binary mask (uint8, 0-255) of detected regions.
- **`segmented`**: RGB image with detected regions isolated on black background.
- **`ratio`**: Percentage of image area covered by detected regions.
- **`num_instances`**: Count of individual particles/rocks detected.

## Customization and Calibration

### Instance Detection Filtering
Instance detection relies on the `pixel_threshold` parameter, which filters contours by minimum area (in pixels):

- **Hydrocarbon Detection**: Small fluorescence spots typically require `pixel_threshold=10` to capture individual particles.
- **Cement Detection**: Microscope-magnified samples typically use `pixel_threshold=500` to focus on substantial crystalline clusters.

Adjust this parameter based on your imaging scale and target specimen size:

```python
detector = HydrocarbonSegmentation(pixel_threshold=20)
```

### Color Threshold Recalibration
When processing new field datasets with different lighting or specimen variations:

1. **Sample Representative Colors**: Open the target image in any color picker tool (Photoshop, Figma, or browser DevTools).
2. **Collect HSL Values**: Record 5-15 representative pixel colors as HSL strings, e.g., `["hsl(309, 100, 49)", "hsl(305, 100, 38)", ...]`.
3. **Analyze Thresholds**: Open **`Auto_Color_Thresholding.ipynb`** and input your color list to generate automatic OpenCV HSV bounds.
4. **Deploy**: Override the subclass configuration at instantiation:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path().absolute() / "src"))
from cement_segmentation import CementSegmentation

detector = CementSegmentation(
    colors_per_class={
        "cement_pink": [
            "hsl(309, 100, 49)",
            "hsl(305, 100, 38)",
            "hsl(313, 100, 17)"
        ]
    },
    pixel_threshold=500  # Adjust based on imaging scale
)
```

Alternatively, permanently update the `DEFAULT_COLORS` dictionary in the subclass source file.

## Creating Custom Segmentation Subclasses

To add detection for new analytes (e.g., calcite via alizarin red indicator), create a subclass inheriting from `ColorSegmentation`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path().absolute() / "src"))
from base_segmentation import ColorSegmentation

class CalciteSegmentation(ColorSegmentation):
    """
    Detects calcite-bearing cuttings stained with alizarin red indicator.
    Alizarin red produces a characteristic red/crimson color on calcite.
    """
    
    DETECTION_LABEL = "Calcite (Alizarin Red)"
    
    DEFAULT_COLORS = {
        "calcite_red": [
            "hsl(350, 100, 50)",
            "hsl(345, 95, 48)",
            "hsl(355, 98, 45)",
            # ... additional calibration samples ...
        ]
    }

# Usage
img_rgb = cv2.cvtColor(cv2.imread("sample.jpg"), cv2.COLOR_BGR2RGB)
detector = CalciteSegmentation(pixel_threshold=300)
mask, segmented, ratio, num_instances = detector.segment(img_rgb)
detector.visualize(img_rgb)
```

### Required Attributes
- **`DETECTION_LABEL`** (str): Human-readable description of the detection target.
- **`DEFAULT_COLORS`** (dict): Mapping of class names to lists of HSL color strings from calibration samples.

### Optional Constructor Parameters
- **`colors_per_class`**: Override `DEFAULT_COLORS` at runtime.
- **`pixel_threshold`**: Minimum contour area in pixels (default: 10).
- **`h_margin`, `s_margin`, `v_margin`**: Tolerance margins for HSV bounds (default: 8, 20, 25).
