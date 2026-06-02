"""
hydrocarbon_segmentation.py
============================
Detects oil-bearing (hydrocarbon) drilling cuttings under UV fluorescence.

Physical principle
------------------
Petroleum and most of its products emit fluorescence under UV irradiation
(polycyclic aromatic hydrocarbons cause luminescence; saturated hydrocarbons
do not). The emitted glow appears cyan/teal for this field dataset.

Reference: Huo et al., J. Petrol. Sci. Eng. 205 (2021) 108898. §3.1.

Usage
-----
    from hydrocarbon_segmentation import HydrocarbonSegmentation

    detector = HydrocarbonSegmentation()
    mask, segmented, ratio = detector.segment(image_rgb)
    detector.visualize(image_rgb)

Recalibrate for a new dataset
------------------------------
    1. Sample 5–15 pixels from UV fluorescence zones with a color picker.
    2. Collect HSL strings.
    3. Pass them in:

        detector = HydrocarbonSegmentation(
            colors_per_class={
                "hydrocarbon_cyan": ["hsl(185, 80, 88)", "hsl(193, 100, 82)", ...]
            }
        )

    Or permanently update DEFAULT_COLORS below.
"""

from base_segmentation import ColorSegmentation


class HydrocarbonSegmentation(ColorSegmentation):

    DETECTION_LABEL = "Hydrocarbon (UV Fluorescence)"

    # -----------------------------------------------------------------------
    # Color samples collected from personal field UV dataset.
    # Sampled with a color picker directly on UV fluorescence images.
    # Observed OpenCV HSV range (computed automatically from these samples):
    #   H = 77–101  (green-cyan → blue-cyan)
    #   S = 20–126
    #   V = 194–255
    # With default margins (h±8, s±20, v±25) → lower=[69,0,169] upper=[109,146,255]
    # -----------------------------------------------------------------------
    DEFAULT_COLORS = {
        "hydrocarbon_cyan": [
            "hsl(167, 49, 87)",   # H_cv=84  S=35  V=238
            "hsl(162, 53, 92)",   # H_cv=81  S=24  V=245
            "hsl(193, 100, 90)",  # H_cv=96  S=52  V=255
            "hsl(200, 100, 86)",  # H_cv=100 S=70  V=255
            "hsl(197, 100, 90)",  # H_cv=99  S=52  V=255
            "hsl(153, 83, 95)",   # H_cv=77  S=20  V=253
            "hsl(189, 100, 96)",  # H_cv=94  S=21  V=255
            "hsl(194, 31, 68)",   # H_cv=97  S=66  V=198
            "hsl(189, 44, 57)",   # H_cv=94  S=126 V=194
            "hsl(203, 70, 76)",   # H_cv=101 S=91  V=237
        ]
    }

    # -----------------------------------------------------------------------
    # Known non-target colors for documentation/validation:
    #   Non-HC dark blue rock : H_cv=118-119, V=94-143  (excluded by V < 169)
    #   Tray reflection       : H_cv=119-130             (excluded by H > 109)
    # -----------------------------------------------------------------------
