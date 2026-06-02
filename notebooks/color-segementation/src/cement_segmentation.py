"""
cement_segmentation.py
=======================
Detects cement rock cuttings stained with phenolphthalein indicator.

Physical principle
------------------
Phenolphthalein is a pH indicator: colourless below pH ≈ 8.2, vivid
pink/magenta above it. Cement paste is strongly alkaline (pH 12–13) so it
turns vivid magenta. Non-cementitious lithologies (shale, silt, coal, sand,
limestone, etc.) are near-neutral and remain unchanged.

Image type: white-light photograph AFTER applying phenolphthalein solution.
(This is NOT a UV image — do not use UV photos here.)

Usage
-----
    from cement_segmentation import CementSegmentation

    detector = CementSegmentation()
    mask, segmented, ratio = detector.segment(image_rgb)
    detector.visualize(image_rgb)

Recalibrate for a new dataset
------------------------------
    1. Apply phenolphthalein, photograph under white light.
    2. Sample 5–15 pixels from clearly stained zones with a color picker.
    3. Pass them in:

        detector = CementSegmentation(
            colors_per_class={
                "cement_pink": ["hsl(308, 100, 45)", "hsl(312, 95, 30)", ...]
            }
        )

Notes on the pale-pink sample (hsl(305, 90, 92))
-------------------------------------------------
This sample (H_cv=152, S=37, V=253) is excluded from DEFAULT_COLORS because
its S=37 is dangerously close to the non-cement lavender (hsl(310, 23, 81)
→ S=27). Including it would force S_lower down to ~17, creating false
positives on light-coloured non-cement minerals.

In practice the pale pink appears only on very thinly stained or edge
cement particles. If your field images show predominantly pale staining
(weakly alkaline cement / old/carbonated cement), reduce the default
s_margin to 3 and add the pale sample explicitly:

    detector = CementSegmentation(
        colors_per_class={
            "cement_pink_vivid": [...default list...],
            "cement_pink_pale":  ["hsl(305, 90, 92)"],
        },
        s_margin=3,
    )
"""

from base_segmentation import ColorSegmentation


class CementSegmentation(ColorSegmentation):

    DETECTION_LABEL = "Cement (Phenolphthalein)"

    # -----------------------------------------------------------------------
    # Color samples from phenolphthalein-stained cement specimens.
    # Sampled under white light with a color picker.
    # Observed OpenCV HSV range (auto-computed from these samples):
    #   H = 149–159  (magenta–pink)
    #   S = 233–255  (highly saturated — key discriminator vs non-cement)
    #   V = 87–250   (dark through bright; very dark = thick/deep staining)
    # With default margins (h±8, s±20, v±25) → lower=[141,213,62] upper=[167,255,255]
    #
    # Non-cement colours and why they are excluded:
    #   hsl(310, 23, 81) → H_cv=155, S=27  — lavender, excluded by S < 213 ✓
    #   hsl(340,  8, 79) → H_cv=170, S=11  — pinkish-grey, excluded by H > 167 ✓
    #   hsl( 42, 13, 60) → H_cv=21,  S=40  — tan/beige, excluded by H < 141 ✓
    #   hsl(  0,  0, 83) → H_cv=0,   S=0   — white/grey, excluded by S ✓
    #   hsl(240,  2, 45) → H_cv=120, S=11  — dark grey, excluded by H and S ✓
    # -----------------------------------------------------------------------
    DEFAULT_COLORS = {
        "cement_pink": [
            "hsl(309, 100, 49)",  # H_cv=155 S=255 V=250  — medium bright magenta
            "hsl(305, 100, 38)",  # H_cv=152 S=255 V=194  — medium magenta
            "hsl(313, 100, 17)",  # H_cv=157 S=255 V= 87  — very dark magenta
            "hsl(306, 100, 33)",  # H_cv=153 S=255 V=168  — medium dark magenta
            "hsl(305, 100, 40)",  # H_cv=153 S=255 V=204  — medium magenta
            "hsl(298,  84, 35)",  # H_cv=149 S=233 V=164  — purple-magenta
            "hsl(309, 100, 28)",  # H_cv=155 S=255 V=143  — dark magenta
            "hsl(318, 100, 35)",  # H_cv=159 S=255 V=178  — pink-magenta
        ]
    }
