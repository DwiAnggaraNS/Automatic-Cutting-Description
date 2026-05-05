import numpy as np

# OpenCV HSV bounds. 
# Note on OpenCV HSV format:
# Hue (H) range is 0-179 (Standard HSL 0-360 is divided by 2. Normalized 0.0-1.0 is multiplied by 180)
# Saturation (S) range is 0-255
# Value (V) range is 0-255

COLOR_THRESHOLDS = {
    "hydrocarbon_cyan": {
        # Captures the cyan/teal/light-blue UV fluorescence of oil-bearing rocks.
        # Hue 73-110 covers green-cyan (H~77) through blue-cyan (H~101),
        #   with ±6 buffer on each side from observed extremes.
        # Saturation 15-255 catches even very pale/desaturated fluorescence (S~20).
        # Value 180-255 ensures only bright fluorescent zones are detected,
        #   safely excluding dark non-hydrocarbon rocks (V~94-143).
        "lower": np.array([73, 15, 180]),
        "upper": np.array([110, 255, 255])
    },
    "oil_yellow": {
        # Based on article values: Hue 0.14-0.18, Saturation 0.05-1.0, Value 0.17-1.0
        # Hue: 0.14 * 180 = 25, 0.18 * 180 = 32
        # Saturation: 0.05 * 255 = 13
        # Value: 0.17 * 255 = 43
        "lower": np.array([25, 13, 43]),
        "upper": np.array([33, 255, 255])
    }
}
