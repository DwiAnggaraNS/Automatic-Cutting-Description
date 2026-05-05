import numpy as np

# OpenCV HSV bounds. 
# Note on OpenCV HSV format:
# Hue (H) range is 0-179 (Standard HSL 0-360 is divided by 2. Normalized 0.0-1.0 is multiplied by 180)
# Saturation (S) range is 0-255
# Value (V) range is 0-255

COLOR_THRESHOLDS = {
    "regular_oil_cyan": {
        # Based on HSL(162-167, 49-53%, 87-92%)
        # Hue: safely set between 75 and 95 (OpenCV) to avoid Dark Blue (118) and Purple (130)
        # Saturation: 100 to 255 to capture vibrant colors
        # Value: 200 to 255 to capture high brightness of fluorescence
        "lower": np.array([75, 100, 200]),
        "upper": np.array([95, 255, 255])
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
