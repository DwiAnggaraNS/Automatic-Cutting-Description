"""
color_utils.py
==============
Utilities for converting HSL color picker values to OpenCV HSV thresholds.

Developer workflow
------------------
1. Open any color picker (browser DevTools, Figma, Photoshop, etc.)
2. Sample 5–15 pixels from representative areas of the target class
3. Record the HSL values:  e.g. ["hsl(309, 100, 49)", "hsl(305, 90, 92)", ...]
4. Pass the list to compute_thresholds_from_hsl() → ready-to-use bounds dict
5. Plug the result into a ColorSegmentation subclass's DEFAULT_COLORS
"""

import re
import numpy as np


# ---------------------------------------------------------------------------
# Parsing & conversion
# ---------------------------------------------------------------------------

def parse_hsl(hsl_str: str) -> tuple:
    """
    Parse an HSL string into (H, S, L).

    Accepts any of:
        "hsl(309, 100, 49)"
        "hsl(309, 100%, 49%)"
        "309, 100, 49"
        "309 100 49"

    Returns:
        (H: float 0-360, S: float 0-100, L: float 0-100)
    """
    nums = re.findall(r"\d+(?:\.\d+)?", str(hsl_str))
    if len(nums) < 3:
        raise ValueError(f"Cannot parse HSL from: {hsl_str!r}")
    return float(nums[0]), float(nums[1]), float(nums[2])


def hsl_to_rgb(h: float, s_pct: float, l_pct: float) -> tuple:
    """
    Convert HSL (H: 0-360, S: 0-100, L: 0-100) to RGB (each 0-255).
    """
    h = h % 360
    s = s_pct / 100
    l = l_pct / 100

    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c / 2

    if   0   <= h < 60:  r1, g1, b1 = c, x, 0
    elif 60  <= h < 120: r1, g1, b1 = x, c, 0
    elif 120 <= h < 180: r1, g1, b1 = 0, c, x
    elif 180 <= h < 240: r1, g1, b1 = 0, x, c
    elif 240 <= h < 300: r1, g1, b1 = x, 0, c
    else:                r1, g1, b1 = c, 0, x

    return round((r1 + m) * 255), round((g1 + m) * 255), round((b1 + m) * 255)


def rgb_to_opencv_hsv(r: int, g: int, b: int) -> tuple:
    """
    Convert RGB (each 0-255) to OpenCV HSV format:
        H: 0-179  (standard 0-360 divided by 2)
        S: 0-255
        V: 0-255
    """
    r_, g_, b_ = r / 255, g / 255, b / 255
    cmax = max(r_, g_, b_)
    cmin = min(r_, g_, b_)
    delta = cmax - cmin

    if delta == 0:
        h = 0
    elif cmax == r_:
        h = 60 * (((g_ - b_) / delta) % 6)
    elif cmax == g_:
        h = 60 * ((b_ - r_) / delta + 2)
    else:
        h = 60 * ((r_ - g_) / delta + 4)

    s = (delta / cmax * 255) if cmax != 0 else 0
    v = cmax * 255

    return round(h / 2), round(s), round(v)


def hsl_str_to_opencv_hsv(hsl_str: str) -> tuple:
    """Convenience: parse an HSL string and convert directly to OpenCV HSV."""
    return rgb_to_opencv_hsv(*hsl_to_rgb(*parse_hsl(hsl_str)))


# ---------------------------------------------------------------------------
# Automatic threshold computation
# ---------------------------------------------------------------------------

def compute_thresholds_from_hsl(
    hsl_list: list,
    h_margin: int = 8,
    s_margin: int = 20,
    v_margin: int = 25,
) -> dict | list:
    """
    Automatically compute OpenCV HSV thresholds from a list of HSL color strings.

    Args:
        hsl_list : List of HSL strings from a color picker.
                   e.g. ["hsl(309, 100, 49)", "hsl(305, 90, 92)", ...]
        h_margin : ±buffer added to each side of the observed H range (0-179).
        s_margin : ±buffer added to each side of the observed S range (0-255).
        v_margin : ±buffer added to each side of the observed V range (0-255).

    Returns:
        Single dict  {"lower": np.array([H,S,V]), "upper": np.array([H,S,V])}
        for most colors.

        List of two dicts if the hue range straddles H=0 (red/near-red tones).
        ColorSegmentation handles both cases automatically.

    Raises:
        ValueError if hsl_list is empty or unparseable.
    """
    if not hsl_list:
        raise ValueError("hsl_list cannot be empty.")

    samples = [hsl_str_to_opencv_hsv(c) for c in hsl_list]
    h_vals = sorted(s[0] for s in samples)
    s_vals = [s[1] for s in samples]
    v_vals = [s[2] for s in samples]

    # S / V bounds: simple linear, clamped to [0, 255]
    s_lo = max(0,   min(s_vals) - s_margin)
    s_hi = min(255, max(s_vals) + s_margin)
    v_lo = max(0,   min(v_vals) - v_margin)
    v_hi = min(255, max(v_vals) + v_margin)

    # H bounds: circular — detect if colors straddle H=0
    n = len(h_vals)
    internal_gaps = [(h_vals[i + 1] - h_vals[i], i) for i in range(n - 1)]
    wraparound_gap = ((180 - h_vals[-1]) + h_vals[0], n - 1)
    max_gap_size, max_gap_idx = max(internal_gaps + [wraparound_gap], key=lambda x: x[0])

    if max_gap_idx == n - 1:
        # Largest gap is the wraparound gap → colors are contiguous, no wrap
        return {
            "lower": np.array([max(0,   h_vals[0]  - h_margin), s_lo, v_lo]),
            "upper": np.array([min(179, h_vals[-1] + h_margin), s_hi, v_hi]),
        }
    else:
        # Colors straddle H=0 → need two separate masks
        hi_group = h_vals[max_gap_idx + 1:]   # high-H cluster (near 179)
        lo_group = h_vals[:max_gap_idx + 1]   # low-H cluster  (near 0)
        return [
            {
                "lower": np.array([max(0,   min(hi_group) - h_margin), s_lo, v_lo]),
                "upper": np.array([min(179, max(hi_group) + h_margin), s_hi, v_hi]),
            },
            {
                "lower": np.array([max(0,   min(lo_group) - h_margin), s_lo, v_lo]),
                "upper": np.array([min(179, max(lo_group) + h_margin), s_hi, v_hi]),
            },
        ]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_threshold_report(class_name: str, hsl_list: list) -> None:
    """
    Print a human-readable calibration report for a color class.
    Useful for developers to verify samples before deploying.

    Example:
        print_threshold_report("cement_pink", ["hsl(309, 100, 49)", ...])
    """
    samples = [hsl_str_to_opencv_hsv(c) for c in hsl_list]
    h_vals = [s[0] for s in samples]
    s_vals = [s[1] for s in samples]
    v_vals = [s[2] for s in samples]
    thresholds = compute_thresholds_from_hsl(hsl_list)

    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  Calibration report: {class_name}")
    print(bar)
    print(f"  {'HSL input':25s}  H    S    V  (OpenCV)")
    print(f"  {'-'*49}")
    for hsl_str, (h, s, v) in zip(hsl_list, samples):
        print(f"  {hsl_str:25s}  {h:3d}  {s:3d}  {v:3d}")
    print(f"  {'-'*49}")
    print(f"  Observed range:  "
          f"H=[{min(h_vals)}, {max(h_vals)}]  "
          f"S=[{min(s_vals)}, {max(s_vals)}]  "
          f"V=[{min(v_vals)}, {max(v_vals)}]")

    if isinstance(thresholds, list):
        print(f"\n  ⚠  Hue wraps around 0 — two masks generated:")
        for i, t in enumerate(thresholds, 1):
            print(f"     Mask {i}: lower={t['lower'].tolist()}  upper={t['upper'].tolist()}")
    else:
        print(f"\n  Threshold lower : {thresholds['lower'].tolist()}")
        print(f"  Threshold upper : {thresholds['upper'].tolist()}")
    print(bar)
