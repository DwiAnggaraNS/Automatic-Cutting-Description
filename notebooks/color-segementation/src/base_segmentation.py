"""
base_segmentation.py
====================
Abstract superclass for HSV color-thresholding + area-filtering segmentation.

Algorithm reference
-------------------
Huo et al., "Novel Lithology Identification Method for Drilling Cuttings
Under PDC Bit Condition", Journal of Petroleum Science and Engineering,
205 (2021) 108898. — Section 3.1: Fluorescence color threshold segmentation.

Subclassing
-----------
    class MyDetector(ColorSegmentation):
        DETECTION_LABEL = "My Rock Type"
        DEFAULT_COLORS = {
            "class_name": ["hsl(h, s, l)", "hsl(h, s, l)", ...]
        }

    # Use with default calibration
    detector = MyDetector()
    mask, segmented, ratio = detector.segment(image_rgb)
    detector.visualize(image_rgb)

    # Override colors at runtime (e.g. for a new field dataset)
    detector = MyDetector(colors_per_class={"class_name": ["hsl(...)", ...]})
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

from color_utils import compute_thresholds_from_hsl, print_threshold_report


class ColorSegmentation:
    """
    Base class for single-image, multi-class HSV color segmentation.

    Core pipeline (same for every subclass):
        1. RGB → HSV conversion
        2. cv2.inRange thresholding per color class
        3. Bitwise-OR across all classes into a combined mask
        4. Contour-based area filtering (reject noise < pixel_threshold)
        5. Apply final mask to produce segmented image + compute area ratio
    """

    # ---- Override in subclasses ----------------------------------------
    DEFAULT_COLORS: dict = {}   # { class_name: ["hsl(h,s,l)", ...] }
    DETECTION_LABEL: str = "Detected Region"
    # -----------------------------------------------------------------------

    def __init__(
        self,
        colors_per_class: dict | None = None,
        pixel_threshold: int = 10,
        h_margin: int = 8,
        s_margin: int = 20,
        v_margin: int = 25,
    ):
        """
        Args:
            colors_per_class : dict { class_name: ["hsl(...)", ...] }
                               If None, uses the subclass DEFAULT_COLORS.
            pixel_threshold  : Minimum contour area in pixels.
                               Contours smaller than this are discarded as noise.
                               (Huo et al. use P ≥ 100 px; empirically 10 px
                               works better for smaller/lower-res images.)
            h_margin         : ±margin added to the auto-computed H range.
            s_margin         : ±margin added to the auto-computed S range.
            v_margin         : ±margin added to the auto-computed V range.
        """
        raw = colors_per_class if colors_per_class is not None else self.DEFAULT_COLORS
        if not raw:
            raise ValueError(
                f"{type(self).__name__} has no color definitions. "
                "Set DEFAULT_COLORS in your subclass or pass colors_per_class."
            )

        self.pixel_threshold = pixel_threshold
        self._margins = dict(h=h_margin, s=s_margin, v=v_margin)

        # Build config: { class_name: bounds_dict_or_list }
        self.config: dict = {
            name: compute_thresholds_from_hsl(
                hsl_list,
                h_margin=h_margin,
                s_margin=s_margin,
                v_margin=v_margin,
            )
            for name, hsl_list in raw.items()
        }

    # ------------------------------------------------------------------
    # Core segmentation algorithm
    # ------------------------------------------------------------------

    def segment(self, image_rgb: np.ndarray) -> tuple:
        """
        Run HSV thresholding + area filtering on an RGB image.

        Args:
            image_rgb : np.ndarray of shape (H, W, 3), dtype uint8, RGB order.

        Returns:
            final_mask    : np.ndarray (H, W), uint8 binary mask (0 or 255).
            segmented_rgb : np.ndarray (H, W, 3) — original pixels where
                            mask=255, black elsewhere.
            ratio         : float — detected area / total area × 100 (%).
            num_instances : int — number of individual rocks/particles detected.
        """
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for bounds in self.config.values():
            # bounds is either a single dict or a list of dicts (hue wraparound)
            for b in (bounds if isinstance(bounds, list) else [bounds]):
                combined = cv2.bitwise_or(
                    combined,
                    cv2.inRange(hsv, b["lower"], b["upper"])
                )

        # Area filtering — Huo et al. Eq. (4): P ≥ pixel_threshold
        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        final_mask = np.zeros_like(combined)
        num_instances = 0
        
        for cnt in contours:
            if cv2.contourArea(cnt) >= self.pixel_threshold:
                cv2.drawContours(final_mask, [cnt], -1, 255, cv2.FILLED)
                num_instances += 1

        segmented_rgb = cv2.bitwise_and(image_rgb, image_rgb, mask=final_mask)
        ratio = float(np.sum(final_mask > 0)) / (image_rgb.shape[0] * image_rgb.shape[1]) * 100

        return final_mask, segmented_rgb, ratio, num_instances

    # ------------------------------------------------------------------
    # Visualization (Jupyter-safe)
    # ------------------------------------------------------------------

    def visualize(self, image_rgb: np.ndarray) -> None:
        """
        Display Original | Segmented (Black BG) | Segmented Overlay in a 1×3 figure.

        Uses display(fig) + plt.close(fig) for full Jupyter + ipywidgets
        compatibility (no figure bleeding between Output widget and notebook).
        """
        final_mask, segmented_rgb, ratio, num_instances = self.segment(image_rgb)

        # Create overlay by drawing transparent mask over original image
        overlay_rgb = image_rgb.copy()
        overlay_mask = np.zeros_like(image_rgb)
        overlay_mask[final_mask == 255] = [0, 255, 0]  # Green overlay
        cv2.addWeighted(overlay_mask, 0.4, overlay_rgb, 1.0, 0, overlay_rgb)

        # Find contours to draw borders on the overlay
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_rgb, contours, -1, (0, 255, 0), 2)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

        axes[0].imshow(image_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(segmented_rgb)
        axes[1].set_title("Segmented (Black Background)")
        axes[1].axis("off")

        axes[2].imshow(overlay_rgb)
        axes[2].set_title(f"Segmented Overlay (Mask + Contour)")
        axes[2].axis("off")

        fig.suptitle(
            f"{self.DETECTION_LABEL} — Detected Instances: {num_instances} | Area proportion: {ratio:.2f}%",
            fontsize=13,
            y=0.98,
        )

        display(fig)   # respects ipywidgets Output context
        plt.close(fig) # prevent figure accumulation / "3-rows" bug

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def print_config(self) -> None:
        """Print active HSV thresholds and settings."""
        print(f"\n{type(self).__name__}  [{self.DETECTION_LABEL}]")
        print(f"  pixel_threshold : {self.pixel_threshold} px")
        print(f"  margins applied : h±{self._margins['h']}  "
              f"s±{self._margins['s']}  v±{self._margins['v']}")
        for name, bounds in self.config.items():
            bl = bounds if isinstance(bounds, list) else [bounds]
            for i, b in enumerate(bl):
                tag = f" (mask {i+1}/{len(bl)})" if len(bl) > 1 else ""
                print(f"  [{name}{tag}]")
                print(f"      lower : {b['lower'].tolist()}")
                print(f"      upper : {b['upper'].tolist()}")

    def print_calibration_report(self, colors_per_class: dict | None = None) -> None:
        """
        Print per-sample calibration details for review.
        Pass colors_per_class to inspect a custom dict,
        or leave None to inspect DEFAULT_COLORS.
        """
        src = colors_per_class or self.DEFAULT_COLORS
        for name, hsl_list in src.items():
            print_threshold_report(name, hsl_list)
