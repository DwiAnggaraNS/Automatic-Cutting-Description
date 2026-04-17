import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import List, Tuple, Dict, Any
import os

try:
    from ensemble_boxes import weighted_boxes_fusion
except ImportError:
    weighted_boxes_fusion = None
    print("Warning: ensemble_boxes not installed. Required for WBF in RockSegmentationPipeline. Install with 'pip install ensemble-boxes'")

class RockSegmentationPipeline:
    """
    A comprehensive pipeline for highly overlapping rock image segmentation, 
    integrating Solidity-based Watershed Post-processing to refine standard inferences without TTA or WBF.
    """

    def __init__(self, solidity_thr: float = 0.85, min_area: int = 300, min_area_after_watershed: int = 150):
        """
        Initializes the pipeline with specific thresholds for smoothing and segmentation.
        
        Args:
            solidity_thr (float): Threshold to classify a contour as a valid rock seed.
            min_area (int): Minimum pixel area of a rock contour to be initially processed.
            min_area_after_watershed (int): Minimum pixel area of a rock contour to be retained after watershed split.
        """
        self.solidity_thr = solidity_thr
        self.min_area = min_area
        self.min_area_after_watershed = min_area_after_watershed

    def process(self, mask_binary: np.ndarray) -> List[np.ndarray]:
        """
        Backward compatible processing method. Applies the watershed post-processing
        to a single binary mask and extracts cleaned polygon coordinates.
        
        Args:
            mask_binary: Boolean or uint8 numpy array of shape (H, W)
            
        Returns:
            List of cleaned polygon coordinates (np.int32 arrays)
        """
        separated_masks = self.apply_solidity_based_watershed(mask_binary)
        clean_polygons = []
        
        for mask_instance in separated_masks:
            contours, _ = cv2.findContours(mask_instance, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < self.min_area_after_watershed:
                    continue
                # Ensure the format matches original output: arrays of shape (N, 2)
                pts = cnt.reshape(-1, 2)
                if len(pts) >= 3:
                    clean_polygons.append(pts)
                    
        return clean_polygons

    def apply_solidity_based_watershed(self, binary_mask: np.ndarray) -> List[np.ndarray]:
        """
        Applies Marker-Based Watershed algorithm using distance transforms and 
        contour solidity validation to safely split under-segmented blobs.
        
        Reference:
        Guo, Q., Wang, Y., Yang, S., & Xiang, Z. (2022). "A method of blasted rock 
        image segmentation based on improved watershed algorithm." Scientific Reports.

        Algorithm Steps:
        1. Morphologically optimize the binary mask to smooth boundaries.
        2. Apply distance transform for localized maxima identification.
        3. Threshold at varying depths to determine logical seed points for discrete rocks.
        4. Validate each seed point using contour solidity (must be >= threshold).
        5. Apply cv2.watershed specifically restricted to the actual blob footprint.
        
        Args:
            binary_mask: A 2D numpy array (0 and 255) representing the fused rock mask.
            
        Returns:
            List of 2D numpy arrays, where each array is a single isolated rock instance.        
        """
        original_mask = (binary_mask > 0).astype(np.uint8) * 255
        
        # 1. Morphological Optimization
        # Smoothes the mask prior to processing so the distance transform doesn't produce jagged local maxima
        kernel = np.ones((5, 5), np.uint8)
        mask_opt = cv2.morphologyEx(original_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask_opt = cv2.morphologyEx(mask_opt, cv2.MORPH_OPEN, kernel, iterations=1)

        # 2. Distance Transform
        dist_transform = cv2.distanceTransform(mask_opt, cv2.DIST_L2, 3)
        cv2.normalize(dist_transform, dist_transform, 0, 255.0, cv2.NORM_MINMAX)
        dist_transform_8u = dist_transform.astype(np.uint8)

        # 3. Dynamic Thresholding for Seeds
        markers_bg = np.zeros_like(original_mask, dtype=np.int32)
        seed_id = 1
        
        # Evaluate from deepest core to shallower core
        thresholds = [192, 128, 64]
        for thresh in thresholds:
            _, thresholded_dist = cv2.threshold(dist_transform_8u, thresh, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresholded_dist, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 10:
                    continue
                    
                # 4. Solidity check on the Seed Point
                convex_hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(convex_hull)
                
                if hull_area == 0:
                    continue
                    
                seed_solidity = float(area) / hull_area
                
                # Validate the seed shape using the solidity metric
                if seed_solidity >= self.solidity_thr:
                    seed_mask = np.zeros_like(original_mask)
                    cv2.drawContours(seed_mask, [contour], -1, 255, thickness=cv2.FILLED)
                    
                    # BEST PRACTICE: Prevent Concentric/Recursive Seeds (Over-segmentation fix)
                    # If this contour encompasses an already registered seed from a deeper threshold,
                    # it represents the base of the identical peak, NOT a separate rock instance.
                    if np.any(markers_bg[(seed_mask == 255)] > 0):
                        continue
                        
                    # Register the new isolated seed point
                    markers_bg[(seed_mask == 255) & (markers_bg == 0)] = seed_id
                    seed_id += 1

        # 5. Define Watershed Boundaries safely
        sure_bg = cv2.dilate(original_mask, kernel, iterations=2)
        unknown = cv2.subtract(sure_bg, (markers_bg > 0).astype(np.uint8) * 255)
        
        markers = markers_bg + 1
        markers[unknown == 255] = 0
        
        # 6. Execute Watershed
        img_color = cv2.cvtColor(original_mask, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_color, markers)
        
        # 7. Extract Individual Instances preserving pixel-perfect outer boundaries
        separated_instances = []
        for marker_val in range(2, markers.max() + 1):
            obj_mask = np.zeros_like(original_mask)
            # Intersection forces the separated mask to NEVER exceed CNN's exact prediction
            obj_mask[(markers == marker_val) & (original_mask == 255)] = 255
            
            if cv2.countNonZero(obj_mask) > self.min_area_after_watershed:
                separated_instances.append(obj_mask)
                
        # Safe fallback: if watershed fails to segment anything valid, return the original whole blob
        if not separated_instances and cv2.countNonZero(original_mask) > 0:
            separated_instances.append(original_mask)
                
        return separated_instances

    def run_pipeline(self, model_results: Any) -> List[Dict[str, Any]]:
        """
        Executes the morphological post-processing workflow on standard YOLO predictions.
        (Refactored to remove TTA and WBF due to geometric distortions).
        
        Args:
            model_results: Ultralytics YOLO Results object containing native predictions.
                            
        Returns:
            List of dictionaries containing {"polygon": ndarray, "class_id": int, "score": float}.
        """
        final_predictions = []
        if not model_results or model_results.masks is None:
            return final_predictions
            
        # 1. Iterate over every native inference prediction natively
        h, w = model_results.orig_shape
        polygons = model_results.masks.xy
        classes = model_results.boxes.cls.cpu().numpy()
        scores = model_results.boxes.conf.cpu().numpy()
        
        for poly, cls_id, base_score in zip(polygons, classes, scores):
            if len(poly) < 3:
                continue
                
            # Create isolated properties for evaluation
            poly_int = poly.astype(np.int32)
            area = cv2.contourArea(poly_int)
            hull = cv2.convexHull(poly_int)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0.0
            
            # BEST PRACTICE: If the polygon generated by YOLO is already 
            # highly solid/convex, it is highly likely a single standalone rock.
            # Running morphological watershed on a perfect contour will only 
            # erode its boundaries destructively.
            if solidity >= self.solidity_thr:
                final_predictions.append({
                    "polygon": poly_int,
                    "class_id": int(cls_id),
                    "score": float(base_score)
                })
                continue
                
            # If the solidity is below threshold, it's considered an irregular, complex 
            # blob that might contain multiple adhered rocks fused together by the CNN.
            # Proceed with explicit marker-based watershed just for this isolated blob.
            isolated_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Draw as filled polygon
            cv2.fillPoly(isolated_mask, [poly_int], 255)
            
            # Initial area check
            if cv2.countNonZero(isolated_mask) < self.min_area:
                continue
                
            # Apply solidity-based watershed on this isolated prediction just in case multiple rocks are fused natively by the single pass
            separated_masks = self.apply_solidity_based_watershed(isolated_mask)
            
            for split_mask in separated_masks:
                contours, _ = cv2.findContours(split_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) < self.min_area_after_watershed:
                        continue
                    
                    pts = cnt.reshape(-1, 2)
                    if len(pts) >= 3:
                        final_predictions.append({
                            "polygon": pts,
                            "class_id": int(cls_id),
                            "score": float(base_score)
                        })
                        
        return final_predictions

class RockVisualizer:
    """
    Handling standardized, high-quality visualizations of rock samples
    mimicking CVAT styles across all scripts and notebooks.
    """
    def __init__(self, thickness=4):
        self.thickness = thickness
        self.class_colors = {
            0: (50, 183, 250),   # Silt 
            1: (250, 250, 55),   # Sandstone 
            2: (61, 245, 61),    # Limestone 
            3: (184, 61, 245),   # Coal 
            4: (204, 153, 51),   # Shalestone 
            5: (250, 125, 187)   # Quartz 
        }
        
        self.class_names = {
            0: "Silt", 1: "Sandstone", 2: "Limestone", 
            3: "Coal", 4: "Shalestone", 5: "Quartz"
        }

    def _get_color(self, class_id):
        return self.class_colors.get(int(class_id), (255, 255, 255))

    def draw_hollow(self, image_bgr, polygons_by_class):
        """
        Draws hollow CVAT-style contours natively directly on an image copy.
        polygons_by_class: format {class_id: [array_of_coords1, array_of_coords2, ...]}
        """
        annotated = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        for class_id, polygons in polygons_by_class.items():
            color = self._get_color(class_id)
            for poly in polygons:
                cv2.polylines(annotated, [poly], isClosed=True, color=color, thickness=self.thickness)
        return annotated

    def draw_mask_only(self, image_shape, polygons_by_class):
        """
        Generates a pure black background mask with colored filled polygons.
        """
        mask_img = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
        for class_id, polygons in polygons_by_class.items():
            color = self._get_color(class_id)
            for poly in polygons:
                cv2.fillPoly(mask_img, [poly], color)
        return mask_img

    def plot_advanced_comparison(self, original_bgr, raw_hollow, proc_hollow, raw_solid, proc_solid, title_suffix=""):
        """
        Builds a comprehensive 2x3 matplotlib figure showing Original, Raw, and Post-Processed
        visualizations side-by-side for both hollow and solid styles.
        """
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(2, 3, figsize=(24, 16))
        
        # Row 1: Hollow Contours
        axes[0, 0].imshow(original_rgb)
        axes[0, 0].set_title(f"Original Image", fontsize=16)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(raw_hollow)
        axes[0, 1].set_title(f"Raw Prediction (Hollow) {title_suffix}", fontsize=16)
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(proc_hollow)
        axes[0, 2].set_title(f"Post-processed (Hollow) {title_suffix}", fontsize=16)
        axes[0, 2].axis('off')
        
        # Row 2: Solid Masks
        axes[1, 0].imshow(original_rgb)
        axes[1, 0].set_title(f"Original Image", fontsize=16)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(raw_solid)
        axes[1, 1].set_title(f"Raw Prediction (Solid) {title_suffix}", fontsize=16)
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(proc_solid)
        axes[1, 2].set_title(f"Post-processed (Solid) {title_suffix}", fontsize=16)
        axes[1, 2].axis('off')
        
        # Consistent legend
        legend_elements = [
            Patch(facecolor=[c/255 for c in color], edgecolor='k', label=self.class_names.get(id, f"ID {id}")) 
            for id, color in self.class_colors.items()
        ]
        axes[0, 2].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1))
        axes[1, 2].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1))
        
        plt.tight_layout()
        return fig

    def plot_comparison(self, original_bgr, proc_hollow, proc_solid, title_suffix=""):
        """
        Builds a concise 1x3 matplotlib figure showing Original, Processed Hollow, 
        and Processed Solid masks side-by-side. Designed primarily for SAHI inference.
        
        Args:
            original_bgr (np.ndarray): Original image array in BGR format.
            proc_hollow (np.ndarray): Processed image with hollow CVAT-style contours.
            proc_solid (np.ndarray): Processed mask-only solid polygons.
            title_suffix (str): Additional text appended to the subplot titles.
            
        Returns:
            matplotlib.figure.Figure: The complete figure to be displayed or saved.
        """
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        axes[0].imshow(original_rgb)
        axes[0].set_title("Original Image", fontsize=16)
        axes[0].axis('off')
        
        axes[1].imshow(proc_hollow)
        axes[1].set_title(f"Processed (Hollow) {title_suffix}", fontsize=16)
        axes[1].axis('off')
        
        axes[2].imshow(proc_solid)
        axes[2].set_title(f"Processed (Solid) {title_suffix}", fontsize=16)
        axes[2].axis('off')
        
        # Consistent legend
        legend_elements = [
            Patch(facecolor=[c/255 for c in color], edgecolor='k', label=self.class_names.get(int(class_id), f"ID {class_id}")) 
            for class_id, color in self.class_colors.items()
        ]
        axes[2].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1))
        
        plt.tight_layout()
        return fig
