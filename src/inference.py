import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import List, Tuple, Dict
import os

try:
    from ensemble_boxes import weighted_boxes_fusion
except ImportError:
    weighted_boxes_fusion = None
    print("Warning: ensemble_boxes not installed. Required for WBF in RockSegmentationPipeline. Install with 'pip install ensemble-boxes'")

class RockSegmentationPipeline:
    """
    A comprehensive pipeline for highly overlapping rock image segmentation, 
    integrating Test-Time Augmentation (TTA) Box Fusion and Solidity-based 
    Watershed Post-processing.
    """

    def __init__(self, wbf_iou_thr: float = 0.55, solidity_thr: float = 0.85, min_area: int = 300):
        """
        Initializes the pipeline with specific thresholds for fusion and segmentation.
        
        Args:
            wbf_iou_thr (float): Intersection over Union threshold for WBF.
            solidity_thr (float): Threshold to classify a contour as a valid rock seed.
            min_area (int): Minimum pixel area of a rock contour to be retained.
        """
        self.wbf_iou_thr = wbf_iou_thr
        self.solidity_thr = solidity_thr
        self.min_area = min_area

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
                if cv2.contourArea(cnt) < self.min_area:
                    continue
                # Ensure the format matches original output: arrays of shape (N, 2)
                pts = cnt.reshape(-1, 2)
                if len(pts) >= 3:
                    clean_polygons.append(pts)
                    
        return clean_polygons

    def apply_weighted_boxes_fusion(
        self, 
        boxes_list: List[List[List[float]]], 
        scores_list: List[List[float]], 
        labels_list: List[List[int]], 
        weights: List[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fuses overlapping bounding boxes from multiple TTA predictions.
        
        Reference: 
        Solovyev, R., Wang, W., & Gabruseva, T. (2020). "Weighted boxes fusion: 
        Ensembling boxes from different object detection models." 
        """
        if weighted_boxes_fusion is None:
            raise ImportError("ensemble_boxes is required to run WBF.")
            
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list, 
            scores_list, 
            labels_list, 
            weights=weights, 
            iou_thr=self.wbf_iou_thr, 
            skip_box_thr=0.0
        )
        return fused_boxes, fused_scores, fused_labels

    def apply_solidity_based_watershed(self, binary_mask: np.ndarray) -> List[np.ndarray]:
        """
        Applies a Marker-Based Watershed algorithm using contour solidity to 
        prevent over-segmentation of complex, adhered blasted rock images.
        
        Reference:
        Guo, Q., Wang, Y., Yang, S., & Xiang, Z. (2022). "A method of blasted rock 
        image segmentation based on improved watershed algorithm." Scientific Reports.

        Algorithm Steps:
        1. Morphological optimization to smooth edges and remove noise.
        2. Distance transformation of the binary mask.
        3. Multiple gray thresholding to find contours.
        4. Calculate Solidity (Area / Convex Hull Area).
        5. Mark contours with solidity > threshold as definitive seed points.
        6. Apply cv2.watershed to segment adhered rocks.
        
        Args:
            binary_mask: A 2D numpy array (0 and 255) representing the fused rock mask.
            
        Returns:
            List of 2D numpy arrays, where each array is a single isolated rock instance.        
        """
        # Ensure mask is uint8
        mask = (binary_mask > 0).astype(np.uint8) * 255
        
        #1. Morphological Optimization
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        #2. Distance Transform
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        cv2.normalize(dist_transform, dist_transform, 0, 255.0, cv2.NORM_MINMAX)
        dist_transform_8u = dist_transform.astype(np.uint8)

        #3. Multiple Gray Thresholds
        # We iterate through different thresholds to avoid missing smaller seeds
        # or over-segmenting larger ones (Guo et al., 2022).
        markers_bg = np.zeros_like(mask, dtype=np.int32)
        seed_id = 1
        
        thresholds = [64, 128, 192]
        for thresh in thresholds:
            _, thresholded_dist = cv2.threshold(dist_transform_8u, thresh, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresholded_dist, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 10:
                    continue
                    
                convex_hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(convex_hull)
                
                if hull_area == 0:
                    continue
                
                # 4. Calculate Solidity            
                solidity = float(area) / hull_area

                # 5. Mark valid seed points
                if solidity >= self.solidity_thr:
                    cv2.drawContours(markers_bg, [contour], -1, (seed_id), thickness=cv2.FILLED)
                    seed_id += 1

        sure_bg = cv2.dilate(mask, kernel, iterations=3)
        unknown = cv2.subtract(sure_bg, (markers_bg > 0).astype(np.uint8) * 255)
        
        markers = markers_bg + 1
        markers[unknown == 255] = 0
        
        # 6. Apply Watershed (Requires 3-channel image)
        img_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_color, markers)
        
        # 7. Extract Individual Instances
        separated_instances = []
        for marker_val in range(2, markers.max() + 1):
            obj_mask = np.zeros_like(mask)
            obj_mask[markers == marker_val] = 255
            
            if cv2.countNonZero(obj_mask) > 0:
                separated_instances.append(obj_mask)
                
        # Fallback if watershed removed everything
        if not separated_instances and cv2.countNonZero(mask) > 0:
            separated_instances.append(mask)
                
        return separated_instances

    def run_pipeline(self, tta_inferences: List[Dict]) -> List[np.ndarray]:
        """
        Executes the entire post-processing workflow precisely in order:
        1. Receive TTA inferences.
        2. Apply WBF to fuse bounding boxes.
        3. (Mocked Step) Reconstruct a single unified binary mask based on fused boxes.
        4. Apply Solidity-Based Watershed to the unified mask.
        
        Args:
            tta_inferences: A list of dictionaries, where each dict represents 
                            one TTA pass containing 'boxes', 'scores', 'labels', 
                            and 'masks'.
                            
        Returns:
            List of final processed binary masks representing separated rocks.
        """
        boxes_list = [infer['boxes'] for infer in tta_inferences]
        scores_list = [infer['scores'] for infer in tta_inferences]
        labels_list = [infer['labels'] for infer in tta_inferences]
        
        fused_boxes, fused_scores, fused_labels = self.apply_weighted_boxes_fusion(
            boxes_list, scores_list, labels_list
        )
        
        unified_binary_mask = self._reconstruct_unified_mask(fused_boxes, tta_inferences)
        final_separated_masks = self.apply_solidity_based_watershed(unified_binary_mask)
        
        return final_separated_masks

    def _reconstruct_unified_mask(self, fused_boxes: np.ndarray, tta_inferences: List[Dict]) -> np.ndarray:
        """
        Mock helper to reconstruct a unified 2D binary mask.
        """
        dummy_shape = tta_inferences[0].get('masks', np.zeros((640, 640))).shape
        return np.zeros(dummy_shape, dtype=np.uint8)

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
