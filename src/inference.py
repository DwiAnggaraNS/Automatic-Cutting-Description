import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

class MaskPostProcessor:
    """
    A robust pipeline for post-processing instance segmentation masks.
    Designed specifically to clean up jagged edges, spikes, and 
    self-intersecting polygons commonly produced by YOLO upscaling.
    """
    def __init__(self, min_area=300, epsilon_ratio=0.008, simplify_tol=2.0):
        self.min_area = min_area
        self.epsilon_ratio = epsilon_ratio
        self.simplify_tol = simplify_tol

    def process(self, mask_binary):
        """
        Process a single binary mask into clean polygon coordinates.
        Args:
            mask_binary: Boolean or uint8 numpy array of shape (H, W)
        Returns:
            List of cleaned polygon coordinates (np.int32 arrays)
        """
        mask = mask_binary.astype(np.uint8)
        
        # 1. Morphological Operations: Close (fill small holes) -> Open (remove external spikes)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        
        # 2. Gaussian Blur & Rethresholding to smooth hard edges
        mask_f = cv2.GaussianBlur(mask.astype(np.float32), (7, 7), 0)
        mask = (mask_f > 0.4).astype(np.uint8)
        
        # 3. Contour Extraction
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        clean_polygons = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_area:
                continue
            
            # 4. Douglas-Peucker Simplification
            eps = self.epsilon_ratio * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)
            
            pts = approx.reshape(-1, 2)
            if len(pts) < 3:
                continue
            
            # 5. Shapely Geometry: Fix intersections and simplify topology safely
            try:
                poly = Polygon(pts)
                poly = make_valid(poly)
                
                # Handle multipolygons strictly created by make_valid tearing parts
                if poly.geom_type == 'MultiPolygon':
                    # Keep the largest part
                    poly = max(poly.geoms, key=lambda a: a.area)
                
                poly = poly.simplify(self.simplify_tol, preserve_topology=True)
                
                if poly.is_empty or poly.area < self.min_area:
                    continue
                
                coords = np.array(poly.exterior.coords, dtype=np.int32)
                clean_polygons.append(coords)
            except Exception:
                # Fallback to the raw approxPolyDP if Shapely fails
                clean_polygons.append(pts)
                
        return clean_polygons

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
