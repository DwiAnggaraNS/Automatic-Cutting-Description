"""
COCO Polygon Simplification Script
===================================
This script simplifies polygon segmentations in COCO format annotations
using the Douglas-Peucker algorithm. It also provides comparison metrics
to evaluate the quality of simplification.

Features:
- Backs up original annotation files
- Simplifies polygons with configurable tolerance
- Calculates comparison metrics (IoU, point reduction, area preservation)
- Generates a detailed report

Usage:
    python coco_polygon_simplification.py [--tolerance TOLERANCE] [--min-points MIN_POINTS]

Author: Auto-generated
Date: 2026-01-23
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse
from datetime import datetime
import numpy as np

try:
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("Warning: shapely not installed. IoU calculations will be skipped.")
    print("Install with: pip install shapely")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def douglas_peucker(points: List[Tuple[float, float]], tolerance: float) -> List[Tuple[float, float]]:
    """
    Simplify a polygon using the Douglas-Peucker algorithm.
    
    Args:
        points: List of (x, y) coordinate tuples
        tolerance: Maximum distance threshold for point removal
        
    Returns:
        Simplified list of (x, y) coordinate tuples
    """
    if len(points) <= 2:
        return points
    
    # Find the point with maximum distance from the line between first and last points
    dmax = 0.0
    index = 0
    end = len(points) - 1
    
    for i in range(1, end):
        d = perpendicular_distance(points[i], points[0], points[end])
        if d > dmax:
            index = i
            dmax = d
    
    # If max distance is greater than tolerance, recursively simplify
    if dmax > tolerance:
        # Recursive call
        rec_results1 = douglas_peucker(points[:index + 1], tolerance)
        rec_results2 = douglas_peucker(points[index:], tolerance)
        
        # Build the result list
        result = rec_results1[:-1] + rec_results2
    else:
        result = [points[0], points[end]]
    
    return result


def perpendicular_distance(point: Tuple[float, float], 
                           line_start: Tuple[float, float], 
                           line_end: Tuple[float, float]) -> float:
    """
    Calculate the perpendicular distance from a point to a line.
    
    Args:
        point: The point (x, y)
        line_start: Start point of the line (x, y)
        line_end: End point of the line (x, y)
        
    Returns:
        Perpendicular distance
    """
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    
    # Handle case where line_start and line_end are the same point
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
    
    # Calculate perpendicular distance using cross product
    numerator = abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1)
    denominator = (dx ** 2 + dy ** 2) ** 0.5
    
    return numerator / denominator if denominator > 0 else 0


def segmentation_to_points(segmentation: List[float]) -> List[Tuple[float, float]]:
    """
    Convert COCO segmentation format (flat list) to list of (x, y) tuples.
    
    Args:
        segmentation: Flat list [x1, y1, x2, y2, ...]
        
    Returns:
        List of (x, y) tuples
    """
    points = []
    for i in range(0, len(segmentation), 2):
        if i + 1 < len(segmentation):
            points.append((segmentation[i], segmentation[i + 1]))
    return points


def points_to_segmentation(points: List[Tuple[float, float]]) -> List[float]:
    """
    Convert list of (x, y) tuples back to COCO segmentation format.
    
    Args:
        points: List of (x, y) tuples
        
    Returns:
        Flat list [x1, y1, x2, y2, ...]
    """
    segmentation = []
    for x, y in points:
        segmentation.extend([round(x, 2), round(y, 2)])
    return segmentation


def calculate_polygon_area(points: List[Tuple[float, float]]) -> float:
    """
    Calculate the area of a polygon using the Shoelace formula.
    
    Args:
        points: List of (x, y) tuples
        
    Returns:
        Area of the polygon
    """
    if len(points) < 3:
        return 0.0
    
    n = len(points)
    area = 0.0
    
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    
    return abs(area) / 2.0


def calculate_iou(original_points: List[Tuple[float, float]], 
                  simplified_points: List[Tuple[float, float]]) -> float:
    """
    Calculate Intersection over Union (IoU) between original and simplified polygons.
    
    Args:
        original_points: Original polygon points
        simplified_points: Simplified polygon points
        
    Returns:
        IoU value (0-1)
    """
    if not SHAPELY_AVAILABLE:
        return -1.0
    
    if len(original_points) < 3 or len(simplified_points) < 3:
        return 0.0
    
    try:
        poly1 = Polygon(original_points)
        poly2 = Polygon(simplified_points)
        
        # Make polygons valid if needed
        if not poly1.is_valid:
            poly1 = make_valid(poly1)
        if not poly2.is_valid:
            poly2 = make_valid(poly2)
        
        # Handle case where make_valid returns a GeometryCollection
        if poly1.geom_type != 'Polygon':
            if hasattr(poly1, 'geoms'):
                polygons = [g for g in poly1.geoms if g.geom_type == 'Polygon']
                if polygons:
                    poly1 = max(polygons, key=lambda p: p.area)
                else:
                    return 0.0
        
        if poly2.geom_type != 'Polygon':
            if hasattr(poly2, 'geoms'):
                polygons = [g for g in poly2.geoms if g.geom_type == 'Polygon']
                if polygons:
                    poly2 = max(polygons, key=lambda p: p.area)
                else:
                    return 0.0
        
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    except Exception as e:
        print(f"Warning: IoU calculation failed: {e}")
        return -1.0


def simplify_polygon(segmentation: List[float], tolerance: float, 
                     min_points: int = 4) -> Tuple[List[float], Dict[str, Any]]:
    """
    Simplify a single polygon segmentation.
    
    Args:
        segmentation: COCO format segmentation (flat list)
        tolerance: Douglas-Peucker tolerance
        min_points: Minimum number of points to keep
        
    Returns:
        Tuple of (simplified_segmentation, metrics_dict)
    """
    original_points = segmentation_to_points(segmentation)
    original_count = len(original_points)
    
    if original_count <= min_points:
        return segmentation, {
            'original_points': original_count,
            'simplified_points': original_count,
            'reduction_percent': 0.0,
            'iou': 1.0,
            'original_area': calculate_polygon_area(original_points),
            'simplified_area': calculate_polygon_area(original_points),
            'area_preservation': 100.0
        }
    
    # Apply Douglas-Peucker simplification
    simplified_points = douglas_peucker(original_points, tolerance)
    
    # Ensure minimum points (for valid polygon)
    while len(simplified_points) < min_points and tolerance > 0.1:
        tolerance *= 0.5
        simplified_points = douglas_peucker(original_points, tolerance)
    
    # If still too few points, use original
    if len(simplified_points) < 3:
        simplified_points = original_points
    
    simplified_count = len(simplified_points)
    
    # Calculate metrics
    original_area = calculate_polygon_area(original_points)
    simplified_area = calculate_polygon_area(simplified_points)
    
    area_preservation = (simplified_area / original_area * 100) if original_area > 0 else 100.0
    
    iou = calculate_iou(original_points, simplified_points)
    
    reduction_percent = ((original_count - simplified_count) / original_count * 100) if original_count > 0 else 0.0
    
    simplified_segmentation = points_to_segmentation(simplified_points)
    
    metrics = {
        'original_points': original_count,
        'simplified_points': simplified_count,
        'reduction_percent': round(reduction_percent, 2),
        'iou': round(iou, 4) if iou >= 0 else 'N/A',
        'original_area': round(original_area, 2),
        'simplified_area': round(simplified_area, 2),
        'area_preservation': round(area_preservation, 2)
    }
    
    return simplified_segmentation, metrics


def process_coco_file(input_path: str, output_path: str, tolerance: float, 
                      min_points: int = 4) -> Dict[str, Any]:
    """
    Process a COCO annotation file and simplify all polygon segmentations.
    
    Args:
        input_path: Path to input COCO JSON file
        output_path: Path to output simplified JSON file
        tolerance: Douglas-Peucker tolerance
        min_points: Minimum points to keep per polygon
        
    Returns:
        Summary statistics dictionary
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    all_metrics = []
    total_annotations = len(coco_data.get('annotations', []))
    
    print(f"\n  Processing {total_annotations} annotations...")
    
    for i, annotation in enumerate(coco_data.get('annotations', [])):
        if 'segmentation' not in annotation:
            continue
        
        new_segmentation = []
        annotation_metrics = []
        
        for seg in annotation['segmentation']:
            if isinstance(seg, list) and len(seg) >= 6:  # Minimum 3 points (6 coords)
                simplified_seg, metrics = simplify_polygon(seg, tolerance, min_points)
                new_segmentation.append(simplified_seg)
                annotation_metrics.append(metrics)
            else:
                new_segmentation.append(seg)
        
        annotation['segmentation'] = new_segmentation
        all_metrics.extend(annotation_metrics)
        
        # Progress indicator
        if (i + 1) % 100 == 0 or (i + 1) == total_annotations:
            print(f"    Processed {i + 1}/{total_annotations} annotations", end='\r')
    
    print()  # New line after progress
    
    # Save simplified data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2)
    
    # Calculate summary statistics
    if all_metrics:
        total_original_points = sum(m['original_points'] for m in all_metrics)
        total_simplified_points = sum(m['simplified_points'] for m in all_metrics)
        avg_reduction = sum(m['reduction_percent'] for m in all_metrics) / len(all_metrics)
        
        iou_values = [m['iou'] for m in all_metrics if isinstance(m['iou'], (int, float)) and m['iou'] >= 0]
        avg_iou = sum(iou_values) / len(iou_values) if iou_values else -1
        
        avg_area_preservation = sum(m['area_preservation'] for m in all_metrics) / len(all_metrics)
        
        summary = {
            'total_polygons': len(all_metrics),
            'total_original_points': total_original_points,
            'total_simplified_points': total_simplified_points,
            'overall_reduction_percent': round((total_original_points - total_simplified_points) / total_original_points * 100, 2) if total_original_points > 0 else 0,
            'average_reduction_percent': round(avg_reduction, 2),
            'average_iou': round(avg_iou, 4) if avg_iou >= 0 else 'N/A (shapely not installed)',
            'average_area_preservation': round(avg_area_preservation, 2),
            'min_iou': round(min(iou_values), 4) if iou_values else 'N/A',
            'max_iou': round(max(iou_values), 4) if iou_values else 'N/A',
        }
    else:
        summary = {'total_polygons': 0}
    
    return summary


def backup_file(file_path: str) -> str:
    """
    Create a backup of a file.
    
    Args:
        file_path: Path to the file to backup
        
    Returns:
        Path to the backup file
    """
    path = Path(file_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.parent / f"{path.stem}_original_{timestamp}{path.suffix}"
    shutil.copy2(file_path, backup_path)
    return str(backup_path)


def generate_quality_assessment(summary: Dict[str, Any]) -> str:
    """
    Generate a quality assessment based on the simplification metrics.
    
    Args:
        summary: Summary statistics dictionary
        
    Returns:
        Quality assessment string
    """
    assessment = []
    
    # IoU assessment
    if isinstance(summary.get('average_iou'), (int, float)):
        iou = summary['average_iou']
        if iou >= 0.99:
            assessment.append("✅ Excellent: IoU >= 0.99 - Simplification preserves polygon shape almost perfectly")
        elif iou >= 0.95:
            assessment.append("✅ Good: IoU >= 0.95 - Simplification preserves polygon shape well")
        elif iou >= 0.90:
            assessment.append("⚠️ Acceptable: IoU >= 0.90 - Minor shape differences, consider lower tolerance")
        elif iou >= 0.80:
            assessment.append("⚠️ Warning: IoU >= 0.80 - Noticeable shape differences, recommend lower tolerance")
        else:
            assessment.append("❌ Poor: IoU < 0.80 - Significant shape loss, use lower tolerance")
    
    # Area preservation assessment
    area_pres = summary.get('average_area_preservation', 100)
    if 98 <= area_pres <= 102:
        assessment.append("✅ Excellent: Area preserved within 2%")
    elif 95 <= area_pres <= 105:
        assessment.append("✅ Good: Area preserved within 5%")
    elif 90 <= area_pres <= 110:
        assessment.append("⚠️ Acceptable: Area preserved within 10%")
    else:
        assessment.append("❌ Warning: Area change exceeds 10%")
    
    # Point reduction assessment
    reduction = summary.get('overall_reduction_percent', 0)
    if reduction >= 70:
        assessment.append(f"📊 High reduction: {reduction}% fewer points - Great for performance")
    elif reduction >= 40:
        assessment.append(f"📊 Moderate reduction: {reduction}% fewer points")
    elif reduction >= 20:
        assessment.append(f"📊 Low reduction: {reduction}% fewer points - Consider higher tolerance")
    else:
        assessment.append(f"📊 Minimal reduction: {reduction}% fewer points - Polygons already simple")
    
    return "\n".join(assessment)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def get_category_colors(num_categories: int) -> Dict[int, Tuple[int, int, int]]:
    """Generate distinct colors for each category."""
    colors = {}
    for i in range(num_categories + 1):
        hue = (i * 137) % 360  # Golden angle for distinct colors
        c = np.uint8([[[hue // 2, 255, 255]]])
        rgb = cv2.cvtColor(c, cv2.COLOR_HSV2BGR)[0][0]
        colors[i] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    return colors


def draw_polygon_cv2(img: np.ndarray, points: List[Tuple[float, float]], 
                     color: Tuple[int, int, int], thickness: int = 2,
                     fill_alpha: float = 0.3) -> np.ndarray:
    """Draw a polygon on image using OpenCV."""
    if len(points) < 3:
        return img
    
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    
    # Create overlay for semi-transparent fill
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], color)
    img = cv2.addWeighted(overlay, fill_alpha, img, 1 - fill_alpha, 0)
    
    # Draw outline
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    
    # Draw points
    for pt in points:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 3, color, -1)
    
    return img


def visualize_comparison(image_path: str, annotations: List[Dict], 
                         categories: Dict[int, str], tolerance: float,
                         output_path: str = None) -> np.ndarray:
    """Create side-by-side comparison visualization."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    h, w = img.shape[:2]
    
    img_original = img.copy()
    img_simplified = img.copy()
    
    colors = get_category_colors(max(categories.keys()) if categories else 10)
    
    total_original_points = 0
    total_simplified_points = 0
    
    for ann in annotations:
        cat_id = ann.get('category_id', 0)
        color = colors.get(cat_id, (0, 255, 0))
        
        for seg in ann.get('segmentation', []):
            if isinstance(seg, list) and len(seg) >= 6:
                original_points = segmentation_to_points(seg)
                total_original_points += len(original_points)
                img_original = draw_polygon_cv2(img_original, original_points, color)
                
                simplified_points = douglas_peucker(original_points, tolerance)
                if len(simplified_points) < 3:
                    simplified_points = original_points
                total_simplified_points += len(simplified_points)
                img_simplified = draw_polygon_cv2(img_simplified, simplified_points, color)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(img_original, (0, 0), (400, 80), (0, 0, 0), -1)
    cv2.rectangle(img_simplified, (0, 0), (400, 80), (0, 0, 0), -1)
    
    cv2.putText(img_original, "ORIGINAL", (10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(img_original, f"Points: {total_original_points}", (10, 60), font, 0.6, (200, 200, 200), 1)
    
    cv2.putText(img_simplified, f"SIMPLIFIED (tol={tolerance})", (10, 30), font, 0.8, (255, 255, 255), 2)
    reduction = ((total_original_points - total_simplified_points) / total_original_points * 100) if total_original_points > 0 else 0
    cv2.putText(img_simplified, f"Points: {total_simplified_points} (-{reduction:.1f}%)", (10, 60), font, 0.6, (200, 200, 200), 1)
    
    combined = np.hstack([img_original, img_simplified])
    cv2.line(combined, (w, 0), (w, h), (255, 255, 255), 2)
    
    if output_path:
        cv2.imwrite(output_path, combined)
        print(f"  Saved: {output_path}")
    
    return combined


def visualize_overlay(image_path: str, annotations: List[Dict], 
                      categories: Dict[int, str], tolerance: float,
                      output_path: str = None) -> np.ndarray:
    """Create overlay visualization showing both original and simplified on same image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    result = img.copy()
    total_original_points = 0
    total_simplified_points = 0
    
    for ann in annotations:
        for seg in ann.get('segmentation', []):
            if isinstance(seg, list) and len(seg) >= 6:
                original_points = segmentation_to_points(seg)
                simplified_points = douglas_peucker(original_points, tolerance)
                if len(simplified_points) < 3:
                    simplified_points = original_points
                
                total_original_points += len(original_points)
                total_simplified_points += len(simplified_points)
                
                # Draw original in red
                pts_orig = np.array(original_points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(result, [pts_orig], True, (0, 0, 255), 1)
                
                # Draw simplified in green
                pts_simp = np.array(simplified_points, dtype=np.int32).reshape((-1, 1, 2))
                overlay = result.copy()
                cv2.fillPoly(overlay, [pts_simp], (0, 255, 0))
                result = cv2.addWeighted(overlay, 0.2, result, 0.8, 0)
                cv2.polylines(result, [pts_simp], True, (0, 255, 0), 2)
                
                for pt in original_points:
                    cv2.circle(result, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)
                for pt in simplified_points:
                    cv2.circle(result, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
    
    # Add legend
    cv2.rectangle(result, (0, 0), (350, 100), (0, 0, 0), -1)
    cv2.putText(result, "Legend:", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.circle(result, (20, 50), 5, (0, 0, 255), -1)
    cv2.putText(result, f"Original ({total_original_points} pts)", (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.circle(result, (20, 80), 5, (0, 255, 0), -1)
    reduction = ((total_original_points - total_simplified_points) / total_original_points * 100) if total_original_points > 0 else 0
    cv2.putText(result, f"Simplified ({total_simplified_points} pts, -{reduction:.1f}%)", (35, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    if output_path:
        cv2.imwrite(output_path, result)
        print(f"  Saved: {output_path}")
    
    return result


def run_visualization(args):
    """Run visualization mode."""
    if not CV2_AVAILABLE:
        print("Error: OpenCV is required for visualization.")
        print("Install with: pip install opencv-python")
        return
    
    base_path = Path(__file__).parent
    dataset_map = {
        'train': ('train', 'train_annotations.json'),
        'val': ('val', 'val_annotations.json'),
        'test': ('test', 'test_annotations.json')
    }
    
    folder, annotation_file = dataset_map[args.dataset]
    annotation_path = base_path / folder / annotation_file
    images_path = base_path / folder / 'images'
    
    if not annotation_path.exists():
        print(f"Error: Annotation file not found: {annotation_path}")
        return
    
    if not images_path.exists():
        print(f"Error: Images folder not found: {images_path}")
        return
    
    print(f"Loading annotations from: {annotation_path}")
    
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)
    
    categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
    images = {img['id']: img['file_name'] for img in coco_data.get('images', [])}
    
    annotations_by_image = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    print(f"Found {len(images)} images, {len(categories)} categories")
    
    # Setup output directory
    output_dir = None
    if args.save_images:
        output_dir = base_path / 'comparison_output' / args.dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    # Get images to process
    if args.image:
        image_id = None
        for img_id, filename in images.items():
            if filename == args.image or Path(filename).stem == Path(args.image).stem:
                image_id = img_id
                break
        
        if image_id is None:
            print(f"Error: Image '{args.image}' not found")
            print(f"Available: {list(images.values())[:5]}...")
            return
        
        images_to_process = [(image_id, images[image_id])]
    else:
        images_to_process = list(images.items())
    
    print(f"\nProcessing {len(images_to_process)} image(s) with tolerance={args.tolerance}")
    print("Press any key for next image, 'q' to quit")
    print("=" * 60)
    
    for img_id, filename in images_to_process:
        image_path = images_path / filename
        
        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue
        
        annotations = annotations_by_image.get(img_id, [])
        if not annotations:
            print(f"Skipping {filename}: No annotations")
            continue
        
        print(f"\nImage: {filename} ({len(annotations)} annotations)")
        
        output_path = None
        if output_dir:
            suffix = '_overlay' if args.overlay else '_comparison'
            output_path = str(output_dir / f"{Path(filename).stem}{suffix}.jpg")
        
        if args.overlay:
            result = visualize_overlay(str(image_path), annotations, categories, args.tolerance, output_path)
        else:
            result = visualize_comparison(str(image_path), annotations, categories, args.tolerance, output_path)
        
        if result is not None and not args.save_images:
            # Resize for display if needed
            h, w = result.shape[:2]
            max_width = 1600
            if w > max_width:
                scale = max_width / w
                result = cv2.resize(result, (int(w * scale), int(h * scale)))
            
            cv2.imshow(f"Comparison - {filename} (any key=next, q=quit)", result)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
    
    cv2.destroyAllWindows()
    
    if args.save_images:
        print(f"\n✅ All comparisons saved to: {output_dir}")
    
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description='Simplify polygon segmentations in COCO format annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run coco_polygon_simplification.py --tolerance 1.5
    python coco_polygon_simplification.py
    python coco_polygon_simplification.py --tolerance 2.0
    python coco_polygon_simplification.py --tolerance 1.0 --min-points 6
    python coco_polygon_simplification.py --dry-run  # Preview only, no changes
    
    # Visualization mode:
    python coco_polygon_simplification.py --visualize --dataset val
    python coco_polygon_simplification.py --visualize --dataset val --image 0007.png
    python coco_polygon_simplification.py --visualize --dataset val --save-images
        """
    )
    
    parser.add_argument('--tolerance', type=float, default=1.5,
                        help='Douglas-Peucker tolerance (default: 1.5). Higher = more simplification')
    parser.add_argument('--min-points', type=int, default=4,
                        help='Minimum points to keep per polygon (default: 4)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run analysis without saving simplified files')
    
    # Visualization arguments
    parser.add_argument('--visualize', action='store_true',
                        help='Enable visualization mode to compare original vs simplified')
    parser.add_argument('--dataset', type=str, choices=['train', 'val', 'test'], default='val',
                        help='Dataset to visualize (default: val)')
    parser.add_argument('--image', type=str, default=None,
                        help='Specific image filename to visualize')
    parser.add_argument('--save-images', action='store_true',
                        help='Save comparison images to comparison_output folder')
    parser.add_argument('--overlay', action='store_true',
                        help='Show overlay view (both polygons on same image)')
    
    args = parser.parse_args()
    
    # Handle visualization mode
    if args.visualize:
        run_visualization(args)
        return
    
    # Define file paths
    base_path = Path(__file__).parent
    annotation_files = [
        base_path / 'test' / 'test_annotations.json',
        base_path / 'train' / 'train_annotations.json',
        base_path / 'val' / 'val_annotations.json',
    ]
    
    print("=" * 70)
    print("COCO Polygon Simplification Tool")
    print("=" * 70)
    print(f"\nSettings:")
    print(f"  Tolerance: {args.tolerance}")
    print(f"  Min points: {args.min_points}")
    print(f"  Dry run: {args.dry_run}")
    print(f"\nShapely available: {SHAPELY_AVAILABLE}")
    
    all_summaries = {}
    
    for file_path in annotation_files:
        if not file_path.exists():
            print(f"\n⚠️ File not found: {file_path}")
            continue
        
        print(f"\n{'=' * 70}")
        print(f"Processing: {file_path.name}")
        print(f"{'=' * 70}")
        
        # Define output path
        if args.dry_run:
            output_path = str(file_path.parent / f"{file_path.stem}_simplified_preview.json")
        else:
            # Create simplified version (original stays untouched)
            output_path = str(file_path.parent / f"{file_path.stem}_simplified.json")
        
        # Process the file
        summary = process_coco_file(
            str(file_path), 
            output_path, 
            args.tolerance, 
            args.min_points
        )
        
        all_summaries[file_path.name] = summary
        
        # Print summary
        print(f"\n  Summary for {file_path.name}:")
        print(f"  {'-' * 50}")
        print(f"  Total polygons processed: {summary.get('total_polygons', 0)}")
        print(f"  Original total points: {summary.get('total_original_points', 0):,}")
        print(f"  Simplified total points: {summary.get('total_simplified_points', 0):,}")
        print(f"  Overall reduction: {summary.get('overall_reduction_percent', 0)}%")
        print(f"  Average IoU: {summary.get('average_iou', 'N/A')}")
        print(f"  Min IoU: {summary.get('min_iou', 'N/A')}")
        print(f"  Max IoU: {summary.get('max_iou', 'N/A')}")
        print(f"  Average area preservation: {summary.get('average_area_preservation', 'N/A')}%")
        
        print(f"\n  Quality Assessment:")
        assessment = generate_quality_assessment(summary)
        for line in assessment.split('\n'):
            print(f"  {line}")
        
        if not args.dry_run:
            print(f"\n  ✅ Simplified file saved: {Path(output_path).name}")
        else:
            # Clean up preview file
            if Path(output_path).exists():
                os.remove(output_path)
            print(f"\n  ℹ️ Dry run - no files were modified")
    
    # Final summary
    print(f"\n{'=' * 70}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 70}")
    
    total_orig = sum(s.get('total_original_points', 0) for s in all_summaries.values())
    total_simp = sum(s.get('total_simplified_points', 0) for s in all_summaries.values())
    total_polygons = sum(s.get('total_polygons', 0) for s in all_summaries.values())
    
    print(f"Files processed: {len(all_summaries)}")
    print(f"Total polygons: {total_polygons:,}")
    print(f"Total original points: {total_orig:,}")
    print(f"Total simplified points: {total_simp:,}")
    
    if total_orig > 0:
        overall_reduction = (total_orig - total_simp) / total_orig * 100
        print(f"Overall point reduction: {overall_reduction:.2f}%")
    
    # Calculate average IoU across all files
    all_ious = [s.get('average_iou') for s in all_summaries.values() 
                if isinstance(s.get('average_iou'), (int, float))]
    if all_ious:
        print(f"Overall average IoU: {sum(all_ious) / len(all_ious):.4f}")
    
    print(f"\n{'=' * 70}")
    print("Recommendations:")
    print(f"{'=' * 70}")
    print("""
If IoU is too low (< 0.95):
    → Decrease tolerance (e.g., --tolerance 0.5)
    
If reduction is too low (< 30%):
    → Increase tolerance (e.g., --tolerance 3.0)
    
To find optimal tolerance, try:
    python coco_polygon_simplification.py --tolerance 0.5 --dry-run
    python coco_polygon_simplification.py --tolerance 1.0 --dry-run
    python coco_polygon_simplification.py --tolerance 2.0 --dry-run
    python coco_polygon_simplification.py --tolerance 3.0 --dry-run
    
Then choose the tolerance with best IoU/reduction trade-off.
""")
    
    # Save report
    report_path = base_path / 'simplification_report.json'
    report = {
        'timestamp': datetime.now().isoformat(),
        'settings': {
            'tolerance': args.tolerance,
            'min_points': args.min_points,
            'dry_run': args.dry_run
        },
        'file_summaries': {str(k): v for k, v in all_summaries.items()},
        'overall': {
            'total_polygons': total_polygons,
            'total_original_points': total_orig,
            'total_simplified_points': total_simp,
            'overall_reduction_percent': round((total_orig - total_simp) / total_orig * 100, 2) if total_orig > 0 else 0
        }
    }
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report saved to: {report_path.name}")


if __name__ == '__main__':
    main()
