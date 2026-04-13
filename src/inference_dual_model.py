import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from ultralytics import YOLO
import sys
from pathlib import Path

# Add project root to sys.path to allow importing MultiModelTrainer
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.MultiModelImageClassification import MultiModelTrainer
from src.inference import MaskPostProcessor

class DualModelPipeline:
    """
    Integration Pipeline for Stage 1 (Segmentor) -> Stage 2 (Classifier)
    This handles predicting masks on an original image, processing those masks
    for smoothness, extracting clean crops, and classifying them.
    """
    def __init__(self, segmentor_path, classifier_configs, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading Segmentor: {segmentor_path}")
        self.segmentor = YOLO(segmentor_path)
        self.postprocessor = MaskPostProcessor(min_area=100) # Using minimum area to cull tiny noisy crops
        self.classifiers = {}
        
        # Setup standard image transforms for PyTorch-based classifiers (EfficientNet, ViT)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load specified classifiers
        for cfg in classifier_configs:
            name = cfg['name']
            path_or_model = cfg['model']
            model_type = cfg.get('type', 'yolo')
            class_names = cfg.get('class_names', {})
            
            print(f"Loading Classifier [{name}] (Type: {model_type})")
            
            if model_type == 'yolo':
                self.classifiers[name] = {
                    'obj': YOLO(path_or_model),
                    'type': 'yolo',
                    'names': class_names
                }
            elif model_type == 'pytorch_timm':
                # Leverage MultiModelTrainer to instantiate architecture properly
                arch_name = cfg.get('architecture', 'tf_efficientnetv2_s')
                num_classes = len(class_names)
                
                # Fetch dynamically size and transforms
                trainer = MultiModelTrainer(model_name=arch_name, num_classes=num_classes, device=self.device, pretrained=False)
                
                # Load the state dict
                if isinstance(path_or_model, str):
                    trainer.model.load_state_dict(torch.load(path_or_model, map_location=self.device))
                else: 
                    # If it's a state_dict directly
                    trainer.model.load_state_dict(path_or_model)
                
                model_obj = trainer.model.eval()
                
                self.classifiers[name] = {
                    'obj': model_obj,
                    'type': 'pytorch_timm',
                    'names': class_names,
                    'transform': trainer.transforms['val']
                }
            else:
                # Standard PyTorch model that was saved entirely using torch.save(model, ...)
                if isinstance(path_or_model, str):
                    model_obj = torch.load(path_or_model, map_location=self.device).eval()
                else:
                    model_obj = path_or_model.to(self.device).eval()
                
                self.classifiers[name] = {
                    'obj': model_obj,
                    'type': 'pytorch',
                    'names': class_names,
                    'transform': self.transform 
                }

    def predict(self, image, classifier_name):
        """
        Predict masks using the segmentor, isolate rock instances, and classify each crop.
        """
        # Ensure image is a numpy array
        if isinstance(image, str):
            image = cv2.imread(image)
            
        original_image = image.copy()
            
        # ─── STAGE 1: SEGMENTOR ─────────────────────────────────────
        results = self.segmentor(image, verbose=False)[0]
        if results.masks is None:
            return results, []

        boxes = results.boxes.xyxy.cpu().numpy()
        polygons = results.masks.xy
        
        final_predictions = []
        classifier_info = self.classifiers[classifier_name]
        clf_model = classifier_info['obj']
        clf_type = classifier_info['type']
        idx_to_name = classifier_info['names']
        
        # ─── STAGE 2: CLASSIFIER ────────────────────────────────────
        for orig_box, orig_poly in zip(boxes, polygons):
            # 1. Rasterize YOLO polygon to binary mask for post-processing
            mask_binary = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask_binary, [np.array(orig_poly, dtype=np.int32)], 1)
            
            # 2. Clean mask with PostProcessor (fixes spikes, jagged edges, topologies)
            clean_polys = self.postprocessor.process(mask_binary)
            
            if not clean_polys:
                continue # Skip if polygon became invalid or too small during cleanup
                
            # Assume handling of the primary/largest restored polygon
            clean_poly = clean_polys[0]
            
            # 3. Create fresh bounding box from cleaned polygon
            x, y, w, h = cv2.boundingRect(clean_poly)
            img_h, img_w = image.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img_w, x + w), min(img_h, y + h)
            
            crop = original_image[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
            
            if clf_type == 'yolo':
                # YOLO Classification Inference
                cls_res = clf_model(crop, verbose=False)[0]
                pred_class_id = int(cls_res.probs.top1)
                conf = float(cls_res.probs.top1conf)
                class_name = idx_to_name.get(pred_class_id, cls_res.names.get(pred_class_id, f"Class_{pred_class_id}"))
            else:
                # Standard PyTorch Classification Inference (Timm or Custom Pytorch)
                pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                
                # Fetch specific transform mapping if exists
                model_transform = classifier_info.get('transform', self.transform)
                input_tensor = model_transform(pil_crop).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = clf_model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs[0], dim=0)
                    conf, pred_class_id = torch.max(probs, 0)
                    pred_class_id = pred_class_id.item()
                    conf = conf.item()
                    class_name = idx_to_name.get(pred_class_id, f"Class_{pred_class_id}")

            final_predictions.append({
                'box': [x1, y1, x2, y2],
                'mask': clean_poly,
                'class_id': pred_class_id,
                'class_name': class_name,
                'conf': conf
            })
            
        return results, final_predictions


class DualModelVisualizer:
    """
    Handles robust visualization rendering hollow contours and label text overlays.
    """
    def __init__(self, color_map=None):
        self.color_map = color_map or {}
        
    def _get_color(self, class_id):
        if class_id not in self.color_map:
            np.random.seed(class_id + 42) # Consistent random color based on ID
            self.color_map[class_id] = tuple(map(int, np.random.randint(50, 255, 3)))
        return self.color_map[class_id]

    def draw(self, image, predictions, thickness=2):
        vis_img = image.copy()
        
        for pred in predictions:
            mask = np.array(pred['mask'], dtype=np.int32)
            class_id = pred['class_id']
            class_name = pred['class_name']
            conf = pred['conf']
            
            color = self._get_color(class_id)
            
            # Draw hollow contour
            cv2.polylines(vis_img, [mask], isClosed=True, color=color, thickness=thickness)
            
            # Draw bounding box and label
            x, y = int(pred['box'][0]), int(pred['box'][1])
            label = f"{class_name} {conf:.2f}"
            
            # Background for text
            (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_img, (x, max(0, y - txt_h - 10)), (x + txt_w, max(0, y)), color, -1)
            
            # Text
            cv2.putText(vis_img, label, (x, max(10, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        
        return vis_img
