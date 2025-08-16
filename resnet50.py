import os
import json
from datetime import datetime
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from pycocotools import mask as mask_utils
from tqdm import tqdm
from PIL import Image
import numpy as np

class DeepLabV3Predictor:
    def __init__(self, model_path, config=None):
        # Default configuration
        self.config = {
            'img_size': 512,
            'confidence_threshold': 0.7,
            'min_area_threshold': 100,  # minimum pixels for valid mask
            'save_low_confidence': False,  # save images that don't meet threshold
            'overlay_alpha': 0.4,
            'device': 'cpu'
        }
        
        # Update with user config
        if config:
            self.config.update(config)
        
        self.model_path = model_path
        self.img_size = self.config['img_size']
        self.confidence_threshold = self.config['confidence_threshold']
        self.device = torch.device(self.config['device'])

        
        self.model = self.load_model()
        self.transform = A.Compose([
            A.Resize(self.img_size, self.img_size), 
            A.Normalize(mean=(0.26274775101559783, 0.29577177815541195, 0.171236798812585), std=(0.2910812701319592, 0.319100625625642, 0.2066569858918171)), 
            ToTensorV2()
        ])
        
        # Class mapping - แก้ไขตามโมเดลของคุณ
        self.class_names = {0: "background", 1: "ใบจุดสีน้ำตาล"}
        self.class_colors = {0: (0, 0, 0), 1: (0, 149, 51)}
        
        print(f"✅ Model loaded with confidence threshold: {self.confidence_threshold}")
        print(f"📊 Configuration: {json.dumps(self.config, indent=2)}")
    
    def load_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model = smp.DeepLabV3Plus(
            encoder_name='resnet50', 
            encoder_weights=None, 
            in_channels=3, 
            classes=2
        )
        state = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_state = {}
        for k, v in state.items():
            if k.startswith('module.'):
                new_state[k.replace('module.', '')] = v
            else:
                new_state[k] = v
        model.load_state_dict(new_state)
        model.to(self.device)
        model.eval()
        return model
    
    def predict_with_confidence(self, image_path):
        """Predict segmentation mask with confidence scores"""
        img = Image.open(image_path).convert('RGB')
        original_size = img.size  # (width, height)
        
        # Transform for inference
        inp = self.transform(image=np.array(img))['image'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(inp)
            # Get probabilities using softmax
            probs = F.softmax(logits, dim=1)
            # Get predicted class and max confidence
            max_probs, pred = torch.max(probs, dim=1)
            
            pred = pred.squeeze(0).cpu().numpy().astype(np.uint8)
            confidence_map = max_probs.squeeze(0).cpu().numpy()
        
        # Resize prediction back to original size
        pred_resized = cv2.resize(pred, original_size, interpolation=cv2.INTER_NEAREST)
        confidence_resized = cv2.resize(confidence_map, original_size, interpolation=cv2.INTER_LINEAR)
        
        return np.array(img), pred_resized, confidence_resized
    
    def evaluate_prediction_quality(self, mask, confidence_map, class_id=1):
        """Evaluate if prediction meets quality thresholds"""
        # Get mask for specific class
        class_mask = (mask == class_id)
        
        if not class_mask.any():
            return {
                'meets_threshold': False,
                'avg_confidence': 0.0,
                'max_confidence': 0.0,
                'area': 0,
                'reason': 'no_detection'
            }
        
        # Calculate confidence metrics for the class
        class_confidences = confidence_map[class_mask]
        avg_confidence = float(np.mean(class_confidences))
        max_confidence = float(np.max(class_confidences))
        area = int(np.sum(class_mask))
        
        # Check thresholds
        meets_confidence = avg_confidence >= self.confidence_threshold
        meets_area = area >= self.config['min_area_threshold']
        
        meets_threshold = meets_confidence and meets_area
        
        # Determine reason if not meeting threshold
        reason = 'passed'
        if not meets_confidence and not meets_area:
            reason = 'low_confidence_and_small_area'
        elif not meets_confidence:
            reason = 'low_confidence'
        elif not meets_area:
            reason = 'small_area'
        
        return {
            'meets_threshold': meets_threshold,
            'avg_confidence': avg_confidence,
            'max_confidence': max_confidence,
            'area': area,
            'reason': reason
        }
    
    def predict(self, image_path):
        """Backward compatibility method"""
        img, mask, confidence = self.predict_with_confidence(image_path)
        return img, mask
    
    def create_overlay(self, image, mask, confidence_map=None):
        """Create overlay visualization with optional confidence visualization"""
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for cls_id, color in self.class_colors.items():
            if cls_id > 0:  # Skip background
                class_mask = mask == cls_id
                if confidence_map is not None:
                    # Modulate color intensity by confidence
                    conf_values = confidence_map[class_mask]
                    if len(conf_values) > 0:
                        avg_conf = np.mean(conf_values)
                        intensity = max(0.3, avg_conf)  # Minimum 30% intensity
                        color = tuple(int(c * intensity) for c in color)
                
                color_mask[class_mask] = color
        
        # Blend with alpha
        alpha = self.config['overlay_alpha']
        overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
        return overlay
    
    def create_confidence_heatmap(self, confidence_map):
        """Create confidence heatmap visualization"""
        # Normalize to 0-255
        conf_normalized = (confidence_map * 255).astype(np.uint8)
        # Apply colormap (warmer colors = higher confidence)
        heatmap = cv2.applyColorMap(conf_normalized, cv2.COLORMAP_JET)
        return heatmap
    
    def mask_to_coco_annotation(self, mask, class_id, annotation_id, image_id, confidence_map=None):
        """Convert binary mask to COCO annotation format - CVAT compatible"""
        # Convert to binary mask for specific class
        binary_mask = (mask == class_id).astype(np.uint8)
        
        if binary_mask.sum() == 0:
            return None
        
        # Encode mask
        encoded_mask = mask_utils.encode(np.asfortranarray(binary_mask))
        area = mask_utils.area(encoded_mask)
        bbox = mask_utils.toBbox(encoded_mask)
        
        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": class_id,
            "segmentation": {
                "size": [int(encoded_mask["size"][0]), int(encoded_mask["size"][1])],
                "counts": encoded_mask["counts"].decode('utf-8')
            },
            "area": float(area),
            "bbox": [float(x) for x in bbox.tolist()],
            "iscrowd": 0
        }
        
        return annotation
    
    def export_to_coco(self, image_paths, output_dir="coco_export"):
        """Export predictions to COCO format with confidence filtering"""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "overlays"), exist_ok=True)
        
        if self.config['save_low_confidence']:
            os.makedirs(os.path.join(output_dir, "low_confidence"), exist_ok=True)
        
        # COCO format structure - Fixed for CVAT compatibility
        coco_data = {
            "info": {
                "description": "DeepLabV3+ Segmentation Results",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "DeepLabV3+ Model",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown License",
                    "url": ""
                }
            ],
            "images": [],
            "annotations": [],
            "categories": [],
            "statistics": {
                "total_processed": 0,
                "passed_threshold": 0,
                "failed_threshold": 0,
                "failure_reasons": {}
            }
        }
        
        # Add categories
        for class_id, class_name in self.class_names.items():
            if class_id > 0:  # Skip background
                coco_data["categories"].append({
                    "id": class_id,
                    "name": class_name,
                    "supercategory": "object"
                })
        
        annotation_id = 1
        valid_image_id = 1
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"Processing {i}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            try:
                # Predict with confidence
                image, mask, confidence_map = self.predict_with_confidence(image_path)
                
                # Evaluate prediction quality
                quality_info = self.evaluate_prediction_quality(mask, confidence_map)
                coco_data["statistics"]["total_processed"] += 1
                
                # Track failure reasons
                reason = quality_info['reason']
                if reason in coco_data["statistics"]["failure_reasons"]:
                    coco_data["statistics"]["failure_reasons"][reason] += 1
                else:
                    coco_data["statistics"]["failure_reasons"][reason] = 1
                
                print(f"  📊 Avg confidence: {quality_info['avg_confidence']:.3f}")
                print(f"  📐 Area: {quality_info['area']} pixels")
                print(f"  ✅ Status: {quality_info['reason']}")
                
                if quality_info['meets_threshold']:
                    # High confidence prediction - save to main dataset
                    coco_data["statistics"]["passed_threshold"] += 1
                    
                    # Save original image
                    image_filename = f"{valid_image_id:06d}.jpg"
                    image_save_path = os.path.join(output_dir, "images", image_filename)
                    Image.fromarray(image).save(image_save_path)
                    
                    # Save overlay
                    overlay = self.create_overlay(image, mask, confidence_map)
                    overlay_filename = f"{valid_image_id:06d}_overlay.jpg"
                    overlay_save_path = os.path.join(output_dir, "overlays", overlay_filename)
                    Image.fromarray(overlay).save(overlay_save_path)
                    
                    # Add image info - CVAT compatible format
                    height, width = image.shape[:2]
                    coco_data["images"].append({
                        "id": valid_image_id,
                        "width": width,
                        "height": height,
                        "file_name": image_filename,
                        "license": 1,
                        "flickr_url": "",
                        "coco_url": "",
                        "date_captured": datetime.now().isoformat()
                    })
                    
                    # Create annotations for each class (skip background)
                    for class_id in range(1, len(self.class_names)):
                        annotation = self.mask_to_coco_annotation(
                            mask, class_id, annotation_id, valid_image_id, confidence_map
                        )
                        if annotation:
                            coco_data["annotations"].append(annotation)
                            annotation_id += 1
                    
                    valid_image_id += 1
                    
                else:
                    # Low confidence prediction
                    coco_data["statistics"]["failed_threshold"] += 1
                    
                    if self.config['save_low_confidence']:
                        # Save to low confidence folder for review
                        low_conf_filename = f"lowconf_{i:06d}_{reason}.jpg"
                        overlay = self.create_overlay(image, mask, confidence_map)
                        low_conf_path = os.path.join(output_dir, "low_confidence", low_conf_filename)
                        Image.fromarray(overlay).save(low_conf_path)
                        
                        print(f"  💾 Saved to low_confidence folder")
                
            except Exception as e:
                print(f"  ❌ Error processing {image_path}: {str(e)}")
                continue
        
        # Save COCO JSON - Main annotation file
        json_path = os.path.join(output_dir, "annotations.json")
        
        # Remove statistics before saving main COCO file (CVAT compatibility)
        coco_export = {
            "info": coco_data["info"],
            "licenses": coco_data["licenses"],
            "images": coco_data["images"],
            "annotations": coco_data["annotations"],
            "categories": coco_data["categories"]
        }
        
        with open(json_path, 'w') as f:
            json.dump(coco_export, f, indent=2)
        
        # Save statistics separately
        stats_path = os.path.join(output_dir, "statistics.json")
        with open(stats_path, 'w') as f:
            json.dump(coco_data["statistics"], f, indent=2)
        
        # Print summary
        stats = coco_data["statistics"]
        print(f"\n🎉 COCO export completed!")
        print(f"📊 Statistics:")
        print(f"  Total processed: {stats['total_processed']}")
        print(f"  Passed threshold: {stats['passed_threshold']}")
        print(f"  Failed threshold: {stats['failed_threshold']}")
        print(f"  Success rate: {stats['passed_threshold']/max(1, stats['total_processed'])*100:.1f}%")
        print(f"\n📈 Failure reasons:")
        for reason, count in stats['failure_reasons'].items():
            print(f"  {reason}: {count}")
        print(f"\n📁 Output directory: {output_dir}")
        print(f"🏷️  Total valid annotations: {len(coco_data['annotations'])}")
        
        return coco_data

def get_images_from_directory(image_dir, extensions=['.jpg', '.jpeg', '.png', '.bmp', '.tiff']):
    """Get all image files from directory"""
    import glob
    
    image_paths = []
    for ext in extensions:
        pattern = os.path.join(image_dir, f"*{ext}")
        image_paths.extend(glob.glob(pattern))
        pattern = os.path.join(image_dir, f"*{ext.upper()}")
        image_paths.extend(glob.glob(pattern))
    
    return sorted(image_paths)

# Usage Example
if __name__ == "__main__":
    MODEL_PATH = '116-cbsd-model.pt'
    
    # Custom configuration
    config = {
        'img_size': 512,
        'confidence_threshold': 0.85,        # Only save predictions with >70% confidence
        'min_area_threshold': 100,          # Minimum 100 pixels
        'save_low_confidence': True,        # Save low confidence for review
        'overlay_alpha': 0.4,
        'device': 'cpu'  # or 'cuda' if available
    }
    
    # Initialize predictor
    predictor = DeepLabV3Predictor(MODEL_PATH, config)
    
    # =========================
    # Option 1: Single image prediction
    # =========================
    single_image_path = "/Users/khemikadeedaungphan/Desktop/cassava-disease/cassava_output/1616821564298.jpg"
    if os.path.exists(single_image_path):
        image, mask, confidence = predictor.predict_with_confidence(single_image_path)
        quality_info = predictor.evaluate_prediction_quality(mask, confidence)
        
        print(f"Single image prediction:")
        print(f"  Average confidence: {quality_info['avg_confidence']:.3f}")
        print(f"  Max confidence: {quality_info['max_confidence']:.3f}")
        print(f"  Area: {quality_info['area']} pixels")
        print(f"  Meets threshold: {quality_info['meets_threshold']}")
        
        if quality_info['meets_threshold']:
            overlay = predictor.create_overlay(image, mask, confidence)
            Image.fromarray(overlay).save("prediction_overlay.jpg")
            print("✅ High confidence prediction saved!")
        else:
            print(f"❌ Prediction rejected: {quality_info['reason']}")
    
    # =========================
    # Option 2: Process entire directory
    # =========================
    IMAGE_DIR = ""
    OUTPUT_DIR = "coco_dataset_output"
    
    if os.path.exists(IMAGE_DIR):
        # Get all images from directory
        image_paths = get_images_from_directory(IMAGE_DIR)
        
        if image_paths:
            print(f"\n🔍 Found {len(image_paths)} images in {IMAGE_DIR}")
            print("Sample images:")
            for i, path in enumerate(image_paths[:5]):
                print(f"  {i+1}. {os.path.basename(path)}")
            if len(image_paths) > 5:
                print(f"  ... and {len(image_paths) - 5} more")
            
            # Export to COCO format with confidence filtering
            coco_data = predictor.export_to_coco(image_paths, OUTPUT_DIR)
            
        else:
            print(f"❌ No images found in {IMAGE_DIR}")
    else:
        print(f"❌ Directory not found: {IMAGE_DIR}")
        print("\n💡 Usage instructions:")
        print("1. Set IMAGE_DIR to your image folder path")
        print("2. Adjust config parameters as needed")
        print("3. Run the script")
    
