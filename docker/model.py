# model.py - Docker Version - Foundation Model Challenge for Ultrasound Image Analysis (FMC_UIA)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import json
import numpy as np
import pandas as pd
import glob
import time
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Optional

# Import local modules
from model_factory import MultiTaskModelFactory


class InferenceDataset(Dataset):
    """Inference dataset class"""
    
    def __init__(self, data_root: str, transforms: Optional[A.Compose] = None):
        super().__init__()
        self.data_root = data_root
        self.transforms = transforms
        self.csv_path = os.path.join(self.data_root, 'csv_files')
        
        if not os.path.isdir(self.csv_path):
            raise FileNotFoundError(f"CSV path not found: {self.csv_path}")
            
        all_csv_files = glob.glob(os.path.join(self.csv_path, '*.csv'))
        if not all_csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.csv_path}")
            
        df_list = [pd.read_csv(csv_file) for csv_file in all_csv_files]
        self.dataframe = pd.concat(df_list, ignore_index=True).reset_index(drop=True)
        print(f"Data loading complete. Total samples: {len(self.dataframe)}")

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict:
        record = self.dataframe.iloc[idx]
        task_id = record['task_id']
        task_name = record['task_name']
        
        # Load image
        image_rel_path = record['image_path']
        image_abs_path = os.path.normpath(os.path.join(self.csv_path, image_rel_path))
        image = cv2.imread(image_abs_path)
        
        if image is None:
            print(f"Warning: Unable to load image {image_abs_path}")
            # Return next sample
            return self.__getitem__((idx + 1) % len(self))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        
        # Get mask_path (if segmentation task)
        mask_path = None
        if task_name == 'segmentation' and 'mask_path' in record and pd.notna(record['mask_path']):
            mask_path = record['mask_path']
        
        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        
        # Return data including metadata
        return {
            'image': image,
            'task_id': task_id,
            'task_name': task_name,
            'image_path': image_rel_path,
            'mask_path': mask_path,
            'original_size': (original_height, original_width),
            'index': idx
        }


def inference_collate_fn(batch):
    """Inference collate function that preserves metadata"""
    images = torch.stack([item['image'] for item in batch], 0)
    task_ids = [item['task_id'] for item in batch]
    task_names = [item['task_name'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    mask_paths = [item['mask_path'] for item in batch]
    original_sizes = [item['original_size'] for item in batch]
    indices = [item['index'] for item in batch]
    
    return {
        'image': images,
        'task_id': task_ids,
        'task_name': task_names,
        'image_path': image_paths,
        'mask_path': mask_paths,
        'original_size': original_sizes,
        'index': indices
    }


class Model:
    """
    Foundation Model for Ultrasound Image Analysis
    Supports four task types: segmentation, classification, Regression, detection
    """
    
    def __init__(self):
        """Initialize model and load pretrained weights"""
        print("Initializing model...")
        
        # Set compute device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Model variables, will be initialized in predict()
        self.model = None
        self.task_configs = None
        self.task_id_to_name = None
        
        # Define data preprocessing transforms (no augmentation for inference)
        self.transforms = A.Compose([
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        
        print("Model initialization complete!\n")
    
    def predict(self, data_root: str, output_dir: str, batch_size: int = 8):
        """
        Perform prediction on input data
        
        Args:
            data_root: Data root directory containing csv_files subdirectory
            output_dir: Output results root directory
            batch_size: Batch size, default is 8
        
        Output:
            - Segmentation tasks: Save predicted masks as image files
            - Classification tasks: Save to classification_predictions.json
            - Detection tasks: Save to detection_predictions.json
            - Regression tasks: Save to regression_predictions.json
        """
        print(f"{'='*60}")
        print(f"Starting prediction...")
        print(f"Data directory: {data_root}")
        print(f"Output directory: {output_dir}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*60}\n")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        print(f"Loading dataset...")
        dataset = InferenceDataset(data_root=data_root, transforms=self.transforms)
        
        # Build task_configs from dataset (dynamic construction)
        print(f"\nBuilding task configurations from dataset...")
        self.task_configs = []
        task_config_map = {}
        
        for _, row in dataset.dataframe.iterrows():
            task_id = row['task_id']
            if task_id not in task_config_map:
                task_config = {
                    'task_id': task_id,
                    'task_name': row['task_name'],
                    'num_classes': int(row['num_classes'])
                }
                task_config_map[task_id] = task_config
                self.task_configs.append(task_config)
        
        print(f"Detected {len(self.task_configs)} task configurations")
        for cfg in sorted(self.task_configs, key=lambda x: x['task_id']):
            print(f"  - {cfg['task_id']}: {cfg['task_name']}, num_classes={cfg['num_classes']}")
        
        # Build task_id to task_name mapping
        self.task_id_to_name = {cfg['task_id']: cfg['task_name'] for cfg in self.task_configs}
        
        # Create and load model
        print(f"\nLoading model...")
        self.model = MultiTaskModelFactory(
            encoder_name='efficientnet-b4',
            encoder_weights=None,
            task_configs=self.task_configs
        ).to(self.device)
        
        # Load trained model weights
        model_path = 'best_model.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        print("Model weights loaded successfully!")
        
        # Create data loader (batch processing for faster inference)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=inference_collate_fn
        )
        
        # Batch inference
        print(f"\nStarting inference...")
        classification_results = []
        detection_results = []
        regression_results = []
        task_counts = {}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Prediction progress"):
                images = batch['image'].to(self.device)
                task_ids = batch['task_id']
                task_names = batch['task_name']
                image_paths = batch['image_path']
                mask_paths = batch['mask_path']
                original_sizes = batch['original_size']
                
                # Process each task in batch
                unique_tasks = list(set(task_ids))
                
                for task_id in unique_tasks:
                    # Get indices for all samples of current task
                    task_indices = [i for i, tid in enumerate(task_ids) if tid == task_id]
                    task_images = images[task_indices]
                    
                    # Model inference
                    outputs = self.model(task_images, task_id=task_id)
                    task_name = task_names[task_indices[0]]
                    
                    # Save prediction results for each sample
                    for i, batch_idx in enumerate(task_indices):
                        pred = outputs[i]
                        image_path = image_paths[batch_idx]
                        mask_path = mask_paths[batch_idx]
                        original_size = original_sizes[batch_idx]
                        
                        # Statistics
                        task_counts[task_id] = task_counts.get(task_id, 0) + 1
                        
                        # Process results by task type
                        if task_name == 'segmentation':
                            self._save_segmentation(pred, image_path, mask_path, output_dir, original_size)
                        
                        elif task_name == 'classification':
                            result = self._process_classification(pred, task_id, image_path)
                            classification_results.append(result)
                        
                        elif task_name == 'Regression':
                            result = self._process_regression(pred, task_id, image_path, original_size)
                            regression_results.append(result)
                        
                        elif task_name == 'detection':
                            result = self._process_detection(pred, task_id, image_path, original_size)
                            detection_results.append(result)
        
        # Save aggregated JSON results
        print("\nSaving prediction results...")
        
        if classification_results:
            json_path = os.path.join(output_dir, 'classification_predictions.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(classification_results, f, indent=2, ensure_ascii=False)
            print(f"  - Classification results: {json_path} ({len(classification_results)} samples)")
        
        if detection_results:
            json_path = os.path.join(output_dir, 'detection_predictions.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(detection_results, f, indent=2, ensure_ascii=False)
            print(f"  - Detection results: {json_path} ({len(detection_results)} samples)")
        
        if regression_results:
            json_path = os.path.join(output_dir, 'regression_predictions.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(regression_results, f, indent=2, ensure_ascii=False)
            print(f"  - Regression results: {json_path} ({len(regression_results)} samples)")
        
        # Print statistics
        print(f"\n{'='*60}")
        print("Prediction complete!")
        print(f"{'='*60}")
        print("\nPrediction count by task:")
        for task_id in sorted(task_counts.keys()):
            task_name_str = self.task_id_to_name.get(task_id, 'unknown')
            count = task_counts[task_id]
            print(f"  - {task_id:<25} ({task_name_str:<15}): {count:>5} samples")
        print(f"\nTotal: {sum(task_counts.values())} samples")
        print()
    
    def _save_segmentation(self, pred, image_path, mask_path, output_dir, original_size):
        """Save segmentation prediction results as image file"""
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        
        # For multi-class segmentation (C, H, W), take argmax
        if pred.ndim == 3:
            mask = np.argmax(pred, axis=0).astype(np.uint8)
        else:
            mask = pred.astype(np.uint8)
        
        # Resize back to original size
        h, w = original_size
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Determine output path
        if mask_path:
            # Use mask path specified in CSV, remove leading '../'
            mask_path_clean = mask_path.replace('../', '')
            output_path = os.path.join(output_dir, mask_path_clean)
        else:
            # Default: replace keywords in image_path
            default_mask_path = image_path.replace('img', 'mask').replace('IMG', 'MASK')
            output_path = os.path.join(output_dir, default_mask_path)
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save mask image
        cv2.imwrite(output_path, mask)
    
    def _process_classification(self, pred, task_id, image_path):
        """Process classification task prediction results"""
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        
        # Get predicted class
        pred_class = int(np.argmax(pred))
        
        # Calculate probability distribution (using softmax)
        # Stable softmax to avoid numerical overflow
        pred_exp = np.exp(pred - np.max(pred))
        pred_probs = pred_exp / np.sum(pred_exp)
        
        return {
            'image_path': image_path,
            'task_id': task_id,
            'predicted_class': pred_class,
            'predicted_probs': pred_probs.tolist()
        }
    
    def _process_regression(self, pred, task_id, image_path, original_size):
        """Process regression task prediction results (keypoint localization)"""
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        
        # Normalized coordinates
        coords = pred.flatten().tolist()
        
        # Convert to pixel coordinates
        h, w = original_size
        pixel_coords = []
        for i in range(0, len(coords), 2):
            x_norm, y_norm = coords[i], coords[i+1]
            x_pixel = x_norm * w
            y_pixel = y_norm * h
            pixel_coords.extend([x_pixel, y_pixel])
        
        return {
            'image_path': image_path,
            'task_id': task_id,
            'predicted_points_normalized': coords,
            'predicted_points_pixels': pixel_coords
        }
    
    def _process_detection(self, pred, task_id, image_path, original_size):
        """Process detection task prediction results"""
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        
        # pred shape: (5, H, W) - grid predictions
        # Find location with highest confidence score
        _, h, w = pred.shape
        scores = pred[4, :, :].flatten()
        best_idx = np.argmax(scores)
        best_h = best_idx // w
        best_w = best_idx % w
        
        # Extract predicted bbox at that location
        bbox_norm = pred[:4, best_h, best_w]
        bbox_norm_list = bbox_norm.tolist()
        
        # Convert to pixel coordinates
        img_h, img_w = original_size
        bbox_pixel = [
            bbox_norm[0] * img_w,
            bbox_norm[1] * img_h,
            bbox_norm[2] * img_w,
            bbox_norm[3] * img_h
        ]
        
        return {
            'image_path': image_path,
            'task_id': task_id,
            'bbox_normalized': bbox_norm_list,
            'bbox_pixels': bbox_pixel
        }


# Docker entry point
if __name__ == '__main__':
    """
    Docker environment entry point
    """
    # Docker path configuration
    tmp_dir = '/myhome/'
    os.makedirs(tmp_dir, exist_ok=True)
    
    data_root = '/input/'      # Docker mounted input data directory
    output_dir = '/output/'    # Docker mounted output directory
    
    print('='*60)
    print('Foundation Model Challenge for Ultrasound Image Analysis (FMC_UIA) - Docker Inference')
    print('='*60)
    
    start_time = time.time()
    
    # Create model and perform prediction
    model = Model()
    model.predict(data_root, output_dir, batch_size=8)
    
    elapsed_time = time.time() - start_time
    print(f"\nTotal time: {elapsed_time:.2f} seconds")
    print("Inference complete!")
