# Evaluation Script for Foundation Model Challenge for Ultrasound Image Analysis (FMC_UIA)

import os
import json
import cv2
import numpy as np
import pandas as pd
import glob
from typing import Dict, List, Tuple
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from tqdm import tqdm
import SimpleITK as sitk
import warnings
warnings.filterwarnings('ignore')


class Evaluator:
    """Evaluator for Foundation Model Challenge for Ultrasound Image Analysis"""
    
    def __init__(self, data_root: str, pred_root: str):
        """
        Initialize evaluator
        
        Args:
            data_root: Data root directory (contains csv_files and ground truth)
            pred_root: Prediction results root directory
        """
        self.data_root = data_root
        self.pred_root = pred_root
        self.csv_path = os.path.join(self.data_root, 'csv_files')
        
        # Load all CSV files
        print("="*80)
        print("Loading dataset information...")
        all_csv_files = glob.glob(os.path.join(self.csv_path, '*.csv'))
        if not all_csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.csv_path}")
        
        df_list = [pd.read_csv(csv_file) for csv_file in all_csv_files]
        self.dataframe = pd.concat(df_list, ignore_index=True).reset_index(drop=True)
        print(f"Data loading complete. Total samples: {len(self.dataframe)}")
        
        # Build task configurations
        self.task_configs = {}
        for _, row in self.dataframe.iterrows():
            task_id = row['task_id']
            if task_id not in self.task_configs:
                self.task_configs[task_id] = {
                    'task_name': row['task_name'],
                    'num_classes': int(row['num_classes'])
                }
        
        print(f"\nDetected {len(self.task_configs)} tasks:")
        for task_id in sorted(self.task_configs.keys()):
            cfg = self.task_configs[task_id]
            print(f"  - {task_id}: {cfg['task_name']}, num_classes={cfg['num_classes']}")
        print("="*80 + "\n")
    
    def evaluate_all(self) -> Dict:
        """Evaluate all tasks"""
        results = {}
        
        # Group by task type
        task_groups = defaultdict(list)
        for task_id, cfg in self.task_configs.items():
            task_groups[cfg['task_name']].append(task_id)
        
        # Evaluate each task type
        if 'segmentation' in task_groups:
            print("\n" + "="*80)
            print("Evaluating segmentation tasks...")
            print("="*80)
            seg_results = self.evaluate_segmentation(task_groups['segmentation'])
            results['segmentation'] = seg_results
        
        if 'classification' in task_groups:
            print("\n" + "="*80)
            print("Evaluating classification tasks...")
            print("="*80)
            cls_results = self.evaluate_classification(task_groups['classification'])
            results['classification'] = cls_results
        
        if 'detection' in task_groups:
            print("\n" + "="*80)
            print("Evaluating detection tasks...")
            print("="*80)
            det_results = self.evaluate_detection(task_groups['detection'])
            results['detection'] = det_results
        
        if 'Regression' in task_groups:
            print("\n" + "="*80)
            print("Evaluating regression tasks...")
            print("="*80)
            reg_results = self.evaluate_regression(task_groups['Regression'])
            results['regression'] = reg_results
        
        return results
    
    def evaluate_segmentation(self, task_ids: List[str]) -> Dict:
        """Evaluate segmentation tasks - DSC and HD (computed separately for each non-background class then averaged)"""
        task_results = {}
        
        for task_id in tqdm(task_ids, desc="Segmentation tasks", unit="task"):
            print(f"\nEvaluating task: {task_id}")
            
            # Get all samples and number of classes for this task
            task_data = self.dataframe[self.dataframe['task_id'] == task_id]
            num_classes = self.task_configs[task_id]['num_classes']
            
            # Collect scores for each class
            dsc_scores_per_class = [[] for _ in range(num_classes)]
            hd_scores_per_class = [[] for _ in range(num_classes)]
            valid_count = 0
            
            for idx, row in tqdm(task_data.iterrows(), total=len(task_data), desc=f"  {task_id} samples", leave=False, unit="sample"):
                # Load ground truth mask
                if pd.notna(row.get('mask_path')):
                    gt_mask_path = os.path.normpath(os.path.join(self.csv_path, row['mask_path']))
                    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                    
                    if gt_mask is None:
                        continue
                    
                    # Load predicted mask
                    pred_mask_rel = row['mask_path'].replace('../', '')
                    pred_mask_path = os.path.join(self.pred_root, pred_mask_rel)
                    
                    if not os.path.exists(pred_mask_path):
                        print(f"  Warning: Prediction file not found {pred_mask_path}")
                        continue
                    
                    pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
                    
                    if pred_mask is None:
                        continue
                    
                    # Ensure consistent size
                    if pred_mask.shape != gt_mask.shape:
                        pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), 
                                              interpolation=cv2.INTER_NEAREST)
                    
                    # Compute DSC and HD (detailed calculation per class)
                    dsc_per_class, dsc_mean = self._compute_dice(pred_mask, gt_mask, num_classes)
                    hd_per_class, hd_mean = self._compute_hausdorff(pred_mask, gt_mask, num_classes)
                    
                    # Collect scores for each class
                    for c in range(num_classes):
                        if dsc_per_class[c] is not None:
                            dsc_scores_per_class[c].append(dsc_per_class[c])
                        if hd_per_class[c] is not None:
                            hd_scores_per_class[c].append(hd_per_class[c])
                    
                    valid_count += 1
            
            # Compute average for each class
            if valid_count > 0:
                class_metrics = {}
                for c in range(num_classes):
                    if dsc_scores_per_class[c]:
                        avg_dsc_c = np.mean(dsc_scores_per_class[c])
                        avg_hd_c = np.mean(hd_scores_per_class[c]) if hd_scores_per_class[c] else 0.0
                        class_metrics[f'class_{c}'] = {
                            'DSC': float(avg_dsc_c),
                            'HD': float(avg_hd_c)
                        }
                
                # Compute overall average for non-background classes
                all_dsc = [class_metrics[f'class_{c}']['DSC'] for c in range(1, num_classes) if f'class_{c}' in class_metrics]
                all_hd = [class_metrics[f'class_{c}']['HD'] for c in range(1, num_classes) if f'class_{c}' in class_metrics]
                
                avg_dsc = np.mean(all_dsc) if all_dsc else 0.0
                avg_hd = np.mean(all_hd) if all_hd else 0.0
                
                task_results[task_id] = {
                    'DSC': float(avg_dsc),
                    'HD': float(avg_hd),
                    'num_samples': valid_count,
                    'per_class': class_metrics
                }
                
                # Print detailed information
                print(f"  Overall average: DSC={avg_dsc:.4f}, HD={avg_hd:.4f} ({valid_count} samples)")
                print(f"  Per-class metrics:")
                for c in range(num_classes):
                    if f'class_{c}' in class_metrics:
                        metrics = class_metrics[f'class_{c}']
                        class_label = "Background" if c == 0 else f"Class {c}"
                        print(f"    {class_label}: DSC={metrics['DSC']:.4f}, HD={metrics['HD']:.4f}")
            else:
                task_results[task_id] = {'DSC': 0.0, 'HD': 0.0, 'num_samples': 0, 'per_class': {}}
                print(f"  No valid samples")
        
        return task_results
    
    def evaluate_classification(self, task_ids: List[str]) -> Dict:
        """Evaluate classification tasks - AUC, F1, Accuracy, MCC"""
        # Load predictions
        pred_file = os.path.join(self.pred_root, 'classification_predictions.json')
        if not os.path.exists(pred_file):
            print(f"  Error: Prediction file not found {pred_file}")
            return {}
        
        with open(pred_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        # Build prediction index {task_id: {image_path: {'class': ..., 'probs': ...}}}
        pred_dict = defaultdict(dict)
        for pred in predictions:
            pred_dict[pred['task_id']][pred['image_path']] = {
                'class': pred['predicted_class'],
                'probs': pred.get('predicted_probs', None)
            }
        
        task_results = {}
        
        for task_id in tqdm(task_ids, desc="Classification tasks", unit="task"):
            print(f"\nEvaluating task: {task_id}")
            
            # Get all samples for this task
            task_data = self.dataframe[self.dataframe['task_id'] == task_id]
            num_classes = self.task_configs[task_id]['num_classes']
            
            y_true = []
            y_pred = []
            y_probs = []
            
            for idx, row in tqdm(task_data.iterrows(), total=len(task_data), desc=f"  {task_id} samples", leave=False, unit="sample"):
                image_path = row['image_path']
                gt_label = int(row['mask'])
                
                if image_path in pred_dict[task_id]:
                    pred_info = pred_dict[task_id][image_path]
                    pred_label = pred_info['class']
                    y_true.append(gt_label)
                    y_pred.append(pred_label)
                    
                    # Collect probabilities for AUC calculation
                    if pred_info['probs'] is not None:
                        y_probs.append(pred_info['probs'])
            
            if len(y_true) > 0:
                y_true = np.array(y_true)
                y_pred = np.array(y_pred)
                
                # Calculate accuracy
                accuracy = np.mean(y_true == y_pred)
                
                # Calculate F1
                f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
                f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                
                # Calculate MCC
                mcc = matthews_corrcoef(y_true, y_pred)
                
                # Calculate AUC (using probabilities)
                auc = 0.0
                if len(y_probs) == len(y_true):
                    y_probs = np.array(y_probs)
                    try:
                        if num_classes == 2:
                            auc = roc_auc_score(y_true, y_probs[:, 1])
                        elif num_classes > 2:
                            auc = roc_auc_score(y_true, y_probs, average='macro', multi_class='ovr')
                    except Exception as e:
                        print(f"    Warning: AUC calculation failed - {str(e)}")
                        auc = 0.0
                else:
                    print(f"    Warning: Missing probability information, cannot calculate AUC")
                
                # Calculate per-class metrics
                per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
                per_class_metrics = {}
                for c in range(num_classes):
                    per_class_metrics[f'class_{c}'] = {
                        'F1': float(per_class_f1[c]) if c < len(per_class_f1) else 0.0
                    }
                
                task_results[task_id] = {
                    'AUC': float(auc),
                    'Accuracy': float(accuracy),
                    'F1_macro': float(f1_macro),
                    'F1_weighted': float(f1_weighted),
                    'MCC': float(mcc),
                    'num_samples': len(y_true),
                    'per_class': per_class_metrics
                }
                
                print(f"  Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, F1(macro): {f1_macro:.4f}, F1(weighted): {f1_weighted:.4f}, MCC: {mcc:.4f} ({len(y_true)} samples)")
                if num_classes <= 10:
                    print(f"  Per-class F1:")
                    for c in range(num_classes):
                        if f'class_{c}' in per_class_metrics:
                            print(f"    Class {c}: F1={per_class_metrics[f'class_{c}']['F1']:.4f}")
            else:
                task_results[task_id] = {
                    'AUC': 0.0, 'Accuracy': 0.0, 'F1_macro': 0.0, 
                    'F1_weighted': 0.0, 'MCC': 0.0, 'num_samples': 0, 'per_class': {}
                }
                print(f"  No valid samples")
        
        return task_results
    
    def evaluate_detection(self, task_ids: List[str]) -> Dict:
        """Evaluate detection tasks - IoU"""
        # Load predictions
        pred_file = os.path.join(self.pred_root, 'detection_predictions.json')
        if not os.path.exists(pred_file):
            print(f"  Error: Prediction file not found {pred_file}")
            return {}
        
        with open(pred_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        # Build prediction index - using pixel coordinates
        pred_dict = defaultdict(dict)
        for pred in predictions:
            pred_dict[pred['task_id']][pred['image_path']] = pred['bbox_pixels']
        
        task_results = {}
        
        for task_id in tqdm(task_ids, desc="Detection tasks", unit="task"):
            print(f"\nEvaluating task: {task_id}")
            
            # Get all samples for this task
            task_data = self.dataframe[self.dataframe['task_id'] == task_id]
            
            iou_scores = []
            
            for idx, row in tqdm(task_data.iterrows(), total=len(task_data), desc=f"  {task_id} samples", leave=False, unit="sample"):
                image_path = row['image_path']
                
                # Ground truth bbox (pixel coordinates)
                cols = ['x_min', 'y_min', 'x_max', 'y_max']
                if all(c in row and pd.notna(row[c]) for c in cols):
                    gt_bbox = [float(row[c]) for c in cols]
                    
                    if image_path in pred_dict[task_id]:
                        pred_bbox = pred_dict[task_id][image_path]
                        
                        # Calculate IoU
                        iou = self._compute_iou(pred_bbox, gt_bbox)
                        iou_scores.append(iou)
            
            if len(iou_scores) > 0:
                avg_iou = np.mean(iou_scores)
                task_results[task_id] = {
                    'IoU': float(avg_iou),
                    'num_samples': len(iou_scores)
                }
                print(f"  IoU: {avg_iou:.4f} ({len(iou_scores)} samples)")
            else:
                task_results[task_id] = {'IoU': 0.0, 'num_samples': 0}
                print(f"  No valid samples")
        
        return task_results
    
    def evaluate_regression(self, task_ids: List[str]) -> Dict:
        """Evaluate regression tasks - MRE (Mean Radial Error)"""
        # Load predictions
        pred_file = os.path.join(self.pred_root, 'regression_predictions.json')
        if not os.path.exists(pred_file):
            print(f"  Error: Prediction file not found {pred_file}")
            return {}
        
        with open(pred_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        # Build prediction index - using pixel coordinates
        pred_dict = defaultdict(dict)
        for pred in predictions:
            pred_dict[pred['task_id']][pred['image_path']] = pred['predicted_points_pixels']
        
        task_results = {}
        
        for task_id in tqdm(task_ids, desc="Regression tasks", unit="task"):
            print(f"\nEvaluating task: {task_id}")
            
            # Get all samples for this task
            task_data = self.dataframe[self.dataframe['task_id'] == task_id]
            num_points = self.task_configs[task_id]['num_classes']
            
            mre_scores = []
            
            for idx, row in tqdm(task_data.iterrows(), total=len(task_data), desc=f"  {task_id} samples", leave=False, unit="sample"):
                image_path = row['image_path']
                
                # Ground truth keypoint coordinates (pixel coordinates)
                gt_coords = []
                for i in range(1, num_points + 1):
                    col = f'point_{i}_xy'
                    if col in row and pd.notna(row[col]):
                        gt_coords.extend(json.loads(row[col]))
                    else:
                        gt_coords.extend([0, 0])
                
                if image_path in pred_dict[task_id]:
                    pred_coords = pred_dict[task_id][image_path]
                    
                    # Calculate MRE
                    mre = self._compute_mre(pred_coords, gt_coords)
                    mre_scores.append(mre)
            
            if len(mre_scores) > 0:
                avg_mre = np.mean(mre_scores)
                task_results[task_id] = {
                    'MRE': float(avg_mre),
                    'num_samples': len(mre_scores)
                }
                print(f"  MRE: {avg_mre:.4f} ({len(mre_scores)} samples)")
            else:
                task_results[task_id] = {'MRE': 0.0, 'num_samples': 0}
                print(f"  No valid samples")
        
        return task_results
    
    # Helper computation functions
    
    def _compute_dice(self, pred: np.ndarray, gt: np.ndarray, num_classes: int) -> Tuple[List[float], float]:
        """
        Compute Dice coefficient (calculated separately for each class)
        
        Returns:
            Tuple[List[float], float]: (DSC list for each class, average DSC for non-background classes)
        """
        dice_scores = []
        
        # Calculate DSC for each class (including background)
        for class_id in range(num_classes):
            pred_class = (pred == class_id).astype(np.uint8)
            gt_class = (gt == class_id).astype(np.uint8)
            
            intersection = np.sum(pred_class * gt_class)
            union = np.sum(pred_class) + np.sum(gt_class)
            
            if union == 0:
                dice = 1.0
            else:
                dice = (2.0 * intersection) / union
            
            dice_scores.append(dice)
        
        # Calculate average DSC for non-background classes
        mean_dice = np.mean(dice_scores[1:]) if len(dice_scores) > 1 else 0.0
        
        return dice_scores, mean_dice
    
    def _compute_hausdorff(self, pred: np.ndarray, gt: np.ndarray, num_classes: int) -> Tuple[List[float], float]:
        """
        Compute Hausdorff distance (calculated separately for each class) - using SimpleITK for better performance
        
        Returns:
            Tuple[List[float], float]: (HD list for each class, average HD for non-background classes)
        """
        hd_scores = []
        
        # Calculate HD for each class (including background)
        for class_id in range(num_classes):
            pred_class = (pred == class_id).astype(np.uint8)
            gt_class = (gt == class_id).astype(np.uint8)
            
            # Check if there are foreground points
            pred_sum = np.sum(pred_class)
            gt_sum = np.sum(gt_class)
            
            if pred_sum == 0 and gt_sum == 0:
                hd = 0.0
            elif pred_sum == 0 or gt_sum == 0:
                h, w = pred.shape
                hd = np.sqrt(h**2 + w**2)
            else:
                # Use SimpleITK to calculate Hausdorff distance
                try:
                    pred_class_255 = (pred_class * 255).astype(np.uint8)
                    gt_class_255 = (gt_class * 255).astype(np.uint8)
                    
                    pred_image = sitk.GetImageFromArray(pred_class_255)
                    gt_image = sitk.GetImageFromArray(gt_class_255)
                    
                    # Calculate Hausdorff distance
                    hausdorff_filter = sitk.HausdorffDistanceImageFilter()
                    hausdorff_filter.Execute(pred_image, gt_image)
                    hd = hausdorff_filter.GetHausdorffDistance()
                except Exception as e:
                    if class_id > 0:
                        print(f"    Warning: HD calculation failed (class {class_id}) - {str(e).split(':')[-1].strip()}")
                    h, w = pred.shape
                    hd = np.sqrt(h**2 + w**2)
            
            hd_scores.append(hd)
        
        # Calculate average HD for non-background classes
        mean_hd = np.mean(hd_scores[1:]) if len(hd_scores) > 1 else 0.0
        
        return hd_scores, mean_hd
    
    def _compute_iou(self, pred_bbox: List[float], gt_bbox: List[float]) -> float:
        """Calculate IoU"""
        # bbox format: [x_min, y_min, x_max, y_max]
        x1_pred, y1_pred, x2_pred, y2_pred = pred_bbox
        x1_gt, y1_gt, x2_gt, y2_gt = gt_bbox
        
        # Calculate intersection
        x1_inter = max(x1_pred, x1_gt)
        y1_inter = max(y1_pred, y1_gt)
        x2_inter = min(x2_pred, x2_gt)
        y2_inter = min(y2_pred, y2_gt)
        
        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        
        # Calculate union
        pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
        gt_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)
        union_area = pred_area + gt_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        iou = inter_area / union_area
        return iou
    
    def _compute_mre(self, pred_coords: List[float], gt_coords: List[float]) -> float:
        """Calculate Mean Radial Error"""
        pred_coords = np.array(pred_coords).reshape(-1, 2)
        gt_coords = np.array(gt_coords).reshape(-1, 2)
        
        # Calculate Euclidean distance for each keypoint
        distances = np.sqrt(np.sum((pred_coords - gt_coords) ** 2, axis=1))
        
        # Return average distance
        mre = np.mean(distances)
        return mre
    
    def print_summary(self, results: Dict, save_path: str = None):
        """Print evaluation summary and optionally save to txt file
        
        Args:
            results: Evaluation results dictionary
            save_path: Path to save txt file (optional)
        """
        # Build summary text
        summary_lines = []
        summary_lines.append("="*80)
        summary_lines.append("Evaluation Summary")
        summary_lines.append("="*80)
        
        if 'segmentation' in results:
            summary_lines.append("\n[Segmentation Tasks]")
            seg_results = results['segmentation']
            all_dsc = [r['DSC'] for r in seg_results.values() if r['num_samples'] > 0]
            all_hd = [r['HD'] for r in seg_results.values() if r['num_samples'] > 0]
            
            if all_dsc:
                summary_lines.append(f"  Average DSC: {np.mean(all_dsc):.4f}")
                summary_lines.append(f"  Average HD:  {np.mean(all_hd):.4f}")
                summary_lines.append(f"  Number of tasks: {len(all_dsc)}")
                summary_lines.append(f"  Task details:")
                for task_id, task_result in seg_results.items():
                    if task_result['num_samples'] > 0:
                        summary_lines.append(f"    {task_id}: DSC={task_result['DSC']:.4f}, HD={task_result['HD']:.4f}, samples={task_result['num_samples']}")
        
        if 'classification' in results:
            summary_lines.append("\n[Classification Tasks]")
            cls_results = results['classification']
            all_acc = [r['Accuracy'] for r in cls_results.values() if r['num_samples'] > 0]
            all_auc = [r['AUC'] for r in cls_results.values() if r['num_samples'] > 0]
            all_f1_macro = [r['F1_macro'] for r in cls_results.values() if r['num_samples'] > 0]
            all_f1_weighted = [r['F1_weighted'] for r in cls_results.values() if r['num_samples'] > 0]
            all_mcc = [r['MCC'] for r in cls_results.values() if r['num_samples'] > 0]
            
            if all_acc:
                summary_lines.append(f"  Average Accuracy:    {np.mean(all_acc):.4f}")
                summary_lines.append(f"  Average AUC:         {np.mean(all_auc):.4f}")
                summary_lines.append(f"  Average F1 (macro):  {np.mean(all_f1_macro):.4f}")
                summary_lines.append(f"  Average F1 (weight): {np.mean(all_f1_weighted):.4f}")
                summary_lines.append(f"  Average MCC:         {np.mean(all_mcc):.4f}")
                summary_lines.append(f"  Number of tasks:     {len(all_acc)}")
                summary_lines.append(f"  Task details:")
                for task_id, task_result in cls_results.items():
                    if task_result['num_samples'] > 0:
                        summary_lines.append(f"    {task_id}: Acc={task_result['Accuracy']:.4f}, AUC={task_result['AUC']:.4f}, F1={task_result['F1_macro']:.4f}, MCC={task_result['MCC']:.4f}, samples={task_result['num_samples']}")
        
        if 'detection' in results:
            summary_lines.append("\n[Detection Tasks]")
            det_results = results['detection']
            all_iou = [r['IoU'] for r in det_results.values() if r['num_samples'] > 0]
            
            if all_iou:
                summary_lines.append(f"  Average IoU: {np.mean(all_iou):.4f}")
                summary_lines.append(f"  Number of tasks: {len(all_iou)}")
                summary_lines.append(f"  Task details:")
                for task_id, task_result in det_results.items():
                    if task_result['num_samples'] > 0:
                        summary_lines.append(f"    {task_id}: IoU={task_result['IoU']:.4f}, samples={task_result['num_samples']}")
        
        if 'regression' in results:
            summary_lines.append("\n[Regression Tasks]")
            reg_results = results['regression']
            all_mre = [r['MRE'] for r in reg_results.values() if r['num_samples'] > 0]
            
            if all_mre:
                summary_lines.append(f"  Average MRE: {np.mean(all_mre):.4f}")
                summary_lines.append(f"  Number of tasks: {len(all_mre)}")
                summary_lines.append(f"  Task details:")
                for task_id, task_result in reg_results.items():
                    if task_result['num_samples'] > 0:
                        summary_lines.append(f"    {task_id}: MRE={task_result['MRE']:.4f}, samples={task_result['num_samples']}")
        
        summary_lines.append("\n" + "="*80)
        
        # Print to console
        summary_text = "\n".join(summary_lines)
        print("\n" + summary_text)
        
        # Save to file (if path is specified)
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(summary_text + "\n")
            print(f"\nEvaluation summary saved to: {save_path}")
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nEvaluation results saved to: {output_path}")


if __name__ == '__main__':
    """
    Usage example
    """
    # Configure paths
    data_root = '/root/baseline/train'
    pred_root = 'predictions_new_new/'
    output_file = 'evaluation_results.json'
    summary_file = 'evaluation_summary.txt'
    
    # Create evaluator
    evaluator = Evaluator(data_root, pred_root)
    
    # Run evaluation
    results = evaluator.evaluate_all()
    
    # Print summary and save as txt
    evaluator.print_summary(results, save_path=summary_file)
    
    # Save detailed results as JSON
    evaluator.save_results(results, output_file)
    
    print("\nEvaluation complete!")

