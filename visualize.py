# Visualization Script for Foundation Model Challenge for Ultrasound Image Analysis (FMC_UIA)

import os
import cv2
import json
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rcParams
from collections import defaultdict

# Set font
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
rcParams['axes.unicode_minus'] = False


class Visualizer:
    """Visualizer for Foundation Model Challenge for Ultrasound Image Analysis"""
    
    def __init__(self, data_root: str, pred_root: str):
        """
        Initialize visualizer
        
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
        
        print(f"\nDetected {len(self.task_configs)} tasks")
        print("="*80 + "\n")
    
    def visualize_all(self, output_dir: str, samples_per_task: int = 1):
        """
        Visualize prediction results for all tasks
        
        Args:
            output_dir: Output directory
            samples_per_task: Number of samples to visualize per task
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Group by task type
        task_groups = defaultdict(list)
        for task_id, cfg in self.task_configs.items():
            task_groups[cfg['task_name']].append(task_id)
        
        # Visualize each task type
        for task_name, task_ids in task_groups.items():
            print(f"\n{'='*80}")
            print(f"Visualizing {task_name} tasks...")
            print(f"{'='*80}")
            
            if task_name == 'segmentation':
                self.visualize_segmentation(task_ids, output_dir, samples_per_task)
            elif task_name == 'classification':
                self.visualize_classification(task_ids, output_dir, samples_per_task)
            elif task_name == 'detection':
                self.visualize_detection(task_ids, output_dir, samples_per_task)
            elif task_name == 'Regression':
                self.visualize_regression(task_ids, output_dir, samples_per_task)
    
    def visualize_segmentation(self, task_ids: list, output_dir: str, samples_per_task: int):
        """Visualize segmentation tasks"""
        for task_id in task_ids:
            print(f"\nVisualizing task: {task_id}")
            
            # Get samples for this task
            task_data = self.dataframe[self.dataframe['task_id'] == task_id]
            num_classes = self.task_configs[task_id]['num_classes']
            
            # Randomly select samples
            samples = task_data.sample(min(samples_per_task, len(task_data)))
            
            for idx, (_, row) in enumerate(samples.iterrows()):
                if pd.notna(row.get('mask_path')):
                    # 加载原图
                    img_path = os.path.normpath(os.path.join(self.csv_path, row['image_path']))
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # 加载真实mask
                    gt_mask_path = os.path.normpath(os.path.join(self.csv_path, row['mask_path']))
                    gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                    
                    # 加载预测mask
                    pred_mask_rel = row['mask_path'].replace('../', '')
                    pred_mask_path = os.path.join(self.pred_root, pred_mask_rel)
                    if not os.path.exists(pred_mask_path):
                        print(f"  警告: 预测文件不存在 {pred_mask_path}")
                        continue
                    pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
                    
                    # 创建可视化
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    axes[0].imshow(image)
                    axes[0].set_title('Original Image', fontsize=12)
                    axes[0].axis('off')
                    
                    axes[1].imshow(image)
                    axes[1].imshow(gt_mask, alpha=0.5, cmap='jet')
                    axes[1].set_title('Ground Truth Mask', fontsize=12)
                    axes[1].axis('off')
                    
                    axes[2].imshow(image)
                    axes[2].imshow(pred_mask, alpha=0.5, cmap='jet')
                    axes[2].set_title('Predicted Mask', fontsize=12)
                    axes[2].axis('off')
                    
                    plt.suptitle(f'{task_id} (Classes: {num_classes})', fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    
                    # 保存
                    save_path = os.path.join(output_dir, f'{task_id}_sample_{idx}.png')
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"  保存: {save_path}")
    
    def visualize_classification(self, task_ids: list, output_dir: str, samples_per_task: int):
        """可视化分类任务"""
        # 加载预测结果
        pred_file = os.path.join(self.pred_root, 'classification_predictions.json')
        if not os.path.exists(pred_file):
            print(f"  错误: 预测文件不存在 {pred_file}")
            return
        
        with open(pred_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        # 建立预测索引
        pred_dict = defaultdict(dict)
        for pred in predictions:
            pred_dict[pred['task_id']][pred['image_path']] = pred
        
        for task_id in task_ids:
            print(f"\n可视化任务: {task_id}")
            
            # 获取该任务的样本
            task_data = self.dataframe[self.dataframe['task_id'] == task_id]
            num_classes = self.task_configs[task_id]['num_classes']
            
            # 随机选择样本
            samples = task_data.sample(min(samples_per_task, len(task_data)))
            
            for idx, (_, row) in enumerate(samples.iterrows()):
                image_path = row['image_path']
                
                if image_path in pred_dict[task_id]:
                    # 加载图像
                    img_path = os.path.normpath(os.path.join(self.csv_path, image_path))
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # 获取预测和真实标签
                    pred_info = pred_dict[task_id][image_path]
                    pred_class = pred_info['predicted_class']
                    pred_probs = pred_info.get('predicted_probs', [])
                    gt_class = int(row['mask'])
                    
                    # 创建可视化
                    fig = plt.figure(figsize=(12, 5))
                    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
                    
                    # 左边：图像
                    ax1 = fig.add_subplot(gs[0])
                    ax1.imshow(image)
                    ax1.set_title(f'Image\nGT: Class {gt_class} | Pred: Class {pred_class}', 
                                 fontsize=11, color='green' if gt_class == pred_class else 'red')
                    ax1.axis('off')
                    
                    # 右边：概率分布
                    ax2 = fig.add_subplot(gs[1])
                    if pred_probs:
                        classes = list(range(num_classes))
                        colors = ['green' if i == gt_class else 'blue' if i == pred_class else 'gray' 
                                 for i in classes]
                        bars = ax2.bar(classes, pred_probs, color=colors, alpha=0.7)
                        ax2.set_xlabel('Class', fontsize=10)
                        ax2.set_ylabel('Probability', fontsize=10)
                        ax2.set_title('Prediction Probabilities', fontsize=11)
                        ax2.set_ylim([0, 1.0])
                        ax2.grid(axis='y', alpha=0.3)
                        
                        # 添加图例
                        from matplotlib.patches import Patch
                        legend_elements = [
                            Patch(facecolor='green', alpha=0.7, label=f'GT: {gt_class}'),
                            Patch(facecolor='blue', alpha=0.7, label=f'Pred: {pred_class}')
                        ]
                        ax2.legend(handles=legend_elements, loc='upper right', fontsize=9)
                    
                    plt.suptitle(f'{task_id} (Classes: {num_classes})', 
                               fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    
                    # 保存
                    save_path = os.path.join(output_dir, f'{task_id}_sample_{idx}.png')
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"  保存: {save_path}")
    
    def visualize_detection(self, task_ids: list, output_dir: str, samples_per_task: int):
        """可视化检测任务"""
        # 加载预测结果
        pred_file = os.path.join(self.pred_root, 'detection_predictions.json')
        if not os.path.exists(pred_file):
            print(f"  错误: 预测文件不存在 {pred_file}")
            return
        
        with open(pred_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        # 建立预测索引
        pred_dict = defaultdict(dict)
        for pred in predictions:
            pred_dict[pred['task_id']][pred['image_path']] = pred['bbox_pixels']
        
        for task_id in task_ids:
            print(f"\n可视化任务: {task_id}")
            
            # 获取该任务的样本
            task_data = self.dataframe[self.dataframe['task_id'] == task_id]
            
            # 随机选择样本
            samples = task_data.sample(min(samples_per_task, len(task_data)))
            
            for idx, (_, row) in enumerate(samples.iterrows()):
                image_path = row['image_path']
                
                # 真实bbox（像素坐标）
                cols = ['x_min', 'y_min', 'x_max', 'y_max']
                if all(c in row and pd.notna(row[c]) for c in cols):
                    gt_bbox = [float(row[c]) for c in cols]
                    
                    if image_path in pred_dict[task_id]:
                        # 加载图像
                        img_path = os.path.normpath(os.path.join(self.csv_path, image_path))
                        image = cv2.imread(img_path)
                        if image is None:
                            continue
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        pred_bbox = pred_dict[task_id][image_path]
                        
                        # 创建可视化
                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                        
                        # 左边：真实bbox
                        axes[0].imshow(image)
                        rect_gt = patches.Rectangle(
                            (gt_bbox[0], gt_bbox[1]), 
                            gt_bbox[2] - gt_bbox[0], 
                            gt_bbox[3] - gt_bbox[1],
                            linewidth=2, edgecolor='green', facecolor='none', label='Ground Truth'
                        )
                        axes[0].add_patch(rect_gt)
                        axes[0].set_title('Ground Truth BBox', fontsize=12)
                        axes[0].axis('off')
                        axes[0].legend(loc='upper right', fontsize=9)
                        
                        # 右边：预测bbox
                        axes[1].imshow(image)
                        rect_pred = patches.Rectangle(
                            (pred_bbox[0], pred_bbox[1]), 
                            pred_bbox[2] - pred_bbox[0], 
                            pred_bbox[3] - pred_bbox[1],
                            linewidth=2, edgecolor='red', facecolor='none', label='Prediction'
                        )
                        axes[1].add_patch(rect_pred)
                        axes[1].set_title('Predicted BBox', fontsize=12)
                        axes[1].axis('off')
                        axes[1].legend(loc='upper right', fontsize=9)
                        
                        plt.suptitle(f'{task_id}', fontsize=14, fontweight='bold')
                        plt.tight_layout()
                        
                        # 保存
                        save_path = os.path.join(output_dir, f'{task_id}_sample_{idx}.png')
                        plt.savefig(save_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        print(f"  保存: {save_path}")
    
    def visualize_regression(self, task_ids: list, output_dir: str, samples_per_task: int):
        """可视化回归任务（关键点）"""
        # 加载预测结果
        pred_file = os.path.join(self.pred_root, 'regression_predictions.json')
        if not os.path.exists(pred_file):
            print(f"  错误: 预测文件不存在 {pred_file}")
            return
        
        with open(pred_file, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        
        # 建立预测索引
        pred_dict = defaultdict(dict)
        for pred in predictions:
            pred_dict[pred['task_id']][pred['image_path']] = pred['predicted_points_pixels']
        
        for task_id in task_ids:
            print(f"\n可视化任务: {task_id}")
            
            # 获取该任务的样本
            task_data = self.dataframe[self.dataframe['task_id'] == task_id]
            num_points = self.task_configs[task_id]['num_classes']
            
            # 随机选择样本
            samples = task_data.sample(min(samples_per_task, len(task_data)))
            
            for idx, (_, row) in enumerate(samples.iterrows()):
                image_path = row['image_path']
                
                # 真实关键点坐标（像素）
                gt_coords = []
                for i in range(1, num_points + 1):
                    col = f'point_{i}_xy'
                    if col in row and pd.notna(row[col]):
                        gt_coords.extend(json.loads(row[col]))
                    else:
                        gt_coords.extend([0, 0])
                
                if image_path in pred_dict[task_id]:
                    # 加载图像
                    img_path = os.path.normpath(os.path.join(self.csv_path, image_path))
                    image = cv2.imread(img_path)
                    if image is None:
                        continue
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    pred_coords = pred_dict[task_id][image_path]
                    
                    # 创建可视化
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # 左边：真实关键点
                    axes[0].imshow(image)
                    gt_points = np.array(gt_coords).reshape(-1, 2)
                    axes[0].scatter(gt_points[:, 0], gt_points[:, 1], 
                                   c='green', s=100, marker='o', edgecolors='white', linewidths=2,
                                   label='Ground Truth')
                    for i, (x, y) in enumerate(gt_points):
                        axes[0].text(x, y-10, str(i+1), color='white', fontsize=10, 
                                    ha='center', fontweight='bold',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7))
                    axes[0].set_title('Ground Truth Keypoints', fontsize=12)
                    axes[0].axis('off')
                    axes[0].legend(loc='upper right', fontsize=9)
                    
                    # 右边：预测关键点
                    axes[1].imshow(image)
                    pred_points = np.array(pred_coords).reshape(-1, 2)
                    axes[1].scatter(pred_points[:, 0], pred_points[:, 1], 
                                   c='red', s=100, marker='x', linewidths=3,
                                   label='Prediction')
                    for i, (x, y) in enumerate(pred_points):
                        axes[1].text(x, y-10, str(i+1), color='white', fontsize=10, 
                                    ha='center', fontweight='bold',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))
                    axes[1].set_title('Predicted Keypoints', fontsize=12)
                    axes[1].axis('off')
                    axes[1].legend(loc='upper right', fontsize=9)
                    
                    plt.suptitle(f'{task_id} (Points: {num_points})', 
                               fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    
                    # 保存
                    save_path = os.path.join(output_dir, f'{task_id}_sample_{idx}.png')
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"  保存: {save_path}")


if __name__ == '__main__':
    """
    使用示例
    """
    # 配置路径
    data_root = '/root/baseline/train'  # 数据根目录
    pred_root = 'predictions_new_new/'  # 预测结果目录
    output_dir = 'visualizations/'      # 可视化输出目录
    
    # 创建可视化器
    visualizer = Visualizer(data_root, pred_root)
    
    # 执行可视化（每个任务可视化1个样本）
    visualizer.visualize_all(output_dir, samples_per_task=1)
    
    print("\n" + "="*80)
    print(f"可视化完成！结果保存在: {output_dir}")
    print("="*80)

