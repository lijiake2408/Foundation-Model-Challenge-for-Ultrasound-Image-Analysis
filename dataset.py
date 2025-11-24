import os
import cv2
import pandas as pd
import numpy as np
import torch
import glob
import random
import json
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
from typing import Optional, Iterator, List
import albumentations as A

class MultiTaskDataset(Dataset):
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
        print(f"Data loaded. Total samples: {len(self.dataframe)}")

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> dict:
        record = self.dataframe.iloc[idx]
        task_id = record['task_id']
        task_name = record['task_name']
        
        # Load image
        image_abs_path = os.path.normpath(os.path.join(self.csv_path, record['image_path']))
        image = cv2.imread(image_abs_path)
        
        # Robustness check: retry next index if image load fails
        if image is None:
            return self.__getitem__((idx + 1) % len(self))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Save original image size BEFORE any transforms (for Regression coordinate normalization)
        original_height, original_width = image.shape[:2]

        # Load raw labels based on task
        label = None
        mask = None
        bboxes = []
        class_labels = []

        if task_name == 'segmentation':
            if pd.notna(record.get('mask_path')):
                mask_path = os.path.normpath(os.path.join(self.csv_path, record['mask_path']))
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        elif task_name == 'classification':
            label = int(record['mask'])

        elif task_name == 'Regression':
            num_points = record['num_classes']
            coords = []
            for i in range(1, num_points + 1):
                col = f'point_{i}_xy'
                if col in record and pd.notna(record[col]):
                    coords.extend(json.loads(record[col]))
                else:
                    coords.extend([0, 0])
            label = np.array(coords, dtype=np.float32)

        elif task_name == 'detection':
            cols = ['x_min', 'y_min', 'x_max', 'y_max']
            if all(c in record and pd.notna(record[c]) for c in cols):
                box_coords = [float(record[c]) for c in cols]
                bboxes = [box_coords + [0]]
                class_labels = [0]

        # Apply augmentations
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask, bboxes=bboxes, class_labels=class_labels)
            image = augmented['image']
            
            if task_name == 'segmentation':
                label = augmented.get('mask')
            elif task_name == 'detection':
                if augmented['bboxes']:
                    label = np.array(augmented['bboxes'][0][:4], dtype=np.float32)
                else:
                    label = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)

        # Format conversion & normalization
        final_label = None
        h, w = image.shape[1], image.shape[2]

        # Ensure label is numpy for processing
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()

        if task_name == 'segmentation':
            if label is None:
                label = np.zeros((h, w), dtype=np.int64)
            final_label = torch.from_numpy(label).long()

        elif task_name == 'classification':
            final_label = torch.tensor(label).long()

        elif task_name in ['Regression', 'detection']:
            if not isinstance(label, np.ndarray):
                label = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)

            # Normalize coordinates to [0, 1]
            if task_name == 'detection' and np.all(label >= 0):
                label[[0, 2]] /= w
                label[[1, 3]] /= h
            elif task_name == 'Regression':
                label[0::2] /= original_width
                label[1::2] /= original_height
            
            final_label = torch.from_numpy(label).float()
        
        return {'image': image, 'label': final_label, 'task_id': task_id}


class MultiTaskUniformSampler(Sampler[List[int]]):
    def __init__(self, dataset: MultiTaskDataset, batch_size: int, steps_per_epoch: Optional[int] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices_by_task = {}

        # Group indices by task_id
        print("\n--- Initializing Sampler ---")
        for idx, task_id in enumerate(tqdm(dataset.dataframe['task_id'], desc="Grouping indices")):
            if task_id not in self.indices_by_task:
                self.indices_by_task[task_id] = []
            self.indices_by_task[task_id].append(idx)
            
        self.task_ids = list(self.indices_by_task.keys())
        
        # Initial shuffle
        for task_id in self.task_ids:
            random.shuffle(self.indices_by_task[task_id])

        # Determine epoch length
        if steps_per_epoch is None:
            self.steps_per_epoch = len(self.dataset) // self.batch_size
        else:
            self.steps_per_epoch = steps_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        task_cursors = {task_id: 0 for task_id in self.task_ids}

        for _ in range(self.steps_per_epoch):
            # Randomly select a task
            task_id = random.choice(self.task_ids)
            indices = self.indices_by_task[task_id]
            cursor = task_cursors[task_id]
            
            start_idx = cursor
            end_idx = start_idx + self.batch_size
            
            if end_idx > len(indices):
                # Wrap around
                batch_indices = indices[start_idx:]
                random.shuffle(indices)
                remaining = self.batch_size - len(batch_indices)
                batch_indices.extend(indices[:remaining])
                task_cursors[task_id] = remaining
            else:
                batch_indices = indices[start_idx:end_idx]
                task_cursors[task_id] = end_idx
            
            yield batch_indices
            
    def __len__(self) -> int:
        return self.steps_per_epoch