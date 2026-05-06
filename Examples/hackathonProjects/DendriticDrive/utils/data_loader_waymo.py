"""
DendriticDrive - Waymo Dataset Loader (Hybrid Mode)
Supports both real Waymo Open Dataset and synthetic demo data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import os


class WaymoDataset(Dataset):
    """
    Hybrid Waymo Dataset Loader
    - In demo mode: generates synthetic point clouds
    - In real mode: loads from Waymo Open Dataset tfrecords
    """
    
    def __init__(self, config, split='train', demo=False):
        """
        Args:
            config: Configuration dict from config.yaml
            split: 'train' or 'val'
            demo: If True, use synthetic data instead of real dataset
        """
        self.config = config
        self.split = split
        self.demo = demo
        self.num_classes = config['data']['num_classes']
        self.num_points = config['data']['num_points_per_cloud']
        
        if demo:
            self.num_samples = config['demo']['num_samples']
            self.points_per_sample = config['demo']['points_per_sample']
            print(f"[DEMO MODE] Using {self.num_samples} synthetic samples")
        else:
            # Real dataset path
            data_path = config['data']['train_path'] if split == 'train' else config['data']['val_path']
            self.data_path = Path(data_path)
            
            if not self.data_path.exists():
                raise FileNotFoundError(
                    f"Waymo dataset not found at {self.data_path}. "
                    f"Use --demo flag to run with synthetic data."
                )
            
            # In a real implementation, you would scan for .tfrecord files here
            self.file_list = self._scan_tfrecords()
            print(f"[REAL MODE] Loaded {len(self.file_list)} Waymo scenes from {data_path}")
    
    def _scan_tfrecords(self):
        """Scan for Waymo .tfrecord files (real implementation would use waymo-open-dataset API)"""
        # Placeholder: In real use, this would return list of tfrecord files
        tfrecords = list(self.data_path.glob("*.tfrecord"))
        if len(tfrecords) == 0:
            print(f"WARNING: No .tfrecord files found in {self.data_path}")
        return tfrecords
    
    def __len__(self):
        if self.demo:
            return self.num_samples
        else:
            return len(self.file_list)
    
    def _generate_synthetic_point_cloud(self, idx):
        """Generate synthetic point cloud for demo purposes"""
        np.random.seed(idx)  # Reproducible
        
        # Generate random 3D points (x, y, z, intensity)
        num_points = self.points_per_sample
        points = np.random.randn(num_points, 4).astype(np.float32)
        
        # Scale to Waymo-like range
        points[:, 0] *= 40  # x: -40 to 40
        points[:, 1] *= 40  # y: -40 to 40
        points[:, 2] = points[:, 2] * 2 + 1  # z: -1 to 3
        points[:, 3] = np.clip(points[:, 3], 0, 1)  # intensity: 0 to 1
        
        # Generate synthetic bounding boxes (labels)
        num_boxes = np.random.randint(1, 10)
        boxes = []
        labels = []
        
        for _ in range(num_boxes):
            # [x, y, z, l, w, h, heading]
            box = np.array([
                np.random.uniform(-30, 30),  # x
                np.random.uniform(-30, 30),  # y
                0.5,  # z (ground level)
                np.random.uniform(3, 5),  # length
                np.random.uniform(1.5, 2.5),  # width
                np.random.uniform(1.5, 2),  # height
                np.random.uniform(-np.pi, np.pi)  # heading
            ], dtype=np.float32)
            boxes.append(box)
            labels.append(np.random.randint(0, self.num_classes))
        
        boxes = np.array(boxes)
        labels = np.array(labels)
        
        return points, boxes, labels
    
    def _load_real_waymo_frame(self, idx):
        """Load real Waymo frame from tfrecord (placeholder for real implementation)"""
        # Real implementation would use waymo-open-dataset library:
        # from waymo_open_dataset import dataset_pb2
        # from waymo_open_dataset.utils import frame_utils
        
        # Example structure (not functional without waymo library):
        # tfrecord_path = self.file_list[idx]
        # dataset = tf.data.TFRecordDataset(tfrecord_path)
        # frame = next(iter(dataset))
        # points, boxes, labels = parse_waymo_frame(frame)
        
        raise NotImplementedError(
            "Real Waymo dataset loading requires waymo-open-dataset library. "
            "Please use --demo mode for this hackathon demonstration."
        )
    
    def __getitem__(self, idx):
        """Get a single sample (point cloud + labels)"""
        if self.demo:
            points, boxes, labels = self._generate_synthetic_point_cloud(idx)
        else:
            points, boxes, labels = self._load_real_waymo_frame(idx)
        
        # Convert to PyTorch tensors
        data = {
            'points': torch.from_numpy(points),
            'boxes': torch.from_numpy(boxes),
            'labels': torch.from_numpy(labels).long()
        }
        
        return data


def get_loaders(config, demo=False):
    """
    Create train and validation data loaders
    
    Args:
        config: Configuration dict from config.yaml
        demo: If True, use synthetic data
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = WaymoDataset(config, split='train', demo=demo)
    val_dataset = WaymoDataset(config, split='val', demo=demo)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"✓ Data loaders created: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    return train_loader, val_loader


def collate_fn(batch):
    """Custom collate function for variable-length point clouds"""
    # In a real implementation, this would handle batching of variable-length data
    points = [item['points'] for item in batch]
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    return {
        'points': points,  # List of tensors
        'boxes': boxes,
        'labels': labels
    }
