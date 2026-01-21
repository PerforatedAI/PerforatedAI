"""
Data Loader for HAM10000 Dataset
Includes synthetic data support for quick testing
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

class HAM10000Dataset(Dataset):
    """
    Basic dataset for HAM10000 skin lesions
    """
    def __init__(self, root_dir, transform=None, synthetic=False, num_samples=100):
        self.root_dir = root_dir
        self.transform = transform
        self.synthetic = synthetic
        self.num_samples = num_samples
        
        if not self.synthetic:
            # In a real scenario, we would parse the CSV and match images
            # For this demo, we'll look for images in the root_dir
            if os.path.exists(root_dir):
                self.images = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            else:
                print(f"Directory {root_dir} not found. Switching to synthetic mode.")
                self.synthetic = True
        
    def __len__(self):
        if self.synthetic:
            return self.num_samples
        return len(self.images)

    def __getitem__(self, idx):
        if self.synthetic:
            # Generate synthetic medical-like image
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img)
            label = np.random.randint(0, 7)
            # Create a synthetic mask
            mask = np.zeros((224, 224), dtype=np.uint8)
            mask[50:150, 50:150] = 1
            mask = Image.fromarray(mask)
        else:
            img_name = os.path.join(self.root_dir, self.images[idx])
            img = Image.open(img_name).convert('RGB')
            label = 0 # Placeholder: actual label should come from CSV
            mask = img.convert('L') # Placeholder mask
            
        if self.transform:
            img = self.transform(img)
            mask_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            mask = mask_transform(mask)
            
        return {"image": img, "label": label, "mask": mask}

def get_loaders(config):
    """
    Returns train and validation loaders
    """
    data_config = config['data']
    
    transform = transforms.Compose([
        transforms.Resize((data_config['image_size'], data_config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Check if real data exists
    synthetic = not os.path.exists(data_config['root_dir'])
    
    train_ds = HAM10000Dataset(data_config['root_dir'], transform=transform, synthetic=synthetic)
    val_ds = HAM10000Dataset(data_config['root_dir'], transform=transform, synthetic=synthetic, num_samples=20)
    
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False)
    
    return train_loader, val_loader
