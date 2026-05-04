import medmnist
from medmnist import INFO
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

def get_loaders(flag, batch_size, root="./data"):
    info = INFO[flag]
    DataClass = getattr(medmnist, info['python_class'])
    
    # Train: NO augmentation (matches MNIST example for clean Train > Val)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    # Val: Same as train (pure)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    
    train_ds = DataClass(split='train', transform=train_transform, download=True, root=root)
    val_ds = DataClass(split='val', transform=val_transform, download=True, root=root)
    test_ds = DataClass(split='test', transform=val_transform, download=True, root=root)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
