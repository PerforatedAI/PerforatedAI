import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import config

def get_dataloaders(batch_size=128, num_workers=2):
    """
    Returns train and test dataloaders for CIFAR-100.
    Resizes images to 224x224 for ResNet compatibility.
    """
    # Standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize(224), # Resize for ResNet
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        normalize,
    ])

    trainset = torchvision.datasets.CIFAR100(root=config.DATA_DIR, train=True,
                                             download=True, transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR100(root=config.DATA_DIR, train=False,
                                            download=True, transform=test_transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    return trainloader, testloader

def plot_comparison(results, save_path="comparison.png"):
    """
    Plots accuracy vs parameters for different models.
    results: dict of {model_name: {'accuracy': float, 'params': int}}
    """
    names = list(results.keys())
    accuracies = [results[n]['accuracy'] for n in names]
    params = [results[n]['params'] / 1e6 for n in names] # in Millions

    plt.figure(figsize=(10, 6))
    
    for name, acc, p in zip(names, accuracies, params):
        plt.scatter(p, acc, s=100, label=name)
        plt.text(p, acc + 0.5, f"{name}\n{acc:.2f}%", ha='center')

    plt.xlabel("Parameters (Millions)")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.title("Accuracy vs Parameters: Dendritic Learning on CIFAR-100")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Comparison plot saved to {save_path}")

def count_parameters(model):
    """
    Counts trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
