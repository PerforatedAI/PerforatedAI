import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from perforatedai import globals_perforatedai as GPA

def get_data_loaders(batch_size=128, dataset_name="mnist"):
    """
    Downloads and prepares the MNIST or CIFAR-10 dataset.
    """
    if dataset_name.lower() == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        # Default to MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch_idx):
    model.train()
    running_loss = 0.0
    total_samples = 0
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        total_loss += loss.item()
        total_samples += 1
        
        if batch_idx % 100 == 99:
            print(f"   [Epoch {epoch_idx+1}] Batch {batch_idx+1}/{len(train_loader)} | Loss: {running_loss / 100:.4f}")
            running_loss = 0.0
            
    return total_loss / total_samples

def evaluate_accuracy(model, test_loader, device):
    """
    Fast evaluation for training loop (keeps model on current device).
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100.0 * correct / total

def train_model(model, train_loader, device, optimizer):
    """
    Performs a single epoch of training as expected by the PerforatedAI API.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        running_loss += loss.item()
        
        if batch_idx % 100 == 99:
            print(f"   Batch {batch_idx+1}/{len(train_loader)} | Loss: {running_loss / 100:.4f}")
            running_loss = 0.0
            
    train_acc = 100.0 * correct / total
    
    # Track the training score optionally via the official API
    GPA.pai_tracker.add_extra_score(train_acc, 'Train Accuracy')
    
    return train_acc
