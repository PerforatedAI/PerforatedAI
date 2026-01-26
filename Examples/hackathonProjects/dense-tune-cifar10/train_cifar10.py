"""
Dense-Tune: Dendritic Vision Optimization
Task: CIFAR-10 Image Classification
Architecture: Deep CNN with SiLU activations

This script follows PerforatedAI hackathon requirements:
- PAI.png auto-generated in PAI/ folder
- Dendrites added automatically when improvement stalls
- training_complete signals when to stop
"""
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import os
import sys

# Ensure the local perforatedai package is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA


class DendriticCIFARNet(nn.Module):
    """Deep CNN for CIFAR-10 (32x32 RGB images)."""
    def __init__(self, dropout=0.25):
        super(DendriticCIFARNet, self).__init__()
        # Conv Block 1
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Conv Block 2
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Conv Block 3
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Fully connected
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Block 1: 32x32 -> 16x16
        x = F.silu(self.conv1(x))
        x = F.silu(self.bn1(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        # Block 2: 16x16 -> 8x8
        x = F.silu(self.conv3(x))
        x = F.silu(self.bn2(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        
        # Block 3: 8x8 -> 4x4
        x = F.silu(self.conv5(x))
        x = F.silu(self.bn3(self.conv6(x)))
        x = F.max_pool2d(x, 2)
        
        # FC layers
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.silu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader), loss.item()))
            
    train_acc = 100.0 * correct / len(train_loader.dataset)
    GPA.pai_tracker.add_extra_score(train_acc, "train")
    return train_acc


def test(model, device, test_loader, optimizer, scheduler, args):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    val_acc = 100.0 * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), val_acc))

    # PAI: Add validation score - this triggers dendrite additions when improvement stalls
    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_acc, model)
    model.to(device)
    
    # Reset optimizer if model was restructured (dendrites added)
    if restructured and not training_complete:
        print("  ✨ Dendrites Added! Resetting optimizer.")
        optimArgs = {'params': model.parameters(), 'lr': args.lr, 'weight_decay': 0.0001}
        schedArgs = {'step_size': 1, 'gamma': args.gamma}
        optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
        
    return model, optimizer, scheduler, training_complete, val_acc


def main():
    parser = argparse.ArgumentParser(description='Dense-Tune: CIFAR-10 Dendritic Optimization')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=200, 
                        help='Max epochs (training stops when dendrites complete)')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--save-name', type=str, default='PAI')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    
    torch.manual_seed(args.seed)
    
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")

    # Data loaders with augmentation
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 2, 'pin_memory': True, 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # CIFAR-10 transforms with augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    dataset_train = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    dataset_test = datasets.CIFAR10('./data', train=False, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    # === PAI Configuration ===
    # IMPORTANT: Set to False for real training (True is only for testing setup)
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_weight_decay_accepted(True)
    GPA.pc.set_unwrapped_modules_confirmed(True)
    GPA.pc.set_verbose(False)
    
    # History mode with improvement thresholds
    GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)
    GPA.pc.set_improvement_threshold([0.01, 0.001, 0.0001, 0])
    
    # Conv dimensions: [batch, neurons, x, y] -> neuron dim is 1
    GPA.pc.set_output_dimensions([-1, 0, -1, -1])

    # Initialize model with PAI
    model = DendriticCIFARNet().to(device)
    model = UPA.initialize_pai(model, save_name=args.save_name)
    
    # Setup optimizer via SGD with momentum (better for CIFAR)
    GPA.pai_tracker.set_optimizer(optim.SGD)
    GPA.pai_tracker.set_scheduler(StepLR)
    optimArgs = {'params': model.parameters(), 'lr': args.lr, 'momentum': 0.9, 'weight_decay': 0.0001}
    schedArgs = {'step_size': 1, 'gamma': args.gamma}
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)

    print(f"\n🚀 Dense-Tune: CIFAR-10 Dendritic Optimization")
    print(f"   Model: DendriticCIFARNet ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"   Device: {device}")
    print(f"   PAI graph will be saved to: {args.save_name}/PAI.png\n")
    
    for epoch in range(1, args.epochs + 1):
        train_acc = train(args, model, device, train_loader, optimizer, epoch)
        model, optimizer, scheduler, training_complete, val_acc = test(
            model, device, test_loader, optimizer, scheduler, args)
        
        print(f"Epoch {epoch:03d} | Train: {train_acc:5.2f}% | Val: {val_acc:5.2f}% | "
              f"Dendrites: {GPA.pai_tracker.member_vars['num_dendrites_added']} | "
              f"Mode: {GPA.pai_tracker.member_vars['mode']}")
        
        if training_complete:
            print("\n✅ Dendritic Optimization Complete!")
            print(f"   Final dendrite count: {GPA.pai_tracker.member_vars['num_dendrites_added']}")
            break

    print(f"\n📊 PAI.png generated in {args.save_name}/")
    print("Run 'python3 benchmark.py' to see final results.")


if __name__ == "__main__":
    main()
