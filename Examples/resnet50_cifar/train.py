import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import get_dataloaders
from src.model import get_resnet50
import wandb
import os

# Try importing PerforatedAI
try:
    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA
    HAS_PAI = True
except ImportError:
    HAS_PAI = False
    print("Warning: PerforatedAI not found. Dendritic mode will fail if enabled.")

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, use_pai=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    acc = 100. * correct / total
    avg_loss = running_loss / len(loader)
    
    print(f"Epoch {epoch}: Train Loss: {avg_loss:.4f} | Train Acc: {acc:.2f}%")
    wandb.log({"train_loss": avg_loss, "train_acc": acc, "epoch": epoch})
    
    if use_pai and HAS_PAI:
        # Add score to tracker
        GPA.pai_tracker.add_extra_score(acc, "train")

def evaluate(model, loader, criterion, device, epoch, use_pai=False, optimizer=None, scheduler=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    acc = 100. * correct / total
    avg_loss = running_loss / len(loader)
    
    print(f"Epoch {epoch}: Val Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}%")
    wandb.log({"val_loss": avg_loss, "val_acc": acc, "epoch": epoch})
    
    restructured = False
    training_complete = False
    
    if use_pai and HAS_PAI:
        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(acc, model)
        if restructured and optimizer:
             # Reset optimizer/scheduler if restructured
             # In PAI examples, they often re-setup the optimizer/scheduler here
             pass
            
    return acc, restructured, training_complete

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='baseline', choices=['baseline', 'dendritic'], help='Mode: baseline or dendritic')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--dataset', type=str, default='cifar100', help='Dataset to use')
    parser.add_argument('--run_name', type=str, default='resnet50_run', help='WandB run name')
    parser.add_argument('--project_name', type=str, default='hackathon-dendritic', help='WandB project name')
    
    # PAI Hyperparameters
    parser.add_argument('--pai_global_candidates', type=int, default=1, help='Number of global candidates')
    parser.add_argument('--pai_improvement_threshold', type=float, default=0.001, help='Improvement threshold')
    parser.add_argument('--pai_max_dendrites', type=int, default=100, help='Max dendrites')
    parser.add_argument('--pai_candidate_multiplier', type=float, default=0.01, help='Candidate weight init multiplier')
    
    args = parser.parse_args()
    
    wandb.init(project=args.project_name, name=args.run_name, config=args)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    trainloader, testloader, num_classes = get_dataloaders(args.batch_size, dataset_name=args.dataset)
    
    model = get_resnet50(num_classes=num_classes).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    
    use_pai = (args.mode == 'dendritic')
    if use_pai:
        if not HAS_PAI:
            raise ImportError("PerforatedAI package is missing! cannot run in dendritic mode.")
        
        print("Initializing PerforatedAI Dendritic Optimization...")
        # PAI Setup
        GPA.pc.set_testing_dendrite_capacity(False)
        GPA.pc.set_verbose(True)
        
        # Tell PAI to skip the unwrapped modules check (avoids pdb breakpoint)
        GPA.pc.set_unwrapped_modules_confirmed(True)
        # Accept weight decay warning (PAI recommends not using weight_decay)
        GPA.pc.set_weight_decay_accepted(True)
        
        # Tell PAI to track but not convert BatchNorm, ReLU, Identity, AdaptiveAvgPool2d
        # These are structural layers that don't need dendrites
        GPA.pc.append_modules_to_track([nn.BatchNorm2d, nn.ReLU, nn.Identity, nn.AdaptiveAvgPool2d, nn.Sequential])
        
        # Apply command line args to PAI Config
        GPA.pc.set_global_candidates(args.pai_global_candidates)
        # improvement_threshold is a list in PAIConfig, we can set all 3 values or just 1.
        # It seems index 0 is most important, but let's set as list.
        # [0.001, 0.0001, 0.0] default
        GPA.pc.set_improvement_threshold([args.pai_improvement_threshold, args.pai_improvement_threshold/10.0, 0.0])
        GPA.pc.set_max_dendrites(args.pai_max_dendrites)
        
        # DEMO CONFIG: Force dendrite addition every 2 epochs
        print("Using FIXED_SWITCH mode to force dendrite addition for demo...")
        GPA.pc.set_switch_mode(GPA.pc.DOING_FIXED_SWITCH)
        GPA.pc.set_fixed_switch_num(2)
        GPA.pc.set_first_fixed_switch_num(2)
        
        GPA.pc.set_candidate_weight_initialization_multiplier(args.pai_candidate_multiplier)
        
        model = UPA.initialize_pai(model)
        
        # Override optimizer with PAI tracker
        GPA.pai_tracker.set_optimizer(optim.SGD)
        GPA.pai_tracker.set_scheduler(optim.lr_scheduler.CosineAnnealingLR)
        
        # Note: PAI recommends NOT using weight_decay, but we'll keep it for comparison
        optimArgs = {"params": model.parameters(), "lr": args.lr, "momentum": 0.9, "weight_decay": 5e-4}
        schedArgs = {"T_max": args.epochs}
        optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, trainloader, criterion, optimizer, device, epoch, use_pai)
        val_acc, restructured, training_complete = evaluate(model, testloader, criterion, device, epoch, use_pai, optimizer, scheduler)
        
        if use_pai and restructured:
            # Re-setup optimizer/scheduler if restructured
             optimArgs = {"params": model.parameters(), "lr": args.lr, "momentum": 0.9, "weight_decay": 5e-4}
             schedArgs = {"T_max": args.epochs} # Note: Resetting scheduler might affect learning rate schedule consistency
             optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)

        if not use_pai or not restructured:
             scheduler.step()
             
        if training_complete:
            print("PerforatedAI indicated training complete.")
            break
            
    wandb.finish()
    if args.mode == 'baseline':
        torch.save(model.state_dict(), 'baseline_resnet50.pth')
    else:
        torch.save(model.state_dict(), 'dendritic_resnet50.pth')

if __name__ == '__main__':
    main()
