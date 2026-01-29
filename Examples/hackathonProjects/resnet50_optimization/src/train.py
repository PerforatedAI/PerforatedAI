import argparse
import sys
import os

# Add the library root to python path so we can import perforatedai
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import get_dataloaders
from src.model import get_resnet
import wandb

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
        GPA.pai_tracker.add_extra_score(acc, "train")


def evaluate(model, loader, criterion, device, epoch, use_pai=False):
    """
    Evaluate the model and handle PAI restructuring.
    
    CRITICAL: Returns the model because add_validation_score may return 
    a restructured model with new dendrites.
    """
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
        # CRITICAL: add_validation_score returns a potentially restructured model
        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(acc, model)
        
        if restructured:
            print(f"*** PAI: Model restructured at epoch {epoch}! Dendrites added. ***")
            
    # CRITICAL FIX: Return the model so main() gets the updated version
    return model, acc, restructured, training_complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='baseline', choices=['baseline', 'dendritic'])
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--arch', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'])
    parser.add_argument('--run_name', type=str, default='resnet_run')
    parser.add_argument('--project_name', type=str, default='hackathon-dendritic')
    
    # PAI Hyperparameters - FIXED: Increased default switch epoch
    parser.add_argument('--pai_global_candidates', type=int, default=1)
    parser.add_argument('--pai_improvement_threshold', type=float, default=0.001)
    parser.add_argument('--pai_max_dendrites', type=int, default=100)
    parser.add_argument('--pai_candidate_multiplier', type=float, default=0.01)
    # FIXED: Changed default from 5 to 10 (or higher based on your needs)
    parser.add_argument('--pai_switch_epoch', type=int, default=10, 
                        help='Epoch to switch to dendritic mode (should be after model starts to plateau)')
    parser.add_argument('--pai_switch_mode', type=str, default='fixed', choices=['fixed', 'dynamic'])
    
    args = parser.parse_args()
    
    wandb.init(project=args.project_name, name=args.run_name, config=args)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    trainloader, testloader, num_classes = get_dataloaders(args.batch_size, dataset_name=args.dataset)
    
    model = get_resnet(num_classes=num_classes, arch=args.arch).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    use_pai = (args.mode == 'dendritic')
    
    if use_pai:
        if not HAS_PAI:
            raise ImportError("PerforatedAI package is missing!")
        
        print("Initializing PerforatedAI Dendritic Optimization...")
        
        # PAI Setup
        GPA.pc.set_testing_dendrite_capacity(False)
        GPA.pc.set_verbose(True)
        GPA.pc.set_unwrapped_modules_confirmed(True)
        GPA.pc.set_weight_decay_accepted(True)
        
        GPA.pc.append_modules_to_track([nn.BatchNorm2d, nn.ReLU, nn.Identity, nn.AdaptiveAvgPool2d, nn.Sequential])
        
        GPA.pc.set_global_candidates(args.pai_global_candidates)
        GPA.pc.set_improvement_threshold([args.pai_improvement_threshold, args.pai_improvement_threshold/10.0, 0.0])
        GPA.pc.set_max_dendrites(args.pai_max_dendrites)
        
        if args.pai_switch_mode == 'fixed':
            print(f"Using FIXED_SWITCH mode. Dendrites added at epoch {args.pai_switch_epoch}...")
            GPA.pc.set_switch_mode(GPA.pc.DOING_FIXED_SWITCH)
            GPA.pc.set_fixed_switch_num(args.pai_switch_epoch)
            GPA.pc.set_first_fixed_switch_num(args.pai_switch_epoch)
        else:
            print("Using DYNAMIC mode. Dendrites will be added upon plateau detection...")
        
        GPA.pc.set_candidate_weight_initialization_multiplier(args.pai_candidate_multiplier)
        
        # Initialize PAI
        model = UPA.initialize_pai(model)
        
        # Setup optimizer through PAI tracker
        GPA.pai_tracker.set_optimizer(optim.SGD)
        GPA.pai_tracker.set_scheduler(optim.lr_scheduler.CosineAnnealingLR)
        
        optimArgs = {"params": model.parameters(), "lr": args.lr, "momentum": 0.9, "weight_decay": 5e-4}
        schedArgs = {"T_max": args.epochs}
        optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
    else:
        # Baseline mode - standard optimizer setup
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, trainloader, criterion, optimizer, device, epoch, use_pai)
        
        # CRITICAL FIX: Capture the returned model!
        model, val_acc, restructured, training_complete = evaluate(
            model, testloader, criterion, device, epoch, use_pai
        )
        
        # CRITICAL FIX: Re-initialize optimizer IMMEDIATELY after restructuring
        if use_pai and restructured:
            print(f"*** Re-initializing optimizer with new dendrite parameters at epoch {epoch} ***")
            
            # The model now has new parameters (dendrites) that need to be trained
            # We MUST create a new optimizer that knows about ALL parameters
            optimArgs = {
                "params": model.parameters(),  # This now includes dendrite parameters!
                "lr": args.lr, 
                "momentum": 0.9, 
                "weight_decay": 5e-4
            }
            schedArgs = {"T_max": args.epochs - epoch}  # Adjust remaining epochs
            optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
            
            # Log the restructuring event
            wandb.log({"restructured": 1, "epoch": epoch})
            
            # Verify new parameters are in optimizer
            num_param_groups = len(optimizer.param_groups)
            total_params = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
            print(f"    Optimizer now tracking {num_param_groups} param groups, {total_params} total parameters")
        
        # Only step scheduler if we didn't restructure (PAI handles this internally when restructuring)
        if not (use_pai and restructured):
            scheduler.step()
             
        if training_complete:
            print("PerforatedAI indicated training complete.")
            break
            
    wandb.finish()
    
    # Save model
    save_name = f'dendritic_{args.arch}.pth' if args.mode == 'dendritic' else f'baseline_{args.arch}.pth'
    torch.save(model.state_dict(), save_name)
    print(f"Model saved to {save_name}")


if __name__ == '__main__':
    main()
