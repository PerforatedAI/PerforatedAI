import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import wandb
from tqdm import tqdm

# Add repository root to sys.path to import perforatedai
current_dir = os.path.dirname(os.path.abspath(__file__))
# current is PerforatedAI/Examples/hackathonProjects/resnet-cifar100-submission
# root is PerforatedAI
# Path to root is ../../..
repo_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.append(repo_root)

# Check if we can import, otherwise try relative path adjustment
try:
    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA
except ImportError:
    # Fallback/Debug
    print(f"Failed to import perforatedai. Added path: {repo_root}")
    # Try adding one more level up if running from deep inside
    repo_root_2 = os.path.abspath(os.path.join(current_dir, "../../"))
    sys.path.append(repo_root_2)
    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA

from model import get_model
from utils import get_dataloaders

def train(args, model, device, train_loader, optimizer, epoch, use_pai=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        pbar.set_postfix({'loss': running_loss / (batch_idx + 1), 'acc': 100. * correct / total})

    acc = 100. * correct / total
    avg_loss = running_loss / len(train_loader)
    
    if args.wandb:
        wandb.log({"train_loss": avg_loss, "train_acc": acc, "epoch": epoch})
        
    print(f"Train Epoch: {epoch} \tLoss: {avg_loss:.4f} \tAccuracy: {acc:.2f}%")
    
    if use_pai:
        # PAI Tracker Update
        GPA.pai_tracker.add_extra_score(acc, "train")

def test(args, model, device, test_loader, optimizer, scheduler, use_pai=False, total_samples=None):
    model.eval()
    test_loss = 0
    correct = 0
    num_samples = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss(reduction='sum')(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            num_samples += target.size(0)

    # Use total_samples if provided (for dry-run), otherwise try .dataset
    if total_samples is None:
        try:
            total_samples = len(test_loader.dataset)
        except AttributeError:
            total_samples = num_samples
    
    test_loss /= total_samples
    acc = 100. * correct / total_samples

    if args.wandb:
        wandb.log({"test_loss": test_loss, "test_acc": acc, "epoch": 0})

    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total_samples} ({acc:.2f}%)\n")


    training_complete = False
    if use_pai:
        # PAI Tracker Validation Update
        # It may restructure the model
        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(acc, model)
        model.to(device)
        
        if restructured:
            print("Model Restructured by PAI!")
            optimArgs = {"params": model.parameters(), "lr": args.lr}
            schedArgs = {"step_size": args.step_size, "gamma": args.gamma}
            optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
            
    return model, optimizer, scheduler, training_complete

def main():
    parser = argparse.ArgumentParser(description="ResNet50 CIFAR-100 with Perforated AI")
    parser.add_argument("--mode", type=str, default="dendritic", choices=["baseline", "dendritic"], help="Mode: baseline or dendritic")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.1, help="LR Gamma")
    parser.add_argument("--step-size", type=int, default=30, help="LR Step Size")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Fast run for debugging")
    
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    if args.wandb:
        # wandb.init(project="resnet-cifar100-pai", config=args) 
        pass

    # Load Data
    train_loader, test_loader = get_dataloaders(args.batch_size)

    # Initialize Model
    model = get_model(num_classes=100).to(device)
    
    use_pai = (args.mode == "dendritic")

    if use_pai:
        print("Initializing Perforated AI Dendrites...")
        # PAI Global Settings
        GPA.pc.set_testing_dendrite_capacity(False)
        GPA.pc.set_verbose(True)
        GPA.pc.set_dendrite_update_mode(True)
        # Avoid PDB drop-in on unwrapped BatchNorms
        GPA.pc.set_unwrapped_modules_confirmed(True)
        
        # Initialize PAI
        model = UPA.initialize_pai(model)
        
        # Optimizer and Scheduler via PAI
        GPA.pai_tracker.set_optimizer(optim.SGD)
        GPA.pai_tracker.set_scheduler(StepLR)
        # Note: PAI expects 'momentum' in optimArgs if using SGD? 
        # Checking mnist example, it used Adadelta which doesn't strictly require momentum.
        # Let's stick to SGD with momentum as standard for ResNet
        optimArgs = {"params": model.parameters(), "lr": args.lr, "momentum": 0.9, "weight_decay": 5e-4}
        schedArgs = {"step_size": args.step_size, "gamma": args.gamma}
        
        optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
        
    else:
        print("Running in Baseline Mode")
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        # train(args, model, device, train_loader, optimizer, epoch, use_pai)
        # We need to pass dry_run to train/test or handle it here.
        # Let's modify train/test signature or wrap loader
        
        curr_train_loader = train_loader
        curr_test_loader = test_loader
        
        if args.dry_run:
            from itertools import islice
            curr_train_loader = list(islice(train_loader, 10)) # 10 batches
            curr_test_loader = list(islice(test_loader, 10))
            print("Dry run: limiting to 10 batches")

        train(args, model, device, curr_train_loader, optimizer, epoch, use_pai)
        model, optimizer, scheduler, training_complete = test(args, model, device, curr_test_loader, optimizer, scheduler, use_pai)
        
        if not use_pai:
            scheduler.step()
        
        if use_pai and training_complete:
            print("PAI Training Complete Triggered.")
            break
        
        if args.dry_run:
            break

    if args.wandb:
        try:
            wandb.finish()
        except:
            pass

if __name__ == "__main__":
    main()
