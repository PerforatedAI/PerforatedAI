from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from types import SimpleNamespace
import os
import sys

# --- IMPORT LIBRARY DARI ROOT ---
# Pastikan path ini benar sesuai struktur folder Anda
try:
    import perforatedai
except ImportError:
    sys.path.append(os.path.abspath("../../../"))

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

import wandb

# --- MODEL DEFINITION ---
class NeuroResNet(nn.Module):
    def __init__(self, num_classes=7):
        super(NeuroResNet, self).__init__()
        # Menggunakan ResNet50 Pretrained
        self.base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Modifikasi Layer Terakhir
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)

    def forward(self, x):
        return self.base(x)

# --- TRAINING LOOP ---
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

    train_acc = 100.0 * correct / len(train_loader.dataset)
    # PAI TRACKER ADD SCORE (WAJIB)
    GPA.pai_tracker.add_extra_score(train_acc, "train")
    model.to(device)
    return train_acc

# --- TESTING LOOP ---
def test(model, device, test_loader, optimizer, scheduler, args):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    val_acc = 100.0 * correct / len(test_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), val_acc))

    # PAI VALIDATION CHECK (WAJIB)
    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_acc, model)
    model.to(device)
    
    # RESTRUCTURE OPTIMIZER IF DENDRITES ADDED
    if restructured and not training_complete:
        print("⚡️ DENDRITIC GROWTH! Restructuring Optimizer...")
        optimArgs = {
            "params": model.parameters(),
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }
        # Gunakan scheduler yang sesuai
        schedArgs = {'mode': 'max', 'patience': 3, 'factor': 0.5}
        optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
        
    return model, optimizer, scheduler, training_complete, val_acc, restructured

# --- CONFIG PARSER ---
def get_parameters_dict():
    parameters_dict = {
        "learning_rate": {"values": [5e-5]},
        "batch_size": {"values": [32]},
        "weight_decay": {"values": [1e-4]},
        "improvement_threshold": {"values": [1]}, 
        "dendrite_mode": {"values": [1]},
    }
    return parameters_dict

def parse_config_string(name_str, parameters_dict):
    prefix = "Dendrites-"
    if not name_str.startswith(prefix):
        return SimpleNamespace(**{"learning_rate": 5e-5}) 
    result = {"dendrite_mode": 1}
    return SimpleNamespace(**result)

# --- MAIN FUNCTION ---
def main(run=None):
    parser = argparse.ArgumentParser(description="NeuroResNet RAF-DB PAI")
    parser.add_argument("--save-name", type=str, default="PAI")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--no-mps", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--sweep-id", type=str, default="main")
    parser.add_argument("--use-config-string", type=str, default=None)
    parser.add_argument("--count", type=int, default=1)
    # --- FIX: INI DIA YANG DULU HILANG! ---
    parser.add_argument("--save-model", action="store_true", default=True, help="Save model at the end") 
    
    args, unknown = parser.parse_known_args()

    # Config Handling
    if args.use_config_string:
        parameters_dict = get_parameters_dict()
        config = parse_config_string(args.use_config_string, parameters_dict)
    elif run is not None:
        config = run.config
        if hasattr(config, 'learning_rate'): args.lr = config.learning_rate
        if hasattr(config, 'batch_size'): args.batch_size = config.batch_size
        if hasattr(config, 'weight_decay'): args.weight_decay = config.weight_decay
    else:
        config = SimpleNamespace(dendrite_mode=1)

    # DEVICE SETUP
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    torch.manual_seed(args.seed)
    if use_cuda: device = torch.device("cuda")
    elif use_mps: device = torch.device("mps")
    else: device = torch.device("cpu")
    print(f"⚙️ Device: {device}")

    # DATASET SETUP
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if not os.path.exists('train'):
        print("❌ ERROR: 'train' folder not found. Please ensure dataset is present.")
        return

    train_ds = datasets.ImageFolder('train', transform)
    test_ds = datasets.ImageFolder('test', transform)
    
    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **kwargs)

    # PAI SETUP
    try:
        GPA.pc.set_testing_dendrite_capacity(False)
        GPA.pc.set_weight_decay_accepted(True)
        GPA.pc.set_unwrapped_modules_confirmed(True)
        GPA.pc.set_verbose(False)
    except:
        pass 

    # INIT MODEL & PAI
    model = NeuroResNet(num_classes=7).to(device)
    model = UPA.initialize_pai(model, save_name=args.save_name)

    # OPTIMIZER SETUP
    GPA.pai_tracker.set_optimizer(optim.AdamW)
    GPA.pai_tracker.set_scheduler(ReduceLROnPlateau)
    optimArgs = {"params": model.parameters(), "lr": args.lr, "weight_decay": args.weight_decay}
    schedArgs = {'mode': 'max', 'patience': 3, 'factor': 0.5}
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)

    # TRACKING VARS
    dendrite_count = 0
    max_val = 0
    max_train = 0
    max_params = 0
    global_max_val = 0
    global_max_train = 0
    global_max_params = 0

    # MAIN TRAINING LOOP
    try:
        for epoch in range(1, args.epochs + 1):
            train_acc = train(args, model, device, train_loader, optimizer, epoch)
            model, optimizer, scheduler, training_complete, val_acc, restructured = test(
                model, device, test_loader, optimizer, scheduler, args
            )

            # UPDATE METRICS
            current_params = UPA.count_params(model)
            if val_acc > max_val:
                max_val = val_acc
                max_train = train_acc
                max_params = current_params
            
            if val_acc > global_max_val:
                global_max_val = val_acc
                global_max_train = train_acc
                global_max_params = current_params

            # LOGGING
            if run is not None:
                try:
                    num_dendrites = GPA.pai_tracker.member_vars.get("num_dendrites_added", 0)
                    mode = GPA.pai_tracker.member_vars.get("mode", "n")
                except:
                    num_dendrites = 0
                    mode = "n"

                metrics = {
                    "ValAcc": val_acc,
                    "TrainAcc": train_acc,
                    "Param Count": current_params,
                    "Dendrite Count": num_dendrites
                }
                
                if restructured:
                    if mode == "n" and (dendrite_count != num_dendrites):
                        dendrite_count = num_dendrites
                        metrics.update({
                            "Arch Max Val": max_val,
                            "Arch Max Train": max_train,
                            "Arch Param Count": max_params,
                            "Arch Dendrite Count": max(0, num_dendrites - 1)
                        })
                        max_val = 0
                
                run.log(metrics)

            if training_complete:
                print("🏁 Training Complete (PAI Stopped Early)")
                if run is not None:
                    run.log({
                        "Final Max Val": global_max_val,
                        "Final Param Count": global_max_params,
                        "Final Dendrite Count": GPA.pai_tracker.member_vars.get("num_dendrites_added", 0)
                    })
                break
                
    except Exception as e:
        print(f"⚠️ Warning: Error during training loop: {e}")

    if args.save_model:
        print("💾 Saving Model...")
        try:
            torch.save(model.state_dict(), "neuroresnet.pt")
            print("✅ Model Saved Successfully!")
        except Exception as e:
            print(f"⚠️ Failed to save model: {e}")
            try:
                if hasattr(model, '_model'):
                    torch.save(model._model.state_dict(), "neuroresnet_base.pt")
                    print("✅ Base Model Saved (Backup)")
            except:
                pass

def run_wrapper():
    try:
        with wandb.init() as wandb_run:
            main(wandb_run)
    except Exception as e:
        print(f"⚠️ WandB Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sweep-id", type=str, default="main")
    parser.add_argument("--count", type=int, default=1)
    args, _ = parser.parse_known_args()

    wandb.login()
    
    sweep_config = {
        'method': 'grid',
        'metric': {'name': 'ValAcc', 'goal': 'maximize'},
        'parameters': get_parameters_dict()
    }
    
    if args.sweep_id == "main":
        sweep_id = wandb.sweep(sweep_config, project="NeuroResNet-RAFDB-Hackathon")
        print(f"Initialized sweep: {sweep_id}")
        wandb.agent(sweep_id, run_wrapper, count=args.count)
    else:
        wandb.agent(args.sweep_id, run_wrapper, count=args.count, project="NeuroResNet-RAFDB-Hackathon")