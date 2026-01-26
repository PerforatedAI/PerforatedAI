from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
from types import SimpleNamespace

# Perforated AI Imports
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

import wandb

# --- Model Setup ---
def get_model(num_classes=100):
    # Using ResNet50 (Industry Standard)
    model = models.resnet50(weights=None)
    
    # Modify the final fully connected layer to match CIFAR-100 classes
    # Standard ResNet50 has 2048 features coming out of the avgpool
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target) # CrossEntropy is standard for ResNet
        loss.backward()
        optimizer.step()
        
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break
                
        # Calculate accuracy for logging
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum()
        
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
            test_loss += F.cross_entropy(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    val_acc = 100.0 * correct / len(test_loader.dataset)
    
    print(
        "\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            val_acc,
        )
    )

    # PAI Integration: Add Validation Score
    # This triggers the logic to check if dendrites should be added
    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
        val_acc, model
    )
    
    model.to(device)
    
    # If restructured, reset optimizer/scheduler to handle new dendritic weights
    if restructured and not training_complete:
        optimArgs = {
            "params": model.parameters(),
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "momentum": 0.9 # Standard momentum for ResNet
        }
        schedArgs = {"step_size": 1, "gamma": args.gamma}
        optimizer, scheduler = GPA.pai_tracker.setup_optimizer(
            model, optimArgs, schedArgs
        )
        
    return model, optimizer, scheduler, training_complete, val_acc, restructured

def main(run=None):
    # --- Configuration ---
    parser = argparse.ArgumentParser(description="PyTorch CIFAR100 Example with WandB and Dendrites")
    parser.add_argument("--save-name", type=str, default="ResNet_CIFAR")
    parser.add_argument("--batch-size", type=int, default=64, help="input batch size")
    parser.add_argument("--test-batch-size", type=int, default=256, help="test batch size")
    parser.add_argument("--epochs", type=int, default=5000, help="max epochs (PAI handles stopping)") # High number for PAI
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.7, help="Learning rate step gamma")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--log-interval", type=int, default=20, help="how many batches to wait before logging")
    parser.add_argument("--save-model", action="store_true", default=False)
    parser.add_argument("--sweep-id", type=str, default="main")
    parser.add_argument("--count", type=int, default=100, help="Number of sweep runs")
    parser.add_argument("--use-config-string", type=str, default=None)
    parser.add_argument("--no-sweep", action="store_true", help="Run single training without sweep")
    args = parser.parse_args()

    # Load Config from WandB or Defaults
    if args.use_config_string:
        parameters_dict = get_parameters_dict()
        config = parse_config_string(args.use_config_string, parameters_dict)
    elif run is not None:
        config = run.config
    else:
        # Default config if not sweeping
        config = SimpleNamespace(
            improvement_threshold=1, # [0.001, 0.0001, 0]
            candidate_weight_initialization_multiplier=0.01,
            pai_forward_function=1, # Relu
            dendrite_mode=2, # 0=None, 1=GD, 2=PB
            dendrite_graph_mode=0
        )

    # Set PAI Global Config
    if config.improvement_threshold == 0:
        thresh = [0.01, 0.001, 0.0001, 0]
    elif config.improvement_threshold == 1:
        thresh = [0.001, 0.0001, 0]
    else:
        thresh = [0]
    GPA.pc.set_improvement_threshold(thresh)
    
    GPA.pc.set_candidate_weight_initialization_multiplier(
        config.candidate_weight_initialization_multiplier
    )

    # PAI Forward Function
    if config.pai_forward_function == 0: pai_forward_function = torch.sigmoid
    elif config.pai_forward_function == 1: pai_forward_function = torch.relu
    elif config.pai_forward_function == 2: pai_forward_function = torch.tanh
    else: pai_forward_function = torch.relu
    GPA.pc.set_pai_forward_function(pai_forward_function)
    
    # Dendrite Mode
    if config.dendrite_mode == 0:
        GPA.pc.set_max_dendrites(0)
    else:
        GPA.pc.set_max_dendrites(10) 
        GPA.pc.set_perforated_backpropagation(True if config.dendrite_mode == 2 else False)
        
    # Set W&B Run Name
    if run is not None:
        name_str = "Dendrites-" + str(config.dendrite_mode) + "_" + "_".join(
            str(run.config[k]) for k in run.config.keys() if k not in ['method', 'metric', 'parameters', 'dendrite_mode']
        )
        run.name = name_str

    # --- Device Setup ---
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    # --- Data Loading (CIFAR-100) ---
    print("Loading CIFAR-100 Dataset...")
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), # CIFAR-100 Normalization
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    dataset1 = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    dataset2 = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=args.test_batch_size, shuffle=False, num_workers=2)

    # --- Model Initialization ---
    model = get_model(num_classes=100).to(device)

    # PAI Global Settings
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_weight_decay_accepted(True)
    GPA.pc.set_verbose(False)
    
    # CRITICAL FIX FOR RESNET:
    # 1. Turn off strict dimension checking (ResNet has mixed Conv and Linear dimensions)
    GPA.pc.set_debugging_output_dimensions(0)
    # 2. Tell the system we don't want to manually confirm unwrapped modules
    GPA.pc.set_unwrapped_modules_confirmed(True)
    
    # Initialize PAI
    model = UPA.initialize_pai(model, save_name=args.save_name, doing_pai=True)

    # Setup Optimizer
    # We use SGD with Momentum for ResNet
    GPA.pai_tracker.set_optimizer(optim.SGD)
    GPA.pai_tracker.set_scheduler(StepLR)
    
    optimArgs = {
        "params": model.parameters(),
        "lr": args.lr,
        "weight_decay": config.weight_decay,
        "momentum": 0.9
    }
    schedArgs = {"step_size": 1, "gamma": args.gamma}
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)

    # --- Tracking Variables ---
    max_val = 0
    max_train = 0
    global_max_val = 0

    for epoch in range(1, args.epochs + 1):
        train_acc = train(args, model, device, train_loader, optimizer, epoch)
        model, optimizer, scheduler, training_complete, val_acc, restructured = test(
            model, device, test_loader, optimizer, scheduler, args
        )

        # Update Maxes
        if val_acc > max_val:
            max_val = val_acc
            max_train = train_acc
            
        if val_acc > global_max_val:
            global_max_val = val_acc

        # Log to WandB
        if run is not None:
            run.log({
                "ValAcc": val_acc,
                "TrainAcc": train_acc,
                "Param Count": UPA.count_params(model),
                "Dendrite Count": GPA.pai_tracker.member_vars["num_dendrites_added"],
            })

        if training_complete:
            print("Training complete!")
            break

    if args.save_model:
        torch.save(model.state_dict(), "resnet_cifar_cnn.pt")

def get_parameters_dict():
    return {
        "improvement_threshold": {"values": [0, 1, 2]},
        "candidate_weight_initialization_multiplier": {"values": [0.1, 0.01]},
        "pai_forward_function": {"values": [0, 1, 2]},
        "dendrite_mode": {"values": [0, 1]}, # 0=Standard, 1=GD (Gradient Dendrites)
        "dendrite_graph_mode": {"values": [0, 1]},
        "weight_decay": {"values": [0, 0.0005, 0.005]},
    }

def parse_config_string(name_str, parameters_dict):
    # Simplified parser for reproducibility
    prefix = "Dendrites-"
    if not name_str.startswith(prefix): raise ValueError("Invalid string")
    rest = name_str[len(prefix):]
    if "_" in rest:
        dend_token, rest = rest.split("_", 1)
    else:
        dend_token, rest = rest, ""
    result = {"dendrite_mode": dend_token}
    tokens = rest.split("_") if rest else []
    excluded = ['method', 'metric', 'parameters', 'dendrite_mode']
    keys = [k for k in parameters_dict.keys() if k not in excluded]
    for key in keys:
        if not tokens: break
        result[key] = tokens.pop(0)
    return SimpleNamespace(**result)

def run():
    try:
        with wandb.init() as wandb_run:
            main(wandb_run)
    except Exception:
        import pdb, traceback
        traceback.print_exc()
        pdb.post_mortem()

if __name__ == "__main__":
    import sys
    
    # Check for --no-sweep flag early
    if "--no-sweep" in sys.argv:
        main(run=None)
    else:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--sweep-id", type=str, default="main")
        parser.add_argument("--count", type=int, default=300)
        args, _ = parser.parse_known_args()
        
        # Initialize WandB
        wandb.login()
        project = "Perforated-ResNet-CIFAR100"
        sweep_config = {"method": "random"}
        metric = {"name": "ValAcc", "goal": "maximize"}
        sweep_config["metric"] = metric
        sweep_config["parameters"] = get_parameters_dict()

        if args.sweep_id == "main":
            sweep_id = wandb.sweep(sweep_config, project=project)
            print("\nInitialized sweep. Use --sweep_id", sweep_id, "to join.\n")
            wandb.agent(sweep_id, run, count=args.count)
        else:
            wandb.agent(args.sweep_id, run, count=args.count, project=project)