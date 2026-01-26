from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from types import SimpleNamespace

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

import wandb

class Net(nn.Module):
    def __init__(self, num_classes, width, dropout1=0.25, dropout2=0.5):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, int(32 * width), 3, 1)
        self.conv2 = nn.Conv2d(int(32 * width), int(64 * width), 3, 1)
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)

        # Calculate flattened size dynamically based on width
        # Input: 28x28
        # After conv1: (28-3+1) = 26x26
        # After conv2: (26-3+1) = 24x24
        # After max_pool2d: 12x12
        # Flattened size: int(64 * width) * 12 * 12
        self.fc1_input_features = int(64 * width) * 12 * 12
        self.fc1 = nn.Linear(self.fc1_input_features, int(128 * width))
        self.fc2 = nn.Linear(int(128 * width), num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

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
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_acc = 100.0 * correct / len(train_loader.dataset)
    GPA.pai_tracker.add_extra_score(train_acc, "train")
    model.to(device) # Ensure model is on device after PAI operations
    return train_acc

def test(model, device, test_loader, optimizer, scheduler, args):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    val_acc = 100.0 * correct / len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), val_acc
        )
    )

    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
        val_acc, model
    )
    model.to(device) # Ensure model is on device after PAI operations
    if restructured and not training_complete:
        optimArgs = {
            "params": model.parameters(),
            "lr": args.lr, # Use initial lr, PAI might adjust internally
            "weight_decay": args.weight_decay, # This will be from config
        }
        schedArgs = {"step_size": 1, "gamma": args.gamma} # This will be from config
        optimizer, scheduler = GPA.pai_tracker.setup_optimizer(
            model, optimArgs, schedArgs
        )
    return model, optimizer, scheduler, training_complete, val_acc, restructured

def parse_config_string(name_str, parameters_dict):
    prefix = "Dendrites-"
    if not name_str.startswith(prefix):
        raise ValueError("name_str must start with 'Dendrites-'")
    rest = name_str[len(prefix) :]
    if "_" in rest:
        dend_token, rest = rest.split("_", 1)
    else:
        dend_token, rest = rest, ""
    result = {"dendrite_mode": dend_token}
    tokens = rest.split("_") if rest else []
    excluded = ["method", "metric", "parameters", "dendrite_mode"]
    keys = [k for k in parameters_dict.keys() if k not in excluded]
    for key in keys:
        if not tokens:
            break
        result[key] = tokens.pop(0)
    if tokens:
        result["_extras"] = tokens
    return SimpleNamespace(**result)

def get_parameters_dict():
    parameters_dict = {
        "dropout": {"values": [0.1, 0.3, 0.5]},
        "weight_decay": {"values": [0, 0.0001, 0.001]},
        "improvement_threshold": {"values": [0, 1, 2]},
        "candidate_weight_initialization_multiplier": {"values": [0.1, 0.01]},
        "pai_forward_function": {"values": [0, 1, 2]},
        "dendrite_mode": {"values": [0, 1, 2]},
        "dendrite_graph_mode": {"values": [0, 1]},
        # Example: Sweep over width multiplier
        "width": {"values": [0.75, 1.0, 1.25]},
    }
    return parameters_dict

def main(run=None):
    parser = argparse.ArgumentParser(description="PyTorch Fashion-MNIST Example with WandB and PAI")
    parser.add_argument("--save-name", type=str, default="FashionPAI")
    parser.add_argument("--dataset", type=str, default="Fashion-MNIST", choices=["Fashion-MNIST", "MNIST"]) # Added MNIST for flexibility
    parser.add_argument("--batch-size", type=int, default=64, metavar="N")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N")
    parser.add_argument("--epochs", type=int, default=10000, metavar="N")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M")
    parser.add_argument("--weight-decay", type=float, default=0.0, metavar="WD") # Default, overridden by sweep
    parser.add_argument("--dropout", type=float, default=0.25, metavar="D") # Default, overridden by sweep
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1, metavar="S")
    parser.add_argument("--log-interval", type=int, default=100, metavar="N") # Increased for less spam
    parser.add_argument("--save-model", action="store_true", default=False)
    parser.add_argument("--sweep-id", type=str, default="main")
    parser.add_argument("--use-config-string", type=str, default=None)
    parser.add_argument("--use-wandb", action="store_true", default=False) # To run without wandb if needed
    parser.add_argument("--count", type=int, default=50, help="Number of sweep runs")
    args = parser.parse_args()

    if args.use_config_string:
        parameters_dict = get_parameters_dict()
        config = parse_config_string(args.use_config_string, parameters_dict)
    elif run is not None and args.use_wandb:
        config = run.config
    else: # Running without wandb or config string - use defaults or args
        config = SimpleNamespace(
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            improvement_threshold=1, # Default
            candidate_weight_initialization_multiplier=0.01, # Default
            pai_forward_function=0, # Default sigmoid
            dendrite_mode=1, # Default GD dendrites
            dendrite_graph_mode=0, # Default
            width=1.0, # Default width
        )
    
    # PAI Configuration
    if config.improvement_threshold == 0: thresh = [0.01, 0.001, 0.0001, 0]
    elif config.improvement_threshold == 1: thresh = [0.001, 0.0001, 0]
    elif config.improvement_threshold == 2: thresh = [0]
    GPA.pc.set_improvement_threshold(thresh)
    GPA.pc.set_candidate_weight_initialization_multiplier(config.candidate_weight_initialization_multiplier)

    if config.pai_forward_function == 0: pai_forward_function = torch.sigmoid
    elif config.pai_forward_function == 1: pai_forward_function = torch.relu
    elif config.pai_forward_function == 2: pai_forward_function = torch.tanh
    else: pai_forward_function = torch.sigmoid
    GPA.pc.set_pai_forward_function(pai_forward_function)
    
    if config.dendrite_mode == 0: GPA.pc.set_max_dendrites(0)
    else: GPA.pc.set_max_dendrites(5) 

    if config.dendrite_mode < 2: GPA.pc.set_perforated_backpropagation(False)
    else:
        GPA.pc.set_perforated_backpropagation(True)
        GPA.pc.set_dendrite_graph_mode(bool(config.dendrite_graph_mode))

    if run is not None and args.use_wandb:
        excluded = ['method', 'metric', 'parameters', 'dendrite_mode', 'improvement_threshold', 'candidate_weight_initialization_multiplier', 'pai_forward_function', 'dendrite_graph_mode']
        keys = sorted([k for k in get_parameters_dict().keys() if k not in excluded and k in run.config])
        name_str = "Dendrites-" + str(run.config.dendrite_mode) + "_" + "_".join(
            str(run.config[k]) for k in keys if k in run.config
        )
        run.name = name_str

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    if args.dataset == "Fashion-MNIST":
        num_classes = 10
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
        dataset1 = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
        dataset2 = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
    elif args.dataset == "MNIST":
        num_classes = 10
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset1 = datasets.MNIST("./data", train=True, download=True, transform=transform)
        dataset2 = datasets.MNIST("./data", train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    GPA.pc.set_testing_dendrite_capacity(False) # Set to True for initial PAI capacity testing
    GPA.pc.set_weight_decay_accepted(True)
    GPA.pc.set_verbose(False) # Set to True for more PAI logging

    model = Net(num_classes, config.width, config.dropout, config.dropout).to(device)
    model = UPA.initialize_pai(model, save_name=args.save_name)

    GPA.pai_tracker.set_optimizer(optim.Adadelta)
    GPA.pai_tracker.set_scheduler(StepLR)
    optimArgs = {"params": model.parameters(), "lr": args.lr, "weight_decay": config.weight_decay}
    schedArgs = {"step_size": 1, "gamma": args.gamma}
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)

    global_max_val = 0
    global_max_train = 0
    global_max_params = 0
    final_dendrite_count = 0

    for epoch in range(1, args.epochs + 1):
        train_acc = train(args, model, device, train_loader, optimizer, epoch)
        model, optimizer, scheduler, training_complete, val_acc, restructured = test(
            model, device, test_loader, optimizer, scheduler, args
        )

        if val_acc > global_max_val:
            global_max_val = val_acc
            global_max_train = train_acc
            global_max_params = UPA.count_params(model)
            final_dendrite_count = GPA.pai_tracker.member_vars.get("num_dendrites_added", 0)

        if run is not None and args.use_wandb:
            run.log({
                "ValAcc": val_acc, "TrainAcc": train_acc,
                "Param Count": UPA.count_params(model),
                "Dendrite Count": GPA.pai_tracker.member_vars.get("num_dendrites_added", 0),
                "Epoch": epoch
            })
            if restructured:
                 run.log({
                    "Arch Max Val": val_acc, "Arch Max Train": train_acc,
                    "Arch Param Count": UPA.count_params(model),
                    "Arch Dendrite Count": GPA.pai_tracker.member_vars.get("num_dendrites_added", 0),
                })
        if training_complete:
            if run is not None and args.use_wandb:
                run.log({
                    "Final Max Val": global_max_val, "Final Max Train": global_max_train,
                    "Final Param Count": global_max_params,
                    "Final Dendrite Count": final_dendrite_count,
                })
            print(f"Training complete. Best Val Acc: {global_max_val:.2f}%")
            break
        scheduler.step() # PAI handles its own scheduler stepping if configured, but for StepLR this is fine.

    if args.save_model:
        torch.save(model.state_dict(), f"{args.save_name}_cnn.pt")

def run_sweep():
    try:
        with wandb.init() as wandb_run:
            main(wandb_run)
    except Exception as e:
        print(f"An error occurred during a sweep run: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    parser_sweep = argparse.ArgumentParser(add_help=False)
    parser_sweep.add_argument("--sweep-id", type=str, default="main")
    parser_sweep.add_argument("--count", type=int, default=50)
    args_sweep, _ = parser_sweep.parse_known_args()
    
    # Basic check for wandb usage
    if '--use-wandb' not in sys.argv and args_sweep.sweep_id == 'main':
        print("Warning: Running sweep without --use-wandb. Weights & Biases will not be used for this sweep.")
        # If not using wandb, just run main once with default args for a quick test
        # For a real sweep, --use-wandb is implied by --sweep-id != 'main' or explicitly set.
        # This part needs careful handling if you want to support non-wandb sweeps with PAI's own logging.
        # For now, let's assume if it's a sweep, wandb is intended.
        # If not a sweep, run main directly.
        if args_sweep.sweep_id == 'main' and '--use-wandb' not in sys.argv:
             main() # Run main directly if not initiating a new sweep and not explicitly using wandb
        elif args_sweep.sweep_id != 'main' and '--use-wandb' not in sys.argv:
             print("To join an existing sweep, --use-wandb is generally expected.")


    if '--use-wandb' in sys.argv or args_sweep.sweep_id != 'main': # if wandb is involved
        wandb.login()
        project_name = "Dendritic Optimization Fashion-MNIST"
        sweep_config = {"method": "random"}
        metric = {"name": "ValAcc", "goal": "maximize"}
        sweep_config["metric"] = metric
        parameters_dict = get_parameters_dict()
        sweep_config["parameters"] = parameters_dict

        if args_sweep.sweep_id == "main":
            sweep_id = wandb.sweep(sweep_config, project=project_name)
            print(f"\nInitialized sweep: {sweep_id}\n")
            wandb.agent(sweep_id, run_sweep, count=args_sweep.count)
        else:
            print(f"Joining existing sweep: {args_sweep.sweep_id}")
            wandb.agent(args_sweep.sweep_id, run_sweep, count=args_sweep.count, project=project_name)
    else: # No wandb, no sweep, just a single run
        main()
