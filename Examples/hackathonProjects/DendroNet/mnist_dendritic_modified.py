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
    def __init__(self, num_classes=10, width=1.0, dropout1=0.25, dropout2=0.5):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, int(32 * width), 3, 1)
        self.conv2 = nn.Conv2d(int(32 * width), int(64 * width), 3, 1)
        self.dropout1 = nn.Dropout(dropout1)
        self.dropout2 = nn.Dropout(dropout2)
        self.fc1 = nn.Linear(9216 * int(width), int(128 * width))
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
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        # Track accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    
    # Calculate training accuracy
    train_acc = 100. * correct / total
    
    # Add training score to PAI tracker
    GPA.pai_tracker.add_extra_score(train_acc, "train")
    model.to(device)
    
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
    val_acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), val_acc))

    # Add validation score to PAI tracker - this may trigger dendrite addition
    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
        val_acc, model
    )
    model.to(device)
    
    # If model was restructured, reset optimizer and scheduler
    if restructured and not training_complete:
        optimArgs = {
            "params": model.parameters(),
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }
        schedArgs = {"step_size": 1, "gamma": args.gamma}
        optimizer, scheduler = GPA.pai_tracker.setup_optimizer(
            model, optimArgs, schedArgs
        )
    
    return model, optimizer, scheduler, training_complete, val_acc, restructured


def parse_config_string(name_str, parameters_dict):
    """Parse a config string back into a SimpleNamespace config object."""
    prefix = "Dendrites-"
    if not name_str.startswith(prefix):
        raise ValueError("name_str must start with 'Dendrites-'")
    
    rest = name_str[len(prefix):]
    if "_" in rest:
        dend_token, rest = rest.split("_", 1)
    else:
        dend_token, rest = rest, ""
    
    result = {"dendrite_mode": int(dend_token)}
    tokens = rest.split("_") if rest else []
    
    excluded = ["method", "metric", "parameters", "dendrite_mode"]
    keys = [k for k in parameters_dict.keys() if k not in excluded]
    
    for key in keys:
        if not tokens:
            break
        val = tokens.pop(0)
        # Convert to appropriate type
        try:
            result[key] = float(val) if '.' in val else int(val)
        except ValueError:
            result[key] = val
    
    if tokens:
        result["_extras"] = tokens
    
    return SimpleNamespace(**result)


def main(run=None):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST with Dendritic Optimization')
    parser.add_argument('--save-name', type=str, default='PAI',
                        help='name for saving PAI results')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                        help='number of epochs to train (default: 10000)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--width', type=float, default=1.0, metavar='W',
                        help='width multiplier for network (default: 1.0)')
    parser.add_argument('--weight-decay', type=float, default=0.0, metavar='WD',
                        help='weight decay (default: 0.0)')
    parser.add_argument('--dropout', type=float, default=0.25, metavar='D',
                        help='dropout rate (default: 0.25)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--sweep-id', type=str, default='main',
                        help='Sweep ID to join, or "main" to create new sweep')
    parser.add_argument('--use-config-string', type=str, default=None,
                        help='Optional: Use a config string instead of wandb sweep')
    parser.add_argument('--use-wandb', action='store_true', default=False,
                        help='Use wandb for experiment tracking')
    parser.add_argument('--count', type=int, default=300,
                        help='Number of sweep runs to perform')
    args = parser.parse_args()

    # Get configuration from wandb run, config string, or defaults
    if args.use_config_string:
        parameters_dict = get_parameters_dict()
        config = parse_config_string(args.use_config_string, parameters_dict)
    elif run is not None:
        config = run.config
    else:
        # Running without wandb - use defaults for dendritic training
        config = SimpleNamespace(
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            improvement_threshold=1,
            candidate_weight_initialization_multiplier=0.01,
            pai_forward_function=0,
            dendrite_mode=2,  # Use CC dendrites (PB) by default
            dendrite_graph_mode=0,
        )

    # Configure Perforated AI settings
    
    # Set improvement threshold
    if config.improvement_threshold == 0:
        thresh = [0.01, 0.001, 0.0001, 0]
    elif config.improvement_threshold == 1:
        thresh = [0.001, 0.0001, 0]
    elif config.improvement_threshold == 2:
        thresh = [0]
    else:
        thresh = [0.001, 0.0001, 0]
    GPA.pc.set_improvement_threshold(thresh)
    
    # Set weight initialization multiplier
    GPA.pc.set_candidate_weight_initialization_multiplier(
        float(config.candidate_weight_initialization_multiplier)
    )

    # Set PAI forward function
    if config.pai_forward_function == 0:
        pai_forward_function = torch.sigmoid
    elif config.pai_forward_function == 1:
        pai_forward_function = torch.relu
    elif config.pai_forward_function == 2:
        pai_forward_function = torch.tanh
    else:
        pai_forward_function = torch.sigmoid
    GPA.pc.set_pai_forward_function(pai_forward_function)
    
    # Set dendrite mode
    dendrite_mode = int(config.dendrite_mode)
    if dendrite_mode == 0:
        GPA.pc.set_max_dendrites(0)
    else:
        GPA.pc.set_max_dendrites(5)
    
    # Set perforated backpropagation
    if dendrite_mode < 2:
        GPA.pc.set_perforated_backpropagation(False)
    else:
        GPA.pc.set_perforated_backpropagation(True)
        if hasattr(config, 'dendrite_graph_mode') and config.dendrite_graph_mode:
            GPA.pc.set_dendrite_graph_mode(True)
        else:
            GPA.pc.set_dendrite_graph_mode(False)

    # Set wandb run name if using wandb
    if run is not None:
        excluded = ['method', 'metric', 'parameters', 'dendrite_mode']
        keys = [k for k in get_parameters_dict().keys() if k not in excluded]
        name_str = "Dendrites-" + str(wandb.config.dendrite_mode) + "_" + "_".join(
            str(wandb.config[k]) for k in keys if k in wandb.config
        )
        run.name = name_str

    # Device configuration
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
        print("Using CUDA")
    elif use_mps:
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Data loader configuration
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1, 'pin_memory': True, 'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Configure PAI global settings
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_weight_decay_accepted(True)
    GPA.pc.set_verbose(False)

    # Create model
    model = Net(
        num_classes=10, 
        width=args.width, 
        dropout1=config.dropout, 
        dropout2=config.dropout
    ).to(device)

    # Initialize Perforated AI
    model = UPA.initialize_pai(model, save_name=args.save_name)

    # Setup optimizer and scheduler through PAI tracker
    GPA.pai_tracker.set_optimizer(optim.Adadelta)
    GPA.pai_tracker.set_scheduler(StepLR)
    
    optimArgs = {
        "params": model.parameters(),
        "lr": args.lr,
        "weight_decay": config.weight_decay,
    }
    schedArgs = {"step_size": 1, "gamma": args.gamma}
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)

    # Initialize tracking variables
    dendrite_count = 0
    max_val = 0
    max_train = 0
    max_params = 0

    global_max_val = 0
    global_max_train = 0
    global_max_params = 0

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_acc = train(args, model, device, train_loader, optimizer, epoch)
        model, optimizer, scheduler, training_complete, val_acc, restructured = test(
            model, device, test_loader, optimizer, scheduler, args
        )

        # Update max values for current architecture
        if val_acc > max_val:
            max_val = val_acc
            max_train = train_acc
            max_params = UPA.count_params(model)

        # Update global max values
        if val_acc > global_max_val:
            global_max_val = val_acc
            global_max_train = train_acc
            global_max_params = UPA.count_params(model)

        # Log to wandb if enabled
        if run is not None:
            run.log({
                "epoch": epoch,
                "ValAcc": val_acc,
                "TrainAcc": train_acc,
                "Param Count": UPA.count_params(model),
                "Dendrite Count": GPA.pai_tracker.member_vars["num_dendrites_added"],
            })

            # Log architecture maximums when dendrites are added
            if restructured:
                if GPA.pai_tracker.member_vars["mode"] == "n" and (
                    dendrite_count != GPA.pai_tracker.member_vars["num_dendrites_added"]
                ):
                    dendrite_count = GPA.pai_tracker.member_vars["num_dendrites_added"]
                    run.log({
                        "Arch Max Val": max_val,
                        "Arch Max Train": max_train,
                        "Arch Param Count": max_params,
                        "Arch Dendrite Count": dendrite_count - 1,
                    })

        # Check if training is complete
        if training_complete:
            print(f"\n{'='*60}")
            print("Training Complete!")
            print(f"{'='*60}")
            print(f"Final Validation Accuracy: {global_max_val:.2f}%")
            print(f"Final Training Accuracy: {global_max_train:.2f}%")
            print(f"Final Parameter Count: {global_max_params:,}")
            print(f"Total Dendrites Added: {GPA.pai_tracker.member_vars['num_dendrites_added']}")
            print(f"{'='*60}\n")
            
            # Log final metrics to wandb
            if run is not None:
                run.log({
                    "Arch Max Val": max_val,
                    "Arch Max Train": max_train,
                    "Arch Param Count": max_params,
                    "Arch Dendrite Count": GPA.pai_tracker.member_vars["num_dendrites_added"],
                })
                run.log({
                    "Final Max Val": global_max_val,
                    "Final Max Train": global_max_train,
                    "Final Param Count": global_max_params,
                    "Final Dendrite Count": GPA.pai_tracker.member_vars["num_dendrites_added"],
                })
            break

        # Step the scheduler
        scheduler.step()

    # Save model if requested
    if args.save_model:
        torch.save(model.state_dict(), "mnist_dendritic.pt")
        print("Model saved to mnist_dendritic.pt")


def get_parameters_dict():
    """Return the parameters dictionary for wandb sweep."""
    parameters_dict = {
        # Hyperparameters to sweep
        "dropout": {"values": [0.1, 0.25, 0.3, 0.5]},
        "weight_decay": {"values": [0, 0.0001, 0.001]},
        
        # Dendritic optimization parameters
        # 0 = [0.01, 0.001, 0.0001, 0], 1 = [0.001, 0.0001, 0], 2 = [0]
        "improvement_threshold": {"values": [0, 1, 2]},
        "candidate_weight_initialization_multiplier": {"values": [0.01, 0.1]},
        
        # 0 = sigmoid, 1 = relu, 2 = tanh
        "pai_forward_function": {"values": [0, 1, 2]},
        
        # dendrite_mode: 0 = no dendrites, 1 = GD dendrites, 2 = CC dendrites (PB)
        "dendrite_mode": {"values": [0, 1, 2]},
        
        # Only used when dendrite_mode == 2 (Perforated Backpropagation)
        "dendrite_graph_mode": {"values": [0, 1]},
    }
    return parameters_dict


def run_sweep():
    """Wrapper function for wandb sweep."""
    try:
        with wandb.init() as wandb_run:
            main(wandb_run)
    except Exception as e:
        import traceback
        print(f"Error during sweep run: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    # Parse minimal args for sweep control
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--use-wandb", action="store_true", default=False)
    parser.add_argument("--sweep-id", type=str, default="main")
    parser.add_argument("--count", type=int, default=300)
    args, _ = parser.parse_known_args()
    
    if args.use_wandb:
        # Initialize wandb
        wandb.login()
        project = "Dendritic-Optimization-MNIST"
        
        # Configure sweep
        sweep_config = {
            "method": "random",
            "metric": {"name": "ValAcc", "goal": "maximize"},
            "parameters": get_parameters_dict()
        }

        if args.sweep_id == "main":
            # Create new sweep
            sweep_id = wandb.sweep(sweep_config, project=project)
            print(f"\n{'='*60}")
            print(f"Initialized new sweep with ID: {sweep_id}")
            print(f"Use --sweep-id {sweep_id} to join on other machines")
            print(f"{'='*60}\n")
            
            # Run the sweep agent
            wandb.agent(sweep_id, run_sweep, count=args.count)
        else:
            # Join existing sweep
            print(f"\n{'='*60}")
            print(f"Joining existing sweep: {args.sweep_id}")
            print(f"{'='*60}\n")
            wandb.agent(args.sweep_id, run_sweep, count=args.count, project=project)
    else:
        # Run single training without wandb
        print(f"\n{'='*60}")
        print("Running single training (no wandb)")
        print(f"{'='*60}\n")
        main()