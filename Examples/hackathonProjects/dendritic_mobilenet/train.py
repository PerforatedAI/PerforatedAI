import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from perforatedai import utils_perforatedai as UPA
from perforatedai import modules_perforatedai as PA
from perforatedai import globals_perforatedai as GPA
import wandb
from tqdm import tqdm
from model import DendriticVisionModel
import os
import argparse

def run_training(config=None, split_ratio=1.0, use_dendrites=True):
    """
    Args:
        config (dict, optional): Configuration dictionary.
        split_ratio (float): Percentage of training data to use (0.0 < ratio <= 1.0).
        use_dendrites (bool): Whether to use Dendritic/PerforatedAI features.
    """
    
    # Ensure config is a dict if not provided
    if config is None:
        config = {}
        
    # Initialize a new wandb run
    # Update config with experiment details
    run_config = config.copy()
    run_config['dataset_split'] = split_ratio
    run_config['mode'] = 'dendritic' if use_dendrites else 'standard'

    with wandb.init(config=run_config, reinit=True) as run:
        config = wandb.config

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        print(f"Mode: {config.mode}, Split: {config.dataset_split*100}%")

        # Data Preparation
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Downloading to a local data folder (relative to this script or repo root)
        data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../data'))
        os.makedirs(data_root, exist_ok=True)
        
        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                                download=True, transform=transform_train)
        
        # Implement Dataset Splitting
        if split_ratio < 1.0:
            train_size = int(len(trainset) * split_ratio)
            # Use fixed seed for reproducibility of splits
            g_cpu = torch.Generator()
            g_cpu.manual_seed(42) 
            indices = torch.randperm(len(trainset), generator=g_cpu)[:train_size]
            trainset = torch.utils.data.Subset(trainset, indices)
            print(f"Training on subset: {len(trainset)} samples")
        else:
            print(f"Training on full dataset: {len(trainset)} samples")
        
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size,
                                                  shuffle=True, num_workers=0) # num_workers=0 for safety on Windows

        testset = torchvision.datasets.CIFAR10(root=data_root, train=False,
                                               download=True, transform=transform_test)
        
        testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size,
                                                 shuffle=False, num_workers=0)

        # Model Initialization
        print(f"Initializing model...")
        dendrite_count = config.dendrite_count if use_dendrites else 0
        model = DendriticVisionModel(num_classes=10, dendrite_count=dendrite_count)
        
        # PerforatedAI Initialization (Conditional)
        name_str = f"{config.mode}_split-{int(config.dataset_split*100)}"
        
        if use_dendrites:
            # Construct dynamic run name
            name_parts = [f"dendrites-{config.dendrite_count}"]
            for k, v in config.items():
                if k not in ['dendrite_count', 'epochs', 'dataset_split', 'mode'] and not k.startswith('_'):
                     name_parts.append(f"{k}-{v}")
            name_str = f"dendritic_{name_str}" # Prefix

            # Custom Dendrite Tracking
            GPA.pc.append_module_ids_to_track(['.features', '.avgpool', '.classifier_top'])
            
            GPA.pc.set_unwrapped_modules_confirmed(True)
            GPA.pc.set_using_safe_tensors(False)
            GPA.pc.set_testing_dendrite_capacity(False)
            
            # Initialize PAI
            model = UPA.initialize_pai(model, save_name=name_str)
        
        wandb.run.name = name_str
        model = model.to(device)

        # Trigger PAI initialization or just warmup
        if use_dendrites:
            print("Running dummy forward/backward pass to initialize PAI shapes...")
            model.train()
            dummy_input = torch.randn(2, 3, 224, 224).to(device)
            dummy_label = torch.randint(0, 10, (2,)).to(device)
            
            temp_optim = optim.SGD(model.parameters(), lr=0.001)
            temp_optim.zero_grad()
            dummy_out = model(dummy_input)
            temp_loss = nn.CrossEntropyLoss()(dummy_out, dummy_label)
            temp_loss.backward()
            temp_optim.step()
        
        # Verify Parameter Count
        params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {params}")
        wandb.log({"total_parameters": params})

        # Optimizer and Loss
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        if use_dendrites:
            GPA.pai_tracker.set_optimizer_instance(optimizer)

        # Training Loop
        for epoch in range(config.epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{config.epochs}")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({"Loss": running_loss/max(total, 1), "Acc": 100.*correct/max(total, 1)})

            train_acc = 100. * correct / total
            train_loss = running_loss / len(trainloader)
            
            if use_dendrites:
                GPA.pai_tracker.add_extra_score(train_acc, "train_acc")

            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_acc = 100. * val_correct / val_total
            val_loss = val_loss / len(testloader)
            
            # PAI Updates and Restructuring
            if use_dendrites:
                GPA.pai_tracker.set_optimizer_instance(optimizer)
                print(f"DEBUG: Calling add_validation_score with val_acc={val_acc}")
                model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_acc, model)
                print(f"DEBUG: add_validation_score returned restructured={restructured}, training_complete={training_complete}")
                model = model.to(device)
                
                if restructured:
                    print("Model restructured (dendrites added). Re-initializing optimizer...")
                    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
                    GPA.pai_tracker.set_optimizer_instance(optimizer)
                
                if training_complete:
                    print("Training complete as determined by PAI Tracker.")
                    break
            
            print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })
    
    return val_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dendritic or Standard MobileNet Training")
    parser.add_argument('--mode', type=str, default='dendritic', choices=['dendritic', 'standard'], help="Training mode")
    parser.add_argument('--split', type=float, default=1.0, help="Dataset split ratio (e.g. 0.5 for 50%)")
    parser.add_argument('--epochs', type=int, default=15, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning Rate")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch Size")
    parser.add_argument('--dendrites', type=int, default=8, help="Dendrite Count")
    
    args = parser.parse_args()

    # Optimal Configuration
    base_config = {
        'dendrite_count': args.dendrites,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'epochs': args.epochs 
    }
    
    # Configure run based on args
    use_dendrites = (args.mode == 'dendritic')
    
    # Set threshold for dendrite addition (only affects dendritic mode)
    if use_dendrites:
         GPA.pc.set_n_epochs_to_switch(3)
         GPA.pc.set_max_dendrites(args.dendrites)

    print(f"Starting Run: Mode={args.mode}, Split={args.split}, LR={args.lr}, Dendrites={args.dendrites}")
    try:
        final_acc = run_training(config=base_config, split_ratio=args.split, use_dendrites=use_dendrites)
        print(f"FINAL_RESULT: {final_acc}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nCRITICAL ERROR: {e}")
        print("FINAL_RESULT: FAILED")


