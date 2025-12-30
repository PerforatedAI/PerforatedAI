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

def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        config = wandb.config

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

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
        # Using absolute path to avoid confusion with CWD
        data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../data'))
        os.makedirs(data_root, exist_ok=True)
        
        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                                download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size,
                                                  shuffle=True, num_workers=0) # num_workers=0 for safety on Windows

        testset = torchvision.datasets.CIFAR10(root=data_root, train=False,
                                               download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size,
                                                 shuffle=False, num_workers=0)

        # Model Initialization
        print(f"Initializing model with dendrite_count={config.dendrite_count}...")
        model = DendriticVisionModel(num_classes=10, dendrite_count=config.dendrite_count)
        
        # Initialize PerforatedAI Tracker
        # This converts eligible layers (like nn.Linear) to PAINeuronModules
        GPA.pc.set_unwrapped_modules_confirmed(True)
        model = UPA.initialize_pai(model)
        model = model.to(device)

        # Trigger PAI initialization (shape inference) with a dummy forward/backward pass
        print("Running dummy forward/backward pass to initialize PAI shapes...")
        model.train() # Ensure train mode
        dummy_input = torch.randn(2, 3, 224, 224).to(device) # Batch size 2
        dummy_label = torch.randint(0, 10, (2,)).to(device)
        
        # Temp execution to trigger hooks
        temp_optim = optim.SGD(model.parameters(), lr=0.001)
        temp_optim.zero_grad()
        dummy_out = model(dummy_input)
        temp_loss = nn.CrossEntropyLoss()(dummy_out, dummy_label)
        temp_loss.backward()
        temp_optim.step()
        
        # Manually add dendrites to the specific output layer if it was converted
        # We iterate to find the 'dendritic_output' which we know is the last layer or by name
        # Since initialize_pai might wrap it, we access it via the model structure
        
        if hasattr(model, 'dendritic_output') and isinstance(model.dendritic_output, PA.PAINeuronModule):
             print(f"Adding {config.dendrite_count} dendrites to dendritic_output layer...")
             for _ in range(config.dendrite_count):
                 model.dendritic_output.create_new_dendrite_module()
        else:
             print("Warning: dendritic_output was not converted to PAINeuronModule or not found.")


        # Verify Parameter Count
        params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {params}")
        wandb.log({"total_parameters": params})

        # Optimizer and Loss
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

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
                
                pbar.set_postfix({"Loss": running_loss/total, "Acc": 100.*correct/total})

            train_acc = 100. * correct / total
            train_loss = running_loss / len(trainloader)
            
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

            print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })

if __name__ == "__main__":
    sweep_configuration = {
        'method': 'bayes',
        'name': 'dendritic-mobilenet-sweep',
        'metric': {'goal': 'maximize', 'name': 'val_accuracy'},
        'parameters': {
            'dendrite_count': {'values': [2, 4, 8]},
            'batch_size': {'values': [32, 64]},
            'learning_rate': {'distribution': 'uniform', 'max': 0.005, 'min': 0.0001},
            'epochs': {'value': 2} # Reduced epochs for faster demo
        }
    }

    # IMPORTANT: Ensure user is logged in to wandb.
    # If not, this might hang or fail.
    # We rely on user having run 'wandb login' as per instructions.
    
    try:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="hackathon-dendritic-vision")
        print(f"Starting sweep agent with ID: {sweep_id}")
        wandb.agent(sweep_id, function=train, count=3) # Limit runs for demo
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        print("This error is likely due to missing WandB authentication.")
        print("Please run 'wandb login' in your terminal and paste your API key.")
        print("If you do not have an account, sign up at https://wandb.ai/site")

