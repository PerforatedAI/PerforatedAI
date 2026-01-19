import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import wandb
import sys
import os

# --- AUTO-FIND PERFORATED AI LIBRARY ---
# If pip install failed:
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'PerforatedAI')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

try:
    try:
        from PerforatedAI import globals_perforatedai as GPA
        from PerforatedAI import utils_perforatedai as UPA
    except ImportError:
        from perforatedai import globals_perforatedai as GPA
        from perforatedai import utils_perforatedai as UPA
    HAS_PAI = True
    print("SUCCESS: PerforatedAI library found!")
except ImportError:
    HAS_PAI = False
    print("WARNING: PerforatedAI library STILL not found. Check folder name.")

# --- 1. SETUP ARGUMENTS ---
parser = argparse.ArgumentParser()
parser.add_argument('--use_dendritic', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--learning_rate', type=float, default=0.001)
args = parser.parse_args()

# --- 2. INIT W&B ---
wandb.init(project="bank-leads-optimization", config=args)
config = wandb.config

# --- 3. DATA LOADING ---
def load_and_process(filename, fit_scaler=False, scaler=None):
    if not os.path.exists(filename):
        print(f"ERROR: {filename} not found!")
        sys.exit(1)
        
    df = pd.read_csv(filename)
    
    # Check for 'y' OR 'deposit'
    if 'deposit' in df.columns:
        target_col = 'deposit'
    elif 'y' in df.columns:
        target_col = 'y'
    else:
        print(f"CRITICAL ERROR: Target column not found. Columns: {df.columns.tolist()}")
        sys.exit(1)

    # Process Data
    df = df.select_dtypes(include=['number']).fillna(0)
    X = df.drop(columns=[target_col], errors='ignore').values
    
    # Convert 'yes'/'no' to 1/0 safely
    # If the CSV was already converted to numbers, this handles it gracefully
    try:
        y_raw = pd.read_csv(filename)[target_col] # Read raw again to get text
        y = y_raw.apply(lambda x: 1 if str(x).lower() in ['yes', '1'] else 0).values
    except:
        y = df[target_col].values # Fallback

    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
        
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return X_tensor, y_tensor, scaler

print("Loading Data...")
X_train, y_train, scaler = load_and_process("train.csv", fit_scaler=True)
X_test, y_test, _ = load_and_process("test.csv", fit_scaler=False, scaler=scaler)
print(f"Data Loaded. Train Size: {len(X_train)}")

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=config.batch_size)

# --- 4. DEFINE MODEL ---
class BankModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 2)
        )
    def forward(self, x):
        return self.layers(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BankModel(X_train.shape[1]).to(device)

# --- 5. EXECUTION LOGIC ---
if config.use_dendritic == 1:
    if not HAS_PAI:
        raise ImportError("Cannot run Dendritic Mode: Library not found!")

    print(">>> MODE: DENDRITIC OPTIMIZATION")
    
    # [README 2.1] Network Conversion
    model = UPA.initialize_pai(model)
    
    # [README 3] Setup Optimizer
    GPA.pai_tracker.set_optimizer(torch.optim.Adam)
    GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
    
    optimArgs = {'params': model.parameters(), 'lr': config.learning_rate}
    schedArgs = {'mode': 'max', 'patience': 5}
    
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
    
    # Set to False for real training
    GPA.pc.set_testing_dendrite_capacity(False) 

    training_complete = False
    
    while not training_complete:
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = nn.CrossEntropyLoss()(output, y_batch)
            loss.backward()
            optimizer.step()
            
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        val_acc = correct / total
        wandb.log({"accuracy": val_acc})
        print(f"Validation Acc: {val_acc}")

        # [README 5.1] Critical Logic
        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_acc, model)
        model.to(device)
        
        if restructured:
            print(">>> NETWORK RESTRUCTURED: ADDING DENDRITES")
            optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)

else:
    print(">>> MODE: STANDARD BASELINE")
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(15):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        val_acc = correct / total
        wandb.log({"accuracy": val_acc})
        print(f"Epoch {epoch} Acc: {val_acc}")

print("Run Complete.")