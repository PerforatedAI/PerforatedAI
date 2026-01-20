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
    print("WARNING: PerforatedAI library STILL not found.")

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
    target_col = 'deposit' if 'deposit' in df.columns else 'y'
    df_num = df.select_dtypes(include=['number']).fillna(0)
    X = df_num.drop(columns=[target_col], errors='ignore').values
    
    try:
        y_raw = df[target_col]
        y = y_raw.apply(lambda x: 1 if str(x).lower() in ['yes', '1'] else 0).values
    except:
        y = df_num[target_col].values 

    if fit_scaler:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)
        
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long), scaler

X_train, y_train, scaler = load_and_process("train.csv", fit_scaler=True)
X_test, y_test, _ = load_and_process("test.csv", fit_scaler=False, scaler=scaler)

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
    print(">>> MODE: DENDRITIC OPTIMIZATION (STABILIZED SEARCH)")
    
    GPA.pc.set_testing_dendrite_capacity(False) 
    GPA.pc.set_n_epochs_to_switch(15) # Wait longer to ensure stability [cite: 96]
    
    model = UPA.initialize_pai(model)
    
    GPA.pai_tracker.set_optimizer(torch.optim.Adam)
    GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
    
    optimArgs = {'params': model.parameters(), 'lr': config.learning_rate}
    schedArgs = {'mode': 'max', 'patience': 5}
    
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
    
    # 5.1 INFINITE TRAINING LOOP [cite: 78]
    training_complete = False
    epoch = -1
    while not training_complete: # [cite: 78]
        epoch += 1
        model.train()
        train_correct, train_total = 0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = nn.CrossEntropyLoss()(output, y_batch)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(output.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
            
        train_acc = train_correct / train_total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        
        val_acc = val_correct / val_total
        wandb.log({"val_accuracy": val_acc, "train_accuracy": train_acc})
        print(f"Epoch {epoch} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        # The system will automatically add the blue line when plateauing [cite: 41, 44]
        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_acc, model)
        
        model.to(device)
        if restructured:
            print(">>> NETWORK RESTRUCTURED: ADDING DENDRITES")
            optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
            
        if training_complete: # [cite: 80]
            break

else:
    print(">>> MODE: STANDARD BASELINE")
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(15):
        model.train()
        train_correct, train_total = 0, 0 # Initialize counters
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            
            # Calculate training accuracy
            _, predicted = torch.max(output.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
        
        train_acc = train_correct / train_total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        
        val_acc = val_correct / val_total
        
        # CRITICAL: Use identical keys to the dendritic block
        wandb.log({"val_accuracy": val_acc, "train_accuracy": train_acc}) 
        
        print(f"Baseline Epoch {epoch} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

print("Run Complete.")