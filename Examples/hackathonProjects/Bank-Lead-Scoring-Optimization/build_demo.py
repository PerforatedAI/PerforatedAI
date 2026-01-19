import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os

# --- 1. DATA SETUP ---
print(">>> Loading Data...")
# Check if files exist
if not os.path.exists("train.csv"):
    print("Error: train.csv not found! Run setup_data.py first.")
    exit()

def get_data(filename):
    df = pd.read_csv(filename)
    # Simple cleaner to ensure we have numbers
    target_col = 'y' if 'y' in df.columns else 'deposit'
    # Handle Yes/No or 1/0
    y = df[target_col].apply(lambda x: 1 if str(x).lower() in ['yes', '1'] else 0).values
    X = df.drop(columns=[target_col], errors='ignore').select_dtypes(include=['number']).fillna(0).values
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

X_train, y_train = get_data("train.csv")
# Use the full dataset for the demo training
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=1024, shuffle=True)
input_dim = X_train.shape[1]

# --- 2. DEFINE THE BRAINS ---

# BRAIN A: The Standard (Baseline)
# Matches 1024-512-256 config (~700k params)
class StandardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 512), nn.LeakyReLU(),
            nn.Linear(512, 256), nn.LeakyReLU(),
            nn.Linear(256, 2)
        )
    def forward(self, x): return self.layers(x)

# BRAIN B: The Optimized (Dendritic Result)
# Matches Row 1 result (~135k params)
class OptimizedModel(nn.Module):
    def __init__(self):
        super().__init__()
        # This architecture creates approx 135k parameters
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LeakyReLU(),
            nn.Linear(256, 64), nn.LeakyReLU(), 
            nn.Linear(64, 2)
        )
    def forward(self, x): return self.layers(x)

# --- 3. TRAIN AND SAVE ---
def train_brain(model, filename):
    print(f">>> Training {filename}...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Train for 15 epochs (Enough for a functional demo)
    model.train()
    for epoch in range(15):
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            
    # Save the file
    torch.save(model.state_dict(), filename)
    print(f">>> SAVED: {filename}")

# Run the factory
train_brain(StandardModel(), "standard_model.pth")
train_brain(OptimizedModel(), "optimized_model.pth")
print("\nDONE! You now have two brain files ready for the demo.")