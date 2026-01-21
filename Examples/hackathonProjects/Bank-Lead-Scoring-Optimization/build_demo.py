import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os

# --- CONFIG ---
NUMERIC_FEATURES = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
JOB_MAPPING = {
    'retired': 2, 'management': 1, 'entrepreneur': 1, 'student': 0, 
    'blue-collar': 0, 'unemployed': -1, 'unknown': 0
}

# --- IMPROVED PREPROCESSING (WITH SCALING) ---
def preprocess(df):
    # 1. Map Jobs
    df['job_code'] = df['job'].map(JOB_MAPPING).fillna(0)
    
    # 2. CRITICAL FIX: Scale the huge numbers so the AI doesn't break
    # We create copies to avoid SettingWithCopy warnings
    df = df.copy()
    df['balance'] = df['balance'] / 1000.0  # Shrink 25000 -> 25.0
    df['age'] = df['age'] / 100.0           # Shrink 65 -> 0.65
    
    X = df[NUMERIC_FEATURES + ['job_code']].fillna(0).values
    return torch.tensor(X, dtype=torch.float32)

def get_data(filename):
    df = pd.read_csv(filename)
    target_col = 'y' if 'y' in df.columns else 'deposit'
    if target_col not in df.columns:
        print(f"Error: {target_col} column missing.")
        exit()
        
    y = df[target_col].apply(lambda x: 1 if str(x).lower() in ['yes', '1'] else 0).values
    X_tensor = preprocess(df)
    return X_tensor, torch.tensor(y, dtype=torch.float32) # Float for BCE Loss

# --- MAIN SETUP ---
print(">>> Loading Data & Scaling Features...")
if not os.path.exists("train.csv"):
    print("Error: train.csv not found!")
    exit()

X_train, y_train = get_data("train.csv")
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=1024, shuffle=True)
input_dim = X_train.shape[1]

# --- DEFINITIONS ---
class StandardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 512), nn.LeakyReLU(),
            nn.Linear(512, 256), nn.LeakyReLU(),
            nn.Linear(256, 1), nn.Sigmoid() # Sigmoid for probability
        )
    def forward(self, x): return self.layers(x)

class OptimizedModel(nn.Module):
    def __init__(self):
        super().__init__()
        # The Winning 271k Architecture
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LeakyReLU(), 
            nn.Linear(512, 512), nn.LeakyReLU(),     
            nn.Linear(512, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.layers(x)

# --- SMART TRAINING ---
def train_brain(model, filename, epochs=30):
    print(f">>> Training {filename}...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # FIX: Use BCELoss for binary classification
    criterion = nn.BCELoss() 
    
    model.train()
    for epoch in range(epochs):
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_b).squeeze()
            
            # FIX: Class Weighting (Manual)
            # Penalize missing a "Yes" (1) much more than missing a "No" (0)
            weight = y_b * 5.0 + 1.0 # 6x penalty for missing a Yes
            loss = (criterion(y_pred, y_b) * weight).mean()
            
            loss.backward()
            optimizer.step()
            
    torch.save(model.state_dict(), filename)
    print(f">>> SAVED: {filename}")

# Build them (More epochs for better convergence)
train_brain(StandardModel(), "standard_model.pth", epochs=20)
train_brain(OptimizedModel(), "optimized_model.pth", epochs=40) # Optimized gets more time to perfect
print("\nDONE! Brains re-engineered with scaling.")