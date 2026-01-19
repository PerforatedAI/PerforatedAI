import torch
import torch.nn as nn
import pandas as pd
import time
import os

# --- CONFIG ---
# Toggle this to test different models
# MODEL_FILE = "standard_model.pth" 
MODEL_FILE = "optimized_model.pth" 
DATA_FILE = "todays_call_list.csv"

# --- RE-DEFINE ARCHITECTURES (Must match above) ---
class StandardModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 512), nn.LeakyReLU(),
            nn.Linear(512, 256), nn.LeakyReLU(),
            nn.Linear(256, 2)
        )
    def forward(self, x): return self.layers(x)

class OptimizedModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LeakyReLU(),
            nn.Linear(256, 64), nn.LeakyReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x): return self.layers(x)

# --- THE APP ---
print(f">>> 📱 BANK MANAGER APP LAUNCHING...")
print(f">>> Loading Brain: {MODEL_FILE}")

# 1. Load Data
if not os.path.exists(DATA_FILE):
    print("Data file not found. Running setup...")
    import setup_data # Fallback
    
df = pd.read_csv(DATA_FILE)
# Simple preprocessing to match training
X_raw = df.select_dtypes(include=['number']).fillna(0).values
X_tensor = torch.tensor(X_raw, dtype=torch.float32)

# 2. Load Model
input_dim = X_tensor.shape[1]
if "optimized" in MODEL_FILE:
    model = OptimizedModel(input_dim)
else:
    model = StandardModel(input_dim)

try:
    model.load_state_dict(torch.load(MODEL_FILE))
except:
    print("ERROR: Model architecture doesn't match file. Check your config.")
    exit()

model.eval()

# 3. Predict & Benchmark
print(">>> Running Prediction on today's call list...")
start_time = time.time()
with torch.no_grad():
    outputs = model(X_tensor)
    probs = torch.softmax(outputs, dim=1)[:, 1] # Probability of 'Yes'
end_time = time.time()

# 4. Display Results
df['Win_Probability'] = probs.numpy()
top_leads = df.sort_values(by='Win_Probability', ascending=False).head(3)

print(f"\nProcessing Complete in {end_time - start_time:.4f} seconds.")
print("="*40)
print("       TOP LEADS TO CALL TODAY")
print("="*40)
# Show the most relevant columns for a bank agent
cols = [c for c in ['age', 'job', 'balance', 'Win_Probability'] if c in df.columns]
print(top_leads[cols].to_string(index=False))
print("="*40)