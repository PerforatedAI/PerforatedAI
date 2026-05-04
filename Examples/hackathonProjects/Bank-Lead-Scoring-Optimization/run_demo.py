import torch
import torch.nn as nn
import pandas as pd
import time
import os

# --- CONFIG ---
MODEL_FILE = "standard_model.pth" 
# MODEL_FILE = "standard_model.pth" 
DATA_FILE = "todays_call_list.csv"

# Matches build_demo.py
NUMERIC_FEATURES = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']
JOB_MAPPING = {
    'retired': 2, 'management': 1, 'entrepreneur': 1, 'student': 0, 
    'blue-collar': 0, 'unemployed': -1, 'unknown': 0
}

# --- ARCHITECTURES (Updated with Sigmoid) ---
class StandardModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 512), nn.LeakyReLU(),
            nn.Linear(512, 256), nn.LeakyReLU(),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.layers(x)

class OptimizedModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LeakyReLU(), 
            nn.Linear(512, 512), nn.LeakyReLU(),     
            nn.Linear(512, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.layers(x)

# --- THE APP ---
print(f">>> 📱 BANK MANAGER APP LAUNCHING...")
print(f">>> Loading Brain: {MODEL_FILE}")

if not os.path.exists(DATA_FILE):
    print("Error: todays_call_list.csv not found.")
    exit()
    
df = pd.read_csv(DATA_FILE)

# --- MATCHING PREPROCESSING ---
df['job_code'] = df['job'].map(JOB_MAPPING).fillna(0)
# SCALING (Must match Training!)
df_scaled = df.copy()
df_scaled['balance'] = df_scaled['balance'] / 1000.0
df_scaled['age'] = df_scaled['age'] / 100.0

X_raw = df_scaled[NUMERIC_FEATURES + ['job_code']].fillna(0).values
X_tensor = torch.tensor(X_raw, dtype=torch.float32)
input_dim = X_tensor.shape[1]

# LOAD MODEL
if "optimized" in MODEL_FILE:
    model = OptimizedModel(input_dim)
else:
    model = StandardModel(input_dim)

try:
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()
except Exception as e:
    print(f"ERROR: Model mismatch. Re-run build_demo.py! {e}")
    exit()

# PREDICT
print(">>> Running Prediction on today's call list...")
start_time = time.time()
with torch.no_grad():
    probs = model(X_tensor).squeeze()
end_time = time.time()

# RESULTS
df['Win_Probability'] = probs.numpy()
top_leads = df.sort_values(by='Win_Probability', ascending=False)

print(f"\nProcessing Complete in {end_time - start_time:.4f} seconds.")
print("="*55)
print(f" {'AGE':<4} {'JOB':<12} {'BALANCE':<10} {'SCORE'}")
print("="*55)
for _, row in top_leads.iterrows():
    print(f" {row['age']:<4} {row['job']:<12} {row['balance']:<10} {row['Win_Probability']:.4f}")
print("="*55)