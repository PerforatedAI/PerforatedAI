"""
SOLAR PREDICTION - PAI WITHOUT RESTRUCTURING
Uses dendrites but prevents architecture changes during training
"""

import pandas as pd
import numpy as np
import os
import torch
import warnings
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
os.environ["WANDB_MODE"] = "disabled"
os.environ["PYTHONBREAKPOINT"] = "0"
sys.breakpointhook = lambda *args, **kwargs: None

print("="*70)
print("SOLAR POWER - PAI WITH STATIC DENDRITES (NO RESTRUCTURING)")
print("="*70)

# Load data
def get_processed_data():
    p1_gen = pd.read_csv('data/Plant_1_Generation_Data.csv')
    p1_wea = pd.read_csv('data/Plant_1_Weather_Sensor_Data.csv')
    p1_gen['DATE_TIME'] = pd.to_datetime(p1_gen['DATE_TIME'], dayfirst=True)
    p1_wea['DATE_TIME'] = pd.to_datetime(p1_wea['DATE_TIME'])
    df1 = pd.merge(p1_gen, p1_wea.drop(columns=['PLANT_ID', 'SOURCE_KEY']), 
                   on='DATE_TIME', how='inner')
    
    p2_gen = pd.read_csv('data/Plant_2_Generation_Data.csv')
    p2_wea = pd.read_csv('data/Plant_2_Weather_Sensor_Data.csv')
    p2_gen['DATE_TIME'] = pd.to_datetime(p2_gen['DATE_TIME'])
    p2_wea['DATE_TIME'] = pd.to_datetime(p2_wea['DATE_TIME'])
    df2 = pd.merge(p2_gen, p2_wea.drop(columns=['PLANT_ID', 'SOURCE_KEY']), 
                   on='DATE_TIME', how='inner')
    
    df = pd.concat([df1, df2], ignore_index=True)
    df['hour'] = df['DATE_TIME'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['DC_POWER'] = df['DC_POWER'] / 1000.0
    
    df = df.dropna()
    df = df[df['IRRADIATION'] > 0].reset_index(drop=True)
    return df

print("\n📁 Loading data...")
df = get_processed_data()
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index).reset_index(drop=True)

print(f"✓ Training: {len(train_df)}, Test: {len(test_df)}")

# Train baseline
print("\n🔧 Training baseline...")
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig

baseline_model = TabularModel(
    data_config=DataConfig(
        target=['DC_POWER'], 
        continuous_cols=['IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'hour_sin', 'hour_cos']
    ),
    model_config=CategoryEmbeddingModelConfig(task="regression", layers="128-64", learning_rate=1e-3),
    optimizer_config=OptimizerConfig(),
    trainer_config=TrainerConfig(batch_size=1024, max_epochs=20, accelerator="auto"),
    experiment_config=ExperimentConfig(project_name="solar", run_name="baseline")
)

baseline_model.fit(train=train_df, validation=test_df)

os.makedirs("models", exist_ok=True)
baseline_model.save_model("models/baseline_model")
torch.save(baseline_model.model.state_dict(), "models/base_weights.pt")

baseline_preds = baseline_model.predict(test_df)['DC_POWER_prediction'].values
y_true = test_df['DC_POWER'].values

baseline_rmse = np.sqrt(mean_squared_error(y_true, baseline_preds))
baseline_r2 = r2_score(y_true, baseline_preds)
print(f"✓ Baseline: RMSE={baseline_rmse:.4f}, R²={baseline_r2:.4f}")

# Configure PAI - DISABLE RESTRUCTURING
print("\n🌳 Training with static dendrites (no restructuring)...")

os.environ["PAIEMAIL"] = "EMAIL"
os.environ["PAITOKEN"] = "TOKEN"

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

# Key change: Keep dendrites static
GPA.pc.set_unwrapped_modules_confirmed(True)
GPA.pc.set_testing_dendrite_capacity(False)
GPA.pc.set_perforated_backpropagation(True)
GPA.pc.set_dendrite_graph_mode(False)
GPA.pc.set_initial_correlation_batches(5)
GPA.pc.set_max_dendrites(8)  # Fewer dendrites

# Force static mode - no evolution
GPA.pc.set_mode('p')  # Start in perforated mode, skip normal mode

tabular_model = TabularModel.load_model("models/baseline_model")

class MockLogger:
    def watch(self, *args, **kwargs): pass
    def log_hyperparams(self, *args, **kwargs): pass
    def log_metrics(self, *args, **kwargs): pass

tabular_model.logger = MockLogger()

datamodule = tabular_model.prepare_dataloader(train=train_df)
lightning_module = tabular_model.prepare_model(datamodule)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lightning_module.to(device)

# Inject PAI
target = lightning_module.model if hasattr(lightning_module, 'model') else lightning_module.backbone
model = UPA.initialize_pai(target, save_name='PAI_SOLAR_STATIC')

if hasattr(lightning_module, 'model'):
    lightning_module.model = model
else:
    lightning_module.backbone = model

# Load baseline weights
baseline_weights = torch.load("models/base_weights.pt", map_location=device)
current_state = lightning_module.state_dict()
filtered_weights = {k: v for k, v in baseline_weights.items() 
                   if k in current_state and current_state[k].shape == v.shape}
lightning_module.load_state_dict(filtered_weights, strict=False)

# Simple training - just gradient descent, no evolution
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lightning_module.parameters(), lr=1e-4)
train_loader = datamodule.train_dataloader()

print("Training (static dendrites)...")
best_loss = float('inf')
patience_counter = 0

for epoch in range(1, 40):
    lightning_module.train()
    total_loss = 0
    
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        optimizer.zero_grad()
        output = lightning_module(batch)
        y_hat = output["logits"] if isinstance(output, dict) else output
        loss = criterion(y_hat, batch["target"].view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(lightning_module.state_dict(), "models/pai_static_best.pt")
        print(f"  Epoch {epoch:2d} | Loss: {avg_loss:.6f} ✓")
        patience_counter = 0
    else:
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:2d} | Loss: {avg_loss:.6f}")
        patience_counter += 1
    
    if patience_counter >= 10:
        print(f"  Stopping at epoch {epoch}")
        break

print(f"\n✓ Static dendrite training complete! Best loss: {best_loss:.6f}")

# Test
print("\n📊 Testing static dendrite model...")
lightning_module.load_state_dict(torch.load("models/pai_static_best.pt", map_location=device))
lightning_module.eval()

test_datamodule = tabular_model.prepare_dataloader(train=test_df, validation=None)
test_loader = test_datamodule.train_dataloader()

pai_preds = []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        output = lightning_module(batch)
        y_hat = output["logits"] if isinstance(output, dict) else output
        pai_preds.extend(y_hat.cpu().numpy().flatten())

y_pai = np.array(pai_preds)

# Align
min_len = min(len(y_true), len(y_pai))
y_true_aligned = y_true[:min_len]
y_baseline_aligned = baseline_preds[:min_len]
y_pai_aligned = y_pai[:min_len]

pai_rmse = np.sqrt(mean_squared_error(y_true_aligned, y_pai_aligned))
pai_r2 = r2_score(y_true_aligned, y_pai_aligned)
baseline_rmse_aligned = np.sqrt(mean_squared_error(y_true_aligned, y_baseline_aligned))
baseline_r2_aligned = r2_score(y_true_aligned, y_baseline_aligned)

# Results
print("\n" + "="*70)
print("RESULTS: STATIC DENDRITES (NO RESTRUCTURING)")
print("="*70)
print(f"Baseline RMSE: {baseline_rmse_aligned:.6f}")
print(f"PAI RMSE:      {pai_rmse:.6f}")
print(f"Improvement:   {((baseline_rmse_aligned-pai_rmse)/baseline_rmse_aligned*100):+.2f}%")
print()
print(f"Baseline R²: {baseline_r2_aligned:.6f}")
print(f"PAI R²:      {pai_r2:.6f}")
print(f"Difference:  {(pai_r2-baseline_r2_aligned):+.6f}")
print("="*70)

if pai_rmse < baseline_rmse_aligned and pai_r2 > 0.5:
    print("\n✅ Static dendrites improved performance!")
    print("This suggests dendritic computation helps, but evolution doesn't")
elif abs(pai_rmse - baseline_rmse_aligned) / baseline_rmse_aligned < 0.05:
    print("\n⚖️  Static dendrites roughly equivalent to baseline")
    print("Dendrites added complexity without clear benefit")
else:
    print("\n❌ Static dendrites still worse than baseline")
    print("This task genuinely doesn't benefit from dendritic architecture")

print("\n💡 CONCLUSION:")
if pai_rmse >= baseline_rmse_aligned:
    print("PerforatedAI is NOT suitable for this solar forecasting task.")
    print("The baseline CategoryEmbedding architecture is already optimal.")
    print("\nFor hackathon: Submit as negative result showing when NOT to use PAI")
else:

    print("Static dendrites show promise! Consider tuning hyperparameters.")
