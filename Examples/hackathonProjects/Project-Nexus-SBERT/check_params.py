"""
Quick Parameter Count Check
"""
import torch
import sys
sys.path.insert(0, 'C:/Users/aakan/Downloads/PerforatedAI')

from sentence_transformers import SentenceTransformer, models
import torch.nn as nn
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

print("\n" + "="*70)
print("  PARAMETER COUNT COMPARISON")
print("="*70)

# Load baseline
print("\n🔵 Loading Baseline...")
try:
    baseline = SentenceTransformer("experiments/baseline/final_model")
    baseline_params = sum(p.numel() for p in baseline.parameters())
    baseline_trainable = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    print(f"   Total params: {baseline_params:,}")
    print(f"   Trainable:    {baseline_trainable:,}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    baseline_params = 0

# Try to load dendritic with PAI
print("\n🔴 Loading Dendritic...")
try:
    # Load checkpoint that has PAI structure
    checkpoint_path = "PAI/latest.pt"
    if not torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    else:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Count parameters in state dict
    dendritic_params = sum(p.numel() for p in checkpoint.values() if isinstance(p, torch.Tensor))
    print(f"   Total params: {dendritic_params:,}")
    
    # Check for dendrite layers
    dendrite_keys = [k for k in checkpoint.keys() if 'dendrite' in k]
    print(f"   Dendrite layers: {len(dendrite_keys)}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    dendritic_params = 0

print("\n" + "="*70)
print("  VERDICT")
print("="*70)

if baseline_params > 0 and dendritic_params > 0:
    diff = dendritic_params - baseline_params
    ratio = dendritic_params / baseline_params
    
    print(f"\nBaseline:  {baseline_params:,} parameters")
    print(f"Dendritic: {dendritic_params:,} parameters")
    print(f"Difference: {diff:,} parameters ({ratio:.2f}x)")
    
    if diff > 0:
        print(f"\n❌ DENDRITIC HAS {diff:,} MORE PARAMETERS")
        print(f"❌ This is NOT a compression/efficiency win")
        print(f"❌ You ADDED parameters, not removed them")
    elif diff < 0:
        print(f"\n✅ DENDRITIC HAS {abs(diff):,} FEWER PARAMETERS")
        print(f"✅ This IS a compression win!")
    else:
        print(f"\n→ SAME PARAMETER COUNT")

print("\n" + "="*70)
