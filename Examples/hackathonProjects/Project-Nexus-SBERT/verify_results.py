"""
Verification Script: Test Real Performance Gains
"""
import time
import torch
from sentence_transformers import SentenceTransformer

print("\n" + "="*70)
print("  VERIFICATION: ARE THE RESULTS REAL?")
print("="*70)

# Test sentences
test_sentences = ["This is a test sentence."] * 1000

print("\n📊 TEST 2: MODEL SIZE COMPARISON")
print("-" * 70)

try:
    baseline_model = SentenceTransformer("experiments/baseline/final_model")
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    print(f"✅ Baseline loaded: {baseline_params:,} parameters")
except Exception as e:
    print(f"❌ Baseline not found: {e}")
    baseline_model = None
    baseline_params = 0

try:
    dendritic_model = SentenceTransformer("experiments/dendritic/final_model")
    dendritic_params = sum(p.numel() for p in dendritic_model.parameters())
    print(f"✅ Dendritic loaded: {dendritic_params:,} parameters")
except Exception as e:
    print(f"❌ Dendritic not found: {e}")
    dendritic_model = None
    dendritic_params = 0

if baseline_params > 0 and dendritic_params > 0:
    param_diff = dendritic_params - baseline_params
    param_ratio = dendritic_params / baseline_params
    print(f"\n📈 Parameter Change:")
    print(f"   Difference: {param_diff:,} parameters")
    print(f"   Ratio: {param_ratio:.4f}x")
    if param_diff > 0:
        print(f"   ⚠️  DENDRITIC HAS MORE PARAMETERS (+{param_diff:,})")
    elif param_diff < 0:
        print(f"   ✅ DENDRITIC HAS FEWER PARAMETERS ({param_diff:,})")
    else:
        print(f"   → SAME NUMBER OF PARAMETERS")

print("\n" + "="*70)
print("📊 TEST 3: INFERENCE SPEED COMPARISON")
print("-" * 70)

if baseline_model is not None:
    print(f"\n🔵 Testing Baseline (1000 sentences)...")
    # Warmup
    _ = baseline_model.encode(test_sentences[:10])
    
    start = time.time()
    _ = baseline_model.encode(test_sentences)
    baseline_time = time.time() - start
    print(f"   ⏱️  Baseline: {baseline_time:.2f}s")
else:
    baseline_time = None

if dendritic_model is not None:
    print(f"\n🔴 Testing Dendritic (1000 sentences)...")
    # Warmup
    _ = dendritic_model.encode(test_sentences[:10])
    
    start = time.time()
    _ = dendritic_model.encode(test_sentences)
    dendritic_time = time.time() - start
    print(f"   ⏱️  Dendritic: {dendritic_time:.2f}s")
else:
    dendritic_time = None

print("\n" + "="*70)
print("🎯 FINAL VERDICT")
print("="*70)

if baseline_time and dendritic_time:
    speedup = baseline_time / dendritic_time
    time_saved = baseline_time - dendritic_time
    percent_faster = ((baseline_time - dendritic_time) / baseline_time) * 100
    
    print(f"\n⚡ Inference Speed:")
    print(f"   Baseline:   {baseline_time:.2f}s")
    print(f"   Dendritic:  {dendritic_time:.2f}s")
    print(f"   Speedup:    {speedup:.2f}x")
    print(f"   Time Saved: {time_saved:.2f}s ({percent_faster:.1f}% faster)")
    
    if speedup > 1.2:
        print(f"\n   ✅ REAL WIN: Dendritic is {speedup:.2f}x faster!")
        print(f"   ✅ You have a legitimate efficiency gain to report.")
    elif speedup > 0.95 and speedup < 1.05:
        print(f"\n   ⚠️  NEUTRAL: Models perform about the same.")
        print(f"   ⚠️  Training time drop was likely warmup, not optimization.")
    else:
        print(f"\n   ❌ PROBLEM: Dendritic is SLOWER than baseline!")
        print(f"   ❌ The graph's time drop was misleading.")
else:
    print("\n❌ Could not run comparison - check if models exist")

print("\n" + "="*70)
