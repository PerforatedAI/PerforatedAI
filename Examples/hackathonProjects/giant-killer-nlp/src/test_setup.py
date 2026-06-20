"""
Quick test script to verify the setup works correctly.
Tests data loading, tokenization, and model creation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("GIANT-KILLER NLP - SETUP VERIFICATION")
print("=" * 60)

# Test 1: Import all modules
print("\n1. Testing imports...")
try:
    from data import get_tokenizer, create_sample_dataset, ToxicityDataset
    from models import create_bert_tiny_model
    print("   ✓ All modules imported successfully")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Load tokenizer
print("\n2. Testing tokenizer...")
try:
    tokenizer = get_tokenizer("prajjwal1/bert-tiny")
    print(f"   ✓ Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"   ✓ Vocab size: {tokenizer.vocab_size}")
except Exception as e:
    print(f"   ✗ Tokenizer failed: {e}")
    sys.exit(1)

# Test 3: Create sample dataset
print("\n3. Testing dataset creation...")
try:
    texts, labels = create_sample_dataset(num_samples=10)
    print(f"   ✓ Created {len(texts)} samples")
    print(f"   ✓ Sample text: '{texts[0][:50]}...'")
    print(f"   ✓ Label distribution: {sum(labels)} toxic, {len(labels) - sum(labels)} non-toxic")
except Exception as e:
    print(f"   ✗ Dataset creation failed: {e}")
    sys.exit(1)

# Test 4: Create dataset object
print("\n4. Testing ToxicityDataset...")
try:
    dataset = ToxicityDataset(texts, labels, tokenizer, max_length=128)
    sample = dataset[0]
    print(f"   ✓ Dataset size: {len(dataset)}")
    print(f"   ✓ Sample keys: {list(sample.keys())}")
    print(f"   ✓ Input shape: {sample['input_ids'].shape}")
    print(f"   ✓ Label: {sample['labels'].item()}")
except Exception as e:
    print(f"   ✗ Dataset object failed: {e}")
    sys.exit(1)

# Test 5: Create model
print("\n5. Testing model creation...")
try:
    import torch
    model = create_bert_tiny_model(num_labels=2)
    print(f"   ✓ Model created: {model.__class__.__name__}")
    print(f"   ✓ Parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    dummy_input = sample['input_ids'].unsqueeze(0)
    dummy_mask = sample['attention_mask'].unsqueeze(0)
    dummy_labels = sample['labels'].unsqueeze(0)
    
    output = model(dummy_input, dummy_mask, dummy_labels)
    print(f"   ✓ Forward pass successful")
    print(f"   ✓ Output shape: {output['logits'].shape}")
    print(f"   ✓ Loss: {output['loss'].item():.4f}")
except Exception as e:
    print(f"   ✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Check PerforatedAI availability
print("\n6. Checking PerforatedAI...")
try:
    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA
    print("   ✓ PerforatedAI is installed and available")
    print("   ✓ Dendritic optimization will be enabled")
except ImportError:
    print("   ⚠ PerforatedAI not installed")
    print("   ⚠ Training will run in baseline mode (without dendrites)")
    print("   ⚠ Install with: pip install perforatedai")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED - SETUP IS READY!")
print("=" * 60)
print("\nNext steps:")
print("1. Run training: python src/train.py --sample-size 1000 --epochs 3")
print("2. Evaluate model: python src/evaluate.py")
