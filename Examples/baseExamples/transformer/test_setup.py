"""
Setup verification script for the Transformer comparison example.
Tests that all required dependencies are installed and functioning correctly.
"""

import sys

def test_pytorch():
    """Test PyTorch installation and device availability."""
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} installed")
        
        # Check device availability
        if torch.backends.mps.is_available():
            print("✓ Apple Metal (MPS) available")
            device = torch.device('mps')
        elif torch.cuda.is_available():
            print("✓ CUDA available")
            device = torch.device('cuda')
        else:
            print("✓ Using CPU (no GPU acceleration)")
            device = torch.device('cpu')
        
        # Test tensor operation on device
        x = torch.randn(10, 10).to(device)
        y = torch.randn(10, 10).to(device)
        z = x @ y
        print(f"✓ Tensor operations working on {device}")
        
        return True
    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        return False


def test_datasets():
    """Test datasets library (Hugging Face)."""
    try:
        from datasets import load_dataset
        print("✓ Hugging Face datasets library installed")
        return True
    except Exception as e:
        print(f"✗ datasets library test failed: {e}")
        return False


def test_other_dependencies():
    """Test other required libraries."""
    missing = []
    
    try:
        import wandb
        print("✓ Weights & Biases (wandb) installed")
    except ImportError:
        missing.append("wandb")
    
    try:
        import numpy
        print("✓ NumPy installed")
    except ImportError:
        missing.append("numpy")
    
    try:
        import tqdm
        print("✓ tqdm installed")
    except ImportError:
        missing.append("tqdm")
    
    if missing:
        print(f"✗ Missing libraries: {', '.join(missing)}")
        return False
    
    return True


def test_perforatedai():
    """Test PerforatedAI library."""
    try:
        from perforatedai import globals_perforatedai as GPA
        from perforatedai import utils_perforatedai as UPA
        print("✓ PerforatedAI library installed and importable")
        return True
    except ImportError as e:
        print(f"✗ PerforatedAI library not found: {e}")
        print("  Note: PerforatedAI is required for dendritic models.")
        print("  Install with: cd PerforatedAI_lib && pip install -e .")
        return False


def test_data_loading():
    """Test that data preparation module works."""
    try:
        from data_preparation import Vocabulary
        vocab = Vocabulary(max_vocab_size=100)
        vocab.build_vocab(["hello world", "test setup"])
        print("✓ Data preparation module working")
        return True
    except Exception as e:
        print(f"✗ Data preparation test failed: {e}")
        return False


def test_model_creation():
    """Test that model creation works."""
    try:
        from model import TransformerLM
        import torch
        
        model = TransformerLM(
            vocab_size=100,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            dropout=0.1
        )
        
        # Test forward pass
        dummy_input = torch.randint(0, 100, (2, 10))
        output = model(dummy_input)
        
        assert output.shape == (2, 10, 100), "Output shape mismatch"
        print("✓ Model creation and forward pass working")
        return True
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        return False


def main():
    print("="*70)
    print("  SETUP VERIFICATION")
    print("="*70)
    print()
    
    tests = [
        ("PyTorch", test_pytorch),
        ("Datasets Library", test_datasets),
        ("Other Dependencies", test_other_dependencies),
        ("PerforatedAI", test_perforatedai),
        ("Data Preparation", test_data_loading),
        ("Model Creation", test_model_creation),
    ]
    
    results = {}
    for name, test_func in tests:
        print(f"\nTesting {name}...")
        print("-" * 70)
        results[name] = test_func()
        print()
    
    # Summary
    print("="*70)
    print("  SUMMARY")
    print("="*70)
    print()
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} {name}")
    
    print()
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! Environment is ready.")
        print("\nYou can now run experiments with:")
        print("  ./run_experiment.sh quick      # Quick test (3 epochs)")
        print("  ./run_experiment.sh full       # Full run (10 epochs)")
        print("  ./run_experiment.sh final      # Final config (30 epochs)")
        return 0
    else:
        print("\n✗ Some tests failed. Please install missing dependencies.")
        print("\nTo install missing packages:")
        print("  pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())

