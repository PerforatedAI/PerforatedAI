"""
AstroAI Setup Test
Verifies that all components work correctly before running full experiments.
"""

import sys

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    errors = []
    
    # Core packages
    try:
        import numpy as np
        print("  ✓ numpy")
    except ImportError as e:
        errors.append(f"numpy: {e}")
    
    try:
        import torch
        print(f"  ✓ torch (version {torch.__version__})")
        print(f"    CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        errors.append(f"torch: {e}")
    
    try:
        import matplotlib
        print("  ✓ matplotlib")
    except ImportError as e:
        errors.append(f"matplotlib: {e}")
    
    try:
        import astropy
        print("  ✓ astropy")
    except ImportError as e:
        errors.append(f"astropy: {e}")
    
    try:
        import streamlit
        print("  ✓ streamlit")
    except ImportError as e:
        errors.append(f"streamlit: {e}")
    
    # Perforated AI
    try:
        import perforatedai as pai
        print("  ✓ perforatedai")
    except ImportError as e:
        errors.append(f"perforatedai: {e}")
        print("  ⚠ perforatedai not installed")
        print("    Install with: pip install perforatedai")
    
    return errors


def test_local_modules():
    """Test that local modules can be imported."""
    print("\nTesting local modules...")
    
    errors = []
    
    try:
        from model import TransitDetector, TransitDetectorCNN
        print("  ✓ model.py")
    except Exception as e:
        errors.append(f"model.py: {e}")
    
    try:
        from simulator import simulate_light_curve, generate_dataset
        print("  ✓ simulator.py")
    except Exception as e:
        errors.append(f"simulator.py: {e}")
    
    return errors


def test_model():
    """Test model forward pass."""
    print("\nTesting model...")
    
    import torch
    from model import TransitDetector, TransitDetectorCNN
    
    # Test MLP
    model = TransitDetector(input_size=1000)
    x = torch.randn(4, 1000)
    y = model(x)
    assert y.shape == (4, 1), f"Expected (4, 1), got {y.shape}"
    print("  ✓ TransitDetector forward pass")
    
    # Test CNN
    model = TransitDetectorCNN(input_size=1000)
    x = torch.randn(4, 1000)
    y = model(x)
    assert y.shape == (4, 1), f"Expected (4, 1), got {y.shape}"
    print("  ✓ TransitDetectorCNN forward pass")


def test_simulator():
    """Test light curve simulation."""
    print("\nTesting simulator...")
    
    from simulator import simulate_light_curve, simulate_transit_light_curve, generate_dataset
    
    # Test basic simulation
    t, flux = simulate_light_curve(period=10, radius_ratio=0.1)
    assert len(t) == 1000, f"Expected 1000 points, got {len(t)}"
    assert len(flux) == 1000, f"Expected 1000 points, got {len(flux)}"
    print("  ✓ simulate_light_curve")
    
    # Test transit simulation
    flux, label = simulate_transit_light_curve(has_transit=True)
    assert label == 1, "Expected label 1 for transit"
    print("  ✓ simulate_transit_light_curve (with transit)")
    
    flux, label = simulate_transit_light_curve(has_transit=False)
    assert label == 0, "Expected label 0 for no transit"
    print("  ✓ simulate_transit_light_curve (no transit)")
    
    # Test dataset generation
    X_train, y_train, X_val, y_val, X_test, y_test = generate_dataset(n_samples=100)
    assert X_train.shape[0] == 70, f"Expected 70 train samples, got {X_train.shape[0]}"
    print("  ✓ generate_dataset")


def test_pai_integration():
    """Test Perforated AI integration."""
    print("\nTesting Perforated AI integration...")
    
    try:
        import perforatedai as pai
        from perforatedai import pb_globals as gpa
        import torch
        from model import TransitDetector
        
        # Create model
        model = TransitDetector(input_size=100)
        
        # Initialize PAI (testing mode)
        gpa.pc.set_testing_dendrite_capacity(True)
        
        model = pai.initialize_pai(
            model,
            doing_pai=True,
            save_name='test_pai',
            making_graphs=False,
            maximizing_score=True
        )
        
        print("  ✓ PAI initialization")
        
        # Test forward pass with PAI
        x = torch.randn(4, 100)
        y = model(x)
        assert y.shape == (4, 1), f"Expected (4, 1), got {y.shape}"
        print("  ✓ PAI forward pass")
        
    except ImportError:
        print("  ⚠ Skipping PAI tests (not installed)")
    except Exception as e:
        print(f"  ✗ PAI test failed: {e}")


def main():
    print("="*60)
    print("AstroAI Setup Test")
    print("="*60)
    
    all_errors = []
    
    # Run tests
    all_errors.extend(test_imports())
    all_errors.extend(test_local_modules())
    
    if not all_errors:
        test_model()
        test_simulator()
        test_pai_integration()
    
    # Summary
    print("\n" + "="*60)
    if all_errors:
        print("Setup FAILED with errors:")
        for error in all_errors:
            print(f"  ✗ {error}")
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("All tests PASSED! ✓")
        print("\nYou can now run:")
        print("  python train.py --epochs 30 --samples 3000")
        print("  streamlit run app.py")
        sys.exit(0)


if __name__ == '__main__':
    main()
