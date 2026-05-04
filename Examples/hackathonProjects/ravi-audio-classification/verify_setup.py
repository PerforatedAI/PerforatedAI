"""
Quick verification script to check if the environment is set up correctly
and to verify reproducibility settings.
"""
import sys
import importlib.util

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        return False, "Not installed"
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown version')
        return True, version
    except ImportError as e:
        return False, str(e)

def main():
    print("="*60)
    print("PerforatedAI ESC-50 Environment Verification")
    print("="*60)
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    if sys.version_info < (3, 8):
        print("⚠️  WARNING: Python 3.8+ is recommended")
    else:
        print("✓ Python version OK")
    
    # Check required packages
    print("\nChecking required packages:")
    packages = [
        ('torch', 'torch'),
        ('torchaudio', 'torchaudio'),
        ('librosa', 'librosa'),
        ('soundfile', 'soundfile'),
        ('numpy', 'numpy'),
        ('sklearn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('tqdm', 'tqdm'),
    ]
    
    all_installed = True
    for package, import_name in packages:
        installed, version = check_package(package, import_name)
        if installed:
            print(f"✓ {package:15s} - {version}")
        else:
            print(f"✗ {package:15s} - {version}")
            all_installed = False
    
    # Check PerforatedAI
    print("\nChecking PerforatedAI:")
    pai_installed, pai_version = check_package('perforatedai', 'perforatedai')
    if pai_installed:
        print(f"✓ perforatedai installed")
        try:
            from perforatedai import globals_perforatedai as GPA
            from perforatedai import utils_perforatedai as UPA
            print("✓ PerforatedAI imports working")
        except ImportError as e:
            print(f"✗ PerforatedAI import failed: {e}")
            all_installed = False
    else:
        print(f"✗ perforatedai not installed")
        print("  Run: pip install -e . from the repository root")
        all_installed = False
    
    # Check for data directory
    print("\nChecking data setup:")
    import os
    data_dir = 'data/ESC-50'
    if os.path.exists(data_dir):
        print(f"✓ ESC-50 data directory found: {data_dir}")
        csv_path = os.path.join(data_dir, 'meta/esc50.csv')
        if os.path.exists(csv_path):
            print(f"✓ ESC-50 metadata found")
        else:
            print(f"⚠️  ESC-50 metadata not found at {csv_path}")
    else:
        print(f"✗ ESC-50 data directory not found: {data_dir}")
        print("  Follow the dataset download instructions in README.md")
    
    # Check preprocessed data
    preprocessed_dir = 'preprocessed'
    if os.path.exists(preprocessed_dir):
        print(f"✓ Preprocessed data directory found")
    else:
        print(f"⚠️  Preprocessed data not found. Run: python preprocess.py")
    
    # Check device availability
    print("\nChecking compute devices:")
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print(f"  Apple Silicon GPU detected")
    
    # Verify reproducibility settings
    print("\nVerifying reproducibility setup:")
    try:
        import config
        seed = config.PREPROCESSING.get('random_state', None)
        if seed is not None:
            print(f"✓ Random seed configured: {seed}")
        else:
            print("✗ Random seed not configured in config.py")
        
        test_fold = config.PREPROCESSING.get('test_fold', None)
        if test_fold == 5:
            print(f"✓ Test fold set to 5 (ESC-50 standard)")
        else:
            print(f"⚠️  Test fold is {test_fold}, standard is 5")
    except Exception as e:
        print(f"✗ Error loading config: {e}")
    
    # Final summary
    print("\n" + "="*60)
    if all_installed:
        print("✓ Environment verification complete - ready to train!")
    else:
        print("⚠️  Some packages are missing. Install them with:")
        print("   pip install -r requirements.txt")
        print("   pip install -e . (from repository root)")
    print("="*60)

if __name__ == '__main__':
    main()
