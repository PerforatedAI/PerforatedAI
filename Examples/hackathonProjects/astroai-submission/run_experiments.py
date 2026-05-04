"""
AstroAI Experiment Runner
Runs baseline and PAI experiments and generates comparison results.
"""

import subprocess
import sys
import os


def check_dependencies():
    """Check if required packages are installed."""
    required = ['torch', 'numpy', 'matplotlib', 'astropy']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Installing missing packages...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
    
    # Check for perforatedai
    try:
        import perforatedai
        print("✓ Perforated AI is installed")
    except ImportError:
        print("⚠ Perforated AI not installed. Installing...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'perforatedai'])


def run_baseline_experiment():
    """Run baseline experiment without PAI."""
    print("\n" + "="*60)
    print("Running Baseline Experiment")
    print("="*60)
    
    subprocess.run([
        sys.executable, 'train.py',
        '--baseline_only',
        '--epochs', '30',
        '--samples', '3000',
        '--model', 'mlp'
    ])


def run_pai_experiment():
    """Run experiment with Perforated AI."""
    print("\n" + "="*60)
    print("Running Perforated AI Experiment")
    print("="*60)
    
    subprocess.run([
        sys.executable, 'train.py',
        '--pai_only',
        '--epochs', '50',
        '--samples', '3000',
        '--model', 'mlp'
    ])


def run_full_comparison():
    """Run full comparison experiment."""
    print("\n" + "="*60)
    print("Running Full Comparison Experiment")
    print("="*60)
    
    subprocess.run([
        sys.executable, 'train.py',
        '--epochs', '30',
        '--samples', '3000',
        '--model', 'mlp'
    ])


def main():
    print("="*60)
    print("AstroAI Experiment Runner")
    print("Perforated AI Dendritic Optimization Hackathon")
    print("="*60)
    
    # Check dependencies
    check_dependencies()
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    print("\nSelect experiment to run:")
    print("1. Baseline only")
    print("2. Perforated AI only")
    print("3. Full comparison (recommended)")
    print("4. Quick test (fewer samples)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        run_baseline_experiment()
    elif choice == '2':
        run_pai_experiment()
    elif choice == '3':
        run_full_comparison()
    elif choice == '4':
        print("\nRunning quick test...")
        subprocess.run([
            sys.executable, 'train.py',
            '--epochs', '10',
            '--samples', '500',
            '--model', 'mlp'
        ])
    else:
        print("Invalid choice. Running full comparison...")
        run_full_comparison()
    
    print("\n" + "="*60)
    print("Experiments Complete!")
    print("Check the 'results/' directory for outputs.")
    print("="*60)


if __name__ == '__main__':
    main()
