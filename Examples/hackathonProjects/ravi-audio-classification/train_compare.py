"""
Train both baseline and PerforatedAI models, then compare results.
Ensures both models are trained with identical parameters for fair comparison.
"""
import os
import sys
import subprocess
import json
import argparse

import config
from compare_results import load_results, print_summary, plot_comparison


def run_training(script_name, description):
    """Run a training script and check for errors."""
    print("\n" + "="*70)
    print(f"{description}")
    print("="*70 + "\n")
    
    result = subprocess.run(
        [sys.executable, script_name],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    if result.returncode != 0:
        print(f"\n❌ Error running {script_name}")
        return False
    
    print(f"\n✅ {description} completed successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Train baseline and PAI models, then compare results'
    )
    parser.add_argument('--baseline-only', action='store_true',
                        help='Only train baseline model')
    parser.add_argument('--pai-only', action='store_true',
                        help='Only train PAI model')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training and just compare existing results')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("ESC-50 Audio Classification: Baseline vs PerforatedAI")
    print("="*70)
    
    # Print current configuration
    print("\nCurrent Configuration:")
    print(f"  Model Type:      {config.MODEL['type']}")
    print(f"  Batch Size:      {config.TRAINING['batch_size']}")
    print(f"  Learning Rate:   {config.TRAINING['learning_rate']}")
    print(f"  Weight Decay:    {config.TRAINING['weight_decay']}")
    print(f"  Max Epochs:      {config.TRAINING['max_epochs']}")
    print(f"  Patience:        {config.TRAINING['patience']}")
    print(f"  Max Dendrites:   {config.PAI['max_dendrites']}")
    
    # Training phase
    if not args.skip_training:
        # Train baseline (unless PAI-only)
        if not args.pai_only:
            if not run_training('train_baseline.py', 'PHASE 1: Training Baseline Model'):
                print("\n⚠️  Baseline training failed. Exiting.")
                return
        
        # Train PAI (unless baseline-only)
        if not args.baseline_only:
            if not run_training('train_perforatedai.py', 'PHASE 2: Training with PerforatedAI'):
                print("\n⚠️  PAI training failed. Exiting.")
                return
    else:
        print("\n⏭️  Skipping training, using existing results...")
    
    # Comparison phase
    print("\n" + "="*70)
    print("PHASE 3: Comparing Results")
    print("="*70 + "\n")
    
    results = load_results()
    
    if not results:
        print("❌ No results found. Please train models first.")
        return
    
    if 'baseline' not in results:
        print("⚠️  Baseline results not found. Run without --pai-only first.")
        return
    
    if 'pai' not in results and not args.baseline_only:
        print("⚠️  PAI results not found. Run without --baseline-only first.")
        return
    
    # Print summary
    print_summary(results)
    
    # Generate comparison plots
    if 'baseline' in results and 'pai' in results:
        print("\nGenerating comparison visualizations...")
        save_path = os.path.join(config.MODELS_DIR, 'comparison.png')
        plot_comparison(results, save_path)
        
        # Also save for hackathon submission
        submission_path = 'Accuracy Improvement.png'
        plot_comparison(results, submission_path)
        print(f"Submission plot saved to: {submission_path}")
        
        # Print final verdict
        baseline_acc = results['baseline']['test_accuracy']
        pai_acc = results['pai']['test_accuracy']
        improvement = pai_acc - baseline_acc
        
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        
        if improvement > 0:
            print(f"✅ SUCCESS! PerforatedAI improved accuracy by {improvement:.2f}%")
            print(f"   Baseline: {baseline_acc:.2f}% → PAI: {pai_acc:.2f}%")
        elif improvement == 0:
            print(f"⚖️  No change. Both models achieved {baseline_acc:.2f}% accuracy.")
        else:
            print(f"⚠️  Baseline performed better by {abs(improvement):.2f}%")
            print(f"   Baseline: {baseline_acc:.2f}% → PAI: {pai_acc:.2f}%")
            print("   (This can happen if dendrites weren't added or hyperparameters need tuning)")
        
        # Check if dendrites were actually added
        if 'dendrites_added' in results['pai']:
            dendrites = results['pai']['dendrites_added']
            print(f"\n   Dendrites added: {dendrites}")
            if dendrites == 0:
                print("   ⚠️  No dendrites were added! Consider:")
                print("      - Increasing max_epochs")
                print("      - Lowering improvement_threshold")
                print("      - Adjusting scheduler patience")
    
    print("\n" + "="*70)
    print("All tasks completed!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
