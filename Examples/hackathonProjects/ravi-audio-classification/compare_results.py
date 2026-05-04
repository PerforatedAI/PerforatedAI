"""
Compare baseline and PerforatedAI results.
Generates comparison visualizations and copies PAI graphs for hackathon submission.
"""
import os
import json
import shutil
import matplotlib.pyplot as plt
import numpy as np

import config


def load_results():
    """Load baseline and PAI results from JSON files."""
    results = {}
    
    baseline_path = os.path.join(config.MODELS_DIR, 'baseline_results.json')
    pai_path = os.path.join(config.MODELS_DIR, 'pai_results.json')
    
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            results['baseline'] = json.load(f)
    
    if os.path.exists(pai_path):
        with open(pai_path, 'r') as f:
            results['pai'] = json.load(f)
    
    return results


def calculate_error_reduction(baseline_acc, improved_acc):
    """Calculate percentage error reduction."""
    baseline_error = 100 - baseline_acc
    improved_error = 100 - improved_acc
    if baseline_error == 0:
        return 0
    return ((baseline_error - improved_error) / baseline_error) * 100


def plot_comparison(results, save_path='comparison.png'):
    """Create comparison visualization."""
    if 'baseline' not in results or 'pai' not in results:
        print("Both baseline and PAI results needed for comparison.")
        return None
    
    baseline = results['baseline']
    pai = results['pai']
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('ESC-50 Audio Classification: Baseline vs PerforatedAI', 
                 fontsize=14, fontweight='bold')
    
    # Color scheme
    colors = {'baseline': '#2196F3', 'pai': '#4CAF50'}
    
    # Plot 1: Accuracy Comparison
    ax1 = axes[0]
    x = np.arange(2)
    accuracies = [baseline['test_accuracy'], pai['test_accuracy']]
    bars = ax1.bar(['Baseline', 'PerforatedAI'], accuracies, 
                   color=[colors['baseline'], colors['pai']])
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Test Accuracy')
    ax1.set_ylim(0, 100)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add improvement annotation
    improvement = pai['test_accuracy'] - baseline['test_accuracy']
    if improvement > 0:
        ax1.annotate(f'+{improvement:.1f}%', 
                    xy=(1, pai['test_accuracy']),
                    xytext=(1.3, pai['test_accuracy'] - 5),
                    fontsize=12, color='green', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='green'))
    
    # Plot 2: Parameter Count
    ax2 = axes[1]
    params = [baseline['num_parameters'], pai['num_parameters']]
    bars2 = ax2.bar(['Baseline', 'PerforatedAI'], params,
                    color=[colors['baseline'], colors['pai']])
    ax2.set_ylabel('Parameters')
    ax2.set_title('Model Size')
    
    # Add value labels
    for bar, p in zip(bars2, params):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(params)*0.02,
                f'{p:,}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Error Reduction
    ax3 = axes[2]
    error_reduction = calculate_error_reduction(
        baseline['test_accuracy'], 
        pai['test_accuracy']
    )
    
    # Pie chart showing error reduction
    baseline_error = 100 - baseline['test_accuracy']
    pai_error = 100 - pai['test_accuracy']
    
    labels = ['Remaining\nError', 'Reduced\nError']
    sizes = [pai_error, baseline_error - pai_error]
    colors_pie = ['#FF5252', '#4CAF50']
    explode = (0, 0.1)
    
    if sizes[1] > 0:  # Only if there's improvement
        ax3.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                autopct='%1.1f%%', shadow=True, startangle=90)
        ax3.set_title(f'Error Reduction: {error_reduction:.1f}%')
    else:
        ax3.text(0.5, 0.5, 'No improvement\ndetected', 
                ha='center', va='center', fontsize=12)
        ax3.set_title('Error Reduction')
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {save_path}")
    
    return fig


def print_summary(results):
    """Print a summary of results."""
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    if 'baseline' in results:
        b = results['baseline']
        print(f"\nBaseline ({b['model']}):")
        print(f"  Test Accuracy:    {b['test_accuracy']:.2f}%")
        print(f"  Val Accuracy:     {b['best_val_accuracy']:.2f}%")
        print(f"  Parameters:       {b['num_parameters']:,}")
        print(f"  Epochs Trained:   {b['epochs_trained']}")
    
    if 'pai' in results:
        p = results['pai']
        print(f"\nPerforatedAI ({p['model']}):")
        print(f"  Test Accuracy:    {p['test_accuracy']:.2f}%")
        print(f"  Val Accuracy:     {p['best_val_accuracy']:.2f}%")
        print(f"  Parameters:       {p['num_parameters']:,}")
        print(f"  Epochs Trained:   {p['epochs_trained']}")
        print(f"  Dendrites Added:  {p.get('dendrites_added', 'N/A')}")
    
    if 'baseline' in results and 'pai' in results:
        b = results['baseline']
        p = results['pai']
        
        improvement = p['test_accuracy'] - b['test_accuracy']
        error_reduction = calculate_error_reduction(
            b['test_accuracy'], 
            p['test_accuracy']
        )
        param_increase = ((p['num_parameters'] - b['num_parameters']) / b['num_parameters']) * 100
        
        print("\n" + "-"*60)
        print("COMPARISON:")
        print(f"  Accuracy Improvement: {improvement:+.2f}%")
        print(f"  Error Reduction:      {error_reduction:.2f}%")
        print(f"  Parameter Increase:   {param_increase:+.2f}%")
        print("-"*60)


def copy_pai_graph():
    """Copy PAI training graph to project root for submission."""
    source = 'PAI_CNN14/PAI_CNN14.png'
    dest = 'PAI_CNN14.png'
    
    if os.path.exists(source):
        shutil.copy(source, dest)
        print(f"\n✓ Copied PAI graph: {source} -> {dest}")
        return True
    else:
        print(f"\n✗ Warning: PAI graph not found at {source}")
        print("  Make sure you've run train_perforatedai.py first")
        return False


def save_metrics_summary(results):
    """Save a text summary of metrics for easy reference."""
    if 'baseline' not in results or 'pai' not in results:
        return
    
    b = results['baseline']
    p = results['pai']
    
    improvement = p['test_accuracy'] - b['test_accuracy']
    error_reduction = calculate_error_reduction(b['test_accuracy'], p['test_accuracy'])
    param_increase = ((p['num_parameters'] - b['num_parameters']) / b['num_parameters']) * 100
    
    summary = f"""ESC-50 Audio Classification Results
{'='*60}

BASELINE CNN14:
  Test Accuracy:    {b['test_accuracy']:.2f}%
  Val Accuracy:     {b['best_val_accuracy']:.2f}%
  Parameters:       {b['num_parameters']:,}
  Epochs Trained:   {b['epochs_trained']}

PERFORATED AI CNN14:
  Test Accuracy:    {p['test_accuracy']:.2f}%
  Val Accuracy:     {p['best_val_accuracy']:.2f}%
  Parameters:       {p['num_parameters']:,}
  Epochs Trained:   {p['epochs_trained']}
  Dendrites Added:  {p.get('dendrites_added', 0)}

IMPROVEMENT:
  Accuracy Improvement: {improvement:+.2f}%
  Error Reduction:      {error_reduction:.2f}%
  Parameter Increase:   {param_increase:+.1f}%
"""
    
    with open('results_summary.txt', 'w') as f:
        f.write(summary)
    
    print("\n✓ Saved metrics summary to: results_summary.txt")


def main():
    print("\n" + "="*60)
    print("Generating Hackathon Submission Artifacts")
    print("="*60)
    
    # Ensure models directory exists
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # Copy PAI training graph (REQUIRED for submission)
    copy_pai_graph()
    
    # Load results
    results = load_results()
    
    if not results:
        print("\n✗ No results found. Please train models first:")
        print("  1. python train_baseline.py")
        print("  2. python train_perforatedai.py")
        return
    
    # Print summary
    print_summary(results)
    
    # Generate comparison plot if both results exist
    if 'baseline' in results and 'pai' in results:
        # Save to models folder
        save_path = os.path.join(config.MODELS_DIR, 'comparison.png')
        plot_comparison(results, save_path)
        
        # Save clean graph for hackathon submission (root directory)
        submission_path = 'Accuracy Improvement.png'
        plot_comparison(results, submission_path)
        print(f"\n✓ Generated clean comparison graph: {submission_path}")
        
        # Save metrics summary
        save_metrics_summary(results)
        
        print("\n" + "="*60)
        print("Submission artifacts ready!")
        print("="*60)
        print("\nFiles for hackathon submission:")
        print("  ✓ PAI_CNN14.png (required PAI training graph)")
        print("  ✓ Accuracy Improvement.png (clean results graph)")
        print("  ✓ results_summary.txt (metrics summary)")
        print("  ✓ models/baseline_results.json")
        print("  ✓ models/pai_results.json")
        print("  ✓ models/baseline_confusion_matrix.png")
        print("  ✓ models/pai_confusion_matrix.png")
    else:
        print("\n✗ Both baseline and PAI results needed for comparison")
        if 'baseline' not in results:
            print("  Missing: baseline_results.json (run train_baseline.py)")
        if 'pai' not in results:
            print("  Missing: pai_results.json (run train_perforatedai.py)")


if __name__ == '__main__':
    main()
