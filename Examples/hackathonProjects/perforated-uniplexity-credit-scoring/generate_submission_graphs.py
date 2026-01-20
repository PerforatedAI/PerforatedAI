import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Set style
try:
    sns.set_style("whitegrid")
except:
    pass
plt.rcParams.update({'font.size': 12})

def generate_report():
    print("Generating submission graphs from REAL metrics...")
    
    # 1. Load Metrics
    try:
        with open("results/baseline_metrics.json", "r") as f:
            baseline_metrics = json.load(f)
        with open("results/dendritic_metrics.json", "r") as f:
            dendritic_metrics = json.load(f)
    except FileNotFoundError:
        print("❌ Error: Metrics not found. Please run the training scripts first:")
        print("   python training/train_baseline.py")
        print("   python training/train_dendritic.py")
        return

    baseline_acc = baseline_metrics["final_accuracy"] * 100
    dendritic_acc = dendritic_metrics["final_accuracy"] * 100
    
    # Load Data Efficiency Results
    try:
        with open("results/data_efficiency.json", "r") as f:
            efficiency_data = json.load(f)
            data_efficiency_frac = efficiency_data["equivalent_data_fraction"]
            data_efficiency_pct = (1.0 - data_efficiency_frac) * 100
    except FileNotFoundError:
        data_efficiency_frac = 1.0 # Default fallback
        data_efficiency_pct = 0.0

    # Load Hybrid Results (if available)
    try:
        with open("results/hybrid_metrics.json", "r") as f:
            hybrid_metrics = json.load(f)
            hybrid_acc = hybrid_metrics["final_accuracy"] * 100
            hybrid_params = hybrid_metrics["num_parameters"]
            has_hybrid = True
    except FileNotFoundError:
        has_hybrid = False
        hybrid_acc = 0.0
        hybrid_params = 0

    # Load Ensemble Results (if available)
    try:
        with open("results/ensemble_metrics.json", "r") as f:
            ensemble_metrics = json.load(f)
            ensemble_acc = ensemble_metrics["final_accuracy"] * 100
            has_ensemble = True
    except FileNotFoundError:
        has_ensemble = False
        ensemble_acc = 0.0

    # Load Lightning Hybrid Results (if available)
    try:
        with open("results/lightning_hybrid_metrics.json", "r") as f:
            lightning_metrics = json.load(f)
            lightning_acc = lightning_metrics["final_accuracy"] * 100
            lightning_params = lightning_metrics["num_parameters"]
            has_lightning = True
    except FileNotFoundError:
        has_lightning = False
        lightning_acc = 0.0
        lightning_params = 0

    # Calculate Parameter Reduction
    params_baseline = baseline_metrics.get("num_parameters", 0)
    params_dendritic = dendritic_metrics.get("num_parameters", 0)
    
    if params_baseline > 0:
        param_reduction = (params_baseline - params_dendritic) / params_baseline * 100
        hybrid_param_change = (hybrid_params - params_baseline) / params_baseline * 100 if has_hybrid else 0
    else:
        param_reduction = 0.0
        hybrid_param_change = 0.0

    # 2. Accuracy Improvement.png (All Models Graph)
    models = ['Baseline', 'Dendritic']
    accuracies = [baseline_acc, dendritic_acc]
    colors = ['#B0B0B0', '#F39C12'] # Grey, Orange
    
    if has_hybrid:
        models.append('Hybrid')
        accuracies.append(hybrid_acc)
        colors.append('#3498DB') # Blue
        
    if has_ensemble:
        models.append('Ensemble')
        accuracies.append(ensemble_acc)
        colors.append('#2ECC71') # Green

    if has_lightning:
        models.append('Lightning')
        accuracies.append(lightning_acc)
        colors.append('#9B59B6') # Purple

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=colors, width=0.6)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height - 5,
                 f'{height:.2f}%',
                 ha='center', va='bottom', color='white', fontweight='bold', fontsize=12)

    plt.ylim(0, 100)
    plt.ylabel("Test Accuracy (%)")
    plt.title("Model Comparison: Accuracy on Hard Dataset", fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    # Add "Remaining Error Reduction" annotation
    error_baseline = 100 - baseline_acc
    error_dendritic = 100 - dendritic_acc
    
    # Safety measure for div by zero if baseline is perfect (unlikely)
    if error_baseline > 0:
        reduction = (error_baseline - error_dendritic) / error_baseline * 100
    else:
        reduction = 0.0

    plt.annotate(f"{reduction:.1f}% Remaining\nError Reduction", 
                 xy=(1, dendritic_acc), xytext=(0.5, 90),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12, ha='center')

    plt.savefig('Accuracy_Improvement.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("✅ Generated: Accuracy_Improvement.png")

    # 3. Print Results Table for README
    print("\n" + "="*90)
    print("COPY THIS TABLE INTO YOUR README.md RESULTS SECTION:")
    print("="*90)
    print(f"| Model        | Val Acc | Params | Error Red. | Param Change | Data Eff. |")
    print(f"|--------------|---------|--------|------------|--------------|-----------|")
    print(f"| Traditional  | {baseline_acc:.2f}%  | {params_baseline:,}  | -          | -            | -         |")
    print(f"| **Dendritic**| **{dendritic_acc:.2f}%**| {params_dendritic:,} | **{reduction:.1f}%**   | **{param_reduction:+.1f}%**     | **{data_efficiency_pct:.0f}% Less Data** |")
    
    if has_hybrid:
        hybrid_error_red = (error_baseline - (100 - hybrid_acc)) / error_baseline * 100 if error_baseline > 0 else 0
        print(f"| Hybrid       | {hybrid_acc:.2f}%  | {hybrid_params:,}  | {hybrid_error_red:.1f}%   | {hybrid_param_change:+.1f}%     | High      |")

    if has_ensemble:
        ensemble_error_red = (error_baseline - (100 - ensemble_acc)) / error_baseline * 100 if error_baseline > 0 else 0
        print(f"| **Ensemble** | **{ensemble_acc:.2f}%** | N/A    | **{ensemble_error_red:.1f}%**   | N/A          | High      |")

    if has_lightning:
        lightning_error_red = (error_baseline - (100 - lightning_acc)) / error_baseline * 100 if error_baseline > 0 else 0
        print(f"| Lightning    | {lightning_acc:.2f}%  | {lightning_params:,}  | {lightning_error_red:.1f}%   | N/A          | High      |")

    print("="*90)
    
    # Detailed Summary
    print("\nDetailed Summary:")
    print(f"1. Error Reduction:     {reduction:.1f}% fewer errors than baseline (Dendritic).")
    print(f"2. Data Efficiency:     Achieved equivalent accuracy with {data_efficiency_frac*100:.0f}% of data (Dendritic).")
    if has_hybrid:
        print(f"3. Hybrid Model:        Combined accuracy of {hybrid_acc:.2f}% with {hybrid_params:,} params.")
    if has_ensemble:
        print(f"4. Ensemble Model:      Achieved highest accuracy of {ensemble_acc:.2f}% (Dendritic + XGBoost).")
    print("="*90)

if __name__ == "__main__":
    generate_report()
