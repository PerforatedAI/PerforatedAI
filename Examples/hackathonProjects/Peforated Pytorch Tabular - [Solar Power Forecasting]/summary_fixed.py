"""
FIXED HACKATHON SUMMARY SCRIPT
Properly analyzes results from aligned comparison
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("="*70)
print("FINAL HACKATHON RESULTS SUMMARY")
print("="*70)

# Load results from aligned comparison
df = pd.read_csv('final_results.csv')

y_true = df['actual'].values
y_baseline = df['baseline_pred'].values
y_perforated = df['perforated_pred'].values

# Calculate metrics
baseline_mae = mean_absolute_error(y_true, y_baseline)
baseline_rmse = np.sqrt(mean_squared_error(y_true, y_baseline))
baseline_r2 = r2_score(y_true, y_baseline)
baseline_corr = np.corrcoef(y_true, y_baseline)[0, 1]

perf_mae = mean_absolute_error(y_true, y_perforated)
perf_rmse = np.sqrt(mean_squared_error(y_true, y_perforated))
perf_r2 = r2_score(y_true, y_perforated)
perf_corr = np.corrcoef(y_true, y_perforated)[0, 1]

print("\n📊 BASELINE MODEL PERFORMANCE")
print("-"*70)
print(f"  MAE:         {baseline_mae:.6f}")
print(f"  RMSE:        {baseline_rmse:.6f}")
print(f"  R² Score:    {baseline_r2:.6f}")
print(f"  Correlation: {baseline_corr:.6f}")

print("\n🌳 PERFORATED AI MODEL PERFORMANCE")
print("-"*70)
print(f"  MAE:         {perf_mae:.6f}")
print(f"  RMSE:        {perf_rmse:.6f}")
print(f"  R² Score:    {perf_r2:.6f}")
print(f"  Correlation: {perf_corr:.6f}")

print("\n📈 IMPROVEMENT")
print("-"*70)
mae_imp = ((baseline_mae - perf_mae) / baseline_mae) * 100
rmse_imp = ((baseline_rmse - perf_rmse) / baseline_rmse) * 100
r2_imp = perf_r2 - baseline_r2

print(f"  MAE:      {mae_imp:+.2f}%")
print(f"  RMSE:     {rmse_imp:+.2f}%")
print(f"  R² Score: {r2_imp:+.4f}")

print("\n" + "="*70)
print("HACKATHON SUBMISSION SUMMARY")
print("="*70)

# Determine submission readiness
if rmse_imp > 5 and perf_r2 > 0.5 and perf_corr > 0.7:
    print("\n✅ SUCCESS! READY FOR SUBMISSION")
    print(f"\n📌 Key Findings:")
    print(f"   • PerforatedAI improved solar power prediction accuracy")
    print(f"   • RMSE improvement: {rmse_imp:.1f}%")
    print(f"   • R² improvement: {r2_imp:.4f}")
    print(f"   • Strong correlation: {perf_corr:.3f}")
    print(f"   • Dataset: Solar generation from 2 plants (34 days)")
    print(f"   • Model: PyTorch Tabular CategoryEmbedding with dendrites")

    print(f"\n💡 Submission Highlights:")
    print(f"   1. Real-world Application: Solar energy forecasting")
    print(f"   2. Multi-plant Generalization: Trained on Plant 1 + 2")
    print(f"   3. Measurable Impact: {rmse_imp:.1f}% accuracy improvement")
    print(f"   4. Dendritic Learning: PAI's adaptive architecture demonstrated")

    submission_status = "READY"

elif rmse_imp > 0 and perf_r2 > 0:
    print("\n⚠️  PARTIAL SUCCESS - CONDITIONAL SUBMISSION")
    print(f"\n📌 Results:")
    print(f"   • RMSE improvement: {rmse_imp:.1f}%")
    print(f"   • R² Score: {perf_r2:.3f} (baseline: {baseline_r2:.3f})")
    print(f"   • Correlation: {perf_corr:.3f}")

    print(f"\n💡 Submission Strategy:")
    print(f"   1. Highlight the {rmse_imp:.1f}% improvement achieved")
    print(f"   2. Discuss PAI's dendritic restructuring process")
    print(f"   3. Note areas for hyperparameter optimization")
    print(f"   4. Present as proof-of-concept for dendritic learning")

    submission_status = "CONDITIONAL"

else:
    print("\n❌ RESULTS NOT SUBMISSION-READY")
    print(f"\n📌 Issues:")
    print(f"   • RMSE change: {rmse_imp:+.1f}%")
    print(f"   • R² Score: {perf_r2:.3f} (baseline: {baseline_r2:.3f})")

    print(f"\n💡 Recommendations:")
    print(f"   1. Review hyperparameter settings (learning rate, dendrite limits)")
    print(f"   2. Try different model architectures")
    print(f"   3. Check for data quality issues")
    print(f"   4. Consider longer training duration")

    submission_status = "NOT READY"

print("\n" + "="*70)
print(f"Test samples compared: {len(y_true):,}")
print(f"Better predictions:    {(df['perforated_error'] < df['baseline_error']).sum():,} ({(df['perforated_error'] < df['baseline_error']).sum()/len(df)*100:.1f}%)")
print(f"Worse predictions:     {(df['perforated_error'] > df['baseline_error']).sum():,} ({(df['perforated_error'] > df['baseline_error']).sum()/len(df)*100:.1f}%)")

# Sample-level analysis
better_samples = df['perforated_error'] < df['baseline_error']
avg_improvement = (df.loc[better_samples, 'baseline_error'].mean() - 
                  df.loc[better_samples, 'perforated_error'].mean())

print(f"\nAverage improvement on better predictions: {avg_improvement:.6f}")

print("\n" + "="*70)
print(f"SUBMISSION STATUS: {submission_status}")
print("="*70)
