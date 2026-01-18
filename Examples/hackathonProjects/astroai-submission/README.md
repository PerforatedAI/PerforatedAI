# PAI Training Results

This directory contains the Perforated AI training results for the AstroAI project.

## Files

- **PAI.png** - Training visualization showing validation scores, training time, and learning rate over epochs
- **PAIbest_test_scores.csv** - Summary of results comparing baseline vs dendritic models

## Results Summary

The graph shows:
- **Blue dashed line**: Point where dendrites were added (epoch 25)
- **Top-left plot**: Validation accuracy improving from ~79% to ~96% after dendrite addition
- **Top-right plot**: Training time per epoch (slightly increased with dendrites)
- **Bottom-left plot**: Learning rate schedule with adaptive adjustments
- **Bottom-right plot**: Placeholder for dendrite correlation scores (requires full PAI license)

## Key Findings

Adding dendritic optimization to the exoplanet transit detection model resulted in:
- **17.4% absolute improvement** in validation accuracy
- **81.5% remaining error reduction**
- Modest increase in training time (~20%)
- Additional 50,000 parameters for dendrite connections

This demonstrates that Perforated AI's dendritic optimization can significantly improve performance on time-series astronomical signal classification tasks.
