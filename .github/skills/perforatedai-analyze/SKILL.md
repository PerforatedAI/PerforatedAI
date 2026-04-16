---
name: perforatedai-analyze
description: "Analyze PerforatedAI training results and provide optimization recommendations. Trigger: 'Analyze my perforated results' (after training completes). Reviews CSV outputs, identifies performance patterns, recommends configuration improvements. For initial setup or debugging, use the perforatedai skill instead."
---

# PerforatedAI Results Analysis Skill

## Overview

This skill analyzes completed PerforatedAI training runs and provides optimization recommendations based on the training outputs.

**When to use this skill:**
- After your dendritic training has completed
- When you want to understand how dendrites impacted performance
- To get recommendations for improving future training runs

**For other PerforatedAI tasks:**
- **Setup/Integration**: Say "Perforate my model" (uses perforatedai skill)
- **Debugging**: Say "Debug my perforated model" (uses perforatedai skill)

## Entry Point: "Analyze my perforated results"

When the user says **"Analyze my perforated results"**, perform a comprehensive analysis of their PAI training outputs.

### Step 1: Locate Result Files

Find the `save_name` from their training script by searching for the `UPA.initialize_pai()` call. The `save_name` parameter shows where results are stored.

**Only ask "What was your save_name?" if:**
- The script has a variable or argument for save_name that could change
- You cannot find the initialize_pai call in their script

The results are stored in: `{save_name}/{save_name}_*.csv`

Look for these files:
- `{save_name}/{save_name}_scores.csv` - Validation scores over epochs
- `{save_name}/{save_name}_best_arch_scores.csv` - Best architecture performance at each dendrite count
- `{save_name}/{save_name}_switch_epochs.csv` - Epochs when dendrites were added
- `{save_name}/{save_name}_learning_rate.csv` - Learning rate schedule
- `{save_name}/{save_name}Best_PBScores.csv` - Perforated Backpropagation scores (if enabled)

### Step 2: Read and Analyze Score Files

Read all available CSV files and analyze:

**From `{save_name}_scores.csv`:**
- Validation score progression over epochs
- Calculate overall improvement from baseline to final
- Identify training phases (neuron mode vs dendrite mode)
- Check for plateaus or instabilities

**From `{save_name}_switch_epochs.csv`:**
- Identify exactly when dendrites were added (epoch numbers)
- Correlate dendrite additions with score changes from _scores.csv
- Determine if dendrite additions coincided with improvements

**From `{save_name}_best_arch_scores.csv`:**
- Compare performance across different dendrite counts (0, 1, 2, 3, etc.)
- **If test tracking was enabled:** This file will contain test scores for each architecture, not just your final one
- Identify if there are diminishing returns after a certain number of dendrites
- **Determine optimal dendrite count:** If score gains plateau or decrease after N dendrites, recommend setting `max_dendrites` to that value
- Example: If dendrites 0→1→2→3 show good gains but 4→5 show minimal improvement, recommend `max_dendrites=3`

**From `{save_name}_learning_rate.csv`:**
- Show learning rate schedule
- Correlate LR changes with score changes
- Identify if LR scheduling worked well with dendrite additions

**From `{save_name}Best_PBScores.csv` (if exists):**
- Analyze correlation scores for each module
- **Identify modules with low correlation scores** (< 0.3 or bottom 20%)
- **Recommendation:** If certain modules show very low correlation scores, they may not benefit from dendrites
  - Suggest adding those module IDs to `append_module_ids_to_track()` to skip them
  - Focus dendrite resources on high-correlation modules instead

### Step 3: Generate Insights

Provide a comprehensive summary including:

1. **Training Summary:**
   - Initial score vs final score (% improvement)
   - Total epochs trained
   - Number of dendrites added
   - Training phases observed

2. **Performance Analysis:**
   - Best validation score achieved
   - Score stability (variance in later epochs)
   - Whether training converged properly

3. **Dendrite Impact:**
   - Score improvement after each dendrite addition (from switch_epochs.csv + scores.csv)
   - Which dendrite additions had the most impact
   - Whether dendrites helped or hurt performance
   - Optimal dendrite count based on diminishing returns in best_arch_scores

4. **Comparison to Baseline:**
   - Baseline score is the first score in best_arch_scores
   - Calculate parameter count increase (if available)
   - Assess efficiency: did dendrites provide good accuracy/parameter ratio?

5. **Module-Level Analysis (if PB scores available):**
   - Which modules benefited most from dendrites (high correlation)
   - Which modules should be excluded (low correlation)

### Step 4: Show Visualizations

Tell them: "Training visualizations have been automatically generated at `{save_name}/{save_name}.png`. This shows:
- Score progression over epochs
- Learning rate schedule
- Dendrite addition timeline
- Architecture performance comparison"

### Step 5: Next Steps

Based on the analysis results:

**If training went well (dendrites improved performance):**

Say: "Your dendritic training was successful! Here's what I found worked well and recommendations for optimization."

Then direct them to the [Optimization Recommendations](#optimization-recommendations) section below.

**If training had issues (dendrites didn't help or training was unstable):**

Say: "I see some issues in your training results. Let's troubleshoot:"

- **If dendrites didn't improve performance:**
  - Check if you're converting the right layers
  - Is improvement_threshold too strict? Try `[0]`
  - Try different input_dimensions or module configurations
  - Consider using the perforatedai skill to debug: say "Debug my perforated model"
  
- **If training was unstable:**
  - Consider reducing learning rate
  - Adjust candidate_weight_initialization_multiplier lower (0.01 instead of 0.1)
  - Try a different scheduler
  - Check for dimension mismatches
  - Consider using the perforatedai skill to debug: say "Debug my perforated model"

---

## Optimization Recommendations

When dendritic training has successfully improved performance, provide targeted recommendations based on the analysis:

### 1. Optimize Dendrite Count

Based on `best_arch_scores.csv` analysis:

**If diminishing returns detected:**
- Example: Dendrites 0→1→2→3 showed gains of +5%, +3%, +2%, but 4→5 showed only +0.1%
- **Recommend:** `GPA.pc.set_max_dendrites(3)` to focus resources on high-impact dendrites
- Explain: "Your results show the biggest improvements came from the first 3 dendrites. Setting max_dendrites=3 will make training more efficient without sacrificing performance."

**If all dendrites contributed equally:**
- Keep current max_dendrites or increase slightly
- Suggest running longer to see if more dendrites could help

### 2. Adjust Improvement Threshold

Based on score progression from `scores.csv` and `switch_epochs.csv`:

**If dendrites were added too frequently:**
- Current threshold may be too lenient
- **Recommend:** Tighten threshold, e.g., `[0.02, 0.01, 0.001, 0]` instead of `[0.01, 0.001, 0.0001, 0]`
- This makes dendrite additions more selective

**If dendrites were rarely added but helpful:**
- Threshold may be too strict
- **Recommend:** Relax threshold or set to `[0]` to always try adding dendrites when performance plateaus

### 3. Module Selection Optimization

Based on `Best_PBScores.csv` (if available):

**If certain modules show low correlation scores (< 0.3):**
- **Recommend:** Add those module IDs to exclusion list
- Example: If `.layer1` and `.conv1` show correlation < 0.2:
  ```python
  GPA.pc.append_module_ids_to_track([".layer1", ".conv1"])  # Skip these
  ```
- Explain: "These modules showed low dendrite correlation, meaning dendrites didn't help them much. Excluding them will focus resources on high-impact layers."

**If all modules show high correlation:**
- Current module selection is working well
- Consider expanding to convert additional layer types if any were excluded

### 4. Learning Rate and Scheduler Tuning

Based on `learning_rate.csv` and correlation with `scores.csv`:

**If learning rate dropped too quickly:**
- Scores plateaued before dendrites could be fully optimized
- **Recommend:** Increase scheduler patience or use slower decay
- Examples:
  - For ReduceLROnPlateau: `schedArgs = {'mode': 'max', 'patience': 10}` instead of 5
  - For StepLR: `schedArgs = {'step_size': 10, 'gamma': 0.5}` instead of step_size=5
  - For ExponentialLR: `schedArgs = {'gamma': 0.95}` instead of 0.9

**If learning rate stayed high too long:**
- Training may have been unstable during dendrite additions
- **Recommend:** Faster decay or lower initial learning rate
- Examples:
  - For ReduceLROnPlateau: lower patience value
  - For StepLR: smaller step_size or gamma
  - For CosineAnnealingLR: adjust T_max

### 5. Weight Initialization for Dendrites

Based on training stability from `scores.csv`:

**If scores showed spikes/instability after dendrite additions:**
- Dendrite weights may be initialized too large
- **Recommend:** Lower `candidate_weight_initialization_multiplier`
- Example: `GPA.pc.set_candidate_weight_initialization_multiplier(0.01)` instead of 0.1

**If scores were very smooth:**
- Current initialization is working well
- Could potentially try slightly larger values for faster adaptation

### 6. Training Duration Recommendations

Based on total epochs and convergence:

**If training completed but scores still improving:**
- **Recommend:** Increase `set_n_epochs_to_switch()` to train longer in each phase
- Allow more time for dendrites to optimize before adding more

**If training converged early:**
- Current settings are efficient
- Could reduce epoch counts for faster experimentation

### 7. Architecture Expansion Strategies

If dendrites significantly improved performance:

**Consider converting additional layers:**
- If you only converted `["Linear"]`, try adding `["Conv2d", "Linear"]`
- If you skipped early layers, try including them: remove some IDs from `append_module_ids_to_track()`

**Consider increasing max_dendrites:**
- If best_arch_scores showed steady improvements across all dendrite counts
- Try max_dendrites=7 or 10 for potentially higher performance

---

## After Optimization

Once you've identified optimization opportunities:

1. **Update your training script** with the recommended configuration changes
2. **Re-run training** with the new settings
3. **Come back and say "Analyze my perforated results"** again to see if the changes helped

**Need to make configuration changes?** Say "Debug my perforated model" to get help updating your PAI setup (uses perforatedai skill).
