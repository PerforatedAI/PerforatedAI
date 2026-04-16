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
- `{save_name}/*noImprove_lr*` files - Check if these exist (indicates no dendrites were added)
- `{save_name}/{save_name}_train_scores.csv` - Training scores if tracked (for overfitting detection)

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
- **Check if dendrites were added before actual plateau:** Look at score history before each switch - if scores were still improving significantly when dendrites were added, they may have been added prematurely

**Check for noImprove_lr files:**
- **If `noImprove_lr` files exist in the save folder:** This means dendrites were NEVER added during training
- This is a critical issue - training completed without ever attempting to add dendrites
- See troubleshooting section for loop settings recommendations

**From `{save_name}_train_scores.csv` (if exists):**
- Compare training scores vs validation/test scores
- **If training scores significantly better than validation/test:** Indicates overfitting
- Calculate the gap between train and validation performance
- Recommend regularization if gap is large (>5-10% difference)

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
   - **CRITICAL:** Check if dendrites were added at all (absence of switch_epochs data or presence of noImprove_lr files)
   - Score improvement after each dendrite addition (from switch_epochs.csv + scores.csv)
   - Which dendrite additions had the most impact
   - Whether dendrites helped or hurt performance
   - Optimal dendrite count based on diminishing returns in best_arch_scores
   - Timing of additions: Were dendrites added prematurely (before plateau) or at appropriate times?

4. **Generalization Analysis (if training scores available):**
   - Train vs validation/test score gap
   - Whether the model is overfitting (train >> val/test)
   - If overfitting is present, recommend regularization strategies

5. **Comparison to Baseline:**
   - Baseline score is the first score in best_arch_scores
   - Calculate parameter count increase (if available)
   - Assess efficiency: did dendrites provide good accuracy/parameter ratio?

6. **Module-Level Analysis (if PB scores available):**
   - Which modules benefited most from dendrites (high correlation)
   - Which modules should be excluded (low correlation)
   - Parameter distribution: Which layers received the most dendrites?

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

- **If NO dendrites were added (noImprove_lr files exist or switch_epochs.csv is empty):**
  - **This means training completed without ever attempting dendrite addition**
  - **Most likely cause:** Your training loop exited before PAI could detect a plateau and attempt adding dendrites
  - **Recommend:** Verify your training loop structure:
    - **CRITICAL:** Training should use an infinite loop (`while True:`) with PAI's `training_complete` flag controlling when to exit
    - The loop should only break when `training_complete` returns `True` after `add_validation_score()`
    - Check `set_n_epochs_to_switch()` allows enough time (e.g., 20-30 epochs per phase)
    - Check `set_n_epochs_for_switch_history()` - needs sufficient history to detect plateau (e.g., 10 epochs)
  - **Example problem:** If you used `for epoch in range(50):` instead of `while True:`, training may have ended before PAI finished
  - **Correct pattern:**
    ```python
    epoch = -1
    while True:
        epoch += 1
        # training code
        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_score, model)
        if training_complete:
            break
    ```
  - The PAI system needs: warmup epochs + history epochs + time to detect plateau before attempting dendrite addition
  - **To control/minimize training time with infinite loop:**
    - Use `GPA.pc.set_max_dendrites(N)` to limit how many dendrites are added (e.g., `set_max_dendrites(3)` stops after 3 dendrites)
    - Training will complete faster since fewer dendrite phases are needed
    - Optionally use `FIXED_SWITCH_MODE` for more consistent/predictable training time:
      ```python
      GPA.pc.set_when_to_switch_mode("FIXED_SWITCH_MODE")
      GPA.pc.set_n_epochs_to_switch(20)  # Adds dendrite every 20 epochs
      ```
    - With FIXED mode, you know exactly when dendrites are added, making total training time predictable
  
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

### 2. Adjust Improvement Threshold and Switch Timing

Based on score progression from `scores.csv` and `switch_epochs.csv`:

**If dendrites were added too frequently:**
- Current threshold may be too lenient
- **Recommend:** Tighten threshold, e.g., `[0.02, 0.01, 0.001, 0]` instead of `[0.01, 0.001, 0.0001, 0]`
- This makes dendrite additions more selective

**If dendrites were added BEFORE scores actually plateaued:**
- Look at score history in the epochs before each dendrite addition
- If scores were still improving significantly (e.g., +2% in the last few epochs), dendrites were added prematurely
- **Premature dendrite addition wastes capacity** - the base network could have improved more first
- **Recommend for FIXED switch mode:** Increase `set_n_epochs_to_switch()` to give more time before adding dendrites
- **Recommend for HISTORY switch mode:** 
  - Raise the improvement threshold (e.g., from `[0.001]` to `[0.005]` or `[0.01]`)
  - Increase `set_n_epochs_for_switch_history()` to require longer plateau (e.g., from 10 to 15 epochs)
- **Goal:** Only add dendrites when base network has truly plateaued

**If dendrites were rarely added but helpful:**
- Threshold may be too strict
- **Recommend:** Relax threshold or set to `[0]` to always try adding dendrites when performance plateaus

### 3. Module Selection Optimization

Based on `Best_PBScores.csv` (if available):

**What PBScores mean:**
- **PBScores measure correlation between dendrite activations and network gradients**
- Higher scores (> 0.02) = dendrites aligned well with learning signal = good dendrite placement
- Lower scores (< 0.01) = dendrites poorly aligned = wasting parameters

**If certain modules show low correlation scores (< 0.3):**
- **Recommend:** Add those module IDs to exclusion list
- Example: If `.layer1` and `.conv1` show correlation < 0.2:
  ```python
  GPA.pc.append_module_ids_to_track([".layer1", ".conv1"])  # Skip these
  ```
- Explain: "These modules showed low dendrite correlation, meaning dendrites didn't help them much. Excluding them will focus resources on high-impact layers."
- **For parameter efficiency:** This is especially important - don't waste parameters on modules that won't benefit

**If all modules show high correlation (> 0.02):**
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

### 8. Regularization and Generalization

Based on comparison between training and validation/test scores:

**If training scores are significantly better than validation/test scores (overfitting):**
- **Example:** Train accuracy 95%, Val accuracy 82% (13% gap)
- **Problem:** Model is memorizing training data rather than learning generalizable features
- **Impact on PAI:** Improvements to training scores won't translate to better validation/test performance

**Recommended regularization techniques:**
- **Add dropout:** Insert dropout layers between converted modules
  ```python
  # Example: Add dropout before Linear layers
  model.dropout = nn.Dropout(0.3)  # Start with 0.3-0.5
  ```
- **Weight decay:** Increase L2 regularization in optimizer
  ```python
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Try 1e-4 to 1e-3
  ```
- **Label smoothing:** Soften target labels to prevent overconfidence (classification only)
  ```python
  # For cross-entropy loss
  criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Try 0.05-0.2
  ```
- **Batch normalization:** Add batch norm layers to reduce internal covariate shift
  ```python
  # Example: After conv/linear layers
  nn.BatchNorm2d(channels)  # For Conv2d
  nn.BatchNorm1d(features)  # For Linear
  ```
- **Gradient clipping:** Prevent exploding gradients during training
  ```python
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  ```
- **Data augmentation:** Add more aggressive augmentation to training data (for vision tasks)
- **Mixup/CutMix:** Mix training examples together (advanced technique)
- **Reduce model complexity:** Use fewer/smaller dendrites or convert fewer layers
- **Early stopping:** Stop when validation score plateaus even if training improves

**Goal:** Get training and validation scores closer together (within 2-5%). This ensures that when dendrites improve training performance, it translates to real test improvements.

### 9. Parameter Efficiency Strategy

Based on `Best_PBScores.csv` analysis and model size considerations:

**IMPORTANT: Dendrites ADD parameters to your model**
- Dendrites are a more efficient way to add capacity than simply making the base network bigger
- BUT they still increase total parameter count
- **Each dendrite adds:** `input_dimensions × output_dimensions` parameters per module

**If you want to use PAI for optimization (keep/reduce model size):**

**CRITICAL: You must reduce the original model FIRST, then perforate:**
- ❌ **WRONG:** Take a large model → perforate it → hope it gets smaller (it won't - it gets bigger)
- ✅ **CORRECT:** Take a large model → reduce width/depth → perforate the smaller model → match original performance with fewer base parameters

**Example workflow:**
```python
# Original model: 512 hidden units, 10M parameters, 85% accuracy
# Step 1: Reduce to 256 hidden units → 2.5M parameters, 78% accuracy (worse)
# Step 2: Perforate reduced model → 4M parameters, 85% accuracy (matched!)
# Result: Same accuracy with 60% fewer parameters
```

**If your perforated model is LARGER than your original:**
- You may have reduced too little or perforated too many layers
- **Recommended strategy:** Perforate fewer layers, focusing on layers closer to the OUTPUT (top of network)
- **Use PBScores to guide this:** 
  - Look at `Best_PBScores.csv` to see which layers got the most dendrites
  - If early layers (close to input) have many dendrites, consider excluding them
  - **Later layers are more efficient:** They have already-processed features, so dendrites there are more impactful per parameter
  
**If you want to push parameter efficiency even further:**
- **Check PBScores first to identify which modules benefit most:**
  - Look at `Best_PBScores.csv` - modules with scores > 0.02 are efficient dendrite users
  - **Only convert modules with good PBScores** - don't waste parameters on low-scoring modules
  - Example: If `.fc` has 0.1 correlation but `.conv1` has 0.01, skip `.conv1`
- **Perforate only the last 1-3 layers** instead of the whole network
  ```python
  # Example: Only perforate final classifier
  GPA.pc.append_module_ids_to_track([".conv1", ".conv2", ".layer1", ".layer2"])  # Skip these
  # Now only .layer3 and .fc will be perforated
  ```
- **Reduce base width more aggressively** and let dendrites compensate
- **Use PBScores to find the highest-impact layers** and only perforate those
- **If PBScores show most modules have low correlation (< 0.01):**
  - This suggests dendrites aren't helping much overall
  - Consider only tracking (excluding) all but the 1-2 highest-scoring modules
  - Focus all dendrite capacity on the modules that actually benefit

**Parameter count visibility:**
- Count parameters before and after: `sum(p.numel() for p in model.parameters())`
- Check `Best_PBScores.csv` to see dendrite distribution across layers
- Calculate efficiency ratio: `(accuracy_gain / parameter_increase) × 100`

---

## After Optimization

Once you've identified optimization opportunities:

1. **Update your training script** with the recommended configuration changes
2. **Re-run training** with the new settings
3. **Come back and say "Analyze my perforated results"** again to see if the changes helped

**Need to make configuration changes?** Say "Debug my perforated model" to get help updating your PAI setup (uses perforatedai skill).
