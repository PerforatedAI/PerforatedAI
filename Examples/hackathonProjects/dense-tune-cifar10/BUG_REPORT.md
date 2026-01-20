# Bug Report: LR Scheduler Step Order After Dendrite Addition

## Summary

During training with PerforatedAI, a PyTorch warning is raised after dendrite additions indicating that `lr_scheduler.step()` is being called before `optimizer.step()`.

## Environment

- PyTorch: 2.x
- Python: 3.9
- PerforatedAI: Latest (from source)
- OS: macOS

## Steps to Reproduce

1. Run training with PAI using `DOING_HISTORY` switch mode
2. Train until PAI detects a plateau and adds dendrites
3. Observe the warning after the optimizer is reset via `setup_optimizer()`

## Expected Behavior

No warnings should appear after dendrite additions.

## Actual Behavior

```
✨ Dendrites Added! Resetting optimizer.
UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. 
In PyTorch 1.1.0 and later, you should call them in the opposite order: 
`optimizer.step()` before `lr_scheduler.step()`. Failure to do this will 
result in PyTorch skipping the first value of the learning rate schedule.
```

## Root Cause Analysis

When `GPA.pai_tracker.setup_optimizer()` is called after restructuring:
1. A new optimizer and scheduler are created
2. The training loop continues and calls `scheduler.step()` at the end of the epoch
3. However, the new optimizer hasn't done its first `step()` yet
4. This causes PyTorch to skip the first LR value in the schedule

## Impact

- The first learning rate value after dendrite addition may be skipped
- This could affect model convergence after restructuring
- The warning clutters training logs

## Suggested Fix

**Option 1:** Return a flag from `setup_optimizer()` indicating the scheduler is fresh:
```python
optimizer, scheduler, is_fresh = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
if not is_fresh:
    scheduler.step()
```

**Option 2:** Call `optimizer.step()` with zero gradients internally in `setup_optimizer()`:
```python
# Inside setup_optimizer, after creating scheduler:
optimizer.step()  # Dummy step to satisfy scheduler
```

**Option 3:** Have the training script skip `scheduler.step()` when `restructured` is True:
```python
if not restructured:
    scheduler.step()
```

## Workaround

In the training script, skip the scheduler step on epochs where restructuring occurred:
```python
if restructured:
    pass  # Skip scheduler.step() this epoch
else:
    scheduler.step()
```

## Affected Code

- `perforatedai/globals_perforatedai.py` - `PAITracker.setup_optimizer()`
- User training scripts that call `scheduler.step()` after `setup_optimizer()`

## Priority

Low - This is a warning, not an error, but it may affect training dynamics.

---

*Reported by: Sathvik Vempati*  
*Date: 2026-01-19*
