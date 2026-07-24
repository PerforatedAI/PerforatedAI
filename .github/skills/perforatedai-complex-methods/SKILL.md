# PerforatedAI Complex Methods

This skill documents solutions for non-trivial integration scenarios with PerforatedAI.

---

## AMP (Automatic Mixed Precision) with PAI p Mode

### The Problem

When using `torch.cuda.amp.GradScaler` with PAI, training will crash in p mode with:

```
AssertionError: No inf checks were recorded for this optimizer.
```

### Why It Happens

PAI uses a backward hook to run a **second, separate `loss.backward()`** for dendrite training in p mode. This hook fires after your main backward call.

In p mode:
- `scaler.scale(loss).backward()` runs — this registers inf checks only for params that receive gradients through this call, which are the **main_module params** (not in the optimizer in p mode)
- PAI's backward hook fires its own unscaled `backward()` — dendrite params (which ARE in the optimizer) receive gradients through this unscaled call
- `scaler.step(optimizer)` checks the optimizer's param groups for recorded inf checks, finds none (dendrite grads came from the unscaled hook backward), and asserts

### The Fix

Bypass the scaler entirely in p mode. PAI's hook handles the dendrite backward correctly without it:

```python
optimizer.zero_grad(set_to_none=True)
if scaler is not None and GPA.pai_tracker.member_vars['mode'] != 'p':
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    optimizer.step()
```

### Why This Is Safe

- In n mode (main network training): AMP runs normally with full scaler benefits
- In p mode (dendrite training): the scaler is bypassed; PAI's hook-driven backward is already unscaled and handles dendrite gradient computation correctly. AMP precision is not needed for the dendrite training phase.

---

## Optimizer Rebuild After Restructure

When `add_validation_score` returns `restructured=True`, the model has been modified and the optimizer must be fully rebuilt. Key points:

1. Rebuild optimizer AND scheduler identically to initial setup — extract this into a shared function so both call sites are guaranteed identical
2. Rebuild the scaler too if using AMP — the old scaler has stale state tied to the previous optimizer
3. Call `GPA.pai_tracker.set_optimizer_instance(optimizer)` after rebuilding so PAI can re-sync its param groups for the new mode

```python
def build_optimizer_and_scheduler(model, data_loader_train, args):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, ...)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    # ... build scheduler ...
    return optimizer, lr_scheduler, scaler

# Initial setup
optimizer, lr_scheduler, scaler = build_optimizer_and_scheduler(student, data_loader_train, args)
GPA.pai_tracker.set_optimizer_instance(optimizer)

# After restructure
elif restructured and not training_complete:
    optimizer, lr_scheduler, scaler = build_optimizer_and_scheduler(student, data_loader_train, args)
    GPA.pai_tracker.set_optimizer_instance(optimizer)
```
