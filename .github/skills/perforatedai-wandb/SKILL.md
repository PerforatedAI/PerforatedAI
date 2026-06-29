---
name: perforatedai-wandb
description: "WandB-specific PerforatedAI integration guardrail skill. Use when users want WandB sweeps/logging with PerforatedAI, or when fixing repeated WandB integration mistakes. Enforces strict adherence to perforatedai/api-docs/wandb.md and includes expandable correction sections."
---

# PerforatedAI WandB Guardrail Skill

This skill is a strict add-on for WandB + PerforatedAI workflows.

## Purpose

- Follow the canonical guidance in `perforatedai/api-docs/wandb.md` exactly.
- Reduce recurring implementation mistakes by enforcing checkpoints.
- Provide structured places to add new correction notes over time.

## When To Use

Use this skill when the user asks for any of the following:
- WandB sweep integration with PerforatedAI
- WandB logging for perforated training
- Fixing a broken or incomplete WandB + PAI setup
- Guardrails/checklist for WandB + PAI edits

Do not use this as a replacement for base PAI integration steps when WandB is not involved.

## Mandatory First Action

Before making any code changes, read the full canonical doc:

- `.github/skills/perforatedai/api-docs/wandb.md`

If any local habit or previous pattern conflicts with `wandb.md`, `wandb.md` wins.

## Global Guardrails (Always Enforce)

CRITICAL PRIORITY CHECK #1 (must be explicitly verified first on every WandB integration):

- Final metrics must always log from global maxima (`global_max_val`, `global_max_train`, `global_max_params`).
- At end-of-training final logging, the extra final arch log must only run when this condition is true:
  - `current_integrated > last_logged_integrated and hasattr(wandb, "run") and wandb.run is not None`
- Never skip or weaken this condition in the final logging block.

1. Never skip reading `wandb.md` first.
2. Never invent alternative WandB flow when `wandb.md` already defines one.
3. Always preserve the user's existing training logic unless a WandB/PAI requirement forces a change.
4. For sweep mode, ensure `wandb.init()` occurs in the sweep training function and config is read from `wandb.config`.
5. Apply dendritic hyperparameters from `wandb.config` before `UPA.perforate_model()`.
6. Keep `save_name` aligned with `wandb.run.name` when available.
7. Use `num_dendrites_integrated` (not `num_dendrites_added`) for architecture-level logging.
8. Avoid duplicate final metric logging for perforated models.
9. Keep edits minimal and targeted; do not introduce formatting-only changes.

## Required Execution Flow

Follow these sections in order. For each section:
- complete the checklist,
- run the verification,
- then proceed.

---

## Section 1: Setup And Imports

### Checklist

- [ ] `import wandb` exists.
- [ ] PAI imports exist:
  - `from perforatedai import globals_perforatedai as GPA`
  - `from perforatedai import utils_perforatedai as UPA`
- [ ] Sweep-related CLI args exist when needed (`--sweep-id`, `--sweep-count`, `--wandb-project`, optional `--wandb-entity`).

### Verification

- Confirm script can parse args in both new-sweep and join-sweep modes.

### Guardrail: Common Mistakes

- Mistake pattern: Missing one or more WandB CLI args.
- Detection rule: `wandb.agent`/`wandb.sweep` present but no `--wandb-project` arg.
- Correction rule: Add required args exactly as documented in `wandb.md`.

### Space For Future Corrections

- Additional mistakes for Section 1:
  - TODO:
  - TODO:

---

## Section 2: Sweep Function Pattern

### Checklist

- [ ] Dedicated `train_with_wandb()` (or equivalent) exists.
- [ ] `wandb.init()` is called inside that function.
- [ ] `config = wandb.config` is used.
- [ ] If needed, silent mode is overridden for visibility (`GPA.pc.set_silent(False)`).

### Verification

- Confirm training function runs under `wandb.agent(..., function=train_with_wandb, ...)`.

### Guardrail: Common Mistakes

- Mistake pattern: Using global/static config instead of `wandb.config` during sweeps.
- Detection rule: Sweep params exist in config but code never reads `wandb.config`.
- Correction rule: Route sweep hyperparameters through `wandb.config` in the sweep function.

### Space For Future Corrections

- Additional mistakes for Section 2:
  - TODO:
  - TODO:

---

## Section 3: Apply PAI Config Before Perforation

### Checklist

- [ ] Dendritic sweep params are read from `wandb.config`.
- [ ] `GPA.pc.set_improvement_threshold(...)` and optional forward-function mapping are set before perforation.
- [ ] `UPA.perforate_model(...)` is called only after required PAI config is applied.

### Verification

- Confirm code path that calls `UPA.perforate_model(...)` has already applied sweep-controlled PAI settings.

### Guardrail: Common Mistakes

- Mistake pattern: Applying PAI sweep params after perforation.
- Detection rule: `UPA.perforate_model(...)` appears before dendritic config setters.
- Correction rule: Move relevant `GPA.pc.set_*` calls to before `UPA.perforate_model(...)`.

### Space For Future Corrections

- Additional mistakes for Section 3:
  - TODO:
  - TODO:

---

## Section 4: save_name And Run Identity

### Checklist

- [ ] `save_name` is derived from `wandb.run.name` when available.
- [ ] Fallback save name exists when run name is unavailable.
- [ ] Local result folders and WandB run identity are consistent.

### Verification

- Confirm a run produces local output folder matching WandB run naming pattern.

### Guardrail: Common Mistakes

- Mistake pattern: Hardcoded save_name causes mixed/overwritten runs.
- Detection rule: Sweep code present but constant string used for `save_name`.
- Correction rule: Use `wandb.run.name` with safe fallback.

### Space For Future Corrections

- Additional mistakes for Section 4:
  - TODO:
  - TODO:

---

## Section 5: Metric Logging During Training

### Checklist

- [ ] Epoch-level metrics are logged (`Train*`, `Val*`, optional `Test*`, LR, params, dendrite count).
- [ ] Logging checks `wandb.run` exists before `wandb.log(...)`.

### Verification

- Confirm no logging crash when WandB run is missing or disabled.

### Guardrail: Common Mistakes

- Mistake pattern: Unconditional `wandb.log(...)` causing runtime errors.
- Detection rule: No run-availability check around logging.
- Correction rule: Guard `wandb.log(...)` calls with run existence checks.

### Space For Future Corrections

- Additional mistakes for Section 5:
  - TODO:
  - TODO:

---

## Section 6: Architecture-Level Logging For Perforated Models

### Checklist

- [ ] Track architecture maxima only in neuron mode (`mode == 'n'`).
- [ ] Log arch metrics when integrated dendrite count increases.
- [ ] Use `num_dendrites_integrated`, not attempted count.
- [ ] Reset arch trackers after each successful architecture log.

### Verification

- Confirm one arch log entry per integration step and no duplicates.

### Guardrail: Common Mistakes

- Mistake pattern: Logging by attempted dendrites (`num_dendrites_added`).
- Detection rule: Arch count sourced from `num_dendrites_added`.
- Correction rule: Switch to `num_dendrites_integrated` and track last logged integrated count.

### Space For Future Corrections

- Additional mistakes for Section 6:
  - TODO:
  - TODO:

---

## Section 7: Final Metrics (No Duplicate Logging)

### Checklist

- [ ] Perforated model: Final max metrics logged in the restructuring/training-complete path.
- [ ] Final metrics are logged from global maxima (`global_max_val`, `global_max_train`, `global_max_params`).
- [ ] In the final logging block, extra final architecture log only runs when `current_integrated > last_logged_integrated and hasattr(wandb, "run") and wandb.run is not None`.
- [ ] Non-perforated model: Final metrics logged after normal training completion.
- [ ] No double logging of the same final metrics.

### Verification

- Confirm one final metrics event per run.

### Guardrail: Common Mistakes

- Mistake pattern: Final metrics logged in both training loop and post-training for perforated runs.
- Detection rule: Two code paths can log `Final Max Val` for perforated model.
- Correction rule: Keep only one final-logging path for perforated runs.
- Mistake pattern: Final metrics logged from last-epoch values instead of global maxima.
- Detection rule: Final logs do not reference tracked global max variables.
- Correction rule: Always log final metrics from global max trackers.
- Mistake pattern: Final logging skips the integrated-count/run-exists gate for extra final architecture log.
- Detection rule: End-of-training block can emit final arch metrics without `current_integrated > last_logged_integrated and hasattr(wandb, "run") and wandb.run is not None`.
- Correction rule: Apply the exact gate only in the final logging block; keep in-loop arch logging behavior as defined in `wandb.md`.

### Space For Future Corrections

- Additional mistakes for Section 7:
  - TODO:
  - TODO:

---

## Section 8: Sweep Launch/Join Commands

### Checklist

- [ ] New-sweep command path is present.
- [ ] Join-existing-sweep path is present.
- [ ] `--wandb-project` is always required and passed in both modes.

### Verification

- Confirm both command patterns from `wandb.md` are represented.

### Guardrail: Common Mistakes

- Mistake pattern: Join flow missing explicit project argument.
- Detection rule: Join path calls `wandb.agent` without project value from args.
- Correction rule: Ensure project argument is explicit in all `wandb.agent` calls.

### Space For Future Corrections

- Additional mistakes for Section 8:
  - TODO:
  - TODO:

---

## Pre-Delivery Self-Check (Must Pass)

Before returning edits to the user, verify all items:

1. Confirm each Section 1-8 checklist is satisfied.
2. Confirm edits match guidance from `.github/skills/perforatedai/api-docs/wandb.md`.
3. Confirm no extra unrelated refactors or formatting changes were introduced.
4. Summarize exactly which WandB/PAI requirements were implemented.

If any item fails, fix it before responding.

---

## Correction Ledger (Append-Only)

Use this ledger to capture recurring mistakes and lock in new guardrails.
Add new entries whenever a WandB integration error is discovered.

Template:

- Date:
- Mistake ID:
- Context (what task/script):
- Incorrect behavior:
- Detection signal:
- Root cause:
- Permanent guardrail added:
- Verification added:

### Entries

- Date: 2026-06-26
  - Mistake ID: WANDB-LOG-001
  - Context: Perforated WandB integration review
  - Incorrect behavior: Tried to skip strict gating condition in final logging and risked incorrect final arch/final metric behavior.
  - Detection signal: End-of-training final logging path existed without enforcing `current_integrated > last_logged_integrated and hasattr(wandb, "run") and wandb.run is not None`.
  - Root cause: Guardrail was implied but not enforced as top-priority explicit requirement.
  - Permanent guardrail added: Added CRITICAL PRIORITY CHECK #1 requiring global-max-based final logging and exact gate in final logging block.
  - Verification added: Section 7 checklist requires exact final-block gate plus global-max final metric logging.
