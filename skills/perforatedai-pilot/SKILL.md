---
name: perforatedai-pilot
description: "Run a controlled two-run pilot study: the user's baseline exactly as it stands, then the same script with PerforatedAI dendrites added and nothing else changed. Produces a one-page comparison report suitable for someone who wasn't in the room. Triggers: 'run a perforated pilot', 'pilot study', 'does perforated help my model', 'evaluate perforated on my model'. Adds strict no-tuning enforcement on top of the perforatedai skill - for ordinary integration or optimization work, use perforatedai and perforatedai-analyze directly."
---

# PerforatedAI Pilot Study

## Scope

A controlled experiment with **one independent variable**:

1. **Run A - baseline.** The user's script exactly as it stands. No changes.
2. **Run B - perforated.** The identical script plus the PerforatedAI integration. Nothing else different.

The deliverable is a filled-in comparison table and a deferred-ideas list. Not a better model.

**This skill is the enforcement layer, not the workflow.** The workflow lives in the **perforatedai**
skill, which already records a baseline in Step 1.4 and holds tuning until after the first result.
Follow it. This file adds the rules that make the resulting number defensible to someone who wasn't
in the room, and it overrides the main skill wherever they disagree.

### Overrides on the main skill

| Main skill says | In a pilot |
|---|---|
| Step 1.4 - reuse an existing baseline number if they have one | Reuse **only** if it came from the current unmodified script, same hardware, same data. Otherwise rerun. A number from an older commit or a different GPU is not a baseline. |
| Step 8 - score tracking instrumentation | Add only what the report needs. Instrumentation is safe; keep it minimal anyway. |
| Step 3 - config per architecture | Take the stock block as written. Do not explore module-selection variants or tune `max_dendrites` hunting for a better number. If the stock config runs, it is the config. |
| Tuning Options section | Out of scope. Do not open it. |
| Follow-up - run perforatedai-analyze | Not during the pilot. Its recommendations are exactly what a pilot defers. It comes after the report. |
| Step 10 - compare, then offer tuning | Compare, write the report, **stop.** |

---

## The one-variable rule

Between Run A and Run B, exactly one thing may differ: the PerforatedAI integration.

**The most common way this task fails is that the agent - you - notices the baseline could be better
and improves it.** The learning rate looks untuned, there's no augmentation, AMP would double the
speed. Those observations may all be correct, and acting on any of them destroys the experiment: the
comparison stops isolating dendrites and starts measuring your unrelated changes.

**The baseline is whatever it currently is, including its flaws.** A weak baseline is a valid
baseline. Do not fix it, do not fix it "just a little," do not fix it in Run B only.

**Forbidden in both runs:**

| Category | Do not touch |
|---|---|
| Optimization | learning rate, schedule, warmup, optimizer, momentum, weight decay, gradient clipping |
| Data | augmentation, normalization, splits, dataset size, sampling, shuffling, batch size |
| Architecture | layer counts, widths, activations, norm layers, dropout, pretrained weights |
| Training length | epoch count, early-stopping patience, checkpoint selection |
| Performance | AMP, `torch.compile`, channels_last, dataloader workers, device changes |
| Code quality | refactors, bug fixes, dead code removal, restructuring the training loop |

That last row is not an oversight. **A change that is objectively an improvement is still a second
variable.** Even a genuine bug fix counts, unless it is applied identically to both runs and
disclosed in the report's Method section.

### Diff gate - run before each of the two runs

```bash
git diff --stat
```

- **Before Run A:** empty, or only files unrelated to training. If the training script has
  uncommitted edits, stop and ask whether they are part of the intended baseline. Record the SHA:
  `git rev-parse --short HEAD`.
- **Before Run B:** only PerforatedAI lines - imports, `GPA.pc.set_*`, `UPA.perforate_model`,
  `setup_optimizer`, `add_validation_score`, the restructure block. Anything else in that diff means
  the experiment is broken. Revert it before running.

Not under git? Copy the script to `<name>_baseline_snapshot.py` before editing and diff against that.

**One exemption:** a parameter-count or timing `print`, applied identically to both runs.

---

## The parking lot

You will spot real improvements. Write them down instead of acting on them.

Keep a running **Deferred Optimizations** list from the moment you read the script. One line per
observation plus expected impact. It ships as a section of the report and is often the second-most
valuable thing the pilot produces.

**When the user proposes scope creep mid-pilot** - and they will:

> "Good call, and I've added it to the deferred list. I'd rather not change it mid-pilot - once the
> two runs differ by more than the dendrites, the comparison stops telling us whether dendrites
> helped. Let's finish the two runs, then that's the obvious first thing to try."

If they hear that and still want it: it's their model. **Apply it identically to both runs, restart
from Run A, and note it in Method.** What you must never do is apply it to one side only, or carry
on without rerunning the baseline.

---

## Sizing and honesty

**Before starting**, check how long one run takes. A pilot needs a model that trains in hours, not
days. If a run takes more than about a day, ask whether there's a smaller configuration - fewer
epochs, a data subset, a smaller variant - to pilot on. A shortened configuration is valid **as long
as both runs use it**, and it gets disclosed.

**Expect Run B to take more epochs and more wall-clock time.** PAI trains to a plateau, adds
dendrites, trains again. That is inherent to the method, not a misconfiguration. Do not truncate
Run B to match Run A, and do not extend Run A to match Run B. Each side gets its natural stopping
point; the cost difference is reported, not engineered away.

**Debugging a crash in Run B is in scope** - "make it run" is not "make it better." Use the
**perforatedai** skill's debug entry point, and the error-by-error reference in
`perforatedai-debugging/SKILL.md`, as freely as you need. If `noImprove_lr*` files appear, no
dendrites were ever added and the run is not a valid Run B: fix it and rerun.

---

## The report

One page. Plain numbers, honest caveats.

```markdown
# Pilot: <model> on <dataset>

**Question:** does adding PerforatedAI dendrites improve <metric>, with nothing else changed?

## Result

| | Baseline | + Perforated | Delta |
|---|---|---|---|
| <metric> (best) | | | |
| Parameters | | | |
| Epochs to best | | | |
| Wall-clock | | | |

<One sentence: did it help, by how much, at what cost.>

## Method

- Baseline: commit `<sha>`, unmodified, <N> epochs, seed <s>, <hardware>.
- Perforated: identical script + PAI integration. Config: <the stock block used>.
- Diff between runs: PAI integration only. <Or: name any disclosed exception.>
- Dendrites added at epochs <...>; score by dendrite count: 0 -> <x>, 1 -> <y>, ...

## Caveats

- **Single seed, single run per arm.** A directional signal, not a significance test. Differences
  smaller than this model's run-to-run variance should not be read as real.
- Neither arm was tuned. Both numbers are likely below what tuning would reach.
- <Anything else honest: shortened schedule, data subset, baseline known to be weak.>

## Deferred optimizations

Observed during the pilot, deliberately not acted on:

1. <observation> - <expected impact>
2. ...
```

**Read the delta honestly.** If the difference is within run-to-run noise, say so rather than
presenting it as a win. A pilot returning "no measurable difference under these conditions" is a
successful pilot - it answered the question. Overselling a marginal delta is worse than a null
result, because the user finds out on their next run.

---

## Stop

When the report is delivered, the pilot is over. Do not start a second configuration, do not begin
working the deferred list, do not run a seed sweep.

The constraints lift here. Hand off:

> "That's the pilot. **perforatedai-analyze** will read the run's CSVs and recommend config changes,
> and the deferred list is there whenever you want to pick it up - both are where the optimization
> work starts."
