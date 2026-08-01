# AGENTS.md

Guide for AI agents working **on the PerforatedAI repo itself**. To add PerforatedAI to _another_ project, use the [Agent Skills](./skills/README.md) instead; those are consumer-facing and don't apply here.

## Orientation

- PyTorch library that adds artificial _dendrites_ to networks. Published pip package: `perforatedai`. Library source is in `perforatedai/` (`globals_perforatedai` → `GPA`, `utils_perforatedai` → `UPA`, plus `tracker_`/`modules_`/`network_`). Integration docs are in `api/`.
- The patented **Perforated Backpropagation™** algorithm lives in a separate `perforatedbp` package that is **not in this repo**; the `perforated_backpropagation` flag defaults to `False`. Do not add, stub, or reference `perforatedbp` internals here.

## Setup & checks

- Dev install: `python -m venv ENV && source ENV/bin/activate && pip install -e .`
- **No repo-wide test suite.** Validate a change by running an example that exercises it end-to-end (e.g. `examples/baseExamples/mnist/`), confirming dendrites are added and training completes, then report what you observed.
- Format with Black (no lint config is checked in).

## Traps

- **Bump `version` in both `setup.py` and `setupCython.py`;** they carry it separately.
- Changing the integration API (`GPA.pc.set_*`, `UPA.perforate_model`, `pai_tracker.*`) breaks downstream users **and** the shipped skills in `skills/`. Update `api/` docs and the affected `SKILL.md` files in the same change.
- Sign off every commit (DCO): `git commit -s`; PRs without it are rejected. Branch from `main`, never commit to it directly. Full checklist: [CONTRIBUTING.md](./CONTRIBUTING.md).
