# Dashboard Package

Customer-deliverable package that adds the Dashboard MCP Server and its Claude Code Skills to a project.

**New here? Start with [QUICKSTART.md](QUICKSTART.md)** — zero to a live dendrite training run. This README is the reference.

## Contents

This directory holds only what's public: the installer and its docs.

- `bootstrap.sh` — **the installer.** The only file published outside the image.
- `README.md` — this file
- `QUICKSTART.md` — the happy path, start here

Everything else it installs — `uninstall.sh`, `dashboard-run.sh`, the `skills/` Claude Code copies into `.claude/skills/`, the MCP Server, and the Dashboard client — **rides inside the Docker image** at `/app/package/` and is extracted onto the host at install time. That source lives in the private repo that builds the image, not here. Shipping it inside the image rather than this repo is what makes it impossible for the Skills and the MCP Server to be at different versions — they are the same artifact.

`bootstrap.sh` is the exception on purpose: an installer welded into the image could not be fixed without rebuilding it, and users pinned to an old version would keep the broken one forever. Publishing it from this repo means a fix ships the moment it's merged.

## Install

From the project root (the Codebase you want to add the Dashboard to):

```sh
curl -fsSL https://raw.githubusercontent.com/PerforatedAI/PerforatedAI/main/Studio_Install/bootstrap.sh -o bootstrap.sh
sh bootstrap.sh
```

Or in one line, if you don't want to read it first:

```sh
curl -fsSL https://raw.githubusercontent.com/PerforatedAI/PerforatedAI/main/Studio_Install/bootstrap.sh | sh
```

This will:

1. Pull the image and smoke-test that it actually runs on this machine — so a bad image fails here, loudly, rather than inside the invisible MCP subprocess later.
2. **Resolve the version.** It reads the image's version label and pins *that* into `.mcp.json`. `latest` is never written there — Claude Code re-resolves the image reference every session, so a floating tag would let the MCP Server drift to a new version while your Skills stayed frozen at this one.
3. Extract the Package from the image: Skills into `.claude/skills/`, launcher and uninstaller into `.perforated_tools/`.
4. Register a `dashboard` entry in the project's `.mcp.json`, pointing at the launcher, mounting the Codebase read-only plus a read-write `.perforated_tools/`.
5. Write `.perforated_tools/installed.json` recording what it installed.

Restart Claude Code (or start a new session) afterwards so it picks up the updated `.mcp.json`.

### Options

| Flag | Default | |
|---|---|---|
| `--version` | `latest` | Install a specific version, e.g. `--version v0.1.0` |
| `--port` | `3002` | Port the MCP Server listens on |

Piping to `sh` means flags go **to `sh`**, not to `curl`:

```sh
curl -fsSL <url> | sh -s -- --version v0.1.0 --port 4000
```

### Upgrading

Re-run the installer. It's idempotent, and it reconciles: Skills a new version renamed or dropped are removed rather than left orphaned in `.claude/skills/`.

There is no `update.sh`, deliberately. Anything shipped inside the image is by definition the *previous* version's logic, and upgrading is precisely when you want the *newest* installer — which is the one at the URL above.

Your own hand-written Skills in `.claude/skills/` are never touched. The installer only removes Skills that `installed.json` says it put there.

## Usage

Once installed, invoke any of the skills from Claude Code:

- `/dashboard` — verify the MCP Server is reachable and open the Dashboard in a browser.
- `/train-my-model` — configure and launch a training run with live dashboard monitoring.
- `/compare-models` — export two or more models and compare their architectures side by side.
- `visualize-model` — export a PyTorch model to ONNX and view its architecture in the Visualizer (triggered by asking to visualize a model, or via `ml_export_model`).
- `perforatedai` — set up or debug a PerforatedAI dendrite integration in a PyTorch model.
- `perforatedai-analyze` — review results from a completed PerforatedAI training run and get optimization recommendations.
- `perforatedai-distributed` — multi-GPU (DataParallel/DDP) setup for PerforatedAI; invoked automatically by the `perforatedai` skill when needed.

## Uninstall

The installer left the uninstaller in your project. Run it there:

```sh
.perforated_tools/uninstall.sh
```

It removes the `dashboard` entry from `.mcp.json`, the Skills it installed, the runtime launcher and log, and the Docker image it recorded at install time — the version you actually installed, not a hardcoded one.

It needs no network (part of its job is deleting the image, so depending on a registry would be perverse), and it leaves alone anything it doesn't own: your own Skills, and your exports and persisted training runs under `.perforated_tools/`.

It reads `.perforated_tools/installed.json` to know what it owns. With no manifest it refuses outright rather than guessing — guessing would mean deleting Skills it never installed.

## MCP Server

The MCP Server is a Python (fastMCP) process running inside the `dashboard-mcp` Docker container. It talks to Claude Code over stdio MCP transport and serves the compiled React build (the Dashboard) as static files over HTTP on the configured port (default `3002`). It mounts the Codebase read-only at `/workspace` and `.perforated_tools/` read-write at `/perforated_tools` — the only channel the container can write output back through.

Tools are grouped into two plugins, each namespacing its tool names with a prefix:

- `dashboard` plugin — `ping`, `dashboard_visualize_model`, `dashboard_show_variant`, `dashboard_remove_variant`, `dashboard_clear_variants`, `dashboard_open_training`, `dashboard_show_training_chart`, `dashboard_hide_training_chart`
- `ml` plugin — `ml_export_model`, `ml_configure_runner`

## Dashboard views

The Dashboard is a single-page app with three routes:

- **Visualizer** (`/visualize/<ModelClassName>`) — a React Flow DAG of one model's forward pass, built from a Model Artifact (`ml_export_model` output). Click a node to open a sidebar with its type, path, output shape, and `__init__` params. Perforated layers are highlighted. Opened via `dashboard_visualize_model` or the `visualize-model` skill, one model per browser tab.
- **Comparison View** (`/compare`) — multiple models side by side, each in its own React Flow panel. Panel membership (which models are shown) is controlled by Claude via `dashboard_show_variant` / `dashboard_remove_variant` / `dashboard_clear_variants`, pushed to all connected browser tabs over the `/events` SSE channel. Use the `compare-models` skill to drive this.
- **Training View** (`/training`) — live view of the in-progress or most recent Training Run. Opened ahead of time by `dashboard_open_training` (called from the `train-my-model` skill) so it's ready before the training script starts posting events. Shows:
  - An **Epoch Progress Bar** (current epoch out of `total_epochs`, from the run's `run_start` event).
  - The **Dendrite Diagram** — a tall schematic on the left, growing a dendrite node per *successful integration* (`dendrite_added`). Privileged: it has no chart id and cannot be hidden. Rendering caps at 5 dendrites; the count beside it stays exact.
  - The permanent **score chart** (validation score, train score, with Switch boundaries marked as vertical phase lines).
  - Optional **Training Charts** — Learning Rate, Epoch Times, Param Counts, PB Scores — in a grid. **All are visible from the start of a run**; Claude curates by *hiding* via `dashboard_hide_training_chart`. `dashboard_show_training_chart` exists to undo a hide. Visibility resets to all-visible at the start of every new run.
  - A **Training Log** scrollback of `log`-type events for the current run (not persisted — cleared on reload). `error`-level log events also trigger a Toast notification.

## How training events reach the dashboard

PerforatedAI (running in the customer's own environment, not inside the container) POSTs Training Events as JSON to an `events_url` the MCP Server exposes at `/training-events`. The `train-my-model` skill wires this up via `ml_configure_runner`, which writes `events_url` into the Perforation Config (`{save_name}/{save_name}_config.json`) so PerforatedAI can read it at runtime. The server updates its in-memory run state from each event and fans it out to all connected Dashboard tabs over the `/training-events` SSE channel.

The event types:

| type | when | carries |
|---|---|---|
| `run_start` | training begins | model class name, timestamp — resets run state, seeds all charts visible |
| `epoch` | after each epoch | validation score, train score, learning rate, normal epoch time, PAI epoch time, and optionally `pb_scores` |
| `switch` | a PerforatedAI learning-phase boundary | switch number, param count, epoch number |
| `dendrite_added` | a dendrite set is **successfully integrated** | epoch, `num_dendrites_integrated` (absolute count) |
| `log` | free-text message | `message`, `level` (`info` \| `warning` \| `error`) |
| `run_end` | training finishes | persists the run |

Every stored event carries its `type`, and that is the only supported way to tell them apart — never infer an event's kind from which fields are present.

`dendrite_added` is **not** the same moment as `switch`. Integration is conditional: every integration implies a switch, but a switch can happen without one, when PerforatedAI tries a dendrite set and rejects it. The gap between the two counts is the point.

`pb_scores` on an `epoch` is a dict of per-layer dendrite candidate scores (`{".conv1": 0.83}`). It is **sparse** — omitted entirely outside dendrite-learning phases — and the PB Scores chart renders those gaps as breaks in the line rather than inventing values.

Only `epoch`, `switch`, and `dendrite_added` events are persisted with the run; `log` events are ephemeral (Training Log scrollback only).

> **`pb_scores` and `dendrite_added` require PerforatedAI to emit them.** The Dashboard is built and tested against this contract, but until PerforatedAI posts them, the PB Scores chart and the Dendrite Diagram stay empty on real runs. The spec to implement is [`perforatedai-event-requirements.md`](../perforatedai-event-requirements.md).

### Adding a graph to the Training View

*This section is for developers working in the private repo that builds the image (where `mcp_server/` and `client/` live) — not for this repo.*

The score chart and the Dendrite Diagram are permanent and cannot be hidden. Everything else is an optional Training Chart — visible by default, hideable by Claude. To add a new one:

1. **Server**: add the new id to the `TrainingChartId` literal in `mcp_server/tools/dashboard/training_state.py`. That literal is the single server-side list: the MCP tools validate against it *and* `run_start` seeds `visible_charts` from it, so a new id is automatically visible by default. There is no second list to update.
2. **Client**: add a chart component (e.g. modeled on `client/src/components/LearningRateChart/`) and register it — with a label and the component — in the `OPTIONAL_CHARTS` map in `client/src/components/TrainingView/TrainingView.tsx`. A chart declares which slices of run state it consumes; `ParamCountsChart` takes `switches` rather than `epochs`.
3. The server enum and the client map are kept in sync by convention, not a shared source of truth — a drift fails safely: the tool call is rejected, or the chart id is accepted but nothing renders.
4. Nothing else is needed to make it show up. Visibility resets to *all charts visible* on the next `run_start`.

If a training script has metrics PerforatedAI itself doesn't know about (e.g. a loop-bound `total_epochs`), it can POST additional events directly to `GPA.pc.events_url`, e.g.:

```python
import requests

requests.post(GPA.pc.events_url, json={"type": "run_config", "total_epochs": epochs})
```

This is a separate event rather than part of `run_start` because the training script's known values aren't available until the loop actually starts.

No real PerforatedAI run needed to develop against the Training View — that repo's `scripts/simulate_training.py` posts a realistic sequence of Training Events to a running MCP Server.
