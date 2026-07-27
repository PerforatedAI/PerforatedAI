# Quickstart

Zero to a live dendrite training run. About 10 minutes, most of it Docker pulling an image.

For what each piece *is*, see [README.md](README.md). This is just the happy path.

---

## Before you start

- **Docker**, running. The MCP Server ships as a container; the installer fails fast if Docker isn't up.
- **Claude Code**, in the project you want to add dendrites to.
- **A PyTorch project** with a model class and a dataloader you can import.

You do **not** need to install PerforatedAI or the Dashboard by hand. The skills walk you through it.

---

## 1. Install

From the root of your project:

```sh
curl -fsSL https://raw.githubusercontent.com/PerforatedAI/PerforatedAI/main/Studio_Install/bootstrap.sh | sh
```

Add `-s -- --port 4000` if something already owns port 3002. (The `-s --` is how you pass flags
through a pipe — they go to `sh`, not to `curl`.)

**Then restart Claude Code.** It reads `.mcp.json` at startup, so a running session won't see the
Dashboard until you do. This is the single most common reason step 2 fails.

To upgrade later, run the same command again. It's idempotent, and your own skills are left alone.

## 2. Check it's alive

```
/dashboard
```

Claude pings the MCP Server and opens the Dashboard in your browser. If it says the server isn't
connected, see [Troubleshooting](#troubleshooting).

## 3. Look at your model

```
Visualize my model
```

Claude asks for your model file, class name, and dataloader, then traces the forward pass and opens
it as an interactive graph. Click any node for its type, shape, and constructor args.

This step is optional, but it is the fastest way to confirm the Dashboard can actually import your
model — which is the same thing dendrite setup needs. Better to find out here.

## 4. Add dendrites

```
Perforate my model
```

The `perforatedai` skill walks you through wrapping your model with PerforatedAI: which layers get
dendrites, what the switching policy is, and how the training loop changes. It writes a Perforation
Config next to your model.

Multi-GPU? It pulls in the `perforatedai-distributed` skill on its own; you don't have to ask.

## 5. Train, and watch it grow

```
/train-my-model
```

Claude collects your save name and training script, wires the training run to the Dashboard, opens
the **Training View**, and hands you the command to run. Start training, and the page fills in live:

- **Dendrites** — the diagram on the left. Each time a dendrite set is *successfully integrated*, a
  new dendrite sprouts, tapping the same inputs as the neuron and feeding back into it. The count
  below it is exact even when the drawing caps out.
- **Score per epoch** — validation and train score, with each switch marked.
- **Learning Rate**, **Epoch Times**, **Param Counts**, **PB Scores** — the diagnostics, in the grid.

All the charts start visible. Ask Claude to hide the noisy ones (`hide the epoch times chart`) and
it will; the choice resets on the next run.

> **Watch the gap.** More switches than dendrites is normal and interesting: it means PerforatedAI
> tried a dendrite set and rejected it because it didn't earn its place. Three switches with two
> dendrites is the model telling you it's saturating.

## 6. Read the results

```
Analyze my training run
```

The `perforatedai-analyze` skill reads the run output and tells you what the dendrites bought you —
accuracy per parameter added, where returns started diminishing, what to try next.

---

## Troubleshooting

**`/dashboard` says the MCP Server isn't connected.**
Almost always a stale Claude Code session. Restart it. If that doesn't do it, check `docker ps` and
look at `.perforated_tools/dashboard.log` — the container's stderr goes there.

**Port already in use.**
Reinstall with `--port`: `curl -fsSL <url> | sh -s -- --port 4000`.

**The Training View says "waiting for training run".**
The page is open and connected; it just hasn't received a `run_start` yet. That's expected until
your training script actually starts. If it stays that way after training begins, PerforatedAI isn't
finding its `events_url` — re-run `/train-my-model`, which rewrites it into the Perforation Config.

**The dendrite diagram never grows.**
Dendrites only appear on *successful integration*, which is not the same as a switch. If the Score
chart shows switches but the diagram stays at zero, PerforatedAI is trying dendrite sets and
rejecting all of them — that's a real result, not a bug. Check the PB Scores chart: flat or absent
candidate scores mean the dendrites aren't learning anything worth keeping.

---

## Uninstall

The installer left the uninstaller in your project:

```sh
.perforated_tools/uninstall.sh
```

Removes the MCP entry, the skills it installed, the launcher, and the Docker image. Your own skills,
and anything else you put in `.perforated_tools/`, are left alone.
