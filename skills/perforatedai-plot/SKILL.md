---
name: perforatedai-plot
description: "Render a single-panel PAI figure of score versus parameter count from sweep CSVs, PAI run folders, or hand-supplied numbers. Trigger: 'make the PAI graph', 'plot params vs score', 'plot my dendrite results'. Builds a JSON spec, then renders it with data_processing/plot_streams.py. Figures only, no analysis. For recommendations use perforatedai-analyze."
---

# PerforatedAI Plot Skill

## Overview

This skill produces one publication-style figure: parameter count on the
x axis, a score on the y axis, one line or scatter stream per model or
configuration. It reads a JSON spec of literal numbers and renders it with
`data_processing/plot_streams.py`. The spec is the frozen record of the
figure, so every number in the PNG can be traced back.

**When to use this skill:**
- After a sweep, to draw the best score per dendrite count per model
- After one or more PAI training runs, to draw score against parameters
- When the user hands you numbers and wants them in the shared PAI style

**Not for:** interpreting results or recommending settings. That is the
`perforatedai-analyze` skill.

## Scripts

All scripts live in `data_processing/` of the PerforatedAI repository.
Run them from that directory or by absolute path.

| Script | Purpose |
| --- | --- |
| `plot_streams.py` | Render a spec to a PNG |
| `spec_from_csv.py` | Build a spec from a sweep CSV or from PAI run folders |
| `pai_style.py` | Shared colors and axes style, imported by the others |
| `example_spec.json` | A working spec showing every optional field |

Plotting is read-only work and runs on the local machine.

## Entry Point

Ask the user one question at a time. Offer a recommended default with each
question and wait for the answer before the next one.

### Step 0: Choose the source

Ask which of the three sources the figure comes from:

1. **Sweep CSV**: a by-dendrite-separate CSV from
   `get_wandb_results.py --mode by-dendrite-separate`. One stream per
   model, one point per dendrite count.
2. **PAI run folders**: folders holding `<save_name>_best_arch_scores.csv`
   from `UPA.perforate_model(save_name=...)`. Each stream is a list of
   folders, one point per folder by default.
3. **Numbers**: values the user supplies, or that you pull from W&B in
   session and write into the spec as literals.

### Step 1: Output name and location

Ask for the PNG stem, for example `yolo26_cityscapes`. Then ask where the
PNG and spec copy should go. Recommend the directory that will hold the
spec. Pass the answer to `plot_streams.py` as an absolute `--out-dir`.

### Step 2: Title

Optional. Always ask.

### Step 3: Y axis

Ask for the label. It is required. Ask for the value format only when the
metric is not a four decimal quantity, the default is `.4f`. The x axis
defaults to "Parameters (millions)" with raw counts in the spec divided by
1e6 at plot time. Ask for a custom x label only if the user mentions a
different x quantity.

### Step 4: Streams

**Sweep CSV.** Ask which stat collapses the runs in each column, recommend
`max`. Then build:

```bash
python3 data_processing/spec_from_csv.py \
    --csv path/to/sweep_by_dendrite_separate.csv \
    --out NAME --spec-path DIR/NAME.json \
    --title "..." --y-label "..." --stat max
```

Stream names come from `model_info.csv` beside the CSV when it exists,
otherwise `model_0`, `model_1`. Ask the user whether to rename them.

**Run folders.** Ask which folders go in each stream and what each stream
is called. Ask whether a stream takes the best point per folder or every
dendrite count from a single folder (`--all-dendrites`). Then build:

```bash
python3 data_processing/spec_from_csv.py \
    --stream Vanilla:runs/nano_plain,runs/small_plain \
    --stream PAI:runs/nano_pai,runs/small_pai \
    --out NAME --spec-path DIR/NAME.json --title "..." --y-label "..."
```

`--metric` names the column of `best_arch_scores.csv`, default
`Max Valid Scores`. `--best min` selects the lowest row for losses.

**Numbers.** Write the spec by hand following `example_spec.json`. When
pulling from W&B, default to the best epoch by the run's primary metric,
confirm that rule with the user, and take parameter counts from
`<save_name>_best_arch_scores.csv` or `<save_name>param_counts.csv`, not
from W&B's fused parameter field. Record run ids, files, and the pull
date in each stream's `source` field.

### Step 5: Anchor and colors

Ask whether there is a vanilla or zero dendrite point that belongs to no
stream. If so it becomes the spec's `anchor`, drawn hollow, and line
streams start from it. Streams take colors in order from
`pai_style.stream_palette`: teal, gray, dark blue, orange, then evenly
spaced hues. Ask only if the user wants a stream pinned to a color, set
with the stream's `color` field.

### Step 6: Render and check

```bash
python3 data_processing/plot_streams.py DIR/NAME.json --out-dir DIR
```

Open the PNG with Read and show the user. Ask whether any labels collide.
If so, add per point `offset` values `[dx, dy, ha, va]` in points and
re-render. The builder turns annotations off when a stream has more than
eight points, flip `annotate` back on if the user wants them.

## Spec shape

```json
{
  "out": "yolo26_cityscapes",
  "title": "Cityscapes Perforated YOLO26",
  "x_axis": {"label": "Steps"},
  "y_axis": {"label": "mAP50-95", "format": ".4f", "lim": [0.25, 0.28]},
  "annotate": true,
  "anchor": {"x": 2506920, "y": 0.26187, "name": "Vanilla",
             "source": "..."},
  "streams": [
    {"name": "PAI", "style": "line", "from_anchor": true,
     "color": "#00FAC9",
     "points": [[2571528, 0.26447],
                {"x": 2636136, "y": 0.27322, "label": "2d",
                 "offset": [10, -8, "left", "top"]}],
     "source": "run id, file, date"},
    {"name": "L2_Full", "style": "scatter", "dash_to_anchor": true,
     "points": [[2585520, 0.26641]]}
  ]
}
```

Omit `x_axis` for the default parameters axis. `title`, `lim`, `format`,
`annotate`, `anchor`, `color`, `from_anchor`, `dash_to_anchor`, `label`
and `offset` are optional. A point is `[x, y]` or an object with `x`, `y`,
and optional `label` and `offset`.

## Style

Every figure the repository produces shares `data_processing/pai_style.py`:
dotted grid on both axes, no top or right spine, framed legend, 200 dpi
with a tight bounding box. Do not set colors, grids, or dpi inline in a
new plotting script, import `pai_style` instead.
