# Data Processing

Scripts for turning PerforatedAI sweep results into CSVs and figures.
Everything here runs locally and reads finished results, nothing trains.

| Script | What it does |
| --- | --- |
| `get_wandb_results.py` | Pull a Weights & Biases sweep into CSVs |
| `combine_csvs.py` | Merge by-dendrite-separate CSVs whose columns differ |
| `process_csv_output.py` | Box plots, scatter plots, and stats CSVs from a sweep |
| `process_csv_output_data_percent.py` | The same for data-efficiency sweeps |
| `spec_from_csv.py` | Build a `plot_streams.py` spec from a sweep CSV or run folders |
| `plot_streams.py` | Render one figure of score against parameters from a spec |
| `pai_style.py` | Shared colors and axes style, imported by the plotting scripts |
| `example_spec.json` | A working spec that shows every optional field |

The plotting side is documented in the Plotting Results section of
[api/wandb.md](../api/wandb.md) and driven by the `perforatedai-plot`
skill in `skills/`.

## get_wandb_results.py

Pull the results of a Weights & Biases sweep into CSVs. We read every run
in the sweep and keep the architecture progression it logged, plus the
final metrics if you ask for them.

### Quick start

```bash
python get_wandb_results.py --url "https://wandb.ai/entity/project/sweeps/SWEEP_ID"
```

### Modes

| `--mode` | Output |
| --- | --- |
| `download` (default) | The raw CSV with every metric. The other modes read this CSV |
| `gen-by-run` | A pivot table for a line graph, one line for each run |
| `by-dendrite` | Scatter plot data grouped by dendrite count |
| `by-dendrite-separate` | Scatter plot data grouped by model type and dendrite count |

### Common options

- `--include-final` keeps the Final metrics too, to check them rather than
  graph them.
- `--dendrite-offset` says where each model starts counting dendrites.
  `"0:2"` means `model_index_0` is pretrained and already has two.
- `--csv` reads an existing raw CSV instead of calling W&B.
- `--output` picks the output filename.
- `--max-completed N` keeps only the first N finished runs with final
  scores. `0` means all.
- `--ignore-models 0 2` drops model indices from the output.

### Examples

```bash
# Download the raw data
python get_wandb_results.py --url "https://wandb.ai/myteam/project/sweeps/abc123"

# Build the by-run comparison
python get_wandb_results.py --url "URL" --mode gen-by-run

# Follow dendrite progression for a pretrained model that starts at 2
python get_wandb_results.py --url "URL" --mode by-dendrite --dendrite-offset "0:2"

# Several models, each with its own offset
python get_wandb_results.py --url "URL" --mode by-dendrite --dendrite-offset "0:2" "1:3"

# Pick the output filename yourself
python get_wandb_results.py --url "URL" --mode by-dendrite --output my_results.csv

# Keep the Final metrics to check they match the last Arch values
python get_wandb_results.py --url "URL" --include-final
```

The output formats and the full workflow are in the Analyzing Sweep
Results section of [api/wandb.md](../api/wandb.md).

## process_csv_output.py

Turn a by-dendrite-separate CSV into summary plots and companion CSVs.
The input comes from `get_wandb_results.py --mode by-dendrite-separate`
and looks like this:

- Row 1 holds metadata labels such as `param_count <column_name>`.
- Row 2 holds the metadata values, the parameter count of each column.
- Row 3 is the header row.
- Row 4 onwards holds the data rows, one for each run.

Unless `--output` says otherwise we write into a folder named after the
CSV stem. It holds candlestick, average, and max plots for each metric,
each with a CSV of the numbers behind it. A `model_info.csv` beside the
input, with `model_id` and `model_name` columns, gives the models readable
names.

```bash
python process_csv_output.py --csv sweep.csv
python process_csv_output.py --csv sweep.csv --output out_dir
python process_csv_output.py --csv sweep.csv --x-break 13000000,22000000
python process_csv_output.py --csv sweep.csv --filter-key data_percent --filter-value 12,25
```

- `--x-break START,END` cuts a range out of the x axis so far-apart
  parameter counts fit on one plot.
- `--filter-key` and `--filter-value` group the plots by a CSV column or
  a run-name token, keeping only the listed values.

## process_csv_output_data_percent.py

The same input, for sweeps that vary the training data fraction. Run
names carry `data_percent_<N>`, `subj_<N>`, `samp_<M>`, and `split_<x>`
tokens, and the script draws score against parameters with one line for
each data fraction, averaging repeats. Extra flags:

- `--dendrite-percent-to-graph 0.6` keeps a dendrite point only if it
  shows up in at least that fraction of the repeats.
- `--require-complete-repeat-dendrites` with `--expected-repeat-count N`
  keeps a point only when every one of the N repeats has it.
- `--sample-splits 50,75,100` and `--subject-splits 50,100` pick and
  order which splits to draw.
