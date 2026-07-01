# Dashboard HTTP API

Base URL: `http://localhost:3002`

---

## `GET /events`

SSE stream for the Comparison View panel list. Immediately sends the current state on connect, then pushes on every change.

**Response:** `text/event-stream`

### Events

| Event | Data type | Description |
|---|---|---|
| `panel_list` | `string[]` | Ordered list of model class names currently in the Comparison View |

**Example:**
```
event: panel_list
data: ["ResNet", "VGG"]
```

---

## `GET /training-events`

SSE stream for the active training run. Immediately sends a `run_state` hydration event on connect.

**Response:** `text/event-stream`

### Events

| Event | Data type | Description |
|---|---|---|
| `run_state` | `RunState \| null` | Full run snapshot; `null` when no run is active |
| `epoch` | `EpochEvent` | Pushed after each epoch completes |
| `switch` | `SwitchEvent` | Pushed when the model switches architecture |
| `log` | `LogEvent` | Pushed for a free-text log line. Not appended to `RunState.events` and not persisted — session-only |
| `chart_visibility` | `string[]` | Full list of currently visible optional Training Chart ids, pushed whenever it changes |

**`RunState`**
```json
{
  "model_class": "ResNet",
  "timestamp": "2026-06-24T10:00:00",
  "total_epochs": 30,
  "visible_charts": [],
  "events": [/* all epoch/switch events seen so far — log events are excluded */]
}
```

**`EpochEvent`** (also the shape stored in `RunState.events`)
```json
{
  "epoch": 3,
  "validation_score": 99.1,
  "train_score": 98.0,
  "learning_rate": 0.001,
  "normal_time": 7.0,
  "pai_time": 5.5
}
```

**`SwitchEvent`** (also stored in `RunState.events`)
```json
{
  "switch_number": 1,
  "epoch": 10,
  "param_count": 1199882
}
```

**`LogEvent`** (not stored in `RunState.events`, not persisted)
```json
{
  "level": "error",
  "message": "loss diverged"
}
```
`level` is one of `info`, `warning`, `error`. An `error`-level `log` event additionally triggers a Toast in the Dashboard — the same event powers both, there's no separate error channel.

---

## `POST /training-events`

Ingest a training lifecycle event. Used by the training script to push progress to connected clients.

**Request:** `application/json`

**Response:** `{"ok": true}` on success, `400` for unknown `type`.

### Payloads

**`run_start`** — resets run state (including `visible_charts` to `[]`), pushes `run_state` to all SSE clients
```json
{ "type": "run_start", "model_class": "ResNet", "timestamp": "2026-06-24T10:00:00", "total_epochs": 30 }
```

**`epoch`** — appends to run state, pushes `epoch` event to all SSE clients
```json
{
  "type": "epoch",
  "epoch": 1,
  "validation_score": 98.5,
  "train_score": 97.1,
  "learning_rate": 0.01,
  "normal_time": 8.4,
  "pai_time": 6.2
}
```

**`switch`** — appends to run state, pushes `switch` event to all SSE clients
```json
{ "type": "switch", "switch_number": 1, "epoch": 10, "param_count": 1199882 }
```

**`log`** — pushes a `log` event to all SSE clients; does not mutate `RunState.events` and is not persisted
```json
{ "type": "log", "level": "warning", "message": "switch 1 triggered at epoch 10" }
```

**`run_end`** — persists the current `RunState` to `${EXPORTS_DIR}/runs/{timestamp}_{model_class}.json`, does not push to SSE clients
```json
{ "type": "run_end" }
```

---

## `GET /runs`

Returns metadata for all persisted Training Runs, sorted newest-first by timestamp. Backed by files written to `${EXPORTS_DIR}/runs/` on `run_end`. Malformed or unreadable files are skipped with a logged warning rather than failing the request.

**Response:** `application/json`

```json
[
  { "filename": "2026-06-30T10-00-00_ResNet.json", "model_class": "ResNet", "timestamp": "2026-06-30T10:00:00", "epoch_count": 30 }
]
```

`epoch_count` is the number of `epoch`-type events in the persisted run (i.e. events with a `validation_score` key).

---

## Training Chart visibility — MCP tools

Not HTTP endpoints, but directly affect the `chart_visibility` SSE event above. Called by Claude, not the training script.

| Tool | Params | Description |
|---|---|---|
| `dashboard_show_training_chart` | `chart_id: "learning_rate" \| "epoch_times"` | Adds a chart id to `visible_charts`; no-op if already visible or no active run |
| `dashboard_hide_training_chart` | `chart_id: "learning_rate" \| "epoch_times"` | Removes a chart id from `visible_charts`; no-op if not currently visible or no active run |

`chart_id` is validated against a fixed enum server-side — unknown ids are rejected. See `docs/adr/0012-server-side-training-chart-id-registry.md` and `docs/adr/0013-training-chart-visibility-scoped-per-run.md`.
