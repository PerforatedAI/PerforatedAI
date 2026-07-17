# PerforatedAI — Training Event requirements

**Audience:** whoever is implementing the event emitters inside the PerforatedAI library.
**Status:** **the Dashboard side is built and tested against this contract.** Both changes below are the only things missing. Implement them and the visuals light up on the first real run — no Dashboard work required.

This document is standalone. You should not need any other file to implement against it.

---

## Context

PerforatedAI already POSTs Training Events to a configurable `events_url` (read from the Perforation
Config at runtime). A Dashboard consumes them and renders live charts of a training run.

The Dashboard is adding a set of new visuals. Most need **no change to PerforatedAI** — the data is
already on the wire. Three of them do. This document specifies only those three.

### What already works — do not re-implement

| Dashboard visual | Data source | Change needed? |
|---|---|---|
| Score chart — the lines | `epoch` event: `validation_score`, `train_score` | No |
| Learning Rate chart | `epoch` event: `learning_rate` | No |
| Epoch Times chart | `epoch` event: `normal_time`, `pai_time` | No |
| Param Counts chart | `switch` event: `param_count` | No |
| **Dendrite Diagram** | **new event** | **Yes — change 1** |
| **PB Scores chart** | **new field on `epoch`** | **Yes — change 2** |
| **Score chart — the switch markers** | **new field on `switch`** | **Yes — change 3** |

The existing `epoch` event, for reference:

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

The existing `switch` event:

```json
{ "type": "switch", "switch_number": 1, "epoch": 10, "param_count": 1199882 }
```

---

## The `epoch` field is an index, not a counter — it can go backwards

This applies to **every** event carrying an `epoch` (`epoch`, `switch`, `dendrite_added`) and it is
the one thing on this page most likely to produce a wrong-looking chart from a correct emitter.

`epoch` is `num_epochs_run` — the index into the tracker's score lists. It is deliberately **not**
`total_epochs_run`. At a cycle switch the library loads the best model from earlier in the cycle and
`num_epochs_run` **rewinds** to that point; the epochs in between are discarded as *overwritten*.
So a real run emits sequences like:

```
… epoch 29 → epoch 30 → [switch] → epoch 28 → epoch 29 → …
```

The same epoch number is legitimately sent more than once, with different scores.

**The Dashboard must treat `epoch` as an authoritative x-position to overwrite at, not as an
ever-growing counter to append to.** Append, and a rewind draws a sawtooth: the line doubles back on
itself and the chart is garbage. Overwrite at the index, and the live chart matches the offline
graph the library already produces.

### Why this is the right definition

It is what PerforatedAI's own graphs plot. `generate_accuracy_plots` draws
`np.arange(len(accuracies))` — the list index, i.e. `num_epochs_run`. The overwritten epochs are not
dropped; the library redraws them as separate overlay lines that restart at zero. Sending
`total_epochs_run` instead would keep the number monotonic (nicer to consume, no rewinds) but would
put every point to the right of where the offline graph puts it, with the drift growing by the
number of epochs discarded at each switch. The two charts would disagree about the same run, which
is worse than a rewind the consumer knows to expect.

If the Dashboard needs a monotonic axis for some visual, derive it — do not ask the library for it.

---

## Change 1 — new `dendrite_added` event

### Why

The Dashboard's centrepiece visual is a diagram of a dendrite layer that sprouts a new dendrite each
time one is successfully integrated. Nothing currently on the wire says an integration happened.

Critically, **`switch` is not a proxy for this.** Integration is conditional — some cycles try a
dendrite set and switch back without keeping it. Every integration implies a switch; not every switch
implies an integration. That distinction is the whole point of the visual, and it is currently
invisible to the Dashboard.

### Where

At the site where the integration counter is incremented:

```python
if should_increment_integrated:
    GPA.pai_tracker.member_vars["num_dendrites_integrated"] += 1
    _pai_log("info", f"Dendrites successfully integrated! Total integrated: {GPA.pai_tracker.member_vars['num_dendrites_integrated']}")
```

POST the event immediately after the increment, so the value sent is the **new** total.

### Contract

`POST {events_url}` — `application/json`

```json
{ "type": "dendrite_added", "epoch": 14, "num_dendrites_integrated": 2 }
```

| Field | Type | Notes |
|---|---|---|
| `type` | string | Literal `"dendrite_added"` |
| `epoch` | int | The epoch at which integration occurred |
| `num_dendrites_integrated` | int | The **absolute** counter value *after* the increment |

### Why absolute, not a delta or a bare signal

The Dashboard hydrates clients that connect mid-run from a server-held run snapshot, and events can in
principle be dropped or replayed. A count derived by tallying events would silently drift wrong. An
absolute value is self-healing and lets the client assert rather than accumulate. Send the counter's
value, not the fact that it changed.

### Note on granularity

`num_dendrites_integrated` is a single global counter on the tracker, not a per-layer breakdown — a
successful integration adds a dendrite to every dendrite layer at once. The Dashboard treats it that
way: one abstract diagram, N dendrites. **No per-layer data is wanted on this event.**

### Keep posting `switch` — this event does not replace it

`dendrite_added` is **in addition to** the `switch` event you already post, not instead of it. Both
fire at the same moment when an integration succeeds; only `switch` fires when one fails.

This matters more than it looks. The Param Counts chart is keyed by **switch number**, and the Score
chart draws a marker per switch — both go blank if `switch` stops being posted. And the single most
interesting fact the Dashboard can show is the **gap** between the two counts: three switches, two
dendrites means one dendrite set was tried and rejected. Suppress `switch` and that gap disappears,
taking the reason this event exists with it.

---

## Change 2 — `pb_scores` on the `epoch` event

### Why

PerforatedAI already writes a `{save_name}Best PBScores.csv` and plots it — the per-layer dendrite
candidate score, one series per dendrite layer (e.g. `.conv1`, `.conv2`, `.fc1`, `.fc2`). The
Dashboard wants that chart live. It has no way to get the numbers.

### Where

Wherever the epoch POST is currently assembled. These are the same numbers already being written to
the Best PBScores CSV at the same point in the cycle — the ask is to serialize them onto the event you
are already sending, not to compute anything new.

### Contract

The existing `epoch` event gains **one optional field**:

```json
{
  "type": "epoch",
  "epoch": 12,
  "validation_score": 98.5,
  "train_score": 97.1,
  "learning_rate": 0.01,
  "normal_time": 8.4,
  "pai_time": 6.2,
  "pb_scores": { ".conv1": 0.83, ".conv2": 0.71, ".fc1": 0.61, ".fc2": 0.44 }
}
```

| Field | Type | Notes |
|---|---|---|
| `pb_scores` | `dict[str, float]` | Layer name → score. **Omit the key entirely** when there are no scores. |

### Sparsity — important

PB scores only exist while candidate dendrites are being trained. Outside those phases the CSV cells
are blank, and the event should reflect that by **omitting `pb_scores` altogether**.

- Do **not** send `{}`.
- Do **not** send `{".conv1": null}`.
- Do **not** carry forward the last value to fill the gap.

The Dashboard renders the absence as a **break in the line**, which is truthful: it means "no candidate
dendrites were being scored," a real and interesting state rather than missing data.

### Dynamic keys

The layer set is model-dependent — `.conv1`/`.fc1` for a small MNIST net, something else entirely for
a ResNet. Send whatever layer names the model actually has; the Dashboard discovers them at runtime,
builds one line per layer, and labels the legend with the key verbatim.

**Keep the key for a given layer identical for the whole run.** The Dashboard assigns each layer a
colour the first time it sees the key and never reassigns it. A layer whose key changes spelling
mid-run (a prefix added, a rename, a normalisation applied on some epochs but not others) becomes a
*second* layer with a second colour and a broken line, rather than a continuation of the first.

Renaming between runs is fine. Renaming within one is not.

---

## Change 3 — `switch_type` on the `switch` event

### Why

The Score chart draws a vertical marker at every switch. Today all it can say is `Switch 1` — which
tells the reader nothing about *what changed*. With the phase named, the marker becomes
`→ Dendrite` or `→ Normal`, and the chart reads as the alternating cycle it actually is.

### Where

Wherever the `switch` event is currently assembled.

### Contract

The existing `switch` event gains **one optional field**:

```json
{
  "type": "switch",
  "switch_number": 1,
  "epoch": 10,
  "param_count": 1199882,
  "switch_type": "p"
}
```

| Field | Type | Notes |
|---|---|---|
| `switch_type` | `"n"` or `"p"` | The phase being **entered**: `"n"` normal training, `"p"` perforated (dendrite) training |

### It names the phase being entered, not the one that ended

This is the part that is easy to get backwards, and getting it backwards mislabels **every marker on
the chart**. The value describes the mode the model is in *after* the switch. A switch that begins a
dendrite-learning phase is `"p"`, even though what preceded it was normal training.

### Absence is not "normal"

The field is optional and the Dashboard treats a missing `switch_type` as *unknown*, not as `"n"` —
it renders the switch number and a neutral grey rather than a confident wrong label. So a partial
rollout degrades honestly. But do not send `"n"` as a placeholder when you don't know; that is a
claim, and it will be drawn as one.

### Expect every switch to be `"n"` in a default (open-source) run — this is not a bug

Implemented. But the alternating `p, n, p …` cycle this section imagines **does not occur in the
default configuration**, and the Dashboard side should not treat uniform `"n"` as a broken emitter.

The defaults are `perforated_backpropagation = False` and `no_extra_n_modes = True`. Under either,
`change_learning_modes` recurses immediately after entering dendrite mode:

```python
# Because open source version is only doing neuron training for
# gradient descent dendrites, switch back to n mode right away
if (not GPA.pc.get_perforated_backpropagation()) or GPA.pc.get_no_extra_n_modes():
    net = change_learning_modes(net, folder, name, doing_pai)
```

So a single switch goes n → p → back to n before control returns. The model never *rests* in
dendrite mode: dendrites are added and then trained by gradient descent during the normal phase. The
emitted `"n"` is therefore a true statement about the phase being entered, not a mislabel — there is
simply only one standing phase to name.

Consequences:

- **Every Score-chart marker reads `→ Normal`.** Truthful, but carries no signal in this config. The
  interesting per-switch fact is whether a `dendrite_added` accompanied it, not the phase name.
- **`→ Dendrite` markers appear only** when a run sets `perforated_backpropagation = True` (or
  `no_extra_n_modes = False`). The recursion above stops firing, a real `p` phase exists, and the
  same emitter code starts sending `"p"` with no further change. The wiring is forward-compatible;
  it is the configuration that collapses the cycle.

---

## Open questions — please answer these

1. **Is a Best PBScore a running maximum, or the current epoch's score?**
   The filename says *Best*, which implies monotonic. This does not change the wire format either
   way, but it changes what the chart means: if it's a running max, the line only ever climbs, and a
   flat continuation through normal-training epochs would arguably have been more truthful than a gap.
   Answer this before implementing, because it may reopen the sparsity decision above.

2. **Does the integration site have the current epoch number in scope?**
   **Answered: yes.** `member_vars["total_epochs_run"]` is in scope at the increment site and is what
   `dendrite_added` sends. The contract stands as written.

3. **Which switch does an integration coincide with?**
   **Answered — and the intuition in the original question was backwards.** Integration is flagged by
   `should_increment_integrated`, which is set when the mode is `"n"` *before* the switch: reaching a
   switch while in normal training with the global best improved means the existing dendrite set
   earned its keep, so it is banked and a new set is started. `dendrite_added` therefore rides the
   transition that *adds* dendrites, not one that returns to normal.

   In a default run this is not externally observable anyway — see "Expect every switch to be `n`"
   above; the switch collapses n → p → n and reports `"n"` regardless. The Dashboard does not depend
   on the pairing either way (the two events are independent), so this remains non-blocking. If the
   simulator wants to model it, it should emit integrations on the dendrite-adding switch.

---

## What "correct" looks like

The Dashboard was built and verified against a simulator that POSTs exactly this contract. If your
implementation produces the same *shape*, it will render correctly. A 30-epoch run with a switch every
10 epochs produced:

| | |
|---|---|
| `epoch` events | 30 — every epoch |
| `epoch` events carrying `pb_scores` | **9** — epochs 8/9/10, 18/19/20, 28/29/30 (the scoring window before each switch) |
| `switch` events | 3 — at epochs 10, 20, 30 |
| `dendrite_added` events | **2** — at epochs 10 and 20; the third switch failed to integrate |
| `num_dendrites_integrated` values | `1`, then `2` — absolute, never a delta |

The two numbers to sanity-check when you first run it for real:

1. **`dendrite_added` count ≤ `switch` count.** If they are always equal, `should_increment_integrated`
   is not being honoured and the event is firing on every cycle.
2. **PB Scores has visible gaps.** A continuous line across every epoch means `pb_scores` is being
   sent (or carried forward) outside dendrite-learning phases. A completely empty chart means the key
   is never being sent at all.

To see it: start the Dashboard, open `/training`, and run training. The dendrite diagram sprouts a
node on each integration and the count below it tracks `num_dendrites_integrated`.
