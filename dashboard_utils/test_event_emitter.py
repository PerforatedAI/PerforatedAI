"""Tests for the dashboard event emitter wire payloads (ticket 01).

Seam under test: DashboardEventEmitter.emit_epoch / emit_switch -> the JSON dict
handed to requests.post. We drive the emitter with a fake `pc` and a fake
`requests` module, and assert on the captured payload(s).
"""

import pytest

import dashboard_utils.event_emitter as ee


class FakePc:
    def __init__(self, debug=False):
        self._debug = debug

    def get_dashboard_debug(self):
        return self._debug

    def get_dashboard_url(self):
        return "http://localhost:3002/"

    def get_dashboard_events_enabled(self):
        return True

    def get_silent(self):
        return True


class FakeRequests:
    def __init__(self):
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})


@pytest.fixture
def captured(monkeypatch):
    fake = FakeRequests()
    monkeypatch.setattr(ee, "requests", fake, raising=False)
    monkeypatch.setattr(ee, "_requests_available", True)
    return fake


@pytest.fixture
def emitter():
    e = ee.DashboardEventEmitter()
    e.reset()
    return e


def payloads(captured, event_type):
    return [c["json"] for c in captured.calls if c["json"]["type"] == event_type]


# --- emit_epoch --------------------------------------------------------------

def test_emit_epoch_uses_new_contract_keys(captured, emitter):
    emitter.emit_epoch(
        FakePc(),
        epoch_index=17,
        phase="n",
        scores={"validation": 0.9891, "running": 0.9887, "train": 0.9776},
        learning_rate=0.0002,
        normal_time=8.4,
        pai_time=None,
    )
    (p,) = payloads(captured, "epoch")
    assert p["type"] == "epoch"
    assert p["epoch_index"] == 17
    assert p["phase"] == "n"
    assert p["scores"] == {"validation": 0.9891, "running": 0.9887, "train": 0.9776}
    assert p["learning_rate"] == 0.0002
    assert p["normal_time"] == 8.4
    assert p["pai_time"] is None


def test_emit_epoch_drops_old_flat_fields(captured, emitter):
    emitter.emit_epoch(
        FakePc(), epoch_index=0, phase="n", scores={"validation": 0.5},
        learning_rate=None,
    )
    (p,) = payloads(captured, "epoch")
    for gone in ("epoch", "validation_score", "train_score"):
        assert gone not in p


def test_emit_epoch_omits_pb_scores_when_absent(captured, emitter):
    emitter.emit_epoch(
        FakePc(), epoch_index=0, phase="n", scores={"validation": 0.5},
    )
    (p,) = payloads(captured, "epoch")
    assert "pb_scores" not in p
    assert "pb_scores_current" not in p


def test_emit_epoch_includes_pb_scores_in_p_phase(captured, emitter):
    emitter.emit_epoch(
        FakePc(), epoch_index=5, phase="p", scores={"validation": 0.5},
        pb_scores={"conv1": 0.008, "fc2": 0.087},
        pb_scores_current={"conv1": 0.006, "fc2": 0.070},
    )
    (p,) = payloads(captured, "epoch")
    assert p["pb_scores"] == {"conv1": 0.008, "fc2": 0.087}
    assert p["pb_scores_current"] == {"conv1": 0.006, "fc2": 0.070}


def test_true_epoch_never_decreases_across_rollback(captured, emitter):
    pc = FakePc()
    for idx in (0, 1, 2, 3):
        emitter.emit_epoch(pc, epoch_index=idx, phase="n", scores={"validation": 0.5})
    # rollback: epoch_index steps backward
    emitter.emit_epoch(pc, epoch_index=1, phase="n", scores={"validation": 0.5})
    emitter.emit_epoch(pc, epoch_index=2, phase="n", scores={"validation": 0.5})
    trues = [p["true_epoch"] for p in payloads(captured, "epoch")]
    assert trues == sorted(trues)
    assert all(b > a for a, b in zip(trues, trues[1:]))


def test_reset_zeros_true_epoch_counter(captured, emitter):
    pc = FakePc()
    emitter.emit_epoch(pc, epoch_index=0, phase="n", scores={"validation": 0.5})
    emitter.emit_epoch(pc, epoch_index=1, phase="n", scores={"validation": 0.5})
    emitter.reset()
    emitter.emit_epoch(pc, epoch_index=0, phase="n", scores={"validation": 0.5})
    assert payloads(captured, "epoch")[-1]["true_epoch"] == 1


# --- emit_switch -----------------------------------------------------------

def test_emit_switch_uses_new_contract_keys(captured, emitter):
    pc = FakePc()
    emitter.emit_epoch(pc, epoch_index=19, phase="n", scores={"validation": 0.5})
    emitter.emit_switch(
        pc, switch_ordinal=2, epoch_index=19, param_count=1192084, switch_type="n",
    )
    (p,) = payloads(captured, "switch")
    assert p["type"] == "switch"
    assert p["switch_ordinal"] == 2
    assert p["epoch_index"] == 19
    assert p["true_epoch"] == 1
    assert p["switch_type"] == "n"
    assert p["param_count"] == 1192084
    for gone in ("switch_number", "epoch"):
        assert gone not in p


def test_emit_switch_omits_unknown_switch_type(captured, emitter):
    emitter.emit_switch(
        FakePc(), switch_ordinal=0, epoch_index=3, param_count=1000, switch_type=None,
    )
    (p,) = payloads(captured, "switch")
    assert "switch_type" not in p


# --- emit_dendrite_added (ticket 02) --------------------------------------

def test_emit_dendrite_added_uses_new_contract_keys(captured, emitter):
    pc = FakePc()
    emitter.emit_epoch(pc, epoch_index=25, phase="n", scores={"validation": 0.5})
    emitter.emit_dendrite_added(
        pc, epoch_index=25, num_dendrites_integrated=1, param_count=1192084,
    )
    (p,) = payloads(captured, "dendrite_added")
    assert p["type"] == "dendrite_added"
    assert p["epoch_index"] == 25
    assert p["true_epoch"] == 1
    assert p["num_dendrites_integrated"] == 1
    assert p["param_count"] == 1192084
    assert "epoch" not in p


# --- emit_run_end (ticket 02) --------------------------------------------

def test_emit_run_end_sends_stored_best_values(captured, emitter):
    emitter.emit_run_end(
        FakePc(), epoch_last_improved=41, global_best_score=0.9921,
    )
    (p,) = payloads(captured, "run_end")
    assert p["type"] == "run_end"
    assert p["epoch_last_improved"] == 41
    assert p["global_best"] == {"epoch_index": 41, "score": 0.9921}


# --- emit_run_start (ticket 02: confirm unchanged) ----------------------

def test_emit_run_start_payload_unchanged(captured, emitter):
    emitter.emit_run_start(FakePc(), "MyModel")
    (p,) = payloads(captured, "run_start")
    assert p["type"] == "run_start"
    assert p["model_class"] == "MyModel"
    assert "timestamp" in p
    assert set(p) == {"type", "model_class", "timestamp"}
