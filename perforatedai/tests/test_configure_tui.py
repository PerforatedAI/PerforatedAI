"""Headless keystroke tests for the interactive configuration TUI.

These drive ``set_perforation_targets`` by monkeypatching ``read_single_key`` to
replay a scripted key sequence, and assert on the rendered frames. When the
script runs out the fake key reader raises ``KeyboardInterrupt``, which the TUI
treats as a clean quit (``SystemExit``), so every test ends in ``SystemExit``.
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from perforatedai import globals_perforatedai as GPA
from perforatedai import configure_perforatedai as CPA


def build_model():
    return nn.Sequential(
        nn.Conv2d(3, 8, 3),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3),
        nn.Flatten(),
        nn.Linear(8, 4),
    )


@pytest.fixture(autouse=True)
def clean_config():
    GPA.pc.set_module_ids_to_perforate([])
    GPA.pc.set_module_ids_to_track([])
    GPA.pc.set_module_names_to_perforate([])
    GPA.pc.set_module_names_to_track([])
    GPA.pc.set_configuration_confirmed(False)
    yield


class Driver:
    def __init__(self, monkeypatch, keys):
        self._keys = list(keys)
        self.frames = []
        monkeypatch.setattr(CPA, "enter_alternate_screen", lambda: None)
        monkeypatch.setattr(CPA, "exit_alternate_screen", lambda: None)
        monkeypatch.setattr(CPA, "read_single_key", self._next_key)

    def _next_key(self):
        if not self._keys:
            raise KeyboardInterrupt
        return self._keys.pop(0)

    def run(self, capsys):
        with pytest.raises(SystemExit):
            CPA.set_perforation_targets(build_model())
        out = capsys.readouterr().out
        self.frames = [CPA.strip_ansi(f) for f in out.split("\x1b[2J\x1b[H") if f.strip()]
        return self.frames

    def last(self, before_quit_lines=1):
        # The final frame before the quit message.
        return self.frames[-1]


def find_frame(frames, needle):
    return next((f for f in frames if needle in f), None)


def test_perforate_by_id_updates_marker_and_budget(monkeypatch, capsys):
    driver = Driver(monkeypatch, ["j", "p"])  # move to Conv2d, perforate
    frames = driver.run(capsys)
    final = frames[-1]
    assert "mode=perforate" in final
    assert "Perforation budget:" in final
    assert "params per dendrite cycle" in final


def test_type_rule_shows_in_overlay_with_count(monkeypatch, capsys):
    # row 0 is the first Conv2d; P = perforate the whole type, y = open type rules
    driver = Driver(monkeypatch, ["P", "y"])
    frames = driver.run(capsys)
    overlay = find_frame(frames, "Type rules")
    assert overlay is not None
    assert "Conv2d → perforate" in overlay
    assert "(2)" in overlay  # two Conv2d modules in the fixture


def test_h_on_targets_opens_marker_legend(monkeypatch, capsys):
    driver = Driver(monkeypatch, ["p", "h"])
    frames = driver.run(capsys)
    legend = find_frame(frames, "Targets legend")
    assert legend is not None
    assert "inherited from an ancestor" in legend
    assert "needs attention" in legend
    assert "Which mode wins" in legend
    assert "inherited  >  by id  >  by type" in legend


def test_unset_modules_warning_and_save_gate(monkeypatch, capsys):
    driver = Driver(monkeypatch, ["s"])  # straight to the save dialog, nothing set
    frames = driver.run(capsys)
    assert find_frame(frames, "still need a mode") is not None
    save = find_frame(frames, "have parameters but no mode")
    assert save is not None
    assert "Start anyway" in save
    assert "Keep editing" in save


def test_tab_to_run_settings_shows_divider(monkeypatch, capsys):
    # Tab to run settings, move to the Dendrite acceptance bucket, expand it.
    driver = Driver(monkeypatch, ["\t", "j", "j", "\r"])
    frames = driver.run(capsys)
    run_frame = find_frame(frames, "run-wide configuration")
    assert run_frame is not None
    divider = find_frame(frames, "rarely changed")
    assert divider is not None


def test_switch_mode_renders_by_name_and_cycles(monkeypatch, capsys):
    # Tab -> run settings, open Switch Strategy (item 1), select switch_mode, cycle.
    driver = Driver(monkeypatch, ["\t", "j", "\r", "j", "\r"])
    frames = driver.run(capsys)
    assert find_frame(frames, "switch_mode = DOING_") is not None
    # after one cycle from DOING_HISTORY the value should have changed
    final = frames[-1]
    assert "switch_mode = DOING_FIXED_SWITCH" in final


def test_quit_confirim_exits(monkeypatch, capsys):
    driver = Driver(monkeypatch, ["q", "y"])
    with pytest.raises(SystemExit):
        CPA.set_perforation_targets(build_model())
    out = capsys.readouterr().out
    assert "training did not start" in out.lower()
