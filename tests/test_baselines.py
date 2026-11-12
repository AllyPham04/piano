"""Conversion of supervised-baseline outputs to the common note format (README §4)."""

from __future__ import annotations

import pytest

from pianoabs.baselines.basic_pitch_runner import events_to_notes as bp_events
from pianoabs.baselines.bytedance_runner import events_to_notes as bd_events


def test_bytedance_events_with_and_without_pedal():
    notes = [{"onset_time": 0.5, "offset_time": 0.8, "midi_note": 60, "velocity": 70},
             {"onset_time": 0.1, "offset_time": 0.3, "midi_note": 64, "velocity": 90}]
    pedal = [{"onset_time": 0.6, "offset_time": 2.0}]
    raw = bd_events(notes, pedal, apply_pedal=False)
    assert raw[["onset_s", "offset_s", "pitch"]].values.tolist() == [[0.1, 0.3, 64], [0.5, 0.8, 60]]
    ext = bd_events(notes, pedal, apply_pedal=True)
    # key 60 released at 0.8 while the predicted pedal is down -> sounds until 2.0
    assert ext[["onset_s", "offset_s", "pitch"]].values.tolist() == [[0.1, 0.3, 64], [0.5, 2.0, 60]]
    assert list(ext.columns) == ["onset_s", "offset_s", "pitch", "velocity"]
    assert bd_events([], pedal, apply_pedal=True).empty


def test_basic_pitch_events_velocity_from_amplitude():
    events = [(1.0, 1.5, 62, 0.5, [0, 1]), (0.2, 0.4, 70, 0.0, None), (0.3, 0.9, 50, 1.2, [])]
    notes = bp_events(events)
    assert notes["onset_s"].tolist() == pytest.approx([0.2, 0.3, 1.0])
    assert notes["pitch"].tolist() == [70, 50, 62]
    assert notes["velocity"].tolist() == [1, 127, 64]   # clipped to 1..127, 0.5 -> 64
