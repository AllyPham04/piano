"""Pedal extension, cropping and excerpt MIDI on hand-made toy MIDI (README §5.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pretty_midi
import pytest

from pianoabs.data.groundtruth import (crop_notes, extend_notes_with_pedal, ground_truth,
                                       make_excerpt_midi, mean_polyphony, midi_to_notes,
                                       notes_to_roll, pedal_fraction, pedal_value_at,
                                       performance_notes, sustain_intervals)


def toy_midi(notes, pedal=()):
    """notes: (pitch, start, end[, velocity]); pedal: (time, value) CC64 events."""
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    for n in notes:
        pitch, start, end = n[:3]
        inst.notes.append(pretty_midi.Note(velocity=n[3] if len(n) > 3 else 80,
                                           pitch=pitch, start=start, end=end))
    for t, v in pedal:
        inst.control_changes.append(pretty_midi.ControlChange(64, v, t))
    pm.instruments.append(inst)
    return pm


def offsets_by(notes: pd.DataFrame) -> list[tuple[int, float, float]]:
    return [(int(p), round(a, 6), round(b, 6))
            for p, a, b in zip(notes["pitch"], notes["onset_s"], notes["offset_s"])]


# ---------------------------------------------------------------- sustain intervals

def test_sustain_intervals_threshold_and_repeated_down():
    events = [(1.0, 64), (1.5, 100), (2.0, 63), (3.0, 127)]  # 64 counts as down
    assert sustain_intervals(events, 64, end_time=5.0) == [(1.0, 2.0), (3.0, 5.0)]


def test_pedal_value_at():
    events = [(1.0, 100), (2.0, 0)]
    assert pedal_value_at(events, 0.5) is None
    assert pedal_value_at(events, 1.0) == 100
    assert pedal_value_at(events, 1.9) == 100
    assert pedal_value_at(events, 2.0) == 0


# ---------------------------------------------------------------- pedal extension

def test_release_under_pedal_extends_to_pedal_up():
    pm = toy_midi([(60, 0.0, 0.5)], pedal=[(0.2, 127), (2.0, 0)])
    notes, _ = performance_notes(pm)
    assert offsets_by(notes) == [(60, 0.0, 2.0)]


def test_release_before_pedal_down_is_unchanged():
    pm = toy_midi([(60, 0.0, 0.1)], pedal=[(0.2, 127), (2.0, 0)])
    notes, _ = performance_notes(pm)
    assert offsets_by(notes) == [(60, 0.0, 0.1)]


def test_key_held_past_pedal_release_keeps_own_offset():
    pm = toy_midi([(60, 0.0, 3.0)], pedal=[(0.2, 127), (1.0, 0)])
    notes, _ = performance_notes(pm)
    assert offsets_by(notes) == [(60, 0.0, 3.0)]


def test_restrike_same_key_cuts_sustained_note():
    pm = toy_midi([(60, 0.0, 0.5), (60, 1.0, 1.2), (64, 0.1, 0.3)],
                  pedal=[(0.2, 127), (3.0, 0)])
    notes, _ = performance_notes(pm)
    assert offsets_by(notes) == [(60, 0.0, 1.0),   # cut at the re-strike
                                 (64, 0.1, 3.0),   # released under the pedal
                                 (60, 1.0, 3.0)]   # extended to pedal up


def test_pedal_still_down_at_end_of_file():
    pm = toy_midi([(60, 0.0, 0.5), (62, 0.0, 4.0)], pedal=[(0.2, 127)])
    notes, intervals = performance_notes(pm)
    assert intervals == [(0.2, 4.0)]
    assert offsets_by(notes)[0] == (60, 0.0, 4.0)


def test_extension_without_pedal_is_identity():
    notes = midi_to_notes(toy_midi([(60, 0.0, 0.5), (67, 0.2, 0.9)]))
    pd.testing.assert_frame_equal(extend_notes_with_pedal(notes, []), notes)


# ---------------------------------------------------------------- cropping

@pytest.fixture
def window_notes():
    # t0 = 10, L = 10 -> window [10, 20)
    return pd.DataFrame({
        "onset_s":  [5.0, 9.0, 10.0, 15.0, 19.9, 20.0],
        "offset_s": [8.0, 12.0, 11.0, 25.0, 20.5, 21.0],
        "pitch":    [40, 41, 42, 43, 44, 45],
        "velocity": [80] * 6,
    })


def test_crop_real_keeps_notes_sounding_before_t0(window_notes):
    out = crop_notes(window_notes, t0=10.0, length=10.0, keep_sounding_before=True)
    got = [(p, round(a, 6), round(b, 6), e) for p, a, b, e in
           zip(out.pitch, out.onset_s, out.offset_s, out.evaluated)]
    assert got == [
        (41, 0.0, 2.0, False),   # started before t0, still sounding: frames only
        (42, 0.0, 1.0, True),    # onset exactly at t0 is inside
        (43, 5.0, 10.0, True),   # offset clipped at L
        (44, 9.9, 10.0, True),
    ]  # 40 ended before t0; 45 starts at the window end (excluded)


def test_crop_synthetic_drops_notes_before_t0(window_notes):
    out = crop_notes(window_notes, t0=10.0, length=10.0, keep_sounding_before=False)
    assert list(out.pitch) == [42, 43, 44]
    assert out.evaluated.all()


def test_ground_truth_variants_agree_on_evaluated_notes(window_notes):
    gt = ground_truth(window_notes, 10.0, 10.0)
    real_eval = gt["real"][gt["real"].evaluated].reset_index(drop=True)
    pd.testing.assert_frame_equal(real_eval, gt["synthetic"])


# ---------------------------------------------------------------- excerpt MIDI

def test_excerpt_midi_carries_pedal_state_and_shifts_times():
    pm = toy_midi([(50, 8.0, 12.0), (60, 10.5, 11.0), (62, 19.0, 25.0), (64, 21.0, 22.0)],
                  pedal=[(9.0, 127), (12.0, 0), (13.0, 100), (30.0, 0)])
    out = make_excerpt_midi(pm, t0=10.0, length=10.0, tail=1.0)
    notes = out.instruments[0].notes
    assert [(n.pitch, round(n.start, 3), round(n.end, 3)) for n in notes] == [
        (60, 0.5, 1.0),
        (62, 9.0, 11.0),  # end capped at the end of the tail (10 + 1 s)
    ]  # 50 started before t0 and 64 after the window: not rendered
    ccs = [(c.number, round(c.time, 3), c.value) for c in out.instruments[0].control_changes]
    assert ccs == [(64, 0.0, 127), (64, 2.0, 0), (64, 3.0, 100)]


def test_excerpt_midi_no_pedal_event_when_pedal_up_at_t0():
    pm = toy_midi([(60, 10.5, 11.0)], pedal=[(1.0, 127), (2.0, 0)])
    out = make_excerpt_midi(pm, t0=10.0, length=10.0, tail=1.0)
    assert out.instruments[0].control_changes == []


def test_excerpt_midi_roundtrip_through_file(tmp_path):
    pm = toy_midi([(60, 10.5, 11.0, 90)], pedal=[(9.0, 127)])
    path = tmp_path / "x.mid"
    make_excerpt_midi(pm, 10.0, 10.0, 1.0).write(str(path))
    back = pretty_midi.PrettyMIDI(str(path))
    n = back.instruments[0].notes[0]
    assert (n.pitch, n.velocity) == (60, 90)
    assert n.start == pytest.approx(0.5, abs=2e-3) and n.end == pytest.approx(1.0, abs=2e-3)


# ---------------------------------------------------------------- rolls and statistics

def test_roll_polyphony_and_pedal_fraction():
    notes = pd.DataFrame({"onset_s": [0.0, 0.5], "offset_s": [1.0, 1.0],
                          "pitch": [21, 108], "velocity": [80, 80]})
    roll = notes_to_roll(notes, length=1.0, frame_s=0.25)  # frames at 0, .25, .5, .75
    assert roll.shape == (88, 4)
    assert roll[0].tolist() == [True] * 4
    assert roll[87].tolist() == [False, False, True, True]
    assert mean_polyphony(notes, 1.0, 0.25) == pytest.approx(1.5)
    assert pedal_fraction([(0.4, 0.9)], t0=0.0, length=1.0, frame_s=0.25) == pytest.approx(0.5)


def test_roll_offset_is_exclusive():
    notes = pd.DataFrame({"onset_s": [0.25], "offset_s": [0.5], "pitch": [60], "velocity": [80]})
    assert notes_to_roll(notes, 1.0, 0.25)[60 - 21].tolist() == [False, True, False, False]
    assert np.sum(notes_to_roll(notes.iloc[:0], 1.0, 0.25)) == 0
