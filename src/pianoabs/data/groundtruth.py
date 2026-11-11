"""Ground-truth construction (README §5.4).

Notes are kept in a ``pandas.DataFrame`` with columns ``onset_s, offset_s, pitch,
velocity`` (+ ``evaluated`` after cropping), which is also the CSV format.

Steps
-----
1. **Sustain-pedal extension** (as in Onsets & Frames). If a key is released while
   the sustain pedal is down (CC64 >= 64), the string keeps sounding, so the
   note's offset is moved to the pedal release. A note is always cut at the next
   onset of the same key (re-striking damps the old vibration):

       offset' = min( release(offset) , next_onset_same_pitch )
       release(t) = end of the pedal interval containing t, or t if the pedal is up

2. **Cropping** to the window [t0, t0 + L):
   * notes with an onset inside the window are *evaluated* (onset metrics);
   * notes that started before t0 and still sound are kept only in the *real*
     GT (they are physically in a real recording) with ``evaluated = False``,
     onset clipped to 0; they count for frame metrics only;
   * all times are shifted by -t0 and offsets clipped at L.

3. **Synthetic GT** (C0–C2) contains only notes with an onset in the window,
   because only those notes are rendered (see :func:`make_excerpt_midi`).

Only CC64 (sustain) is used. Sostenuto (CC66) and una corda (CC67) are ignored
both in the GT and in the rendered MIDI, so audio and labels stay consistent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi

NOTE_COLUMNS = ["onset_s", "offset_s", "pitch", "velocity"]
GT_COLUMNS = NOTE_COLUMNS + ["evaluated"]
SUSTAIN_CC = 64


# ---------------------------------------------------------------- reading MIDI

def midi_to_notes(pm: pretty_midi.PrettyMIDI) -> pd.DataFrame:
    """All notes of all (non-drum) instruments, sorted by onset then pitch."""
    rows = [(n.start, n.end, n.pitch, n.velocity)
            for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    notes = pd.DataFrame(rows, columns=NOTE_COLUMNS)
    notes = notes.astype({"onset_s": float, "offset_s": float, "pitch": int, "velocity": int})
    return notes.sort_values(["onset_s", "pitch"], kind="stable").reset_index(drop=True)


def cc64_events(pm: pretty_midi.PrettyMIDI) -> list[tuple[float, int]]:
    """Sorted ``(time, value)`` sustain-pedal events of all instruments."""
    events = [(cc.time, cc.value) for inst in pm.instruments
              for cc in inst.control_changes if cc.number == SUSTAIN_CC]
    return sorted(events)


# ---------------------------------------------------------------- pedal

def sustain_intervals(events: list[tuple[float, int]], threshold: int = 64,
                      end_time: float = 0.0) -> list[tuple[float, float]]:
    """Turn CC64 events into ``[(down, up), ...]`` intervals where value >= threshold.

    Repeated "down" values (e.g. 70 then 100) do not start a new interval. A pedal
    still down at the end of the file is closed at ``end_time``.
    """
    intervals: list[tuple[float, float]] = []
    down_at = None
    for t, value in sorted(events):
        if value >= threshold and down_at is None:
            down_at = t
        elif value < threshold and down_at is not None:
            intervals.append((down_at, t))
            down_at = None
    if down_at is not None:
        intervals.append((down_at, max(end_time, down_at)))
    return intervals


def pedal_value_at(events: list[tuple[float, int]], t: float) -> int | None:
    """Value of the last CC64 event at or before time ``t`` (None if there is none)."""
    value = None
    for time, v in events:
        if time > t:
            break
        value = v
    return value


def pedal_down_mask(intervals: list[tuple[float, float]], times: np.ndarray) -> np.ndarray:
    """Boolean array: is the pedal down at each time (down <= t < up)?"""
    mask = np.zeros(len(times), dtype=bool)
    for down, up in intervals:
        mask |= (times >= down) & (times < up)
    return mask


def extend_notes_with_pedal(notes: pd.DataFrame,
                            intervals: list[tuple[float, float]]) -> pd.DataFrame:
    """Apply sustain-pedal extension (see module docstring, step 1)."""
    notes = notes.sort_values(["onset_s", "pitch"], kind="stable").reset_index(drop=True)
    onsets = notes["onset_s"].to_numpy()
    offsets = notes["offset_s"].to_numpy(copy=True)

    # (a) key released while the pedal is down -> sound until the pedal is released
    for i, off in enumerate(offsets):
        for down, up in intervals:
            if down <= off < up:
                offsets[i] = up
                break

    # (b) cut at the next onset of the same key
    for _, idx in notes.groupby("pitch").indices.items():
        idx = idx[np.argsort(onsets[idx], kind="stable")]
        for a, b in zip(idx[:-1], idx[1:]):
            offsets[a] = min(offsets[a], onsets[b])

    out = notes.copy()
    out["offset_s"] = offsets
    # Two onsets of the same key at the same instant would leave a zero-length note.
    return out[out["offset_s"] > out["onset_s"]].reset_index(drop=True)


def performance_notes(pm: pretty_midi.PrettyMIDI, threshold: int = 64) -> tuple[pd.DataFrame, list]:
    """Pedal-extended notes of a full performance and its sustain intervals."""
    notes = midi_to_notes(pm)
    end_time = float(max(notes["offset_s"].max() if len(notes) else 0.0, pm.get_end_time()))
    intervals = sustain_intervals(cc64_events(pm), threshold, end_time)
    return extend_notes_with_pedal(notes, intervals), intervals


# ---------------------------------------------------------------- cropping

def crop_notes(notes: pd.DataFrame, t0: float, length: float,
               keep_sounding_before: bool) -> pd.DataFrame:
    """Crop (pedal-extended) notes to [t0, t0 + length), see module docstring, step 2."""
    end = t0 + length
    onset, offset = notes["onset_s"], notes["offset_s"]
    in_window = (onset >= t0) & (onset < end)
    sounding_before = (onset < t0) & (offset > t0)
    keep = in_window | sounding_before if keep_sounding_before else in_window

    out = notes.loc[keep, NOTE_COLUMNS].copy()
    out["onset_s"] = (out["onset_s"] - t0).clip(lower=0.0)
    out["offset_s"] = (out["offset_s"] - t0).clip(upper=length)
    out["evaluated"] = in_window[keep].to_numpy()
    return out.sort_values(["onset_s", "pitch"], kind="stable").reset_index(drop=True)


def ground_truth(extended_notes: pd.DataFrame, t0: float, length: float) -> dict[str, pd.DataFrame]:
    """Both GT variants: ``synthetic`` (C0–C2) and ``real`` (C3)."""
    return {
        "synthetic": crop_notes(extended_notes, t0, length, keep_sounding_before=False),
        "real": crop_notes(extended_notes, t0, length, keep_sounding_before=True),
    }


# ---------------------------------------------------------------- excerpt MIDI for rendering

def make_excerpt_midi(pm: pretty_midi.PrettyMIDI, t0: float, length: float, tail: float,
                      threshold: int = 64, program: int = 0) -> pretty_midi.PrettyMIDI:
    """MIDI of the window, for FluidSynth (C1/C2), with times shifted by -t0.

    * Raw (not pedal-extended) notes with an onset in [t0, t0 + length): the
      synthesiser applies the pedal itself. Note ends are capped at the end of
      the release tail (the audio is trimmed there anyway).
    * CC64 events in (t0, t0 + length + tail), plus the pedal state at t0
      carried over as a CC64 event at time 0 if the pedal is down (README §5.4).
    """
    end, render_end = t0 + length, t0 + length + tail
    out = pretty_midi.PrettyMIDI(resolution=960, initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=program, name="piano")
    for src in pm.instruments:
        if src.is_drum:
            continue
        for n in src.notes:
            if t0 <= n.start < end:
                inst.notes.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch,
                                                   start=n.start - t0,
                                                   end=min(n.end, render_end) - t0))
    events = cc64_events(pm)
    value_at_t0 = pedal_value_at(events, t0)
    if value_at_t0 is not None and value_at_t0 >= threshold:
        inst.control_changes.append(pretty_midi.ControlChange(SUSTAIN_CC, value_at_t0, 0.0))
    for t, value in events:
        if t0 < t < render_end:
            inst.control_changes.append(pretty_midi.ControlChange(SUSTAIN_CC, value, t - t0))
    inst.notes.sort(key=lambda n: (n.start, n.pitch))
    out.instruments.append(inst)
    return out


# ---------------------------------------------------------------- piano rolls and statistics

def frame_times(length: float, frame_s: float) -> np.ndarray:
    """Frame sample times 0, frame_s, 2 frame_s, ... < length (625 frames for 10 s / 16 ms)."""
    return np.arange(int(round(length / frame_s))) * frame_s


def notes_to_roll(notes: pd.DataFrame, length: float, frame_s: float,
                  pitch_min: int = 21, pitch_max: int = 108) -> np.ndarray:
    """Binary piano roll (88, n_frames): frame t is on if onset <= t < offset."""
    times = frame_times(length, frame_s)
    roll = np.zeros((pitch_max - pitch_min + 1, len(times)), dtype=bool)
    for onset, offset, pitch in zip(notes["onset_s"], notes["offset_s"], notes["pitch"]):
        if not pitch_min <= pitch <= pitch_max:
            continue
        i0 = np.searchsorted(times, onset, side="left")
        i1 = np.searchsorted(times, offset, side="left")
        roll[int(pitch) - pitch_min, i0:i1] = True
    return roll


def mean_polyphony(notes: pd.DataFrame, length: float, frame_s: float) -> float:
    """Average number of sounding notes per frame (README §5.3)."""
    return float(notes_to_roll(notes, length, frame_s).sum(axis=0).mean())


def pedal_fraction(intervals: list[tuple[float, float]], t0: float, length: float,
                   frame_s: float) -> float:
    """Share of the window's frames with the sustain pedal down (README §5.3)."""
    return float(pedal_down_mask(intervals, t0 + frame_times(length, frame_s)).mean())


def write_notes_csv(notes: pd.DataFrame, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    notes.to_csv(path, index=False, float_format="%.6f", lineterminator="\n")


def read_notes_csv(path) -> pd.DataFrame:
    return pd.read_csv(path)
