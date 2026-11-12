r"""From activations to notes (README §3.5).

1. Normalise A by its 99th percentile in the excerpt (so thresholds are
   comparable across excerpts with different loudness).
2. **Onset** at (p, t) when A[p,t] > θ_on and A[p,t] - A[p,t-1] > θ_rise
   (piano notes start with a rise in energy). Rule for keys that are already
   sounding: a *re-onset* also needs the current rising run (consecutive frames
   with ΔA > θ_rise) to have risen by at least ``reonset_rise_ratio · θ_on``
   since it started. Without this, with θ_rise = 0, every small wiggle of a
   sustained note would start a new note. The ratio is tuned on validation
   with the other thresholds, from {1.0, 0.5, 0.25, 0.1}: the E0 diagnosis
   showed that 38% of onsets re-strike a pedal-sustained key, and a fixed
   ratio of 1.0 missed most of them.
3. **Offset** when A falls below θ_off = θ_on / 2 (hysteresis), or at a new
   onset of the same key.
4. Merge same-pitch notes separated by a gap shorter than 30 ms, then drop
   notes shorter than 30 ms.
5. Velocity: peak normalised activation in the note, clipped to [0, 1], mapped
   linearly to 1..127.

Times: frame t is centred at t · hop / sr (STFT with center=True).
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi

NOTE_COLUMNS = ["onset_s", "offset_s", "pitch", "velocity"]


def normalise_activations(A: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """A / (its ``percentile``-th percentile); a silent excerpt stays all-zero."""
    scale = float(np.percentile(A, percentile))
    return A / scale if scale > 0 else np.zeros_like(A)


def _key_events(a: np.ndarray, theta_on: float, theta_rise: float, theta_off: float,
                reonset_rise: float) -> list[tuple[int, int, float]]:
    """Notes of one key as (onset_frame, offset_frame, peak) from a normalised row."""
    notes = []
    active, onset, peak = False, 0, 0.0
    run_start_value = 0.0           # value just before the current rising run
    prev = 0.0                      # A[t-1], with A[-1] = 0
    for t, value in enumerate(a):
        rising = value - prev > theta_rise
        if not rising:
            run_start_value = value
        if rising and value > theta_on:
            if not active:
                active, onset, peak = True, t, value
            elif value - run_start_value >= reonset_rise:
                notes.append((onset, t, peak))       # cut the old note at the re-onset
                onset, peak = t, value
                run_start_value = value               # one onset per rising run
        if active:
            peak = max(peak, value)
            if value < theta_off:
                notes.append((onset, t, peak))
                active = False
        prev = value
    if active:
        notes.append((onset, len(a), peak))
    return notes


def extract_notes(A: np.ndarray, frame_s: float, theta_on: float, theta_rise: float,
                  theta_off_ratio: float = 0.5, reonset_rise_ratio: float = 1.0,
                  min_note_s: float = 0.03, merge_gap_s: float = 0.03,
                  percentile: float = 99.0, pitch_min: int = 21) -> pd.DataFrame:
    """Note list (onset_s, offset_s, pitch, velocity) from raw activations A (88, T)."""
    An = normalise_activations(A, percentile)
    theta_off = theta_on * theta_off_ratio
    rows = []
    for key, row in enumerate(An):
        events = _key_events(row, theta_on, theta_rise, theta_off, reonset_rise_ratio * theta_on)
        merged: list[list] = []
        for onset, offset, peak in events:
            gap = (onset - merged[-1][1]) * frame_s if merged else None
            if merged and 0 < gap < merge_gap_s:
                merged[-1][1] = offset
                merged[-1][2] = max(merged[-1][2], peak)
            else:
                merged.append([onset, offset, peak])
        for onset, offset, peak in merged:
            if (offset - onset) * frame_s < min_note_s:
                continue
            velocity = int(round(1 + 126 * min(max(peak, 0.0), 1.0)))
            rows.append((onset * frame_s, offset * frame_s, key + pitch_min, velocity))
    notes = pd.DataFrame(rows, columns=NOTE_COLUMNS)
    return notes.sort_values(["onset_s", "pitch"], kind="stable").reset_index(drop=True)


def extract_with_config(A: np.ndarray, cfg: dict, theta_on: float, theta_rise: float,
                        reonset_rise_ratio: float) -> pd.DataFrame:
    n = cfg["notes"]
    return extract_notes(A, cfg["features"]["hop"] / cfg["data"]["sample_rate"], theta_on,
                         theta_rise, n["theta_off_ratio"], reonset_rise_ratio,
                         n["min_note_s"], n["merge_gap_s"], n["percentile"],
                         cfg["model"]["pitch_min"])


def threshold_grid(cfg: dict) -> list[tuple[float, float, float]]:
    """All (θ_on, θ_rise, re-onset ratio) triples, in a fixed order.

    Ties in tuning go to the first entry, so for equal F1 the smaller θ_on and
    the strictest re-onset ratio (1.0, listed first) are preferred.
    """
    n = cfg["notes"]
    return list(product(n["theta_on_grid"], n["theta_rise_grid"], n["reonset_rise_ratio_grid"]))


def notes_to_midi(notes: pd.DataFrame, path: str | Path, program: int = 0) -> None:
    """Write a note list as a single-track MIDI file."""
    pm = pretty_midi.PrettyMIDI(resolution=960, initial_tempo=120.0)
    inst = pretty_midi.Instrument(program=program, name="transcription")
    for onset, offset, pitch, velocity in zip(notes["onset_s"], notes["offset_s"],
                                              notes["pitch"], notes["velocity"]):
        inst.notes.append(pretty_midi.Note(velocity=int(velocity), pitch=int(pitch),
                                           start=float(onset), end=float(offset)))
    pm.instruments.append(inst)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))
