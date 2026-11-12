"""Transcription metrics on the 10 s window (README §6), built on ``mir_eval``.

* **Onset P/R/F1** (primary): onset within ±50 ms and same pitch (±50 cents),
  offsets ignored; only GT notes with ``evaluated = True`` (onset in window).
* **Onset+offset F1**: additionally the offset must lie within
  max(0.2 · reference duration, 50 ms).
* **Frame F1**: 16 ms grid, binary piano rolls; the reference contains *all*
  sounding GT notes (including ``evaluated = False`` ones in the real GT).
* **Octave error rate** = (# false-positive predicted onsets within ±50 ms of an
  evaluated GT onset whose pitch differs by ±12, ±24 or ±36) / (# false
  positives). **Harmonic error rate**: same with pred = GT + 19 or + 28
  semitones (3rd and 5th harmonic). NaN when there are no false positives.

Predictions are made on excerpt + tail (11 s) and cropped to the window first.
"""

from __future__ import annotations

import warnings

import mir_eval
import numpy as np
import pandas as pd

from pianoabs.data.groundtruth import notes_to_roll


def midi_to_hz(pitch) -> np.ndarray:
    return 440.0 * 2.0 ** ((np.asarray(pitch, dtype=float) - 69.0) / 12.0)


def crop_predictions(notes: pd.DataFrame, excerpt_s: float) -> pd.DataFrame:
    """Keep predicted notes with onset < excerpt_s; clip offsets to the window."""
    out = notes[notes["onset_s"] < excerpt_s].copy()
    out["offset_s"] = out["offset_s"].clip(upper=excerpt_s)
    return out[out["offset_s"] > out["onset_s"]].reset_index(drop=True)


def evaluated_reference(gt: pd.DataFrame) -> pd.DataFrame:
    """GT notes that count for onset metrics (onset inside the window)."""
    return gt[gt["evaluated"].astype(bool)].reset_index(drop=True)


def _intervals_pitches(notes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    intervals = notes[["onset_s", "offset_s"]].to_numpy(dtype=float).reshape(-1, 2)
    return intervals, midi_to_hz(notes["pitch"].to_numpy())


def _prf(ref: pd.DataFrame, est: pd.DataFrame, onset_tolerance: float,
         offset_ratio: float | None, offset_min_tolerance: float = 0.05) -> tuple[float, float, float]:
    if len(ref) == 0 or len(est) == 0:
        return 0.0, 0.0, 0.0
    ref_i, ref_p = _intervals_pitches(ref)
    est_i, est_p = _intervals_pitches(est)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_i, ref_p, est_i, est_p, onset_tolerance=onset_tolerance,
            offset_ratio=offset_ratio, offset_min_tolerance=offset_min_tolerance)
    return float(p), float(r), float(f)


def onset_prf(gt: pd.DataFrame, pred: pd.DataFrame, onset_tolerance: float = 0.05) -> dict:
    """Note-onset P/R/F1 on evaluated GT notes (``pred`` already cropped)."""
    ref = evaluated_reference(gt)
    p, r, f = _prf(ref, pred, onset_tolerance, None)
    return {"onset_P": p, "onset_R": r, "onset_F1": f, "n_ref": len(ref), "n_pred": len(pred)}


def frame_f1(gt: pd.DataFrame, pred: pd.DataFrame, excerpt_s: float, frame_s: float) -> float:
    """Multi-pitch F1 = 2TP / (2TP + FP + FN) over all (key, frame) cells of the window."""
    ref = notes_to_roll(gt, excerpt_s, frame_s)
    est = notes_to_roll(pred, excerpt_s, frame_s)
    tp = np.sum(ref & est)
    denom = 2 * tp + np.sum(~ref & est) + np.sum(ref & ~est)
    return float(2 * tp / denom) if denom else 1.0


def false_positive_errors(gt: pd.DataFrame, pred: pd.DataFrame, onset_tolerance: float,
                          octave_intervals: list[int], harmonic_intervals: list[int]) -> dict:
    """Count false-positive onsets that are octave / harmonic errors (see module docstring)."""
    ref = evaluated_reference(gt)
    n_pred = len(pred)
    matched: set[int] = set()
    if len(ref) and n_pred:
        ref_i, ref_p = _intervals_pitches(ref)
        est_i, est_p = _intervals_pitches(pred)
        pairs = mir_eval.transcription.match_notes(ref_i, ref_p, est_i, est_p,
                                                   onset_tolerance=onset_tolerance,
                                                   offset_ratio=None)
        matched = {e for _, e in pairs}
    fp = [i for i in range(n_pred) if i not in matched]
    ref_on, ref_pitch = ref["onset_s"].to_numpy(), ref["pitch"].to_numpy()
    n_oct = n_harm = 0
    for i in fp:
        near = np.abs(ref_on - pred["onset_s"].iloc[i]) <= onset_tolerance
        diffs = set((pred["pitch"].iloc[i] - ref_pitch[near]).tolist())
        n_oct += bool(diffs & set(octave_intervals))
        n_harm += bool(diffs & set(harmonic_intervals))
    n_fp = len(fp)
    return {"n_fp": n_fp, "n_octave_fp": n_oct, "n_harmonic_fp": n_harm,
            "octave_err_rate": n_oct / n_fp if n_fp else float("nan"),
            "harmonic_err_rate": n_harm / n_fp if n_fp else float("nan")}


def excerpt_metrics(gt: pd.DataFrame, pred: pd.DataFrame, cfg: dict) -> dict:
    """All per-excerpt metrics of README §6 for one prediction (cropped here)."""
    e, excerpt_s = cfg["eval"], cfg["data"]["excerpt_s"]
    pred = crop_predictions(pred, excerpt_s)
    out = onset_prf(gt, pred, e["onset_tolerance"])
    _, _, out["onoff_F1"] = _prf(evaluated_reference(gt), pred, e["onset_tolerance"],
                                 e["offset_ratio"], e["offset_min_tolerance"])
    out["frame_F1"] = frame_f1(gt, pred, excerpt_s, e["frame_s"])
    out.update(false_positive_errors(gt, pred, e["onset_tolerance"],
                                     e["octave_intervals"], e["harmonic_intervals"]))
    return out
