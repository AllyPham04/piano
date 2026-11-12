"""ByteDance high-resolution piano transcription (Kong et al., 2021), README §4.

Supervised and piano-specific: trained on the MAESTRO (v2) *train* split, so on
MAESTRO-derived material it serves as an upper bound, not a fair competitor.

Package: ``piano_transcription_inference`` (``PianoTranscription``), default
settings. Two practical details:

* the package downloads its checkpoint with ``wget``, which does not exist on
  Windows; we download it with Python and pass ``checkpoint_path`` instead;
* it predicts key releases and sustain-pedal events separately. Our ground
  truth is pedal-extended (§5.4), so by default we apply the *predicted* pedal
  to the predicted notes with the same rule as for the GT
  (``apply_predicted_pedal``). This only moves offsets; onsets are unchanged.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from pianoabs.data.groundtruth import NOTE_COLUMNS, extend_notes_with_pedal
from pianoabs.data.maestro import download


def ensure_checkpoint(cfg: dict) -> Path:
    b = cfg["baselines"]["bytedance"]
    path = Path(b["checkpoint_path"])
    if not path.exists() or path.stat().st_size < 1.6e8:   # same size check as the package
        if path.exists():
            path.unlink()
        download(b["checkpoint_url"], path)
    return path


def events_to_notes(note_events: list[dict], pedal_events: list[dict],
                    apply_pedal: bool) -> pd.DataFrame:
    """ByteDance event dicts -> our note table, optionally pedal-extended."""
    notes = pd.DataFrame(
        [(e["onset_time"], e["offset_time"], int(e["midi_note"]), int(e["velocity"]))
         for e in note_events], columns=NOTE_COLUMNS)
    notes = notes.astype({"onset_s": float, "offset_s": float, "pitch": int, "velocity": int})
    if apply_pedal and len(notes):
        intervals = sorted((float(p["onset_time"]), float(p["offset_time"])) for p in pedal_events)
        notes = extend_notes_with_pedal(notes, intervals)
    return notes.sort_values(["onset_s", "pitch"], kind="stable").reset_index(drop=True)


class BytedanceTranscriber:
    """Loads the model once; ``transcribe`` returns (notes, runtime_s)."""

    def __init__(self, cfg: dict, device: str):
        import torch
        from piano_transcription_inference import PianoTranscription

        self.cfg = cfg
        self.device = device
        self.model = PianoTranscription(device=torch.device(device),
                                        checkpoint_path=str(ensure_checkpoint(cfg)))

    def transcribe(self, audio: np.ndarray) -> tuple[pd.DataFrame, float]:
        start = time.perf_counter()
        out = self.model.transcribe(audio.astype(np.float32), None)
        runtime = time.perf_counter() - start
        notes = events_to_notes(out["est_note_events"], out["est_pedal_events"],
                                self.cfg["baselines"]["bytedance"]["apply_predicted_pedal"])
        return notes, runtime
