"""Writing predictions in the common format (README §8.2). Deliberately free of torch,
so that runners in an isolated environment (Basic Pitch) can import it.

Every prediction is three files next to each other:
``{excerpt_id}.csv`` (onset_s, offset_s, pitch, velocity), ``.mid``, and ``.json``
(runtime, device and the settings that produced it).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pianoabs import paths
from pianoabs.data.groundtruth import write_notes_csv
from pianoabs.model.notes import notes_to_midi


def read_meta(csv_path: Path) -> dict | None:
    meta = Path(csv_path).with_suffix(".json")
    return json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else None


def write_prediction(cfg: dict, method: str, variant: str, condition: str, excerpt_id: str,
                     notes: pd.DataFrame, runtime_s: float, device: str,
                     extra: dict | None = None) -> Path:
    """Note CSV + MIDI + a small JSON with runtime and device (used by ``evaluate``)."""
    csv = paths.prediction_path(cfg, method, variant, condition, excerpt_id, ".csv")
    write_notes_csv(notes, csv)
    notes_to_midi(notes, csv.with_suffix(".mid"))
    csv.with_suffix(".json").write_text(json.dumps(
        {"runtime_s": runtime_s, "device": device, **(extra or {})}), encoding="utf-8")
    return csv


def is_up_to_date(csv_path: Path, setting: dict) -> bool:
    """True if a prediction exists and was made with exactly ``setting``."""
    meta = read_meta(csv_path)
    return Path(csv_path).exists() and meta is not None and all(
        meta.get(k) == v for k, v in setting.items())
