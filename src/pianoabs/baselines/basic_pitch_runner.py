"""Spotify Basic Pitch (Bittner et al., 2022), README §4: supervised, instrument-agnostic.

Default settings: ``predict(audio_path, model)`` -> note events
``(start_s, end_s, pitch, amplitude, pitch_bends)``; velocity = round(127 · amplitude).

Basic Pitch may pull in TensorFlow, which can clash with PyTorch, so this
module imports **no torch code** and can run in its own environment::

    <basic-pitch python> -m pianoabs.baselines.basic_pitch_runner \
        --config configs/default.yaml --condition C1 --manifest test

(``pip install -e . --no-deps`` plus pandas, pyyaml, pretty_midi, soundfile,
tqdm in that environment.) It writes predictions in the common format.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import time

import numpy as np
import pandas as pd

from pianoabs import paths
from pianoabs.config import apply_overrides, load_config, parse_set_args
from pianoabs.data.groundtruth import NOTE_COLUMNS
from pianoabs.data.sampling import load_manifest
from pianoabs.predictions import is_up_to_date, write_prediction

METHOD, VARIANT = "basic_pitch", "default"


def events_to_notes(note_events) -> pd.DataFrame:
    rows = [(float(s), float(e), int(p), int(np.clip(round(127 * float(a)), 1, 127)))
            for s, e, p, a, *_ in note_events]
    notes = pd.DataFrame(rows, columns=NOTE_COLUMNS)
    return notes.sort_values(["onset_s", "pitch"], kind="stable").reset_index(drop=True)


def run(cfg: dict, condition: str, which: str, max_excerpts: int | None = None) -> int:
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model, predict

    model = Model(ICASSP_2022_MODEL_PATH)
    setting = {"method": METHOD, "basic_pitch_version": importlib.metadata.version("basic-pitch")}
    written = 0
    for eid in load_manifest(paths.manifest_path(cfg, which), max_excerpts)["excerpt_id"]:
        csv = paths.prediction_path(cfg, METHOD, VARIANT, condition, eid, ".csv")
        if is_up_to_date(csv, setting):
            continue
        start = time.perf_counter()
        _, _, note_events = predict(str(paths.audio_path(cfg, condition, eid)), model)
        runtime = time.perf_counter() - start
        write_prediction(cfg, METHOD, VARIANT, condition, eid, events_to_notes(note_events),
                         runtime, "cpu", setting)
        written += 1
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--condition", default="C1")
    p.add_argument("--manifest", default="test", choices=["val", "test"])
    p.add_argument("--set", action="append")
    a = p.parse_args()
    cfg = apply_overrides(load_config(a.config), parse_set_args(a.set))
    n = run(cfg, a.condition, a.manifest, cfg["run"]["max_excerpts"])
    print(f"basic_pitch: wrote {n} new prediction files ({a.condition}, {a.manifest})")


if __name__ == "__main__":
    main()
