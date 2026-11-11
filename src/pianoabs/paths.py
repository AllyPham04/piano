"""Single source of truth for the file layout of README §8.

All modules build file names through these functions, so the layout can be read
(and changed) in one place. Every path is relative to the repository root.

Deviation agreed in the plan: predictions get a ``{variant}`` level
(``outputs/predictions/{method}/{variant}/{condition}/``) so that ablations of
"ours" do not overwrite each other. Baselines use variant ``default``.
"""

from __future__ import annotations

from pathlib import Path


def data_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["data_dir"])


def raw_dir(cfg: dict) -> Path:
    """Downloaded MAESTRO MIDI zip / metadata (git-ignored)."""
    return data_dir(cfg) / "raw"


def manifest_path(cfg: dict, which: str) -> Path:
    """``which`` is ``test`` or ``val`` -> data/manifests/excerpts_{which}.csv."""
    return data_dir(cfg) / "manifests" / f"excerpts_{which}.csv"


def gt_path(cfg: dict, kind: str, excerpt_id: str) -> Path:
    """``kind`` is ``synthetic`` (C0–C2) or ``real`` (C3), see README §5.4."""
    assert kind in ("synthetic", "real"), kind
    return data_dir(cfg) / "gt" / kind / f"{excerpt_id}.csv"


def excerpt_midi_path(cfg: dict, excerpt_id: str) -> Path:
    """Cropped excerpt MIDI (notes with onset in the window + pedal), input to FluidSynth."""
    return data_dir(cfg) / "excerpt_midi" / f"{excerpt_id}.mid"


def audio_path(cfg: dict, condition: str, excerpt_id: str) -> Path:
    return data_dir(cfg) / "audio" / condition / f"{excerpt_id}.wav"


def _outputs(cfg: dict) -> Path:
    return Path(cfg["paths"]["outputs_dir"])


def prediction_path(cfg: dict, method: str, variant: str, condition: str,
                    excerpt_id: str, ext: str = ".csv") -> Path:
    return _outputs(cfg) / "predictions" / method / variant / condition / f"{excerpt_id}{ext}"


def activation_path(cfg: dict, variant: str, condition: str, excerpt_id: str) -> Path:
    return _outputs(cfg) / "activations" / variant / condition / f"{excerpt_id}.npy"


def params_path(cfg: dict, variant: str, condition: str, excerpt_id: str) -> Path:
    return _outputs(cfg) / "params" / variant / condition / f"{excerpt_id}.json"


def results_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["results_dir"])


def figures_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["figures_dir"])
