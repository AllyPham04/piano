"""MAESTRO v3.0.0: download the MIDI-only archive and metadata, load performances (README §5.1).

Only the ~56 MB MIDI zip and the metadata CSV are downloaded. The 101 GB archive
with audio is never touched; real audio for C3 comes from the HF mirror (§5.2).
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import pretty_midi

from pianoabs import paths

MAESTRO_DIRNAME = "maestro-v3.0.0"  # top-level folder inside the MIDI zip


def download(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest`` unless it already exists (written via a .part file)."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f)
    tmp.replace(dest)
    return dest


def _safe_extract(zip_path: Path, out_dir: Path) -> None:
    """Extract a zip, refusing members that would land outside ``out_dir``."""
    out_dir = out_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = (out_dir / member).resolve()
            if out_dir not in target.parents and target != out_dir:
                raise ValueError(f"unsafe path in zip: {member}")
        zf.extractall(out_dir)


def ensure_maestro(cfg: dict) -> tuple[pd.DataFrame, Path]:
    """Download (once) and return ``(metadata, midi_root)``.

    ``midi_root / row.midi_filename`` is the MIDI file of a metadata row.
    """
    raw = paths.raw_dir(cfg)
    data_cfg = cfg["data"]
    csv_path = download(data_cfg["metadata_csv_url"], raw / "maestro-v3.0.0.csv")
    midi_root = raw / MAESTRO_DIRNAME
    if not midi_root.exists():
        zip_path = download(data_cfg["midi_zip_url"], raw / "maestro-v3.0.0-midi.zip")
        _safe_extract(zip_path, raw)
    return load_metadata(csv_path), midi_root


def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    """Metadata with columns canonical_composer, canonical_title, split, year,
    midi_filename, audio_filename, duration. Sorted by midi_filename so that
    iteration order (and therefore sampling) never depends on file order."""
    meta = pd.read_csv(csv_path)
    expected = {"canonical_composer", "canonical_title", "split", "year",
                "midi_filename", "audio_filename", "duration"}
    missing = expected - set(meta.columns)
    if missing:
        raise ValueError(f"metadata CSV is missing columns {missing}")
    return meta.sort_values("midi_filename").reset_index(drop=True)


def load_performance(midi_root: Path, midi_filename: str) -> pretty_midi.PrettyMIDI:
    return pretty_midi.PrettyMIDI(str(midi_root / midi_filename))
