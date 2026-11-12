"""Condition C3: real MAESTRO recordings from the Hugging Face mirror (README §5.2).

``B-K/MAESTRO-test`` (inspected 2026-11-12, see scripts/inspect_hf_mirror.py):
177 rows in 9 Parquet shards, one row group of 19–20 rows each, columns
``canonical_composer, canonical_title, split, year, midi (bytes),
audio (struct: bytes, path), duration``. There is **no MIDI file name**, so rows
are matched to MAESTRO metadata by the **md5 of the MIDI bytes** against the
local MIDI files of the test split.

Audio is **streamed**: the shards are read through ``HfFileSystem`` one row
(batch) at a time; only the rows of manifest excerpts are decoded, cropped to
the window [t0, t0 + excerpt + tail), converted to 16 kHz mono, peak-normalised
and written as WAV. Nothing else is kept on disk.

The real recording contains notes struck before t0, so C3 is scored against
the *real* GT (README §5.4).
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from pianoabs import paths
from pianoabs.audio import fix_length, peak_normalize, write_wav

REPO = "datasets/{repo_id}/data/*.parquet"


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def local_midi_md5(meta: pd.DataFrame, midi_root: Path, split: str = "test") -> dict[str, str]:
    """md5 of each local MAESTRO MIDI file of ``split`` -> midi_filename."""
    rows = meta[meta["split"] == split]
    return {md5((midi_root / f).read_bytes()): f for f in rows["midi_filename"]}


def shard_files(cfg: dict) -> list[str]:
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    return sorted(fs.glob(REPO.format(repo_id=cfg["data"]["hf_real"]["repo_id"])))


def index_mirror(cfg: dict, md5_to_file: dict[str, str]) -> pd.DataFrame:
    """One row per mirror row: shard, row index, md5, matched midi_filename, duration.

    Reads only the small ``midi`` and ``duration`` columns (HTTP range requests).
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    rows = []
    for shard in shard_files(cfg):
        with fs.open(shard, "rb", block_size=1 << 20) as fh:
            table = pq.ParquetFile(fh).read(columns=["midi", "duration"])
        for i, (midi, duration) in enumerate(zip(table.column("midi").to_pylist(),
                                                 table.column("duration").to_pylist())):
            h = md5(midi)
            rows.append({"shard": shard, "row": i, "md5": h, "duration": duration,
                         "midi_filename": md5_to_file.get(h)})
    return pd.DataFrame(rows)


def crop_to_window(audio: np.ndarray, sr: int, t0: float, cfg: dict) -> np.ndarray:
    """[t0, t0 + excerpt + tail) at the original rate -> 16 kHz mono, fixed length, peak 0.9."""
    import librosa

    d = cfg["data"]
    length = d["excerpt_s"] + d["tail_s"]
    seg = audio[int(round(t0 * sr)): int(round((t0 + length) * sr))]
    if seg.ndim == 2:
        seg = seg.mean(axis=1)
    seg = librosa.resample(seg.astype(np.float32), orig_sr=sr, target_sr=d["sample_rate"])
    seg = fix_length(seg, int(round(length * d["sample_rate"])))
    return peak_normalize(seg, d["peak_norm"])


def fetch_excerpts(cfg: dict, index: pd.DataFrame, manifest: pd.DataFrame,
                   condition: str = "C3") -> dict[str, str]:
    """Stream the needed shards row by row and write the C3 WAV of every excerpt.

    Returns {excerpt_id: status}. Existing WAVs are skipped (idempotent).
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    status: dict[str, str] = {}
    todo = manifest[[not paths.audio_path(cfg, condition, e).exists()
                     for e in manifest["excerpt_id"]]]
    for eid in set(manifest["excerpt_id"]) - set(todo["excerpt_id"]):
        status[eid] = "exists"
    wanted = todo.merge(index, on="midi_filename", how="left")
    for eid in wanted[wanted["shard"].isna()]["excerpt_id"]:
        status[eid] = "not in mirror"
    wanted = wanted.dropna(subset=["shard"])

    fs = HfFileSystem()
    for shard, group in wanted.groupby("shard"):
        by_row = {int(r.row): r for r in group.itertuples(index=False)}
        with fs.open(shard, "rb", block_size=8 << 20) as fh:
            pf = pq.ParquetFile(fh)
            for i, batch in enumerate(pf.iter_batches(batch_size=1, columns=["audio"])):
                if i in by_row:
                    r = by_row[i]
                    audio_bytes = batch.column("audio").to_pylist()[0]["bytes"]
                    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
                    write_wav(paths.audio_path(cfg, condition, r.excerpt_id),
                              crop_to_window(audio, sr, float(r.start_s), cfg),
                              cfg["data"]["sample_rate"])
                    status[r.excerpt_id] = f"written (shard {Path(shard).name}, row {i}, sr {sr})"
                if i >= max(by_row):
                    break
    return status
