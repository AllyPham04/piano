"""Reproducible excerpt sampling and frozen manifests (README §5.3).

Protocol (test: N = 30, seed 0; validation: N = 6, seed 1):

1. For each performance of the split (sorted by file name), draw 3 start times
   uniformly in [5 s, duration - 15 s] (rounded to 1 ms) -> candidate pool.
   Windows without any note onset are discarded (onset F1 would be undefined).
2. For each candidate: mean polyphony (sounding notes per 16 ms frame, after
   pedal extension, counting every note sounding in the window, including
   notes struck before t0) and pedal fraction (share of frames with CC64 >= 64).
3. Polyphony terciles over the pool -> low / mid / high. Shuffle the pool with
   the seed and pick N/3 per tercile (low first), at most one excerpt per
   performance.

Manifests are **frozen**: once a manifest file exists it is never overwritten.
Re-running the sampler only *verifies* that it reproduces the saved file and
raises an error otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from pianoabs.data.groundtruth import (ground_truth, mean_polyphony, pedal_fraction,
                                       performance_notes)
from pianoabs.data.maestro import load_performance

BINS = ["low", "mid", "high"]
MANIFEST_COLUMNS = [
    "excerpt_id", "split", "midi_filename", "audio_filename", "canonical_composer",
    "canonical_title", "year", "start_s", "duration_s", "n_notes_evaluated",
    "mean_polyphony", "polyphony_bin", "pedal_fraction", "seed",
]


class ManifestMismatchError(RuntimeError):
    """The sampler no longer reproduces a frozen manifest."""


def candidate_pool(meta: pd.DataFrame, midi_root: Path, split: str,
                   rng: np.random.Generator, cfg: dict) -> pd.DataFrame:
    """Step 1–2: candidate windows with their polyphony and pedal statistics."""
    s = cfg["data"]["sampling"]
    length, frame_s = cfg["data"]["excerpt_s"], s["frame_s"]
    threshold = cfg["data"]["pedal_threshold"]
    performances = meta[meta["split"] == split].sort_values("midi_filename")

    rows = []
    for perf in tqdm(performances.itertuples(index=False), total=len(performances),
                     desc=f"candidates ({split})"):
        lo, hi = s["start_min_s"], perf.duration - s["end_margin_s"]
        if hi <= lo:  # performance shorter than 20 s
            continue
        starts = np.round(rng.uniform(lo, hi, size=s["candidates_per_performance"]), 3)
        notes, intervals = performance_notes(load_performance(midi_root, perf.midi_filename),
                                             threshold)
        for t0 in starts:
            gt = ground_truth(notes, float(t0), length)
            n_eval = int(gt["synthetic"].shape[0])
            if n_eval < s["min_evaluated_notes"]:
                continue
            rows.append({
                "split": split,
                "midi_filename": perf.midi_filename,
                "audio_filename": perf.audio_filename,
                "canonical_composer": perf.canonical_composer,
                "canonical_title": perf.canonical_title,
                "year": int(perf.year),
                "start_s": float(t0),
                "duration_s": float(length),
                "n_notes_evaluated": n_eval,
                "mean_polyphony": mean_polyphony(gt["real"], length, frame_s),
                "pedal_fraction": pedal_fraction(intervals, float(t0), length, frame_s),
            })
    return pd.DataFrame(rows)


def assign_polyphony_bins(pool: pd.DataFrame) -> pd.DataFrame:
    """Label each candidate low / mid / high by the terciles of mean polyphony."""
    q1, q2 = np.quantile(pool["mean_polyphony"], [1 / 3, 2 / 3])
    out = pool.copy()
    out["polyphony_bin"] = np.where(out["mean_polyphony"] <= q1, "low",
                                    np.where(out["mean_polyphony"] <= q2, "mid", "high"))
    return out


def select_excerpts(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Step 3: n/3 excerpts per tercile, at most one per performance."""
    if n % len(BINS):
        raise ValueError(f"n={n} must be divisible by {len(BINS)}")
    per_bin = n // len(BINS)
    shuffled = pool.iloc[rng.permutation(len(pool))]
    used: set[str] = set()
    chosen = []
    for b in BINS:
        count = 0
        for _, row in shuffled[shuffled["polyphony_bin"] == b].iterrows():
            if row["midi_filename"] in used:
                continue
            chosen.append(row)
            used.add(row["midi_filename"])
            count += 1
            if count == per_bin:
                break
        if count < per_bin:
            raise RuntimeError(f"only {count} distinct performances in bin '{b}', need {per_bin}")
    return pd.DataFrame(chosen).reset_index(drop=True)


def build_manifest(meta: pd.DataFrame, midi_root: Path, which: str, cfg: dict) -> pd.DataFrame:
    """Run the whole protocol for ``which`` in {"test", "val"}."""
    spec = cfg["data"]["sampling"]["manifests"][which]
    rng = np.random.default_rng(spec["seed"])
    pool = assign_polyphony_bins(candidate_pool(meta, midi_root, spec["split"], rng, cfg))
    manifest = select_excerpts(pool, spec["n_excerpts"], rng)
    manifest["excerpt_id"] = [f"{which}_{i:03d}" for i in range(len(manifest))]
    manifest["seed"] = spec["seed"]
    return manifest[MANIFEST_COLUMNS]


def save_or_verify_manifest(manifest: pd.DataFrame, path: Path) -> str:
    """Write a new manifest, or check that an existing (frozen) one is reproduced.

    Returns ``"created"`` or ``"verified"``. Never overwrites an existing file.
    """
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
        return "created"
    frozen = pd.read_csv(path)
    same = (len(frozen) == len(manifest)
            and list(frozen["excerpt_id"]) == list(manifest["excerpt_id"])
            and list(frozen["midi_filename"]) == list(manifest["midi_filename"])
            and np.allclose(frozen["start_s"], manifest["start_s"], atol=1e-6))
    if not same:
        raise ManifestMismatchError(
            f"{path} is frozen but sampling now gives a different result. The file was NOT "
            "changed. Find out why (metadata, code or config change) before going on.")
    return "verified"


def load_manifest(path: Path, max_excerpts: int | None = None) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    return manifest.head(max_excerpts) if max_excerpts else manifest
