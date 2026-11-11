"""Tercile selection and manifest freezing on a synthetic candidate pool (README §5.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pianoabs.data.sampling import (ManifestMismatchError, assign_polyphony_bins,
                                    save_or_verify_manifest, select_excerpts)


def fake_pool(n_perf=20, per_perf=3, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "midi_filename": np.repeat([f"perf_{i:02d}.midi" for i in range(n_perf)], per_perf),
        "start_s": rng.uniform(5, 100, n_perf * per_perf).round(3),
        "mean_polyphony": rng.uniform(1, 10, n_perf * per_perf),
    })


def test_terciles_are_balanced():
    pool = assign_polyphony_bins(fake_pool())
    counts = pool["polyphony_bin"].value_counts()
    assert set(counts.index) == {"low", "mid", "high"}
    assert counts.max() - counts.min() <= 1
    low_max = pool.loc[pool.polyphony_bin == "low", "mean_polyphony"].max()
    high_min = pool.loc[pool.polyphony_bin == "high", "mean_polyphony"].min()
    assert low_max < high_min


def test_selection_per_bin_and_one_per_performance():
    pool = assign_polyphony_bins(fake_pool())
    chosen = select_excerpts(pool, 9, np.random.default_rng(0))
    assert chosen["polyphony_bin"].value_counts().to_dict() == {"low": 3, "mid": 3, "high": 3}
    assert chosen["midi_filename"].is_unique


def test_selection_is_deterministic_for_a_seed():
    pool = assign_polyphony_bins(fake_pool())
    a = select_excerpts(pool, 9, np.random.default_rng(1))
    b = select_excerpts(pool, 9, np.random.default_rng(1))
    c = select_excerpts(pool, 9, np.random.default_rng(2))
    pd.testing.assert_frame_equal(a, b)
    assert not a["start_s"].equals(c["start_s"])


def test_selection_fails_when_too_few_performances():
    pool = assign_polyphony_bins(fake_pool(n_perf=2))
    with pytest.raises(RuntimeError):
        select_excerpts(pool, 9, np.random.default_rng(0))


def test_manifest_is_frozen(tmp_path):
    path = tmp_path / "excerpts_val.csv"
    m = pd.DataFrame({"excerpt_id": ["val_000", "val_001"],
                      "midi_filename": ["a.midi", "b.midi"], "start_s": [5.0, 6.123]})
    assert save_or_verify_manifest(m, path) == "created"
    before = path.read_bytes()
    assert save_or_verify_manifest(m.copy(), path) == "verified"

    changed = m.copy()
    changed.loc[1, "start_s"] = 7.0
    with pytest.raises(ManifestMismatchError):
        save_or_verify_manifest(changed, path)
    assert path.read_bytes() == before  # never overwritten
