"""Note extraction (README §3.5) and the optimiser (§3.3–3.4)."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pretty_midi
import pytest
import torch

from pianoabs.model.c0 import synthesize_c0
from pianoabs.model.fit import fit_excerpt
from pianoabs.model.notes import (_key_events, extract_notes, normalise_activations,
                                  notes_to_midi, threshold_grid)

ON, OFF = 0.3, 0.15


def events(row, theta_rise=0.0, reonset=ON):
    return [(a, b) for a, b, _ in _key_events(np.asarray(row, float), ON, theta_rise, OFF, reonset)]


# ---------------------------------------------------------------- single-key state machine

def test_clean_pulse():
    assert events([0] * 5 + [1] * 10 + [0] * 5) == [(5, 15)]


def test_hysteresis_keeps_note_through_dip():
    # dips to 0.2 (below theta_on, above theta_off = 0.15), rises back by 0.2 < theta_on
    row = [0, 1, 0.8, 0.2, 0.2, 0.4, 0.3, 0.1, 0]
    assert events(row) == [(1, 7)]


def test_reonset_cut_at_restrike_frame():
    # decay to 0.4, then a re-strike to 1.0 (rise 0.6 >= theta_on): two notes, cut at frame 5
    row = [0, 1.0, 0.8, 0.6, 0.4, 1.0, 0.8, 0.0]
    assert events(row) == [(1, 5), (5, 7)]


def test_small_wiggles_never_retrigger():
    wiggle = [0, 1.0, 0.95, 0.97, 0.93, 0.96, 0.9, 0.0]   # theta_rise = 0
    assert events(wiggle) == [(1, 7)]


def test_theta_rise_blocks_slow_ramps():
    ramp = list(np.linspace(0, 1, 101))  # +0.01 per frame
    assert events(ramp, theta_rise=0.02) == []
    assert events(ramp, theta_rise=0.0)[0][0] == 31  # first frame above 0.3


def test_note_still_sounding_at_end():
    assert events([0, 1, 1, 1]) == [(1, 4)]


# ---------------------------------------------------------------- full extraction

def test_extract_notes_merge_drop_velocity():
    A = np.zeros((88, 40))
    A[39, 2:12] = 1.0            # key 60: note frames 2..11 ...
    A[39, 13:25] = 1.0           # ... gap of one frame (16 ms < 30 ms) -> merged
    A[50, 30:31] = 1.0           # key 71: a one-frame note (16 ms) -> dropped
    A[60, 20:30] = 0.5           # key 81: peak 0.5 -> velocity 64
    notes = extract_notes(A, frame_s=0.016, theta_on=0.3, theta_rise=0.0, percentile=100)
    got = [(int(p), round(a, 3), round(b, 3), int(v)) for a, b, p, v in
           zip(notes.onset_s, notes.offset_s, notes.pitch, notes.velocity)]
    assert got == [(60, 0.032, 0.4, 127), (81, 0.32, 0.48, 64)]


def test_normalise_by_percentile():
    A = np.arange(100, dtype=float).reshape(1, 100)
    assert normalise_activations(A, 99.0).max() == pytest.approx(99 / 98.01)
    assert normalise_activations(np.zeros((2, 3))).sum() == 0


def test_threshold_grid():
    from pianoabs.config import load_config
    grid = threshold_grid(load_config("configs/default.yaml"))
    assert len(grid) == 10 * 5 * 4
    assert grid[0] == (0.1, 0.0, 1.0)      # strictest re-onset ratio first (wins ties)
    assert grid[-1] == (1.0, 0.2, 0.1)


def test_notes_to_midi_roundtrip(tmp_path):
    notes = pd.DataFrame({"onset_s": [0.5], "offset_s": [1.0], "pitch": [64], "velocity": [90]})
    notes_to_midi(notes, tmp_path / "x.mid")
    n = pretty_midi.PrettyMIDI(str(tmp_path / "x.mid")).instruments[0].notes[0]
    assert (n.pitch, n.velocity) == (64, 90) and n.start == pytest.approx(0.5, abs=1e-3)


# ---------------------------------------------------------------- optimiser

def tiny_cfg(default_cfg, stage1=160, stage2=20):
    """~180 steps: with the amplitude scale, A must climb from 1e-3 to ~1 (Z: -6.9 -> ~0.5)."""
    cfg = copy.deepcopy(default_cfg)
    cfg["fit"].update(stage1_steps=stage1, stage2_steps=stage2)
    cfg["data"].update(excerpt_s=0.8, tail_s=0.2)          # 1 s -> T = 63 frames
    return cfg


def tiny_targets(cfg):
    notes = pd.DataFrame({"onset_s": [0.1, 0.4], "offset_s": [0.7, 0.8],
                          "pitch": [48, 67], "velocity": [100, 80], "evaluated": [True, True]})
    return synthesize_c0(notes, cfg)[0]


def test_fit_reduces_loss_and_is_deterministic(default_cfg):
    cfg = tiny_cfg(default_cfg)
    targets = tiny_targets(cfg)
    r1 = fit_excerpt(targets, cfg)
    r2 = fit_excerpt(targets, cfg)
    losses = [h["total"] for h in r1.loss_history]
    assert len(losses) == 180 and losses[-1] < 0.5 * losses[0]
    assert r1.activations.shape == (88, 63)
    np.testing.assert_array_equal(r1.activations, r2.activations)


def test_stage1_freezes_physical_parameters(default_cfg):
    cfg = tiny_cfg(default_cfg, stage1=15, stage2=0)
    params = fit_excerpt(tiny_targets(cfg), cfg).params
    assert params["delta_cents"] == 0.0
    assert params["a"] == pytest.approx([np.log(3e-4), 0, 0])
    cfg2 = tiny_cfg(default_cfg, stage1=5, stage2=10)
    params2 = fit_excerpt(tiny_targets(cfg2), cfg2).params
    assert params2["a"] != pytest.approx([np.log(3e-4), 0, 0])  # learned in stage 2


def test_B0_variant_keeps_B_zero(default_cfg):
    from pianoabs.config import variant_config
    cfg = variant_config(tiny_cfg(default_cfg, 5, 10), "B0")
    params = fit_excerpt(tiny_targets(cfg), cfg).params
    assert max(params["B_per_key"]) == 0.0


def test_amplitude_scale_makes_activations_dimensionless(default_cfg):
    cfg = tiny_cfg(default_cfg)
    targets = tiny_targets(cfg)
    res = fit_excerpt(targets, cfg)
    expected = torch.quantile(targets[2048].flatten(), 0.999).item()
    assert res.params["amplitude_scale"] == pytest.approx(expected, rel=1e-5)
    # loud notes now have A of order 1, not of order the raw magnitude (~0.1)
    assert res.activations.max() > 0.3
    cfg_off = copy.deepcopy(cfg)
    cfg_off["model"]["amplitude_scale_quantile"] = None
    assert fit_excerpt(targets, cfg_off).params["amplitude_scale"] == 1.0


def test_outputs_recomputed_when_config_changes(default_cfg, tmp_path):
    """Idempotent skip only for matching fingerprints; stale thresholds are refused."""
    from pianoabs import paths
    from pianoabs.experiments import (activation_fingerprint, compute_activations,
                                      predict_manifest, tune_thresholds)

    cfg = copy.deepcopy(default_cfg)
    cfg["paths"].update(outputs_dir=str(tmp_path / "out"), results_dir=str(tmp_path / "res"))
    cfg["fit"].update(stage1_steps=3, stage2_steps=3)
    assert compute_activations(cfg, "ours", "full", "C0", "val_004", "cpu") is True
    assert compute_activations(cfg, "ours", "full", "C0", "val_004", "cpu") is False  # skipped
    assert activation_fingerprint(cfg, "ours", "full") != activation_fingerprint(cfg, "ours", "B0")

    cfg["notes"].update(theta_on_grid=[0.3], theta_rise_grid=[0.0], reonset_rise_ratio_grid=[1.0])
    cfg["run"]["max_excerpts"] = 1
    import pandas as pd
    manifest = pd.read_csv("data/manifests/excerpts_val.csv")
    single = tmp_path / "m.csv"
    manifest[manifest.excerpt_id == "val_004"].to_csv(single, index=False)
    cfg["paths"]["data_dir"] = "data"
    import pianoabs.experiments as ex
    original = ex.manifest_for
    ex.manifest_for = lambda c, w, m=None: pd.read_csv(single)
    try:
        tune_thresholds(cfg, "ours", "full", "C0")
        assert predict_manifest(cfg, "ours", "full", "C0", "val") == 1
        assert predict_manifest(cfg, "ours", "full", "C0", "val") == 0   # up to date
        cfg["fit"]["stage2_steps"] = 4                                  # config change
        with pytest.raises(RuntimeError):
            predict_manifest(cfg, "ours", "full", "C0", "val")          # stale thresholds
        assert compute_activations(cfg, "ours", "full", "C0", "val_004", "cpu") is True
    finally:
        ex.manifest_for = original
    assert paths.prediction_path(cfg, "ours", "full", "C0", "val_004").exists()


@pytest.mark.slow
def test_e0_single_excerpt_recovers_notes(default_cfg):
    """E0 on one validation excerpt (C0, full 1500 steps): onset F1 >= 0.95 on CPU."""
    from pianoabs.eval.metrics import crop_predictions, onset_prf
    from pianoabs.model.notes import extract_with_config

    gt = pd.read_csv("data/gt/synthetic/val_000.csv")
    result = fit_excerpt(synthesize_c0(gt, default_cfg)[0], default_cfg)
    best = max(onset_prf(gt, crop_predictions(extract_with_config(result.activations, default_cfg,
                                                                  on, rise, ratio), 10.0))["onset_F1"]
               for on, rise, ratio in threshold_grid(default_cfg))
    assert best >= 0.95
