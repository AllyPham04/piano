"""Metrics on toy notes (README §6), statistics, and the NMF baseline (§4)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pianoabs.baselines.nmf import kl_divergence, kl_nmf_activations, nmf_dictionary
from pianoabs.eval.metrics import (crop_predictions, excerpt_metrics, false_positive_errors,
                                   frame_f1, onset_prf)
from pianoabs.eval.stats import bootstrap_ci, planned_tests, summarise, wilcoxon_paired


def notes(rows, evaluated=None):
    df = pd.DataFrame(rows, columns=["onset_s", "offset_s", "pitch", "velocity"])
    if evaluated is not None:
        df["evaluated"] = evaluated
    return df


GT = notes([(1.0, 2.0, 48, 80), (3.0, 3.5, 60, 80), (0.0, 0.5, 72, 80)],
           evaluated=[True, True, False])   # the 72 started before the window


# ---------------------------------------------------------------- onset metrics

def test_perfect_prediction():
    pred = GT[GT.evaluated].drop(columns="evaluated")
    s = onset_prf(GT, pred)
    assert (s["onset_P"], s["onset_R"], s["onset_F1"], s["n_ref"]) == (1.0, 1.0, 1.0, 2)


def test_onset_tolerance_and_pitch():
    pred = notes([(1.04, 2.0, 48, 80),    # 40 ms late: match
                  (3.06, 3.5, 60, 80)])   # 60 ms late: no match
    s = onset_prf(GT, pred)
    assert s["onset_P"] == 0.5 and s["onset_R"] == 0.5
    assert onset_prf(GT, notes([(1.0, 2.0, 49, 80)]))["onset_F1"] == 0.0  # wrong pitch


def test_non_evaluated_notes_do_not_count_for_onsets():
    pred = notes([(0.0, 0.5, 72, 80)])
    s = onset_prf(GT, pred)
    assert s["onset_P"] == 0.0 and s["n_ref"] == 2


def test_empty_prediction():
    s = onset_prf(GT, notes([]))
    assert s["onset_F1"] == 0.0 and s["n_pred"] == 0


# ---------------------------------------------------------------- frame F1 and cropping

def test_frame_f1_counts_all_sounding_notes():
    gt = notes([(0.0, 1.0, 60, 80)], evaluated=[False])
    assert frame_f1(gt, notes([(0.0, 1.0, 60, 80)]), 2.0, 0.25) == 1.0
    # half the frames: TP = 2, FN = 2 -> F1 = 4 / 6
    assert frame_f1(gt, notes([(0.0, 0.5, 60, 80)]), 2.0, 0.25) == pytest.approx(2 / 3)


def test_crop_predictions():
    pred = notes([(9.5, 10.6, 60, 80), (10.2, 10.8, 62, 80)])
    out = crop_predictions(pred, 10.0)
    assert out[["onset_s", "offset_s", "pitch"]].values.tolist() == [[9.5, 10.0, 60]]


# ---------------------------------------------------------------- octave / harmonic errors

def test_octave_and_harmonic_errors():
    gt = notes([(1.0, 2.0, 48, 80)], evaluated=[True])
    pred = notes([(1.00, 2.0, 48, 80),   # TP
                  (1.02, 2.0, 60, 80),   # FP, +12: octave error
                  (1.00, 2.0, 67, 80),   # FP, +19: 3rd harmonic
                  (1.00, 2.0, 36, 80),   # FP, -12: octave error below
                  (3.00, 3.5, 50, 80)])  # FP, unrelated
    e = false_positive_errors(gt, pred, 0.05, [12, 24, 36, -12, -24, -36], [19, 28])
    assert (e["n_fp"], e["n_octave_fp"], e["n_harmonic_fp"]) == (4, 2, 1)
    assert e["octave_err_rate"] == 0.5 and e["harmonic_err_rate"] == 0.25


def test_error_rates_nan_without_false_positives():
    gt = notes([(1.0, 2.0, 48, 80)], evaluated=[True])
    e = false_positive_errors(gt, notes([(1.0, 2.0, 48, 80)]), 0.05, [12], [19])
    assert e["n_fp"] == 0 and math.isnan(e["octave_err_rate"])


def test_excerpt_metrics_schema(default_cfg):
    pred = notes([(1.0, 2.0, 48, 80), (3.0, 3.1, 60, 80)])
    m = excerpt_metrics(GT, pred, default_cfg)
    for key in ("onset_P", "onset_R", "onset_F1", "onoff_F1", "frame_F1", "octave_err_rate",
                "harmonic_err_rate", "n_ref", "n_pred"):
        assert key in m
    assert m["onset_F1"] == 1.0
    assert m["onoff_F1"] == 0.5  # second note ends 0.4 s early (> max(0.2*0.5, 0.05))


# ---------------------------------------------------------------- statistics

def test_bootstrap_ci():
    mean, lo, hi, n = bootstrap_ci([0.5] * 10)
    assert (mean, lo, hi, n) == (0.5, 0.5, 0.5, 10)
    x = np.random.default_rng(0).normal(0.6, 0.1, 30)
    mean, lo, hi, n = bootstrap_ci(x, seed=1)
    assert lo < mean < hi and (hi - lo) < 0.1
    assert bootstrap_ci(x, seed=1) == bootstrap_ci(x, seed=1)
    assert bootstrap_ci([0.2, float("nan"), 0.4])[3] == 2


def _per_excerpt(n=12):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        base = rng.uniform(0.3, 0.7)
        for method, variant, shift in [("ours", "full", 0.1), ("nmf_harmonic", "default", 0.0),
                                       ("ours", "B0", 0.1)]:
            rows.append({"excerpt_id": f"e{i}", "condition": "C1", "method": method,
                         "variant": variant, **{m: base + shift for m in
                                                ("onset_P", "onset_R", "onset_F1", "onoff_F1",
                                                 "frame_F1", "octave_err_rate",
                                                 "harmonic_err_rate", "runtime_s")}})
    return pd.DataFrame(rows)


def test_wilcoxon_and_planned_tests():
    per = _per_excerpt()
    row = wilcoxon_paired(per, "C1", "onset_F1", ("ours", "full"), ("nmf_harmonic", "default"))
    assert row["n_pairs"] == 12 and row["median_diff"] == pytest.approx(0.1)
    assert row["p_value"] < 0.01
    tie = wilcoxon_paired(per, "C1", "onset_F1", ("ours", "full"), ("ours", "B0"))
    assert tie["median_diff"] == 0.0 and math.isnan(tie["p_value"])
    tests = planned_tests(per)
    assert set(tests["comparison"]) == {"ours vs NMF (RQ1)", "B learned vs B=0 (RQ2)"}


def test_summarise_long_format():
    s = summarise(_per_excerpt(), n_resamples=100)
    assert set(s.columns) == {"condition", "method", "variant", "metric", "n", "mean",
                              "ci_low", "ci_high"}
    assert len(s) == 3 * 8


# ---------------------------------------------------------------- NMF

def test_nmf_dictionary_shapes(default_cfg):
    W = nmf_dictionary(default_cfg, "nmf_harmonic")
    assert W.shape == (1025, 88) and np.allclose(W.max(axis=0), 1.0)
    assert not np.allclose(W, nmf_dictionary(default_cfg, "nmf_inharmonic"))


def test_nmf_recovers_active_keys(default_cfg):
    W = nmf_dictionary(default_cfg, "nmf_harmonic")
    H_true = np.zeros((88, 20))
    H_true[39, 5:15] = 0.1       # key 60
    H_true[46, 10:20] = 0.05     # key 67
    V = W @ H_true + 1e-6
    H = kl_nmf_activations(V, W, n_iter=200)
    assert H.shape == (88, 20) and np.all(H >= 0)
    energy = H.sum(axis=1)
    assert set(np.argsort(energy)[-2:]) == {39, 46}
    assert kl_divergence(V, W @ H) < kl_divergence(V, W @ kl_nmf_activations(V, W, n_iter=5))
