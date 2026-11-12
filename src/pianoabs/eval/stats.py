"""Aggregation and statistics (README §6).

* Mean and **95% bootstrap CI** over excerpts (10,000 resamples, percentile
  method) for every (condition, method, variant) and metric.
* **Wilcoxon signed-rank test** for planned paired comparisons (same excerpts,
  two methods/variants), reported with the median paired difference. With
  N = 30 these tests are exploratory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

METRICS = ["onset_P", "onset_R", "onset_F1", "onoff_F1", "frame_F1",
           "octave_err_rate", "harmonic_err_rate", "runtime_s"]
GROUP = ["condition", "method", "variant"]


def bootstrap_ci(values, n_resamples: int = 10000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float, float, int]:
    """(mean, ci_low, ci_high, n) of the mean, resampling excerpts with replacement.

    NaN values (e.g. octave error rate without false positives) are dropped.
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_resamples, len(x)))].mean(axis=1)
    alpha = (1 - level) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return float(x.mean()), float(lo), float(hi), len(x)


def summarise(per_excerpt: pd.DataFrame, n_resamples: int = 10000, seed: int = 0) -> pd.DataFrame:
    """Long-format summary: one row per (condition, method, variant, metric)."""
    rows = []
    for key, group in per_excerpt.groupby(GROUP, sort=True):
        for metric in METRICS:
            mean, lo, hi, n = bootstrap_ci(group[metric], n_resamples, seed)
            rows.append({**dict(zip(GROUP, key)), "metric": metric, "n": n,
                         "mean": mean, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def wilcoxon_paired(per_excerpt: pd.DataFrame, condition: str, metric: str,
                    a: tuple[str, str], b: tuple[str, str]) -> dict | None:
    """Wilcoxon signed-rank test of metric(a) - metric(b) over shared excerpts.

    ``a`` and ``b`` are (method, variant). Returns None if either is missing.
    """
    def series(method, variant):
        sel = per_excerpt[(per_excerpt.condition == condition) & (per_excerpt.method == method)
                          & (per_excerpt.variant == variant)]
        return sel.set_index("excerpt_id")[metric]

    sa, sb = series(*a), series(*b)
    if sa.empty or sb.empty:
        return None
    paired = pd.concat([sa, sb], axis=1, keys=["a", "b"]).dropna()
    diff = (paired["a"] - paired["b"]).to_numpy()
    row = {"condition": condition, "metric": metric, "method_a": a[0], "variant_a": a[1],
           "method_b": b[0], "variant_b": b[1], "n_pairs": len(diff),
           "mean_a": float(paired["a"].mean()) if len(diff) else float("nan"),
           "mean_b": float(paired["b"].mean()) if len(diff) else float("nan"),
           "median_diff": float(np.median(diff)) if len(diff) else float("nan"),
           "statistic": float("nan"), "p_value": float("nan")}
    if len(diff) and np.any(diff != 0):
        res = sps.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        row["statistic"], row["p_value"] = float(res.statistic), float(res.pvalue)
    return row


# Planned comparisons (README §6–§7): (label, metric, a, b, conditions)
PLANNED = [
    ("ours vs NMF (RQ1)", "onset_F1", ("ours", "full"), ("nmf_harmonic", "default"), ["C1", "C2", "C3"]),
    ("B learned vs B=0 (RQ2)", "onset_F1", ("ours", "full"), ("ours", "B0"), ["C1", "C3"]),
    ("B learned vs B=0 (RQ2)", "octave_err_rate", ("ours", "full"), ("ours", "B0"), ["C1", "C3"]),
    ("decay learned vs fixed (A2)", "onset_F1", ("ours", "full"), ("ours", "rho_fixed"), ["C1"]),
    ("multi- vs single-resolution (A4)", "onset_F1", ("ours", "full"), ("ours", "single_res"), ["C1"]),
]


def planned_tests(per_excerpt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, metric, a, b, conditions in PLANNED:
        for condition in conditions:
            row = wilcoxon_paired(per_excerpt, condition, metric, a, b)
            if row is not None:
                rows.append({"comparison": label, **row})
    return pd.DataFrame(rows)
