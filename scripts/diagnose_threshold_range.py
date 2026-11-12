"""Is the §3.5 threshold grid too narrow? Validation only.

For each method, compare the best mean onset F1 within the README grid
(θ_on <= 0.5, θ_rise <= 0.05) with a wider grid, and describe the false
positives at the best wide-grid setting (octave / harmonic share).

Usage (from the repository root): python scripts/diagnose_threshold_range.py [condition]
"""

import sys
from itertools import product

import numpy as np

from pianoabs.config import load_config
from pianoabs.eval.metrics import crop_predictions, excerpt_metrics, onset_prf
from pianoabs.experiments import gt_for, load_activations, manifest_for
from pianoabs.model.notes import extract_with_config

condition = sys.argv[1] if len(sys.argv) > 1 else "C1"
cfg = load_config("configs/default.yaml")
ids = list(manifest_for(cfg, "val")["excerpt_id"])
wide_on = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5]
wide_rise = [0.0, 0.02, 0.05, 0.1, 0.2]
ratios = [1.0, 0.5, 0.25, 0.1]

for method, variant in [("ours", "full"), ("nmf_harmonic", "default")]:
    data = [(load_activations(cfg, method, variant, condition, e), gt_for(cfg, condition, e)) for e in ids]
    results = {}
    for on, rise, ratio in product(wide_on, wide_rise, ratios):
        f1 = np.mean([onset_prf(gt, crop_predictions(extract_with_config(A, cfg, on, rise, ratio), 10.0))
                      ["onset_F1"] for A, gt in data])
        results[(on, rise, ratio)] = f1
    readme = {k: v for k, v in results.items() if k[0] <= 0.5 and k[1] <= 0.05}
    best_readme = max(readme, key=readme.get)
    best_wide = max(results, key=results.get)
    print(f"\n{method}/{variant} on {condition} validation ({len(ids)} excerpts)")
    print(f"  README grid best: F1={readme[best_readme]:.4f} at {best_readme}")
    print(f"  wide grid best:   F1={results[best_wide]:.4f} at {best_wide}")
    print("  F1 vs theta_on (best over rise/ratio):",
          {on: round(max(v for k, v in results.items() if k[0] == on), 4) for on in wide_on})
    ms = [excerpt_metrics(gt, extract_with_config(A, cfg, *best_wide), cfg) for A, gt in data]
    n_fp = sum(m["n_fp"] for m in ms)
    print(f"  at wide best: P={np.mean([m['onset_P'] for m in ms]):.3f} "
          f"R={np.mean([m['onset_R'] for m in ms]):.3f}  false positives={n_fp}, "
          f"octave={sum(m['n_octave_fp'] for m in ms)}, harmonic={sum(m['n_harmonic_fp'] for m in ms)}")
