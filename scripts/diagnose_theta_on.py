"""Onset F1 / precision on validation for theta_on values below the §3.5 grid,
for several re-onset ratios (theta_rise = 0). Validation only.

Usage (from the repository root): python scripts/diagnose_theta_on.py [condition]
"""

import sys

import numpy as np

from pianoabs.config import load_config
from pianoabs.eval.metrics import crop_predictions, onset_prf
from pianoabs.experiments import gt_for, load_activations, manifest_for
from pianoabs.model.notes import extract_with_config

condition = sys.argv[1] if len(sys.argv) > 1 else "C0"
cfg = load_config("configs/default.yaml")
data = [(load_activations(cfg, "ours", "full", condition, e), gt_for(cfg, condition, e))
        for e in manifest_for(cfg, "val")["excerpt_id"]]
for ratio in (1.0, 0.25, 0.1):
    cells = []
    for on in (0.03, 0.05, 0.075, 0.1, 0.15):
        s = [onset_prf(g, crop_predictions(extract_with_config(A, cfg, on, 0.0, ratio), 10.0))
             for A, g in data]
        cells.append(f"on={on}: F1={np.mean([x['onset_F1'] for x in s]):.4f} "
                     f"P={np.mean([x['onset_P'] for x in s]):.3f}")
    print(f"ratio {ratio}: " + " | ".join(cells))
