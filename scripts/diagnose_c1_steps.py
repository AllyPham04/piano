"""Why does a longer fit score worse on C1? Validation only.

Fits validation excerpts on a condition and, at several step counts, reports
the best onset F1 over the threshold grid (an optimistic per-excerpt oracle,
for diagnosis only), with P/R and the number of predicted notes, plus the
reconstruction loss and the learned physical parameters at the end.

Usage (from the repository root):
    python scripts/diagnose_c1_steps.py --condition C1 --excerpts val_000 val_001 [--set fit.lambda_sp=1.0]
"""

import argparse

import numpy as np

from pianoabs.config import apply_overrides, load_config, parse_set_args
from pianoabs.eval.metrics import crop_predictions, onset_prf
from pianoabs.experiments import gt_for, load_targets
from pianoabs.model.fit import fit_excerpt
from pianoabs.model.notes import extract_with_config, threshold_grid

parser = argparse.ArgumentParser()
parser.add_argument("--condition", default="C1")
parser.add_argument("--excerpts", nargs="+", default=["val_000", "val_001"])
parser.add_argument("--steps", nargs="+", type=int, default=[100, 300, 600, 1000, 1500])
parser.add_argument("--set", action="append")
args = parser.parse_args()

cfg = apply_overrides(load_config("configs/default.yaml"), parse_set_args(args.set))
print("overrides:", args.set)
for eid in args.excerpts:
    gt = gt_for(cfg, args.condition, eid)
    res = fit_excerpt(load_targets(cfg, args.condition, eid, "cpu"), cfg, "cpu",
                      snapshot_steps=tuple(args.steps))
    recon = {h["step"] + 1: h["recon"] for h in res.loss_history}
    print(f"\n{eid} ({args.condition}), n_ref={int(gt.evaluated.sum())}, runtime {res.runtime_s:.0f} s")
    for step in args.steps:
        A = res.snapshots[step]
        best = max(((onset_prf(gt, crop_predictions(extract_with_config(A, cfg, *t), 10.0)), t)
                    for t in threshold_grid(cfg)), key=lambda x: x[0]["onset_F1"])
        s, t = best
        print(f"  step {step:5d}: recon={recon[step]:.4f}  mean(A)={A.mean():.5f} "
              f"p99(A)={np.percentile(A, 99):.4f}  best F1={s['onset_F1']:.3f} "
              f"(P={s['onset_P']:.3f} R={s['onset_R']:.3f} n_pred={s['n_pred']}) at {t}")
    p = res.params
    B = np.array(p["B_per_key"])
    print(f"  final: delta={p['delta_cents']:.2f} cents, a={np.round(p['a'], 3).tolist()}, "
          f"c={np.round(p['c'], 3).tolist()}, B[A0, C4, C8]={B[0]:.2e}, {B[39]:.2e}, {B[87]:.2e}")
