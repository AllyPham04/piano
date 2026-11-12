"""Diagnose ours on validation excerpts: which GT onsets are missed and why.

For each validation excerpt, extract notes with the tuned thresholds and report
P/R/F1, then describe the missed GT notes (false negatives) by:
  * repeat: same key struck again within 0.5 s before (re-strike while sounding)
  * velocity and the activation level the GT predicts at the onset
  * normalised activation peak of the fitted A around the onset (±2 frames)
Validation only, never test.

Usage (from the repository root): python scripts/diagnose_e0.py [condition]
"""

import json
import sys

import mir_eval
import numpy as np
import pandas as pd

from pianoabs.config import load_config
from pianoabs.eval.metrics import (_intervals_pitches, crop_predictions,
                                   evaluated_reference, onset_prf)
from pianoabs.experiments import gt_for, load_activations, manifest_for
from pianoabs.model.notes import extract_with_config, normalise_activations

condition = sys.argv[1] if len(sys.argv) > 1 else "C0"
cfg = load_config("configs/default.yaml")
thr = json.load(open(f"results/thresholds/ours__full__{condition}.json"))
frame_s = 0.016
all_fn = []
for eid in manifest_for(cfg, "val")["excerpt_id"]:
    A = load_activations(cfg, "ours", "full", condition, eid)
    An = normalise_activations(A, 99.0)
    gt = gt_for(cfg, condition, eid)
    pred = crop_predictions(extract_with_config(A, cfg, thr["theta_on"], thr["theta_rise"],
                                                thr.get("reonset_rise_ratio", 1.0)), 10.0)
    s = onset_prf(gt, pred)
    ref = evaluated_reference(gt)
    ri, rp = _intervals_pitches(ref)
    ei, ep = _intervals_pitches(pred)
    matched = {r for r, _ in mir_eval.transcription.match_notes(ri, rp, ei, ep, 0.05, offset_ratio=None)}
    print(f"{eid}: P={s['onset_P']:.3f} R={s['onset_R']:.3f} F1={s['onset_F1']:.3f} "
          f"n_ref={s['n_ref']} n_pred={s['n_pred']}  p99(A)={np.percentile(A, 99):.4f} max(A)={A.max():.4f}")
    for i, note in ref.iterrows():
        if i in matched:
            continue
        same = ref[(ref.pitch == note.pitch) & (ref.onset_s < note.onset_s)]
        gap = note.onset_s - same.onset_s.max() if len(same) else np.inf
        t = int(round(note.onset_s / frame_s))
        k = int(note.pitch) - 21
        win = An[k, max(t - 2, 0): t + 3]
        all_fn.append({"excerpt": eid, "pitch": int(note.pitch), "velocity": int(note.velocity),
                       "repeat_gap_s": gap, "peak_norm_A": float(win.max()),
                       "rise_norm_A": float(win.max() - An[k, max(t - 3, 0)])})
fn = pd.DataFrame(all_fn)
print(f"\nmissed onsets: {len(fn)}")
print("  re-strike of a sounding key within 0.5 s:", int((fn.repeat_gap_s < 0.5).sum()))
print("  normalised activation peak < theta_on:", int((fn.peak_norm_A < thr["theta_on"]).sum()))
print("  velocity quartiles of missed notes:", fn.velocity.quantile([0.25, 0.5, 0.75]).tolist())
print(fn.sort_values("peak_norm_A").head(12).round(3).to_string(index=False))
