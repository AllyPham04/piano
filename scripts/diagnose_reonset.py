"""How identifiable are re-strikes in C0, and how sensitive is onset F1 to the
re-onset rule? Validation only.

1. For every GT onset that re-strikes a still-sounding key, compare the C0
   activation just before and at the onset: if it does not rise, no rule on A
   can detect it (the onset is unidentifiable by construction).
2. Recall on non-re-strike notes vs re-strike notes at the tuned thresholds.
3. Mean onset F1 over the threshold grid for several reonset_rise_ratio values.

Usage (from the repository root): python scripts/diagnose_reonset.py
"""

import json

import mir_eval
import numpy as np

from pianoabs.config import load_config
from pianoabs.eval.metrics import (_intervals_pitches, crop_predictions,
                                   evaluated_reference, onset_prf)
from pianoabs.experiments import gt_for, load_activations, manifest_for
from pianoabs.model.c0 import gt_activation_roll, n_frames_for
from pianoabs.model.notes import extract_with_config, threshold_grid

cfg = load_config("configs/default.yaml")
thr = json.load(open("results/thresholds/ours__full__C0.json"))
ids = list(manifest_for(cfg, "val")["excerpt_id"])
data = {eid: (load_activations(cfg, "ours", "full", "C0", eid), gt_for(cfg, "C0", eid)) for eid in ids}

n_re = n_re_rise = 0
hit = {"re-strike": [0, 0], "other": [0, 0]}
for eid, (A, gt) in data.items():
    roll = gt_activation_roll(gt, n_frames_for(cfg), 0.016, cfg["c0"]["tau_s"])
    ref = evaluated_reference(gt)
    pred = crop_predictions(extract_with_config(A, cfg, thr["theta_on"], thr["theta_rise"],
                                                thr.get("reonset_rise_ratio", 1.0)), 10.0)
    ri, rp = _intervals_pitches(ref)
    ei, ep = _intervals_pitches(pred)
    matched = {r for r, _ in mir_eval.transcription.match_notes(ri, rp, ei, ep, 0.05, offset_ratio=None)}
    for i, note in ref.iterrows():
        earlier = gt[(gt.pitch == note.pitch) & (gt.onset_s < note.onset_s) & (gt.offset_s >= note.onset_s - 1e-6)]
        kind = "re-strike" if len(earlier) else "other"
        hit[kind][0] += i in matched
        hit[kind][1] += 1
        if kind == "re-strike":
            t = int(np.ceil(note.onset_s / 0.016 - 1e-9))
            row = roll[int(note.pitch) - 21]
            n_re += 1
            n_re_rise += bool(row[t] > row[t - 1] * 1.001)

print(f"re-strikes of a sounding key: {n_re}; GT activation rises at the onset: {n_re_rise} "
      f"({n_re - n_re_rise} cannot be detected from A by any rule)")
for kind, (h, n) in hit.items():
    print(f"recall on {kind:9s} notes: {h}/{n} = {h / n:.3f}")

print("\nmean val onset F1 (best over the theta grid) per reonset_rise_ratio:")
pairs = sorted({(on, rise) for on, rise, _ in threshold_grid(cfg)})
for ratio in (1.0, 0.5, 0.25, 0.1, 0.05):
    best = max((np.mean([onset_prf(gt, crop_predictions(extract_with_config(A, cfg, on, rise, ratio),
                                                        10.0))["onset_F1"] for A, gt in data.values()]),
                on, rise) for on, rise in pairs)
    print(f"  ratio {ratio:<5}: F1 = {best[0]:.4f} at theta_on={best[1]}, theta_rise={best[2]}")
