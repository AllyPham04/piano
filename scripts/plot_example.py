"""Example figure (input, reconstruction, piano roll) and loss curves for fitted excerpts.

Usage (from the repository root):
    python scripts/plot_example.py --condition C0 --variant full --excerpts test_000 test_017
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch

from pianoabs import paths
from pianoabs.config import load_config, variant_config
from pianoabs.experiments import gt_for, load_targets
from pianoabs.model.spectral_piano import SpectralPiano
from pianoabs.viz.plots import plot_loss_curves, plot_transcription_example

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--condition", default="C0")
parser.add_argument("--variant", default="full")
parser.add_argument("--excerpts", nargs="+", required=True)
args = parser.parse_args()

cfg = variant_config(load_config(args.config), args.variant)
out_dir = paths.figures_dir(cfg) / "sanity"
histories = {}
for eid in args.excerpts:
    params = json.loads(paths.params_path(cfg, args.variant, args.condition, eid).read_text())
    A = np.load(paths.activation_path(cfg, args.variant, args.condition, eid))
    model = SpectralPiano.from_fitted(cfg, params, A)
    with torch.no_grad():
        recon = model.spectrogram(4096).numpy()
    target = load_targets(cfg, args.condition, eid, "cpu")[4096].numpy()
    pred = pd.read_csv(paths.prediction_path(cfg, "ours", args.variant, args.condition, eid))
    out = plot_transcription_example(
        target, recon, gt_for(cfg, args.condition, eid), pred,
        out_dir / f"example_{args.condition}_{args.variant}_{eid}.png",
        title=f"{eid} ({args.condition}, ours/{args.variant})")
    print("wrote", out)
    histories[eid] = params["loss_history"]
print("wrote", plot_loss_curves(histories, out_dir / f"loss_{args.condition}_{args.variant}.png"))
