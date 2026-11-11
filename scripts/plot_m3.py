"""M3 figures: model templates with/without inharmonicity, and a C0 spectrogram
with its ground truth (validation excerpt val_000).

Usage (from the repository root): python scripts/plot_m3.py
"""

import pandas as pd
import torch

from pianoabs.config import load_config
from pianoabs.model.c0 import synthesize_c0
from pianoabs.model.spectral_piano import SpectralPiano
from pianoabs.viz.plots import plot_gt_over_spec, plot_templates

cfg = load_config("configs/default.yaml")
sr, hop = cfg["data"]["sample_rate"], cfg["features"]["hop"]
pitch, n_fft = 33, 4096  # A1, 55 Hz: low string, where inharmonicity matters most

templates, partials = {}, {}
for label, kwargs in [("B = 0 (harmonic)", {"B_mode": "zero"}),
                      ("B = 3e-4 (initial value)", {"B_init": 3e-4}),
                      ("B = 1e-3", {"B_init": 1e-3})]:
    m = SpectralPiano(n_frames=1, **kwargs)
    with torch.no_grad():
        templates[label] = m.templates(n_fft)[pitch - 21].numpy()
        partials[label] = m.partial_freqs()[pitch - 21].numpy()
print("wrote", plot_templates(templates, partials, sr, n_fft, pitch,
                              "figures/sanity/m3_templates_key33.png"))

gt = pd.read_csv("data/gt/synthetic/val_000.csv")
specs, _ = synthesize_c0(gt, cfg)
print("wrote", plot_gt_over_spec(specs[4096].numpy(), sr, 4096, hop, gt,
                                 "figures/sanity/gt_over_spec_C0_val_000.png",
                                 title="val_000 (C0): model-synthesised |S| 4096 with synthetic GT",
                                 gamma=cfg["features"]["gamma"]))
