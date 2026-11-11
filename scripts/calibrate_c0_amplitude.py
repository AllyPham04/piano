"""Choose c0.amplitude so that C0 spectrograms have the same level as C1 audio.

Statistic: 99.9th percentile of the window-sum-normalised |STFT| over the 10 s
window, per excerpt and resolution. amplitude = median over validation excerpts
of (C1 statistic / C0 statistic with amplitude 1). Validation excerpts only.

Also times one forward + backward pass of the full model (T = 688) on CPU.

Usage (from the repository root): python scripts/calibrate_c0_amplitude.py
"""

import time

import numpy as np
import pandas as pd
import torch

from pianoabs.audio import load_mono
from pianoabs.config import load_config
from pianoabs.features import multires_stft
from pianoabs.model.c0 import n_frames_for, synthesize_c0
from pianoabs.model.spectral_piano import SpectralPiano

cfg = load_config("configs/default.yaml")
cfg["c0"]["amplitude"] = 1.0
n_ffts, hop, T10 = cfg["features"]["n_ffts"], cfg["features"]["hop"], 625

rows = []
for eid in pd.read_csv("data/manifests/excerpts_val.csv")["excerpt_id"]:
    c1 = multires_stft(load_mono(f"data/audio/C1/{eid}.wav", 16000), n_ffts, hop)
    c0, _ = synthesize_c0(pd.read_csv(f"data/gt/synthetic/{eid}.csv"), cfg)
    for n in n_ffts:
        q1 = float(np.quantile(c1[n][:, :T10].numpy(), 0.999))
        q0 = float(np.quantile(c0[n][:, :T10].numpy(), 0.999))
        rows.append({"excerpt_id": eid, "n_fft": n, "c1_q999": q1, "c0_q999_amp1": q0,
                     "ratio": q1 / q0})
d = pd.DataFrame(rows)
print(d.pivot(index="excerpt_id", columns="n_fft", values="ratio").round(4))
print("\nmedian ratio per resolution:", d.groupby("n_fft")["ratio"].median().round(4).to_dict())
print("median ratio overall:", round(float(d["ratio"].median()), 4))

model = SpectralPiano.from_config(cfg, n_frames_for(cfg))
target = {n: torch.rand(n // 2 + 1, n_frames_for(cfg)) * 0.01 for n in n_ffts}
times = []
for _ in range(6):
    t0 = time.perf_counter()
    pred = model()
    loss = sum((torch.log1p(100 * pred[n]) - torch.log1p(100 * target[n])).abs().mean() for n in n_ffts)
    loss.backward()
    times.append(time.perf_counter() - t0)
print(f"\nCPU forward+backward, T={n_frames_for(cfg)}: median {1000 * np.median(times[1:]):.0f} ms/step "
      f"(torch threads={torch.get_num_threads()})")
