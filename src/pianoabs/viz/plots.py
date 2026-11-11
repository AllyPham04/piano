"""Plotting helpers. Figures of README §8.4 are added in M9; this module starts
with the M2 sanity plot (ground truth drawn over the spectrogram)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: works locally, on Kaggle and in tests
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pianoabs.features import bin_frequencies, compress, stft_magnitude  # noqa: E402


def midi_to_hz(pitch):
    return 440.0 * 2.0 ** ((np.asarray(pitch, dtype=float) - 69.0) / 12.0)


def save_figure(fig, path: str | Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_gt_over_spectrogram(audio: np.ndarray, sample_rate: int, gt: pd.DataFrame,
                             out_path: str | Path, title: str = "", hop: int = 256,
                             n_fft: int = 4096, gamma: float = 100.0,
                             excerpt_s: float = 10.0) -> Path:
    """Log-frequency spectrogram with each GT note drawn as a line at its f0.

    If labels and audio are aligned, every line sits on the lowest bright
    partial of a note and starts exactly at its attack.
    Solid = evaluated note (onset in window); dashed = sounding before t0.
    """
    import torch

    mag = stft_magnitude(torch.as_tensor(audio), n_fft, hop)
    spec = compress(mag, gamma).numpy()
    freqs = bin_frequencies(n_fft, sample_rate)
    times = np.arange(spec.shape[1]) * hop / sample_rate
    keep = (freqs >= 25) & (freqs <= 4500)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.pcolormesh(times, freqs[keep], spec[keep], shading="nearest", cmap="magma")
    ax.set_yscale("log")
    for note in gt.itertuples(index=False):
        f0 = midi_to_hz(note.pitch)
        evaluated = bool(getattr(note, "evaluated", True))
        ax.plot([note.onset_s, note.offset_s], [f0, f0], color="cyan",
                lw=1.4 if evaluated else 1.0, ls="-" if evaluated else "--", alpha=0.9)
        ax.plot(note.onset_s, f0, marker="|", color="white", ms=8)
    ax.axvline(excerpt_s, color="white", ls=":", lw=1)
    ax.set_xlabel("time (s)  [dotted line: end of evaluated window]")
    ax.set_ylabel("frequency (Hz, log)")
    ax.set_title(title or "ground truth over spectrogram")
    return save_figure(fig, out_path)
