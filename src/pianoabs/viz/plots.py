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
    """Compute |STFT| of ``audio`` and call :func:`plot_gt_over_spec`."""
    import torch

    mag = stft_magnitude(torch.as_tensor(audio), n_fft, hop)
    return plot_gt_over_spec(mag.numpy(), sample_rate, n_fft, hop, gt, out_path, title,
                             gamma, excerpt_s)


def plot_gt_over_spec(mag: np.ndarray, sample_rate: int, n_fft: int, hop: int,
                      gt: pd.DataFrame, out_path: str | Path, title: str = "",
                      gamma: float = 100.0, excerpt_s: float = 10.0) -> Path:
    """Log-frequency spectrogram (compressed) with each GT note drawn as a line at its f0.

    If labels and spectrogram are aligned, every line sits on the lowest bright
    partial of a note and starts exactly at its attack.
    Solid = evaluated note (onset in window); dashed = sounding before t0.
    """
    import torch

    spec = compress(torch.as_tensor(mag), gamma).numpy()
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


def _log_spec_panel(ax, mag: np.ndarray, sample_rate: int, n_fft: int, hop: int,
                    gamma: float, title: str, fmin: float = 25.0, fmax: float = 4500.0) -> None:
    import torch

    spec = compress(torch.as_tensor(mag), gamma).numpy()
    freqs = bin_frequencies(n_fft, sample_rate)
    times = np.arange(spec.shape[1]) * hop / sample_rate
    keep = (freqs >= fmin) & (freqs <= fmax)
    ax.pcolormesh(times, freqs[keep], spec[keep], shading="nearest", cmap="magma")
    ax.set_yscale("log")
    ax.set_ylabel("Hz (log)")
    ax.set_title(title, fontsize=9, loc="left")


def plot_transcription_example(target_mag: np.ndarray, recon_mag: np.ndarray,
                               gt: pd.DataFrame, pred: pd.DataFrame, out_path: str | Path,
                               sample_rate: int = 16000, n_fft: int = 4096, hop: int = 256,
                               gamma: float = 100.0, excerpt_s: float = 10.0,
                               title: str = "") -> Path:
    """Input spectrogram, model reconstruction, and GT vs predicted piano roll.

    Piano roll colours (16 ms frames, 10 s window): green = TP (both), red = FP
    (predicted only), blue = FN (GT only). Black dots: GT onsets; white crosses:
    predicted onsets.
    """
    from matplotlib.colors import ListedColormap

    from pianoabs.data.groundtruth import notes_to_roll

    frame_s = hop / sample_rate
    ref = notes_to_roll(gt, excerpt_s, frame_s)
    est = notes_to_roll(pred[pred["onset_s"] < excerpt_s], excerpt_s, frame_s)
    code = np.zeros(ref.shape, dtype=int)
    code[ref & est], code[~ref & est], code[ref & ~est] = 1, 2, 3
    pitches = np.concatenate([gt["pitch"].to_numpy(), pred["pitch"].to_numpy(), [60]])
    lo, hi = int(pitches.min()) - 3, int(pitches.max()) + 3

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1.2]})
    _log_spec_panel(axes[0], target_mag, sample_rate, n_fft, hop, gamma, "input |S| (4096)")
    _log_spec_panel(axes[1], recon_mag, sample_rate, n_fft, hop, gamma, "model reconstruction")
    cmap = ListedColormap(["white", "#2ca02c", "#d62728", "#1f77b4"])
    times = np.arange(ref.shape[1]) * frame_s
    keys = np.arange(21, 109)
    sel = (keys >= lo) & (keys <= hi)
    axes[2].pcolormesh(times, keys[sel], code[sel], cmap=cmap, vmin=0, vmax=3, shading="nearest")
    ev = gt[gt["evaluated"].astype(bool)] if "evaluated" in gt else gt
    axes[2].plot(ev["onset_s"], ev["pitch"], "k.", ms=3)
    p = pred[pred["onset_s"] < excerpt_s]
    axes[2].plot(p["onset_s"], p["pitch"], "wx", ms=3, mew=0.8)
    axes[2].set_ylabel("MIDI pitch")
    axes[2].set_xlabel("time (s)")
    axes[2].set_title("piano roll: green TP, red FP, blue FN; dots GT onsets, x predicted onsets",
                      fontsize=9, loc="left")
    for ax in axes[:2]:
        ax.axvline(excerpt_s, color="white", ls=":", lw=1)
    fig.suptitle(title)
    return save_figure(fig, out_path)


def plot_loss_curves(histories: dict[str, list[dict]], out_path: str | Path) -> Path:
    """Total and reconstruction loss per step; dashed line marks the stage-2 start."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, hist in histories.items():
        steps = [h["step"] for h in hist]
        for ax, key in zip(axes, ("total", "recon")):
            ax.plot(steps, [h[key] for h in hist], lw=1, label=label)
    stage2 = next((h["step"] for h in next(iter(histories.values())) if h["stage"] == 2), None)
    for ax, key in zip(axes, ("total loss", "reconstruction L1 (compressed)")):
        if stage2 is not None:
            ax.axvline(stage2, color="0.5", ls="--", lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_title(key, fontsize=9, loc="left")
    axes[0].legend(fontsize=7)
    return save_figure(fig, out_path)


def plot_templates(templates: dict[str, np.ndarray], partials: dict[str, np.ndarray],
                   sample_rate: int, n_fft: int, pitch: int, out_path: str | Path,
                   fmax: float = 2000.0) -> Path:
    """One key's spectral template under several parameter settings.

    ``templates[label]`` is W_{p,r}(f) (F,), ``partials[label]`` the partial
    frequencies f_{p,k}; dotted grey lines mark the harmonic positions k f0.
    """
    freqs = bin_frequencies(n_fft, sample_rate)
    keep = freqs <= fmax
    labels = list(templates)
    fig, axes = plt.subplots(len(labels), 1, figsize=(12, 2.2 * len(labels)), sharex=True)
    f0 = midi_to_hz(pitch)
    for ax, label in zip(np.atleast_1d(axes), labels):
        for k in range(1, int(fmax / f0) + 1):
            ax.axvline(k * f0, color="0.75", ls=":", lw=0.8)
        ax.plot(freqs[keep], templates[label][keep], lw=1.0)
        fp = partials[label]
        ax.plot(fp[fp <= fmax], np.full((fp <= fmax).sum(), 1.05), "v", ms=3, color="C3")
        ax.set_ylabel("W (unit max)")
        ax.set_title(label, fontsize=9, loc="left")
        ax.set_ylim(0, 1.15)
    np.atleast_1d(axes)[-1].set_xlabel(
        f"frequency (Hz)  [key {pitch}, n_fft={n_fft}; dotted: k·f0, red: model partials f_pk]")
    return save_figure(fig, out_path)
