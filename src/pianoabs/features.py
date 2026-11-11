r"""Multi-resolution STFT magnitudes and loss compression (README §3.1).

* STFT at ``n_fft`` in {1024, 2048, 4096}, Hann window (``win_length = n_fft``),
  shared hop of 256 samples (16 ms) and ``center=True``. With centring, every
  resolution has T = 1 + floor(N / hop) frames, so all resolutions are aligned
  frame by frame and share one activation matrix A (88 x T).
* Magnitudes are divided by the window sum, ``|X| / sum(w)``. Without this the
  peak of a sinusoid grows with ``n_fft`` (4x from 1024 to 4096), while the
  model templates are unit-max at every resolution. After normalisation a
  sinusoid of amplitude a peaks at a/2 at every resolution, so one A fits all.
* Compression for the loss: ``c(S) = log(1 + gamma * S)``.
"""

from __future__ import annotations

import numpy as np
import torch


def stft_magnitude(audio: torch.Tensor, n_fft: int, hop: int,
                   normalize: str = "window_sum") -> torch.Tensor:
    """|STFT| of a mono signal, shape (F, T) with F = n_fft // 2 + 1."""
    window = torch.hann_window(n_fft, device=audio.device, dtype=audio.dtype)
    spec = torch.stft(audio, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                      window=window, center=True, return_complex=True)
    mag = spec.abs()
    if normalize == "window_sum":
        mag = mag / window.sum()
    elif normalize != "none":
        raise ValueError(f"unknown normalize={normalize!r}")
    return mag


def multires_stft(audio: np.ndarray | torch.Tensor, n_ffts: list[int], hop: int,
                  normalize: str = "window_sum", device: str = "cpu") -> dict[int, torch.Tensor]:
    """``{n_fft: |STFT| (F_r, T)}`` for every resolution; all share the same T."""
    x = torch.as_tensor(audio, dtype=torch.float32, device=device)
    return {n: stft_magnitude(x, n, hop, normalize) for n in n_ffts}


def bin_frequencies(n_fft: int, sample_rate: int) -> np.ndarray:
    """Centre frequency (Hz) of each STFT bin, 0 ... sample_rate / 2."""
    return np.arange(n_fft // 2 + 1) * sample_rate / n_fft


def compress(spec: torch.Tensor, gamma: float) -> torch.Tensor:
    """Loss compression c(S) = log(1 + gamma * S)."""
    return torch.log1p(gamma * spec)
