r"""Differentiable spectral piano model (README §3.2).

The model predicts magnitude spectrograms at several STFT resolutions r from a
latent piano roll and a handful of physical parameters:

.. math::

    \hat S_r(f, t) = \sum_p W_{p,r}(f)\, A_{p,t} + \beta_r(f), \qquad A = \mathrm{softplus}(Z)

Keys p = 21..108 (88 keys), u = (p - 60) / 24.

* **Fundamental** (global tuning δ and optional per-key detune δ_p, both in cents):
  f0_p = 440 · 2^((p-69)/12) · 2^((δ + δ_p)/1200),   δ_p = 20 · tanh(raw_p)
* **Partials of a stiff string** (inharmonicity B_p):
  f_{p,k} = k · f0_p · sqrt(1 + B_p k²),   k = 1..K_p,   K_p = min(40, floor((fs/2) / f0_p))
* **Inharmonicity curve**, smooth over keys:
  log B_p = a0 + a1 u + a2 u²,   clamped to [1e-6, 5e-2]
* **Partial amplitudes** (spectral roll-off over partial index):
  g_{p,k} = exp(-ρ_p (k-1)),   log ρ_p = c0 + c1 u
* **Template of key p at resolution r**: a Gaussian per partial approximating
  the Hann main lobe, σ_r = κ · fs / n_fft_r:
  W_{p,r}(f) = Σ_k g_{p,k} exp(-(f - f_{p,k})² / (2 σ_r²)),
  partials above Nyquist removed, each template scaled to unit max.
* **Noise floor** β_r(f) = softplus(raw) ≥ 0, one value per frequency bin.

Learnable: Z (88 x T), δ, (δ_p), a0..a2, c0..c1, β_r. The templates are rebuilt
from the physical parameters at every call, so gradients flow from the
spectrogram error to B, ρ and δ.

Magnitudes are assumed to add (as in NMF transcription); beating and phase
cancellation between partials are ignored (README §12).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def inverse_softplus(y: float) -> float:
    """x such that softplus(x) = y, i.e. x = log(exp(y) - 1)."""
    return math.log(math.expm1(y))


def midi_to_hz(pitch: torch.Tensor) -> torch.Tensor:
    return 440.0 * 2.0 ** ((pitch - 69.0) / 12.0)


class SpectralPiano(nn.Module):
    """Spectral piano model; see module docstring for the equations."""

    def __init__(self, n_frames: int, sample_rate: int = 16000,
                 n_ffts: tuple[int, ...] = (1024, 2048, 4096),
                 pitch_min: int = 21, pitch_max: int = 108, k_max: int = 40,
                 kappa: float = 0.6, B_mode: str = "learned", B_init: float = 3e-4,
                 B_clamp: tuple[float, float] = (1e-6, 5e-2), rho_init: float = 0.3,
                 learn_rho: bool = True, per_key_detune: bool = False,
                 max_detune_cents: float = 20.0, A_init: float = 1e-3,
                 beta_init: float = 1e-4):
        super().__init__()
        if B_mode not in ("learned", "fixed", "zero"):
            raise ValueError(f"B_mode must be learned | fixed | zero, got {B_mode!r}")
        self.sample_rate = sample_rate
        self.n_ffts = tuple(int(n) for n in n_ffts)
        self.k_max, self.kappa = k_max, kappa
        self.B_mode, self.B_clamp = B_mode, tuple(B_clamp)
        self.per_key_detune, self.max_detune_cents = per_key_detune, max_detune_cents

        pitches = torch.arange(pitch_min, pitch_max + 1, dtype=torch.float32)
        self.register_buffer("pitches", pitches)
        self.register_buffer("u", (pitches - 60.0) / 24.0)
        self.register_buffer("k", torch.arange(1, k_max + 1, dtype=torch.float32))
        for n in self.n_ffts:
            freqs = torch.arange(n // 2 + 1, dtype=torch.float32) * sample_rate / n
            self.register_buffer(f"freqs_{n}", freqs)
        n_keys = len(pitches)

        # Latent piano roll: softplus(Z) = A_init everywhere at the start.
        self.Z = nn.Parameter(torch.full((n_keys, n_frames), inverse_softplus(A_init)))
        # Physical parameters.
        self.delta = nn.Parameter(torch.zeros(()))                       # cents
        self.detune_raw = nn.Parameter(torch.zeros(n_keys), requires_grad=per_key_detune)
        self.a = nn.Parameter(torch.tensor([math.log(B_init), 0.0, 0.0]),
                              requires_grad=(B_mode == "learned"))
        self.c = nn.Parameter(torch.tensor([math.log(rho_init), 0.0]), requires_grad=learn_rho)
        # Noise floor per resolution and bin.
        self.beta_raw = nn.ParameterDict({
            str(n): nn.Parameter(torch.full((n // 2 + 1,), inverse_softplus(beta_init)))
            for n in self.n_ffts})

    @classmethod
    def from_config(cls, cfg: dict, n_frames: int) -> "SpectralPiano":
        m = cfg["model"]
        return cls(n_frames=n_frames, sample_rate=cfg["data"]["sample_rate"],
                   n_ffts=tuple(cfg["features"]["n_ffts"]), pitch_min=m["pitch_min"],
                   pitch_max=m["pitch_max"], k_max=m["k_max"], kappa=m["kappa"],
                   B_mode=m["B_mode"], B_init=m["B_init"], B_clamp=tuple(m["B_clamp"]),
                   rho_init=m["rho_init"], learn_rho=m["learn_rho"],
                   per_key_detune=m["per_key_detune"],
                   max_detune_cents=m["max_detune_cents"], A_init=m["A_init"],
                   beta_init=m["beta_init"])

    # ------------------------------------------------------------ physical quantities

    def detune_cents(self) -> torch.Tensor:
        """δ + δ_p for every key (88,)."""
        cents = self.delta.expand_as(self.pitches)
        if self.per_key_detune:
            cents = cents + self.max_detune_cents * torch.tanh(self.detune_raw)
        return cents

    def f0(self) -> torch.Tensor:
        """Fundamental frequency of each key in Hz (88,)."""
        return midi_to_hz(self.pitches) * 2.0 ** (self.detune_cents() / 1200.0)

    def B(self) -> torch.Tensor:
        """Inharmonicity per key (88,): exp(a0 + a1 u + a2 u²), clamped; 0 if B_mode='zero'."""
        if self.B_mode == "zero":
            return torch.zeros_like(self.u)
        log_b = self.a[0] + self.a[1] * self.u + self.a[2] * self.u ** 2
        return torch.exp(log_b).clamp(*self.B_clamp)

    def rho(self) -> torch.Tensor:
        """Spectral roll-off per key (88,): exp(c0 + c1 u)."""
        return torch.exp(self.c[0] + self.c[1] * self.u)

    def partial_freqs(self) -> torch.Tensor:
        """f_{p,k} = k f0_p sqrt(1 + B_p k²), shape (88, K_max)."""
        k = self.k
        return k * self.f0()[:, None] * torch.sqrt(1.0 + self.B()[:, None] * k ** 2)

    def partial_mask(self, partial_freqs: torch.Tensor) -> torch.Tensor:
        """1 for partials k <= K_p that lie below Nyquist, else 0 (88, K_max).

        K_p = min(K_max, floor((fs/2) / f0_p)); the extra check on f_{p,k} itself
        removes inharmonic partials that were pushed above Nyquist.
        """
        nyquist = self.sample_rate / 2.0
        k_p = torch.clamp(torch.floor(nyquist / self.f0().detach()), max=self.k_max)
        return ((self.k[None, :] <= k_p[:, None]) & (partial_freqs < nyquist)).float()

    def partial_gains(self) -> torch.Tensor:
        """g_{p,k} = exp(-ρ_p (k - 1)), shape (88, K_max)."""
        return torch.exp(-self.rho()[:, None] * (self.k[None, :] - 1.0))

    # ------------------------------------------------------------ spectra

    def templates_dense(self, n_fft: int) -> torch.Tensor:
        """Reference implementation of W_{p,r}(f): every partial at every bin.

        Simple but slow (an 88 x 40 x F tensor); kept to test :meth:`templates`.
        """
        freqs = getattr(self, f"freqs_{n_fft}")                       # (F,)
        sigma = self.kappa * self.sample_rate / n_fft
        fpk = self.partial_freqs()                                      # (88, K)
        gains = self.partial_gains() * self.partial_mask(fpk)           # (88, K)
        diff = freqs[None, None, :] - fpk[:, :, None]                   # (88, K, F)
        W = (gains[:, :, None] * torch.exp(-0.5 * (diff / sigma) ** 2)).sum(dim=1)
        return W / W.amax(dim=1, keepdim=True).clamp_min(1e-12)

    def templates(self, n_fft: int, n_sigma: float = 5.0) -> torch.Tensor:
        """W_{p,r}(f): one unit-max spectral template per key, shape (88, F_r).

        Same equation as :meth:`templates_dense`, but each Gaussian is evaluated
        only at the bins within ``n_sigma`` standard deviations of its partial
        (σ = 0.6 bins, so 7 bins per partial). Beyond 5σ a Gaussian is below
        4e-6 of its peak, so the result matches the dense version while being
        ~100x cheaper. ``scatter_add`` sums overlapping partials and is
        differentiable, so gradients still reach f_{p,k} (B, δ) and g_{p,k} (ρ).
        """
        bin_hz = self.sample_rate / n_fft
        sigma = self.kappa * bin_hz
        n_bins = n_fft // 2 + 1
        half = math.ceil(n_sigma * self.kappa)                          # half-width in bins
        fpk = self.partial_freqs()                                      # (88, K)
        gains = self.partial_gains() * self.partial_mask(fpk)           # (88, K)

        centre = torch.round(fpk.detach() / bin_hz).long()             # nearest bin
        offsets = torch.arange(-half, half + 1, device=fpk.device)
        bins = centre[:, :, None] + offsets                             # (88, K, 2h+1)
        inside = (bins >= 0) & (bins < n_bins)
        bins = bins.clamp(0, n_bins - 1)
        diff = bins.to(fpk.dtype) * bin_hz - fpk[:, :, None]
        values = gains[:, :, None] * torch.exp(-0.5 * (diff / sigma) ** 2) * inside

        W = torch.zeros(fpk.shape[0], n_bins, dtype=fpk.dtype, device=fpk.device)
        W = W.scatter_add(1, bins.reshape(fpk.shape[0], -1), values.reshape(fpk.shape[0], -1))
        return W / W.amax(dim=1, keepdim=True).clamp_min(1e-12)

    def activations(self) -> torch.Tensor:
        """A = softplus(Z), the non-negative latent piano roll (88, T)."""
        return F.softplus(self.Z)

    def noise_floor(self, n_fft: int) -> torch.Tensor:
        """β_r(f) = softplus(raw) ≥ 0, shape (F_r,)."""
        return F.softplus(self.beta_raw[str(n_fft)])

    def spectrogram(self, n_fft: int, A: torch.Tensor | None = None) -> torch.Tensor:
        """Ŝ_r = W_rᵀ A + β_r, shape (F_r, T). ``A`` defaults to softplus(Z)."""
        A = self.activations() if A is None else A
        return self.templates(n_fft).T @ A + self.noise_floor(n_fft)[:, None]

    def forward(self, A: torch.Tensor | None = None) -> dict[int, torch.Tensor]:
        """Predicted magnitude spectrogram at every resolution: ``{n_fft: (F_r, T)}``."""
        A = self.activations() if A is None else A
        return {n: self.spectrogram(n, A) for n in self.n_ffts}

    # ------------------------------------------------------------ export

    def physical_params(self) -> dict:
        """JSON-friendly summary of the fitted physical parameters."""
        with torch.no_grad():
            return {
                "delta_cents": float(self.delta),
                "a": [float(x) for x in self.a],
                "c": [float(x) for x in self.c],
                "B_mode": self.B_mode,
                "B_per_key": np.asarray(self.B().cpu()).tolist(),
                "rho_per_key": np.asarray(self.rho().cpu()).tolist(),
                "detune_cents_per_key": np.asarray(self.detune_cents().cpu()).tolist(),
            }
