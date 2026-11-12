r"""Label-free NMF baseline with fixed harmonic templates (README §4).

The dictionary W (F x 88) is the spectral template of §3.2 at n_fft = 2048,
**fixed**: B = 0 (purely harmonic, ``nmf_harmonic``) or B = 3e-4 for all keys
(``nmf_inharmonic``), default roll-off ρ = 0.3, δ = 0. Only the activations
H (88 x T) are learned, with the multiplicative updates of Lee & Seung that
minimise the generalised KL divergence D(V || WH):

.. math::  H \leftarrow H \odot \frac{W^\top (V / (WH))}{W^\top \mathbf{1}}

on the (window-sum-normalised) STFT magnitude V, 200 iterations. H stays
non-negative by construction. Notes are extracted from H exactly as for our
model (§3.5), with thresholds tuned separately on validation.

The differences to "ours" are therefore: fixed physics (no learned B, δ, ρ),
one resolution, KL instead of compressed L1, no sparsity/smoothness terms and
no noise floor.
"""

from __future__ import annotations

import numpy as np
import torch

from pianoabs.model.spectral_piano import SpectralPiano

NMF_B_MODE = {"nmf_harmonic": "zero", "nmf_inharmonic": "fixed"}


def nmf_dictionary(cfg: dict, method: str) -> np.ndarray:
    """Fixed templates W (F, 88) at the NMF resolution."""
    n_fft = cfg["nmf"]["n_fft"]
    model_cfg = {**cfg, "model": {**cfg["model"], "B_mode": NMF_B_MODE[method]},
                 "features": {**cfg["features"], "n_ffts": [n_fft]}}
    model = SpectralPiano.from_config(model_cfg, n_frames=1)
    with torch.no_grad():
        return model.templates(n_fft).T.numpy().astype(np.float64)


def kl_nmf_activations(V: np.ndarray, W: np.ndarray, n_iter: int = 200, eps: float = 1e-10,
                       seed: int = 0) -> np.ndarray:
    """Learn H >= 0 for fixed W with KL multiplicative updates. V: (F, T), W: (F, K)."""
    rng = np.random.default_rng(seed)
    V = np.asarray(V, dtype=np.float64)
    H = rng.uniform(0.5, 1.5, size=(W.shape[1], V.shape[1])) * V.mean()
    denom = W.sum(axis=0)[:, None] + eps          # Wᵀ 1
    for _ in range(n_iter):
        H *= (W.T @ (V / (W @ H + eps))) / denom
    return H.astype(np.float32)


def kl_divergence(V: np.ndarray, WH: np.ndarray, eps: float = 1e-10) -> float:
    """Generalised KL divergence D(V || WH) = Σ V log(V / WH) - V + WH."""
    V, WH = V + eps, WH + eps
    return float(np.sum(V * np.log(V / WH) - V + WH))
