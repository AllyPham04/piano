r"""Objective and per-excerpt optimisation (README §3.3–3.4).

Objective (A = softplus(Z), c(S) = log(1 + γ S)):

.. math::

    L = \sum_r \frac{1}{|F||T|} \| c(S_r) - c(\hat S_r) \|_1
        + λ_{sp}\, \overline{A}
        + λ_{tv}\, \overline{|A_{:,t+1} - A_{:,t}|}
        + λ_B (a_1^2 + a_2^2)

* the L1 error on compressed magnitudes treats quiet and loud partials more
  evenly than a raw-magnitude loss;
* λ_sp (sparsity) prefers few active keys, so harmonics are not explained by
  extra notes; λ_tv (temporal smoothness) discourages flicker; λ_B keeps the
  B(p) curve close to a constant in log space (only when B is learned).

Optimisation, per excerpt, with Adam and no training data:

* stage 1 (300 steps): only Z and β; physical parameters frozen at their
  initial values, so the templates are computed once;
* stage 2 (1200 steps): δ (δ_p), a and c join at lr 0.01; cosine decay of all
  learning rates over this stage.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from pianoabs.features import compress, multires_stft
from pianoabs.model.spectral_piano import SpectralPiano
from pianoabs.utils import seed_everything


@dataclass
class FitResult:
    activations: np.ndarray           # (88, T) float32, A = softplus(Z)
    params: dict                      # physical parameters (JSON-friendly)
    loss_history: list[dict] = field(default_factory=list)
    runtime_s: float = 0.0
    device: str = "cpu"
    model: SpectralPiano | None = None
    snapshots: dict[int, np.ndarray] = field(default_factory=dict)  # step -> activations


def audio_targets(audio: np.ndarray, cfg: dict, device: str) -> dict[int, torch.Tensor]:
    """Target magnitude spectrograms of an audio excerpt at every resolution."""
    f = cfg["features"]
    return multires_stft(audio, f["n_ffts"], f["hop"], f["normalize"], device=device)


def amplitude_scale(targets: dict[int, torch.Tensor], cfg: dict) -> float:
    """Per-excerpt amplitude scale s: a high quantile of the target |S| (README §3.2 note).

    Uses the resolution ``model.amplitude_scale_n_fft`` if it is fitted, else the
    first available one; ``amplitude_scale_quantile: null`` disables scaling (s = 1).
    """
    m = cfg["model"]
    q = m["amplitude_scale_quantile"]
    if q is None:
        return 1.0
    ref = targets.get(m["amplitude_scale_n_fft"], next(iter(targets.values())))
    return max(float(torch.quantile(ref.detach().flatten().float().cpu(), q)), 1e-8)


def loss_bin_masks(cfg: dict, n_ffts: list[int], device: str) -> dict[int, torch.Tensor]:
    """Which frequency bins enter the loss (all, or only those <= loss_fmax_hz)."""
    fmax = cfg["features"]["loss_fmax_hz"]
    sr = cfg["data"]["sample_rate"]
    masks = {}
    for n in n_ffts:
        freqs = torch.arange(n // 2 + 1, device=device) * sr / n
        masks[n] = freqs <= (fmax if fmax is not None else sr / 2)
    return masks


def objective(model: SpectralPiano, targets_c: dict[int, torch.Tensor],
              masks: dict[int, torch.Tensor], cfg: dict,
              templates: dict[int, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    """All loss terms; ``targets_c`` are the compressed targets c(S_r)."""
    fit, gamma = cfg["fit"], cfg["features"]["gamma"]
    A = model.activations()
    pred = model(A, templates)
    recon = sum((targets_c[n][masks[n]] - compress(pred[n][masks[n]], gamma)).abs().mean()
                for n in model.n_ffts)
    sparsity = A.mean()
    tv = (A[:, 1:] - A[:, :-1]).abs().mean()
    b_prior = (model.a[1] ** 2 + model.a[2] ** 2) if model.B_mode == "learned" else A.new_zeros(())
    total = (recon + fit["lambda_sp"] * sparsity + fit["lambda_tv"] * tv
             + fit["lambda_B"] * b_prior)
    return {"total": total, "recon": recon, "sparsity": sparsity, "tv": tv, "B_prior": b_prior}


def _set_lr(optimizer: torch.optim.Optimizer, base_lrs: list[float], factor: float) -> None:
    for group, base in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base * factor


def fit_excerpt(targets: dict[int, torch.Tensor], cfg: dict, device: str = "cpu",
                keep_model: bool = False, snapshot_steps: tuple[int, ...] = ()) -> FitResult:
    """Fit the spectral piano model to one excerpt's target spectrograms.

    ``snapshot_steps``: optional step numbers (1-based count of completed steps)
    at which a copy of the activations is kept, for diagnostics.
    """
    snapshots: dict[int, np.ndarray] = {}

    def maybe_snapshot(done: int) -> None:
        if done in snapshot_steps:
            snapshots[done] = model.activations().detach().cpu().numpy().astype(np.float32)
    seed_everything(cfg["seed"])
    fit = cfg["fit"]
    n_frames = next(iter(targets.values())).shape[1]
    model = SpectralPiano.from_config(cfg, n_frames).to(device)
    model.amplitude_scale.fill_(amplitude_scale(targets, cfg))
    gamma = cfg["features"]["gamma"]
    targets_c = {n: compress(targets[n].to(device), gamma) for n in model.n_ffts}
    masks = loss_bin_masks(cfg, list(model.n_ffts), device)

    start = time.perf_counter()
    history: list[dict] = []

    def log(step: int, stage: int, terms: dict) -> None:
        history.append({"step": step, "stage": stage,
                        **{k: round(float(v.detach()), 6) for k, v in terms.items()}})

    # ---- stage 1: Z and β only; templates fixed (computed once)
    optimizer = torch.optim.Adam([
        {"params": [model.Z], "lr": fit["lr_Z"]},
        {"params": list(model.beta_raw.parameters()), "lr": fit["lr_beta"]},
    ])
    with torch.no_grad():
        frozen_templates = {n: model.templates(n) for n in model.n_ffts}
    for step in range(fit["stage1_steps"]):
        optimizer.zero_grad()
        terms = objective(model, targets_c, masks, cfg, frozen_templates)
        terms["total"].backward()
        optimizer.step()
        log(step, 1, terms)
        maybe_snapshot(step + 1)

    # ---- stage 2: + physical parameters, cosine-decayed learning rates
    physical = model.physical_parameters()
    if physical:
        optimizer.add_param_group({"params": physical, "lr": fit["lr_physical"]})
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    n2 = fit["stage2_steps"]
    for i in range(n2):
        if fit["cosine_decay"]:
            _set_lr(optimizer, base_lrs, 0.5 * (1.0 + math.cos(math.pi * i / n2)))
        optimizer.zero_grad()
        terms = objective(model, targets_c, masks, cfg)
        terms["total"].backward()
        optimizer.step()
        log(fit["stage1_steps"] + i, 2, terms)
        maybe_snapshot(fit["stage1_steps"] + i + 1)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    runtime = time.perf_counter() - start
    return FitResult(
        activations=model.activations().detach().cpu().numpy().astype(np.float32),
        params=model.physical_params(), loss_history=history, runtime_s=runtime,
        device=device, model=model if keep_model else None, snapshots=snapshots)
