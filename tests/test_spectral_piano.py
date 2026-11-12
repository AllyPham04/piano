"""Spectral piano model (README §3.2) and C0 synthesis (§5.5)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from pianoabs.model.c0 import gt_activation_roll, n_frames_for, synthesize_c0
from pianoabs.model.spectral_piano import SpectralPiano, inverse_softplus

SR = 16000


def key(pitch: int) -> int:
    return pitch - 21


@pytest.fixture
def model():
    return SpectralPiano(n_frames=8)


# ---------------------------------------------------------------- shapes and init

def test_template_shapes_unit_max_nonnegative(model):
    for n in (1024, 2048, 4096):
        W = model.templates(n)
        assert W.shape == (88, n // 2 + 1)
        assert torch.all(W >= 0)
        assert torch.allclose(W.amax(dim=1), torch.ones(88))


@pytest.mark.parametrize("kwargs", [{}, {"B_init": 5e-3, "rho_init": 0.05}, {"B_mode": "zero"}])
def test_banded_templates_match_dense(kwargs):
    m = SpectralPiano(n_frames=4, **kwargs)
    with torch.no_grad():
        m.delta.fill_(7.3)  # partials off the bin centres
    for n in (1024, 2048, 4096):
        assert torch.allclose(m.templates(n), m.templates_dense(n), atol=1e-5)


def test_banded_template_gradients_match_dense():
    grads = []
    for method in ("templates", "templates_dense"):
        m = SpectralPiano(n_frames=4)
        weights = torch.linspace(0, 1, 1025)
        (getattr(m, method)(2048) * weights).sum().backward()
        grads.append((m.a.grad.clone(), m.c.grad.clone(), m.delta.grad.clone()))
    for g_band, g_dense in zip(*grads):
        assert torch.allclose(g_band, g_dense, rtol=1e-3, atol=1e-5)


def test_spectrogram_shapes_and_init(model):
    specs = model()
    assert {n: tuple(s.shape) for n, s in specs.items()} == {
        1024: (513, 8), 2048: (1025, 8), 4096: (2049, 8)}
    assert torch.allclose(model.activations(), torch.full((88, 8), 1e-3), rtol=1e-4)
    assert torch.allclose(model.noise_floor(2048), torch.full((1025,), 1e-4), rtol=1e-3)


def test_inverse_softplus():
    for y in (1e-4, 1e-3, 0.5, 3.0):
        assert torch.nn.functional.softplus(torch.tensor(inverse_softplus(y))).item() == pytest.approx(y, rel=1e-5)


# ---------------------------------------------------------------- f0 and partials

def test_f0_and_global_tuning(model):
    assert model.f0()[key(69)].item() == pytest.approx(440.0)
    assert model.f0()[key(21)].item() == pytest.approx(27.5)
    with torch.no_grad():
        model.delta.fill_(1200.0)  # one octave up
    assert model.f0()[key(69)].item() == pytest.approx(880.0)


def test_per_key_detune_is_bounded():
    m = SpectralPiano(n_frames=4, per_key_detune=True, max_detune_cents=20.0)
    with torch.no_grad():
        m.detune_raw.fill_(100.0)
    assert m.detune_cents().max().item() == pytest.approx(20.0)


def test_partials_harmonic_when_B_zero():
    m = SpectralPiano(n_frames=4, B_mode="zero")
    fpk = m.partial_freqs()
    f0 = m.f0()
    assert torch.allclose(fpk, m.k[None, :] * f0[:, None])


def test_partials_follow_stiff_string_formula(model):
    B = 3e-4
    assert torch.allclose(model.B(), torch.full((88,), B), rtol=1e-5)
    f0 = 440.0 * 2 ** ((40 - 69) / 12)
    k = np.arange(1, 41)
    expected = k * f0 * np.sqrt(1 + B * k ** 2)
    assert np.allclose(model.partial_freqs()[key(40)].detach().numpy(), expected, rtol=1e-5)
    # partials are stretched: the 20th partial lies above 20 * f0
    assert expected[19] > 20 * f0 * 1.05


def test_template_peaks_sit_on_inharmonic_partials():
    """Local maxima of the 4096-point template are within one bin of f_{p,k}."""
    m = SpectralPiano(n_frames=4, B_init=1e-3, rho_init=0.05)
    W = m.templates(4096)[key(33)].detach().numpy()
    freqs = np.arange(2049) * SR / 4096
    for k in (1, 5, 10, 15):
        f = m.partial_freqs()[key(33), k - 1].item()
        near = (freqs > f - 20) & (freqs < f + 20)
        peak = freqs[near][np.argmax(W[near])]
        assert abs(peak - f) <= SR / 4096, (k, f, peak)
    # ... and not on the harmonic positions k * f0 for high k
    f0 = m.f0()[key(33)].item()
    harmonic_bin = int(round(15 * f0 * 4096 / SR))
    inharmonic_bin = int(round(m.partial_freqs()[key(33), 14].item() * 4096 / SR))
    assert inharmonic_bin - harmonic_bin > 3
    assert W[inharmonic_bin] > 10 * W[harmonic_bin]


def test_nyquist_mask():
    m = SpectralPiano(n_frames=4, B_mode="zero")
    mask = m.partial_mask(m.partial_freqs())
    n_partials = mask.sum(dim=1)
    assert n_partials[key(21)] == 40                  # A0: 27.5 Hz, capped at K_max
    assert n_partials[key(108)] == 1                  # C8: 4186 Hz, 2nd partial > 8 kHz
    assert n_partials[key(96)] == math.floor(8000 / (440 * 2 ** ((96 - 69) / 12)))
    # with inharmonicity, partials pushed above Nyquist are removed too
    mi = SpectralPiano(n_frames=4, B_init=5e-2)
    fpk = mi.partial_freqs()
    assert torch.all(fpk[mi.partial_mask(fpk).bool()] < 8000)


def test_B_clamp_and_curve():
    m = SpectralPiano(n_frames=4)
    with torch.no_grad():
        m.a.copy_(torch.tensor([10.0, 0.0, 0.0]))
    assert torch.allclose(m.B(), torch.full((88,), 5e-2))
    with torch.no_grad():
        m.a.copy_(torch.tensor([math.log(3e-4), 1.0, 0.0]))
    B = m.B()
    assert B[key(84)].item() == pytest.approx(3e-4 * math.e, rel=1e-4)  # u = 1
    assert torch.all(B[1:] >= B[:-1])  # increasing with a1 > 0


def test_variant_modes_freeze_parameters():
    assert SpectralPiano(n_frames=4, B_mode="zero").a.requires_grad is False
    assert SpectralPiano(n_frames=4, B_mode="fixed").a.requires_grad is False
    assert SpectralPiano(n_frames=4, learn_rho=False).c.requires_grad is False
    assert SpectralPiano(n_frames=4).a.requires_grad is True


# ---------------------------------------------------------------- gradients

def test_gradients_reach_all_learnable_parameters():
    m = SpectralPiano(n_frames=6, per_key_detune=True)
    loss = sum(torch.log1p(100 * s).sum() for s in m().values())
    loss.backward()
    for name in ("Z", "delta", "detune_raw", "a", "c"):
        g = getattr(m, name).grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, name
    for n, p in m.beta_raw.items():
        assert p.grad is not None and p.grad.abs().sum() > 0, n


def _one_key_target(pitch: int, **kwargs) -> dict:
    """Spectrogram of a single active key from a model with the given parameters."""
    m = SpectralPiano(n_frames=1, **kwargs)
    A = torch.zeros(88, 1)
    A[key(pitch)] = 1.0
    with torch.no_grad():
        return {n: s for n, s in m(A).items()}, A


@pytest.mark.parametrize("name, target_kwargs, param, expect_sign", [
    ("B larger", {"B_init": 4e-4}, "a", -1),   # dL/da0 < 0  -> a0 (log B) increases
    ("B smaller", {"B_init": 2e-4}, "a", +1),
])
def test_gradient_points_towards_true_B(name, target_kwargs, param, expect_sign):
    target, A = _one_key_target(45, **target_kwargs)
    m = SpectralPiano(n_frames=1)
    loss = sum((torch.log1p(100 * m(A)[n]) - torch.log1p(100 * target[n])).abs().mean()
               for n in m.n_ffts)
    loss.backward()
    assert np.sign(m.a.grad[0].item()) == expect_sign


def test_gradient_points_towards_true_tuning():
    target, A = _one_key_target(57)
    m = SpectralPiano(n_frames=1)
    with torch.no_grad():
        m.delta.fill_(-8.0)  # model is 8 cents flat
    loss = sum((torch.log1p(100 * m(A)[n]) - torch.log1p(100 * target[n])).abs().mean()
               for n in m.n_ffts)
    loss.backward()
    assert m.delta.grad.item() < 0  # gradient descent raises delta towards 0


def test_from_fitted_roundtrip(default_cfg):
    import json

    m = SpectralPiano.from_config(default_cfg, n_frames=5)
    with torch.no_grad():
        m.Z.normal_()
        m.delta.fill_(3.0)
        m.a.copy_(torch.tensor([-7.5, 0.4, 0.1]))
        m.beta_raw["2048"].fill_(-6.0)
    params = json.loads(json.dumps(m.physical_params()))  # through JSON, as on disk
    A = m.activations().detach().numpy()
    rebuilt = SpectralPiano.from_fitted(default_cfg, params, A)
    for n in m.n_ffts:
        assert torch.allclose(rebuilt()[n], m()[n].detach(), rtol=1e-4, atol=1e-7)


# ---------------------------------------------------------------- C0

def test_gt_activation_roll_decays_and_stops():
    notes = pd.DataFrame({"onset_s": [0.016], "offset_s": [0.08], "pitch": [60],
                          "velocity": [127]})
    roll = gt_activation_roll(notes, n_frames=8, frame_s=0.016, tau_s=1.5)
    row = roll[key(60)]
    assert row[0] == 0 and row[1] == pytest.approx(1.0)
    assert row[2] == pytest.approx(math.exp(-0.016 / 1.5), rel=1e-5)
    assert row[5:].sum() == 0  # offset at 0.08 s = frame 5
    assert roll.sum() == pytest.approx(row.sum())


def test_restrike_creates_rise():
    notes = pd.DataFrame({"onset_s": [0.0, 0.5], "offset_s": [0.5, 1.0], "pitch": [60, 60],
                          "velocity": [100, 100]})
    row = gt_activation_roll(notes, 64, 0.016, 1.5)[key(60)]
    i = int(np.ceil(0.5 / 0.016))
    assert row[i] > row[i - 1] * 1.2


def test_synthesize_c0(default_cfg):
    cfg = {**default_cfg, "c0": {**default_cfg["c0"], "amplitude": 0.1}}
    notes = pd.DataFrame({"onset_s": [1.0], "offset_s": [2.0], "pitch": [69], "velocity": [127]})
    specs, roll = synthesize_c0(notes, cfg)
    T = n_frames_for(cfg)
    assert T == 688 and roll.shape == (88, T)
    S = specs[4096]
    assert S.shape == (2049, T)
    t = int(1.0 / 0.016) + 1
    assert int(S[:, t].argmax()) == round(440 * 4096 / SR)
    assert S[:, t].max().item() == pytest.approx(0.1 * math.exp(-0.016 / 1.5) * 1 + 1e-4, rel=0.02)
    # B0 variant config must not change the C0 data
    cfg_b0 = {**cfg, "model": {**cfg["model"], "B_mode": "zero"}}
    assert torch.allclose(synthesize_c0(notes, cfg_b0)[0][4096], S)
