"""Multi-resolution STFT (README §3.1) and FluidSynth rendering (§5.5)."""

from __future__ import annotations

import numpy as np
import pretty_midi
import pytest
import soundfile as sf

from pianoabs.audio import fix_length, peak_normalize
from pianoabs.features import bin_frequencies, multires_stft


def test_all_resolutions_share_frame_count():
    sr, hop = 16000, 256
    audio = np.random.default_rng(0).standard_normal(11 * sr).astype(np.float32)
    specs = multires_stft(audio, [1024, 2048, 4096], hop)
    frames = {n: s.shape[1] for n, s in specs.items()}
    assert set(frames.values()) == {1 + len(audio) // hop}
    assert {n: s.shape[0] for n, s in specs.items()} == {1024: 513, 2048: 1025, 4096: 2049}


@pytest.mark.parametrize("n_fft", [1024, 2048, 4096])
def test_sinusoid_peak_is_half_amplitude_at_every_resolution(n_fft):
    """Window-sum normalisation: amplitude a -> peak a/2, independent of n_fft."""
    sr, a = 16000, 0.6
    f = bin_frequencies(n_fft, sr)[100]  # exactly on a bin centre
    t = np.arange(2 * sr) / sr
    mag = multires_stft(a * np.sin(2 * np.pi * f * t), [n_fft], 256)[n_fft]
    middle = mag[:, mag.shape[1] // 2]
    assert int(middle.argmax()) == 100
    assert float(middle.max()) == pytest.approx(a / 2, rel=1e-3)


def test_fix_length_and_peak_normalize():
    x = np.array([0.1, -0.5, 0.2], dtype=np.float32)
    assert fix_length(x, 5).tolist() == pytest.approx([0.1, -0.5, 0.2, 0, 0])
    assert fix_length(x, 2).tolist() == pytest.approx([0.1, -0.5])
    assert np.max(np.abs(peak_normalize(x, 0.9))) == pytest.approx(0.9)
    with pytest.raises(ValueError):
        peak_normalize(np.zeros(3), 0.9)


def _fluidsynth_ready(cfg) -> bool:
    from pianoabs.data.render import find_fluidsynth, find_soundfont
    try:
        find_fluidsynth(cfg)
        find_soundfont(cfg, "fluidr3")
        return True
    except FileNotFoundError:
        return False


def test_render_toy_midi(default_cfg, tmp_path):
    if not _fluidsynth_ready(default_cfg):
        pytest.skip("fluidsynth or FluidR3_GM.sf2 not available")
    from pianoabs.data.render import find_soundfont, render_midi

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=69, start=0.5, end=1.5))
    pm.instruments.append(inst)
    mid, wav = tmp_path / "a.mid", tmp_path / "a.wav"
    pm.write(str(mid))
    render_midi(mid, wav, find_soundfont(default_cfg, "fluidr3"), default_cfg)

    audio, sr = sf.read(str(wav))
    assert sr == 16000 and audio.ndim == 1 and len(audio) == 11 * 16000
    assert np.max(np.abs(audio)) == pytest.approx(0.9, rel=1e-4)
    assert np.max(np.abs(audio[: int(0.45 * sr)])) < 0.01   # silent before the onset
    # energy near 440 Hz (A4) dominates
    spec = np.abs(np.fft.rfft(audio[int(0.6 * sr): int(1.4 * sr)]))
    freqs = np.fft.rfftfreq(int(0.8 * sr), 1 / sr)
    assert abs(freqs[spec.argmax()] - 440.0) < 5.0
