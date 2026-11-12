"""C3 helpers (README §5.2): md5 matching and cropping. No network access."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pianoabs.data.hf_real_audio import crop_to_window, local_midi_md5, md5


def test_md5_lookup(tmp_path):
    (tmp_path / "2004").mkdir()
    (tmp_path / "2004" / "a.midi").write_bytes(b"MThd-a")
    (tmp_path / "2004" / "b.midi").write_bytes(b"MThd-b")
    meta = pd.DataFrame({"split": ["test", "train"],
                         "midi_filename": ["2004/a.midi", "2004/b.midi"]})
    lookup = local_midi_md5(meta, tmp_path)
    assert lookup == {md5(b"MThd-a"): "2004/a.midi"}   # train file not included


def test_crop_to_window_resamples_and_normalises(default_cfg):
    sr = 44100
    t = np.arange(30 * sr) / sr
    stereo = np.stack([np.sin(2 * np.pi * 440 * t), 0.5 * np.sin(2 * np.pi * 440 * t)], axis=1)
    stereo[: 12 * sr] *= 0.1          # quiet before 12 s
    out = crop_to_window(stereo.astype(np.float32), sr, t0=12.0, cfg=default_cfg)
    assert out.shape == (11 * 16000,)
    assert np.abs(out).max() == pytest.approx(0.9, rel=1e-4)
    spec = np.abs(np.fft.rfft(out))
    assert abs(np.fft.rfftfreq(len(out), 1 / 16000)[spec.argmax()] - 440) < 1.0


def test_crop_near_end_is_zero_padded(default_cfg):
    sr = 16000
    audio = np.ones(20 * sr, dtype=np.float32)
    out = crop_to_window(audio, sr, t0=15.0, cfg=default_cfg)   # only 5 s left
    assert out.shape == (11 * 16000,) and np.all(out[6 * sr:] == 0)
