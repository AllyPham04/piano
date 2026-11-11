"""Small audio I/O helpers shared by all conditions (README §3.1, §5.5).

Every excerpt is stored as mono, 16 kHz, float32 WAV of exactly
``excerpt_s + tail_s`` seconds, peak-normalised to 0.9.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_mono(path: str | Path, sample_rate: int) -> np.ndarray:
    """Read an audio file, average channels to mono, resample to ``sample_rate``."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    return audio.astype(np.float32)


def fix_length(audio: np.ndarray, n_samples: int) -> np.ndarray:
    """Trim or zero-pad to exactly ``n_samples``."""
    if len(audio) >= n_samples:
        return audio[:n_samples]
    return np.pad(audio, (0, n_samples - len(audio)))


def peak_normalize(audio: np.ndarray, peak: float = 0.9) -> np.ndarray:
    """Scale so that max |x| = ``peak``. Raises on silent input (likely a rendering bug)."""
    current = float(np.max(np.abs(audio)))
    if current == 0.0:
        raise ValueError("cannot peak-normalise silent audio")
    return (audio * (peak / current)).astype(np.float32)


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sample_rate, subtype="FLOAT")
