"""FluidSynth rendering of excerpt MIDI files (README §5.5, conditions C1/C2).

The FluidSynth command-line tool is used rather than ``pyfluidsynth`` (fewer
binding problems)::

    fluidsynth -ni -g 1.0 -r 16000 -O float -F out.wav soundfont.sf2 excerpt.mid

``-O float`` writes 32-bit float WAV so loud chords cannot clip before we
peak-normalise (the only change to the README command; reverb and chorus stay
at FluidSynth's defaults). The output is then converted to mono, trimmed or
padded to excerpt + tail (11 s) and peak-normalised to 0.9.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from pianoabs.audio import fix_length, load_mono, peak_normalize, write_wav


def find_soundfont(cfg: dict, name: str) -> Path:
    """First existing path in ``data.render.soundfonts[name]``."""
    candidates = cfg["data"]["render"]["soundfonts"][name]
    for candidate in candidates:
        if Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(f"soundfont '{name}' not found; looked in {candidates}")


def find_fluidsynth(cfg: dict) -> str:
    """Path of the fluidsynth executable.

    Order: environment variable ``PIANOABS_FLUIDSYNTH`` (machine-specific path,
    e.g. on Windows), then ``data.render.fluidsynth_bin`` (default ``fluidsynth``,
    found on PATH as on Kaggle after apt-get).
    """
    configured = os.environ.get("PIANOABS_FLUIDSYNTH") or cfg["data"]["render"]["fluidsynth_bin"]
    found = shutil.which(configured) or (configured if Path(configured).is_file() else None)
    if found is None:
        raise FileNotFoundError(
            f"fluidsynth not found ('{configured}'). Kaggle: apt-get install -y fluidsynth. "
            "Windows: unzip the release from github.com/FluidSynth/fluidsynth/releases and "
            "set the environment variable PIANOABS_FLUIDSYNTH to fluidsynth.exe.")
    return found


def render_midi(midi_path: Path, wav_path: Path, soundfont: Path, cfg: dict) -> None:
    """Render one excerpt MIDI to a mono, fixed-length, peak-normalised WAV."""
    data, r = cfg["data"], cfg["data"]["render"]
    sr = data["sample_rate"]
    with tempfile.TemporaryDirectory() as tmp:
        raw_wav = Path(tmp) / "raw.wav"
        cmd = [find_fluidsynth(cfg), "-ni", "-q", "-g", str(r["gain"]), "-r", str(sr),
               *r["extra_args"], "-F", str(raw_wav), str(soundfont), str(midi_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not raw_wav.exists():
            raise RuntimeError(f"fluidsynth failed ({result.returncode}): {result.stderr[-500:]}")
        audio = load_mono(raw_wav, sr)
    n_samples = int(round((data["excerpt_s"] + data["tail_s"]) * sr))
    audio = peak_normalize(fix_length(audio, n_samples), data["peak_norm"])
    write_wav(wav_path, audio, sr)
