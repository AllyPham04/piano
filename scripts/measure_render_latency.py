"""Measure FluidSynth's onset latency: render notes at known times and find where
the sound starts (first sample above -40 dB of the note's peak).

Also runs the onset-envelope lag estimator of check_rendered_audio.py on the same
file, to tell estimator bias apart from real rendering latency.

Usage (from the repository root): python scripts/measure_render_latency.py
"""

import tempfile
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf

from pianoabs.config import load_config
from pianoabs.data.render import find_soundfont, render_midi

cfg = load_config("configs/default.yaml")
sr, hop = 16000, 256
onsets = [0.5, 2.0, 3.5, 5.0, 6.5, 8.0]
pitches = [36, 48, 60, 72, 84, 96]

def render(notes):
    """notes: list of (pitch, start, end) -> rendered 11 s mono audio."""
    pm = pretty_midi.PrettyMIDI(resolution=960)
    inst = pretty_midi.Instrument(program=0)
    for p, s, e in notes:
        inst.notes.append(pretty_midi.Note(velocity=90, pitch=p, start=s, end=e))
    pm.instruments.append(inst)
    with tempfile.TemporaryDirectory() as tmp:
        mid, wav = Path(tmp) / "t.mid", Path(tmp) / "t.wav"
        pm.write(str(mid))
        render_midi(mid, wav, find_soundfont(cfg, "fluidr3"), cfg)
        return sf.read(str(wav), dtype="float32")[0]


print("one note per render (onset 0.5 s):")
print("pitch  audio_start  latency_ms  peak_after_ms")
for p in pitches:
    audio = render([(p, 0.5, 1.5)])
    a = np.abs(audio)
    first = int(np.argmax(a > a.max() * 10 ** (-40 / 20)))
    peak = int(np.argmax(a))
    print(f"{p:5d}  {first / sr:11.4f}  {1000 * (first / sr - 0.5):10.1f}  {1000 * (peak / sr - 0.5):13.1f}")

audio = render([(p, t, t + 1.0) for t, p in zip(onsets, pitches)])
env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop)
target = np.zeros_like(env)
target[np.round(np.array(onsets) * sr / hop).astype(int)] = 1.0
lags = np.arange(-6, 7)
best = lags[int(np.argmax([np.dot(np.roll(target, k), env) for k in lags]))]
print(f"\nonset-envelope estimator on this file: best lag = {best * hop / sr * 1000:.0f} ms")
