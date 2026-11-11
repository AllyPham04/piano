"""M2 check of rendered audio: length, peak, RMS, and audio-vs-GT onset alignment.

Alignment: cross-correlate the librosa onset-strength envelope with an impulse
train at the GT onsets; the best lag (in ms) should be ~0 if labels and audio agree.

Usage (from the repository root): python scripts/check_rendered_audio.py [condition]
"""

import sys

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

condition = sys.argv[1] if len(sys.argv) > 1 else "C1"
sr, hop = 16000, 256
rows = []
for which in ("val", "test"):
    for eid in pd.read_csv(f"data/manifests/excerpts_{which}.csv")["excerpt_id"]:
        audio, file_sr = sf.read(f"data/audio/{condition}/{eid}.wav", dtype="float32")
        gt = pd.read_csv(f"data/gt/synthetic/{eid}.csv")
        env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop)
        target = np.zeros_like(env)
        idx = np.round(gt["onset_s"].to_numpy() * sr / hop).astype(int)
        target[idx[idx < len(target)]] = 1.0
        lags = np.arange(-6, 7)  # +-96 ms
        scores = [np.dot(np.roll(target, lag), env) for lag in lags]
        rows.append({
            "excerpt_id": eid, "sr": file_sr, "seconds": len(audio) / file_sr,
            "peak": float(np.abs(audio).max()),
            "rms_db": float(20 * np.log10(np.sqrt(np.mean(audio[: 10 * sr] ** 2)) + 1e-12)),
            "best_lag_ms": float(lags[int(np.argmax(scores))] * hop / sr * 1000),
        })
d = pd.DataFrame(rows)
print(d.describe().round(3).to_string())
print("\nlag histogram (ms):", d["best_lag_ms"].value_counts().sort_index().to_dict())
