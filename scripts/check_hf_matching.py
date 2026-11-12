"""Match the 177 HF mirror rows to local MAESTRO test MIDI files by md5 (M7 check).

Reads only the small midi/duration columns. Also compares the mirror duration with
the MAESTRO metadata duration of the matched file, and shows audio.path for one row.

Usage (from the repository root): python scripts/check_hf_matching.py
"""

import pandas as pd

from pianoabs.config import load_config
from pianoabs.data.hf_real_audio import index_mirror, local_midi_md5
from pianoabs.data.maestro import ensure_maestro

cfg = load_config("configs/default.yaml")
meta, midi_root = ensure_maestro(cfg)
lookup = local_midi_md5(meta, midi_root)
print("local test MIDI files:", len(lookup))
index = index_mirror(cfg, lookup)
matched = index["midi_filename"].notna()
print(f"mirror rows: {len(index)}, matched by md5: {matched.sum()}, unmatched: {(~matched).sum()}")
print("duplicate matches:", index["midi_filename"].dropna().duplicated().sum())

m = index[matched].merge(meta[["midi_filename", "duration"]], on="midi_filename",
                         suffixes=("_hf", "_meta"))
diff = (m["duration_hf"] - m["duration_meta"]).abs()
print(f"|duration_hf - duration_meta|: max {diff.max():.3f} s, median {diff.median():.3f} s")

manifest = pd.read_csv("data/manifests/excerpts_test.csv")
need = manifest.merge(index, on="midi_filename", how="left")
print(f"test manifest excerpts found in mirror: {need['shard'].notna().sum()} / {len(manifest)}")
print("shards needed:", sorted(need["shard"].dropna().map(lambda s: s.split('/')[-1]).unique()))
index.to_csv("data/hf_mirror_index.csv", index=False)
print("wrote data/hf_mirror_index.csv")
