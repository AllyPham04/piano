"""Print a summary of the frozen manifests and the MAESTRO metadata (M2 check).

Usage (from the repository root): python scripts/inspect_manifests.py
"""

import pandas as pd

meta = pd.read_csv("data/raw/maestro-v3.0.0.csv")
print("metadata performances per split:")
print(meta.groupby("split").agg(n=("midi_filename", "size"), hours=("duration", lambda d: d.sum() / 3600)).round(1))

for which in ("val", "test"):
    m = pd.read_csv(f"data/manifests/excerpts_{which}.csv")
    print(f"\n=== {which}: {len(m)} excerpts, {m.midi_filename.nunique()} distinct performances")
    print(m.groupby("polyphony_bin", sort=False).agg(
        n=("excerpt_id", "size"),
        poly_min=("mean_polyphony", "min"), poly_mean=("mean_polyphony", "mean"),
        poly_max=("mean_polyphony", "max"), notes_mean=("n_notes_evaluated", "mean"),
        pedal_mean=("pedal_fraction", "mean")).round(2))
    print(m[["excerpt_id", "canonical_composer", "start_s", "n_notes_evaluated",
             "mean_polyphony", "polyphony_bin", "pedal_fraction"]].head(6).to_string(index=False))
