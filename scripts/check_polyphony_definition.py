"""Compare mean polyphony computed from the real GT (all sounding notes) with the
synthetic GT (only notes with an onset in the window, i.e. what C0-C2 audio contains).

Usage (from the repository root): python scripts/check_polyphony_definition.py
"""

import pandas as pd

from pianoabs.data.groundtruth import mean_polyphony

for which in ("val", "test"):
    m = pd.read_csv(f"data/manifests/excerpts_{which}.csv")
    rows = []
    for r in m.itertuples(index=False):
        syn = pd.read_csv(f"data/gt/synthetic/{r.excerpt_id}.csv")
        rows.append((r.excerpt_id, r.polyphony_bin, r.mean_polyphony,
                     mean_polyphony(syn, 10.0, 0.016)))
    d = pd.DataFrame(rows, columns=["id", "bin", "poly_real", "poly_synth"])
    d["ratio"] = d.poly_synth / d.poly_real
    print(f"\n=== {which}: ratio synth/real  median={d.ratio.median():.2f}  min={d.ratio.min():.2f}")
    print(d.groupby("bin", sort=False)[["poly_real", "poly_synth"]].mean().round(2))
    print("Spearman rank correlation real vs synth:",
          round(d.poly_real.corr(d.poly_synth, method="spearman"), 3))
    print(d.sort_values("ratio").head(4).round(2).to_string(index=False))
