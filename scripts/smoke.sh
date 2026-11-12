#!/usr/bin/env bash
# End-to-end smoke check (CPU, a few minutes). Grows with each milestone.
# Usage (from the repository root): bash scripts/smoke.sh
# Windows: set PIANOABS_FLUIDSYNTH to fluidsynth.exe first.
set -euo pipefail
CFG=configs/smoke.yaml
python -m pytest -q -m "not slow"
python -m pianoabs.cli info --config $CFG
# M2: data (validation manifest only, first 2 excerpts)
python -m pianoabs.cli prepare-data --config $CFG --manifests val --conditions C1
python -m pianoabs.cli sanity-plot --config $CFG --excerpt val_000 --condition C1
# M4: ours on C0 and C1 (tune on the 2 validation excerpts, then run on them)
for COND in C0 C1; do
  python -m pianoabs.cli tune --config $CFG --condition $COND
  python -m pianoabs.cli run --config $CFG --condition $COND
done
# M5: NMF baseline on C1, then metrics + summary + statistics
python -m pianoabs.cli tune --config $CFG --method nmf_harmonic --condition C1
python -m pianoabs.cli run --config $CFG --method nmf_harmonic --condition C1
python -m pianoabs.cli evaluate --config $CFG
