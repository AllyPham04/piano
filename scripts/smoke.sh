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
