#!/usr/bin/env bash
# End-to-end smoke check (CPU, a few minutes). Grows with each milestone.
# Usage (from the repository root): bash scripts/smoke.sh
set -euo pipefail
python -m pytest -q -m "not slow"
python -m pianoabs.cli info --config configs/smoke.yaml
