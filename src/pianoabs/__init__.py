"""pianoabs: label-free piano transcription via differentiable analysis-by-synthesis.

See README.md for the full specification. Sub-packages:

- ``data``      MAESTRO MIDI, excerpt sampling, ground truth, rendering, real audio
- ``model``     differentiable spectral piano model, optimisation, note extraction
- ``baselines`` NMF, Basic Pitch, ByteDance runners
- ``eval``      mir_eval metrics, octave/harmonic errors, bootstrap CIs, Wilcoxon
- ``viz``       figures
"""

__version__ = "0.1.0"
