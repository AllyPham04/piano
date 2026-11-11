# Label-Free Piano Transcription via Differentiable Analysis-by-Synthesis

> **Status:** planned / in progress (Nov 2026) · **Author:** Pham Ngoc Linh · **Compute:** local CPU for smoke tests, Kaggle GPU (T4) for full runs
>
> A short research project. It asks how far piano music can be transcribed **without any labels**, by fitting a small, physics-informed, **differentiable** model of piano sound to a recording with gradient descent ("analysis-by-synthesis"). The results are compared with a classical label-free baseline (NMF) and with supervised deep-learning transcribers.

---

## Table of contents

1. [Motivation](#1-motivation)
2. [Research questions and hypotheses](#2-research-questions-and-hypotheses)
3. [Method](#3-method)
4. [Baselines](#4-baselines)
5. [Data](#5-data)
6. [Evaluation](#6-evaluation)
7. [Experiments](#7-experiments)
8. [Required outputs (deliverables)](#8-required-outputs-deliverables)
9. [Repository structure](#9-repository-structure)
10. [Environment and compute](#10-environment-and-compute)
11. [Timeline and definition of done](#11-timeline-and-definition-of-done)
12. [Risks and mitigations](#12-risks-and-mitigations)
13. [Licences and attribution](#13-licences-and-attribution)
14. [References](#14-references)

---

## 1. Motivation

Automatic music transcription (AMT) turns audio into a symbolic score, for example MIDI notes with an onset, an offset, a pitch and a velocity. The best piano transcribers are supervised neural networks trained on large aligned datasets such as MAESTRO. They work very well on recordings that sound like their training data, but they need labels and their internal representations are hard to interpret.

**Analysis-by-synthesis** takes another route. We write down a *generative* model of how piano notes produce sound. Then we look for the notes that, when synthesised, best explain the recording. If the synthesiser is **differentiable**, the search can be done with gradient descent. The latent variables are the notes themselves (a piano roll), so they are interpretable and can be **intervened on**: change one note, re-synthesise, and hear the difference.

This project builds a deliberately small version of that idea:

- a **spectral** (not waveform) piano model with a handful of physically meaningful parameters per key: fundamental frequency, string **inharmonicity** *B*, partial decay;
- **per-recording optimisation**, with no training set and no labels;
- a careful, honest comparison with label-free and supervised baselines, plus an error analysis.

It is a small-scale exploration of the themes in CaCTüS 2027 Project 15 ("Piano Transcription with Differentiable Synthesizers", MPI-IS Empirical Inference): differentiable synthesis, label-free transcription and intervenable latent spaces.

---

## 2. Research questions and hypotheses

| ID | Question | Hypothesis |
|---|---|---|
| **RQ1** | How well does pure analysis-by-synthesis recover notes compared with (a) NMF with fixed harmonic templates, which is also label-free, and (b) supervised transcribers? | It beats NMF on note-onset F1, because it learns the template parameters, and falls clearly below the supervised models. |
| **RQ2** | Does modelling and **learning string inharmonicity** *B* reduce **octave errors**? | Yes. Without *B*, upper partials of low notes drift away from integer multiples, and the model "explains" them with extra notes an octave or more higher. |
| **RQ3** | How does performance degrade with **polyphony** (simultaneous notes), **sustain pedal** and **timbre mismatch** (synthetic piano → real recording)? | F1 drops steadily as polyphony rises; the real-audio condition is the hardest. |
| **RQ4** (qualitative) | Is the latent piano roll **intervenable**? Do edited notes re-synthesise into the expected spectral change? | Yes, by construction. This is shown as a demo, not measured. |

These are a pilot study's expectations, not claims. Report whatever comes out, including negative results.

---

## 3. Method

### 3.1 Signal representation

- Audio: **mono, 16 kHz, float32**, peak-normalised to 0.9 for each excerpt.
- STFT magnitude at **three resolutions** *r* ∈ {1024, 2048, 4096} (`n_fft`), Hann window (`win_length = n_fft`), shared **hop = 256 samples (16 ms)** and `center=True`, so every resolution has the same number of frames *T*.
- Frequency axis: bins up to 8 kHz (Nyquist). Optionally ignore bins above 5 kHz in the loss (configurable).
- Compression for the loss: `c(S) = log(1 + γ·S)`, with γ = 100 (configurable).

### 3.2 Differentiable spectral piano model

Keys *p* ∈ {21, …, 108} (A0–C8, 88 keys).

**Fundamental frequency** (with a learnable global tuning offset δ in cents, plus an optional small per-key offset δ_p, bounded to ±20 cents through `tanh`):

$$f_{0,p} = 440 \cdot 2^{(p-69)/12} \cdot 2^{(\delta + \delta_p)/1200}$$

**Partial frequencies with inharmonicity** (stiff-string model):

$$f_{p,k} = k \, f_{0,p} \sqrt{1 + B_p k^2}, \qquad k = 1, \dots, K_p, \quad K_p = \min\!\left(K_{\max}, \left\lfloor \tfrac{f_s/2}{f_{0,p}} \right\rfloor\right), \; K_{\max} = 40$$

**Inharmonicity as a smooth function of the key**, so that 88 keys do not each get an unconstrained *B* (*u* = (p − 60)/24):

$$\log B_p = a_0 + a_1 u + a_2 u^2, \qquad a_0^{\text{init}} = \log(3\cdot10^{-4}),\; a_1^{\text{init}} = a_2^{\text{init}} = 0$$

Clamp *B_p* to [1e-6, 5e-2].

**Partial amplitudes** (spectral envelope), also smooth across keys:

$$g_{p,k} = \exp\!\big(-\rho_p (k-1)\big), \qquad \log \rho_p = c_0 + c_1 u$$

An optional richer variant adds a learnable non-negative residual per (octave band, *k*).

**Spectral template of key *p* at resolution *r*.** Each partial is a Gaussian approximation of the Hann-window main lobe, with width σ_r = κ · f_s / n_fft_r (κ ≈ 0.6, configurable):

$$W_{p,r}(f) = \sum_{k=1}^{K_p} g_{p,k}\, \exp\!\left(-\frac{(f - f_{p,k})^2}{2\sigma_r^2}\right)$$

Zero out partials above Nyquist. Normalise each template to unit max.

**Spectrogram model** (assumes magnitudes add, as NMF-based transcription does):

$$\hat S_r(f,t) = \sum_{p} W_{p,r}(f)\, A_{p,t} + \beta_r(f)$$

- `A = softplus(Z)`, with `Z ∈ R^{88×T}`: the **latent piano roll** (activations).
- `β_r(f) ≥ 0`: a learnable noise floor per frequency bin (softplus).

**Learnable parameters:** `Z` (88×T), δ, (δ_p), a₀, a₁, a₂, c₀, c₁, β_r. Everything else is fixed.

### 3.3 Objective

$$\mathcal{L} = \sum_{r} \frac{1}{|F||T|}\left\| c(S_r) - c(\hat S_r) \right\|_1 + \lambda_{\text{sp}}\, \overline{A} + \lambda_{\text{tv}}\, \overline{|A_{:,t+1} - A_{:,t}|} + \lambda_B \,(a_1^2 + a_2^2)$$

- λ_sp (sparsity, encourages few notes): default 1e-2; swept in ablation A3.
- λ_tv (temporal smoothness): default 1e-3.
- λ_B: weak prior that keeps the *B* curve smooth (default 1e-3).

### 3.4 Optimisation

- **Per excerpt**, no training set. Adam.
- Stage 1 (warm-up, 300 steps): optimise only `Z` and β, with physical parameters frozen at their initial values. lr(Z) = 0.05.
- Stage 2 (joint, 1,200 steps): add δ, a, c. lr(physical) = 0.01. Cosine decay.
- Initialise `Z` so that `softplus(Z) ≈ 1e-3`. Fixed seed.
- Log the loss curve; save the final parameters to JSON.
- Device: CUDA if available, else CPU. Smoke-test mode: 2 excerpts, 100 steps.

### 3.5 From activations to notes

1. Normalise `A` by its 99th percentile in each excerpt.
2. **Onset** at (p, t) when `A[p,t] > θ_on` **and** `A[p,t] − A[p,t−1] > θ_rise` (piano notes start with a rise in energy).
3. **Offset** when `A[p,t]` falls below `θ_off` (hysteresis, θ_off < θ_on), or at a new onset of the same key.
4. Drop notes shorter than 30 ms. Merge same-pitch gaps shorter than 30 ms.
5. Velocity (optional): map the peak activation linearly to 1–127.
6. **Thresholds are tuned only on validation-split excerpts (§5.3), never on test.** Grid: θ_on ∈ {0.1, …, 0.5}, θ_rise ∈ {0, 0.02, 0.05}, θ_off = θ_on / 2. Pick the setting that maximises mean onset F1. Tune separately for each method that needs thresholds (ours, NMF).

Output: a note list `onset_s, offset_s, pitch, velocity` (CSV) and a `.mid` file.

### 3.6 Intervention demo (RQ4)

Take one excerpt and its fitted model. (a) Delete one transcribed note. (b) Transpose one note by +1 semitone. (c) Change one note's velocity. Re-synthesise the spectrogram and plot original vs edited vs difference. Optionally invert to audio with Griffin-Lim, for listening only.

---

## 4. Baselines

| Name | Type | Description |
|---|---|---|
| `nmf_harmonic` | label-free | Same templates as §3.2 but **fixed**, with B = 0 (purely harmonic) and default decay. Only the activations H (88×T) are learned, with multiplicative updates minimising the KL divergence on the 2048-point STFT magnitude, 200 iterations. Notes are extracted as in §3.5, with its own tuned thresholds. |
| `nmf_inharmonic` (optional) | label-free | As above, but with fixed B_p = 3e-4 for all keys. |
| `basic_pitch` | supervised, instrument-agnostic | Spotify **Basic Pitch** (`pip install basic-pitch`), default settings, `predict(audio_path)` → note events. |
| `bytedance_hr` | supervised, piano-specific (upper bound) | Kong et al. high-resolution piano transcription (`pip install piano_transcription_inference`), `PianoTranscription(device=...)`. Trained on MAESTRO (train split), so it serves as an upper bound. Note this in the report. |
| `oracle_synth` | sanity check | On condition C0 (audio generated by our own model from the ground-truth MIDI), our method should recover the notes almost perfectly. If it does not, there is a bug or an identifiability problem. |

If Basic Pitch is hard to install next to PyTorch (TensorFlow conflicts), run it in a separate Kaggle notebook or virtual environment and save its predictions as CSV in the same format.

---

## 5. Data

### 5.1 MAESTRO v3.0.0 (primary source of MIDI ground truth)

- Homepage: <https://magenta.tensorflow.org/datasets/maestro>
- About 200 hours of virtuosic piano performances on Yamaha Disklaviers, with audio and MIDI aligned to about 3 ms. The MIDI contains key velocities and sustain (CC64), sostenuto (CC66) and una corda (CC67) pedal events.
- Licence: **CC BY-NC-SA 4.0**. Non-commercial use only; cite the paper.

| File | URL | Size |
|---|---|---|
| **MIDI-only archive (download this)** | <https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip> | ~56 MB (81 MB unzipped) |
| Metadata CSV | <https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.csv> | small |
| Metadata JSON | <https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.json> | small |
| Full archive with WAV (**do not download**; too large) | <https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip> | ~101 GB |

**Metadata columns:** `canonical_composer`, `canonical_title`, `split` (train / validation / test), `year`, `midi_filename`, `audio_filename`, `duration` (seconds).

**Splits (v3):** train 962 performances / 159.2 h; validation 137 / 19.4 h; **test 177 / 20.0 h**.

**MIDI format:** standard MIDI files (`.midi`), one piano track; read with `pretty_midi`. Notes: `start`, `end`, `pitch`, `velocity`. Control changes: `number`, `value`, `time`.

### 5.2 Real piano audio: MAESTRO test set mirror on Hugging Face

- <https://huggingface.co/datasets/B-K/MAESTRO-test>: a mirror of the **MAESTRO v3.0.0 test split** (177 rows, Parquet, ~4.06 GB), with a MIDI column and an audio column (FLAC). Licence CC BY-NC-SA 4.0.
- The exact column names and dtypes are **not documented**. **Inspect the schema first**, for example with `datasets.load_dataset("B-K/MAESTRO-test", split="test", streaming=True)` and printing the features of the first row, and adapt.
- **Use streaming**, or download only the Parquet shards you need, then crop the required 10-second windows and save them as 16 kHz mono WAV. Do not keep the full dataset on disk.
- **Matching rows to metadata:** if there is a filename column, use it. Otherwise hash the MIDI bytes (md5) and match them against the local MAESTRO MIDI files of the test split.
- Fallback if the mirror is unavailable: drop condition C3 and say so in the report. Do not download the 101 GB archive.

### 5.3 Excerpt sampling protocol (reproducible)

- **Test set (evaluation):** from the MAESTRO **test** split, draw **N = 30 excerpts of 10.0 s** with `seed = 0`:
  1. For each test performance, draw 3 random start times uniformly in [5 s, duration − 15 s] → candidate pool.
  2. For each candidate, compute **mean polyphony**: the average number of sounding notes per 16 ms frame, *after pedal extension* (§5.4). Also compute the **pedal fraction**: the share of the window with CC64 ≥ 64.
  3. Split candidates into **polyphony terciles** (low / mid / high) and sample 10 per tercile, at most 1 excerpt per performance.
- **Validation set (threshold tuning only):** 6 excerpts drawn the same way from the **validation** split (2 per tercile), `seed = 1`.
- Save the manifests (§8) and never change them after the first results.

### 5.4 Ground truth construction

- Read the full performance MIDI with `pretty_midi`.
- **Sustain-pedal extension:** while CC64 ≥ 64, extend each note's offset until the pedal is released or the same key is struck again, whichever comes first (standard practice, as in Onsets & Frames).
- **Crop to the window [t0, t0 + 10 s):**
  - Notes with an onset inside the window are **evaluated** (shift times by −t0; clip offsets at 10 s).
  - Notes that started before t0 but are still sounding are **not counted** for onset metrics, but they **are counted** for frame-level metrics (clipped to the window).
- Save as CSV: `onset_s, offset_s, pitch, velocity, evaluated (bool)`.
- **Consistency between audio and ground truth.** For the synthetic conditions C0–C2, render **only the notes whose onset falls inside the window**. Carry over the pedal state at t0: if CC64 ≥ 64 at t0, insert a CC64 event at time 0 of the excerpt MIDI. For these conditions, the frame-level ground truth must therefore also contain only those notes. For the real condition C3, earlier notes are physically present in the audio, so they stay in the frame-level ground truth. Store a separate GT file for each case (`gt/synthetic/` and `gt/real/`).

### 5.5 Audio conditions

| ID | Name | How the audio is produced | Purpose |
|---|---|---|---|
| **C0** | `model_synth` | Our own spectral model with the initial physical parameters, driven by the ground-truth piano roll. The spectrogram is synthesised directly; Griffin-Lim is used only for listening. | Identifiability and sanity check |
| **C1** | `render_fluidr3` | Render the **cropped excerpt MIDI** (notes and pedal events in the window, times shifted) with **FluidSynth** + the **FluidR3_GM** soundfont, program 0 (Acoustic Grand), 16 kHz, then convert to mono. | Main controlled condition: exact labels, realistic timbre |
| **C2** (optional) | `render_salamander` | Same as C1 with the **Salamander Grand Piano** soundfont (Yamaha C5) | Timbre-mismatch robustness |
| **C3** | `real_maestro` | The real MAESTRO recording (§5.2), cropped to the window and resampled to 16 kHz mono | Real-world condition |
| **D** (demo) | `own_recording` | 1–2 short pieces played by the author, recorded on a phone, converted to 16 kHz mono WAV | Qualitative demo only; no metrics |

**Soundfonts**

- **FluidR3_GM** (MIT licence). On Debian/Ubuntu or Kaggle: `apt-get install -y fluidsynth fluid-soundfont-gm`. The file is usually at `/usr/share/sounds/sf2/FluidR3_GM.sf2`; verify with `find / -name "FluidR3*.sf2" 2>/dev/null`.
- **Salamander Grand Piano** by Alexander Holm: originally CC BY 3.0 and now public domain. An SF2 conversion is listed on MuseScore's soundfont page: <https://musescore.org/en/handbook/2/soundfonts-and-sfz-files>. Download it manually if C2 is run.

**Rendering command** (CLI; avoids `pyfluidsynth` problems):

```bash
fluidsynth -ni -g 1.0 -r 16000 -F out.wav /path/to/soundfont.sf2 excerpt.mid
```

Then convert to mono, trim or pad to exactly 10.0 s plus a 1 s release tail (the tail is excluded from evaluation), and peak-normalise.

### 5.6 File formats used throughout

| Artefact | Format |
|---|---|
| Audio | WAV, 16 kHz, mono, float32 or PCM16 |
| Ground-truth and predicted notes | CSV with columns `onset_s,offset_s,pitch,velocity` (+ `evaluated` for ground truth), plus a `.mid` export |
| Activations | `.npy` float32, shape (88, T) |
| Fitted parameters | JSON: δ, a₀–a₂, c₀–c₁, loss history, runtime |
| Metrics | CSV (schema in §8) |

---

## 6. Evaluation

All metrics are computed with **`mir_eval`** on the 10-second window. Times are in seconds; convert pitches to Hz with `librosa.midi_to_hz` (or `mir_eval.util.midi_to_hz`).

| Metric | Definition |
|---|---|
| **Note onset P / R / F1** (primary) | `mir_eval.transcription.precision_recall_f1_overlap(..., onset_tolerance=0.05, offset_ratio=None)`, on ground-truth notes with `evaluated = True` |
| Note with offset F1 | Same, with `offset_ratio=0.2`, `offset_min_tolerance=0.05` |
| Frame-level F1 | Multi-pitch F1 on a 16 ms frame grid, from binary piano rolls (all sounding ground-truth notes; `mir_eval.multipitch` or a direct computation) |
| **Octave error rate** | Share of false-positive predicted onsets lying within ±50 ms of a ground-truth onset whose pitch differs by a multiple of 12 semitones (±12, ±24, ±36). Also report a "harmonic error rate" for intervals of 19 and 28 semitones (3rd and 5th harmonics). |
| Runtime | Wall-clock seconds per excerpt, plus the device |

**Aggregation and statistics**

- Report the **mean and 95% bootstrap CI** (10,000 resamples over excerpts) for each method × condition.
- For paired comparisons (ours vs NMF; B learned vs B = 0), use the **Wilcoxon signed-rank test** over excerpts and report the median difference. With N = 30 this is exploratory: do not over-claim.
- Break results down by **polyphony tercile** and by **pedal fraction** (above vs below the median).

---

## 7. Experiments

| ID | Conditions | Methods / variants | Answers |
|---|---|---|---|
| **E0** | C0 | ours (full) | Sanity check: onset F1 should be ≥ 0.95 |
| **E1** | C1 | ours, nmf_harmonic, basic_pitch, bytedance_hr | RQ1 |
| **E2** | C2 (optional), C3 | same as E1 | RQ1, RQ3 |
| **A1** | C1 (+ C3 if time allows) | ours with **B = 0 fixed** vs **B learned** | RQ2 |
| **A2** | C1 | partial decay **fixed** vs **learned** | Ablation |
| **A3** | C1, validation only → then test | λ_sp ∈ {0, 1e-3, 1e-2, 1e-1} | Sensitivity |
| **A4** | C1 | **single** resolution (2048) vs **multi**-resolution loss | Ablation |
| **D1** | own recording | ours + bytedance_hr | Qualitative demo |
| **D2** | 1 excerpt from C1 | intervention demo (§3.6) | RQ4 |

**Compute budget (estimate):** about 30 s per excerpt for 1,500 steps on a T4. E1 + E2 + A1–A4 ≈ 300 runs ≈ 2.5–3 h of GPU time, which fits in one Kaggle session. If time is short, run the ablations on C1 only.

---

## 8. Required outputs (deliverables)

### 8.1 Data artefacts

- `data/manifests/excerpts_test.csv` and `excerpts_val.csv`, with columns:
  `excerpt_id, split, midi_filename, audio_filename, canonical_composer, canonical_title, year, start_s, duration_s, n_notes_evaluated, mean_polyphony, polyphony_bin, pedal_fraction, seed`
- `data/gt/synthetic/{excerpt_id}.csv` and `data/gt/real/{excerpt_id}.csv` (ground-truth notes, §5.4)
- `data/audio/{condition}/{excerpt_id}.wav`

### 8.2 Predictions

- `outputs/predictions/{method}/{condition}/{excerpt_id}.csv` and `.mid`
- `outputs/activations/{variant}/{condition}/{excerpt_id}.npy` (our model only)
- `outputs/params/{variant}/{condition}/{excerpt_id}.json`

### 8.3 Metrics

- `results/metrics_per_excerpt.csv` with columns:
  `excerpt_id, condition, method, variant, onset_P, onset_R, onset_F1, onoff_F1, frame_F1, octave_err_rate, harmonic_err_rate, n_ref, n_pred, polyphony_bin, pedal_fraction, runtime_s, device`
- `results/summary.csv`: mean, CI-low and CI-high for each (condition, method, variant)
- `results/stats.csv`: Wilcoxon results for the planned comparisons

### 8.4 Figures (`figures/`, PNG at 300 dpi + PDF)

| File | Content |
|---|---|
| `fig1_pipeline.png` | Schematic: audio → multi-resolution STFT → differentiable model ↔ loss → piano roll → notes |
| `fig2_example.png` | One excerpt: input spectrogram, reconstructed spectrogram, ground-truth vs predicted piano roll (TP / FP / FN coloured) |
| `fig3_f1_by_method.png` | Onset F1 (mean ± 95% CI) by method × condition |
| `fig4_f1_vs_polyphony.png` | Onset F1 by polyphony tercile for each method |
| `fig5_octave_errors.png` | Octave error rate, B = 0 vs B learned (paired points per excerpt) |
| `fig6_learned_B.png` | Learned B(p) curves across excerpts vs key number (log scale), with typical literature range for grand pianos as a reference band (cite the source) |
| `fig7_intervention.png` | Intervention demo (§3.6) |
| `fig8_loss_curves.png` | Example optimisation curves |

### 8.5 Audio examples (`examples/`)

For 2–3 excerpts: the input audio, a re-synthesis of our transcription (MIDI → FluidSynth), and the Basic Pitch / ByteDance re-synthesis, so a listener can compare.

### 8.6 Write-up

- `report/RESULTS.md`, 1–2 pages: question, method in 5 lines, main table, 3–4 figures, **key findings (with numbers and CIs)**, limitations and next steps (for example amortised inference with a VAE encoder; waveform-level DDSP synthesis; learning shared piano parameters across recordings).
- This `README.md`, updated with a short "Results" section and a link to the report.
- A 2–3 line summary suitable for a CV entry.

---

## 9. Repository structure

```
piano-abs-transcription/
├── README.md
├── pyproject.toml / requirements.txt
├── configs/
│   ├── default.yaml          # all hyper-parameters from §3
│   └── smoke.yaml            # 2 excerpts, 100 steps, CPU
├── src/pianoabs/
│   ├── data/
│   │   ├── maestro.py        # download MIDI zip + metadata, load performances
│   │   ├── sampling.py       # excerpt sampling (§5.3), manifests
│   │   ├── groundtruth.py    # pedal extension + cropping (§5.4)
│   │   ├── render.py         # FluidSynth rendering (C1/C2)
│   │   └── hf_real_audio.py  # C3: streaming the HF mirror, cropping, resampling
│   ├── model/
│   │   ├── spectral_piano.py # templates + spectrogram model (§3.2)
│   │   ├── fit.py            # objective + optimisation (§3.3–3.4)
│   │   └── notes.py          # activations → notes (§3.5)
│   ├── baselines/
│   │   ├── nmf.py
│   │   ├── basic_pitch_runner.py
│   │   └── bytedance_runner.py
│   ├── eval/
│   │   ├── metrics.py        # mir_eval wrappers, octave/harmonic errors
│   │   └── stats.py          # bootstrap CI, Wilcoxon
│   ├── viz/plots.py
│   └── cli.py                # entry points (see below)
├── scripts/                  # thin wrappers that call cli.py
├── notebooks/
│   ├── 01_data_and_sanity.ipynb
│   ├── 02_run_experiments_kaggle.ipynb
│   └── 03_figures_and_report.ipynb
├── tests/                    # pytest: shapes, pedal extension, metrics on toy data, E0 on 1 excerpt
├── data/  outputs/  results/  figures/  examples/  report/   (git-ignored except small CSVs and figures)
```

**CLI entry points (suggested)**

```bash
python -m pianoabs.cli prepare-data   --config configs/default.yaml   # download MIDI, sample, GT, render C1/C2
python -m pianoabs.cli fetch-real     --config configs/default.yaml   # C3 from HF
python -m pianoabs.cli tune           --method ours --split val        # thresholds
python -m pianoabs.cli run            --method ours --condition C1 --variant full
python -m pianoabs.cli evaluate       --all
python -m pianoabs.cli figures
```

---

## 10. Environment and compute

- **Python** 3.10 or 3.11. Set seeds for `random`, `numpy` and `torch`.
- **Core:** `torch`, `torchaudio`, `numpy`, `scipy`, `librosa`, `soundfile`, `pretty_midi`, `mir_eval`, `pandas`, `matplotlib`, `pyyaml`, `tqdm`, `datasets` (for the HF mirror), `pytest`.
- **Baselines:** `basic-pitch` (may pull in TensorFlow; isolate it if needed), `piano_transcription_inference`.
- **System:** `fluidsynth` and `fluid-soundfont-gm` (via `apt-get`), `ffmpeg`.
- **Local machine:** CPU only, for development and `smoke.yaml`.
- **Kaggle:** GPU T4 for full runs. Persist outputs to `/kaggle/working`. Keep each run idempotent: skip excerpts whose outputs already exist, so an interrupted session can resume.
- Log config, git commit hash, device and package versions to `results/run_info.json`.

---

## 11. Timeline and definition of done

| Day | Work | Done when |
|---|---|---|
| **1** | Data pipeline: MIDI download, metadata, excerpt sampling, pedal-extended GT, FluidSynth rendering (C1); spectral model forward pass; unit tests | Manifests exist; 30 + 6 WAVs in C1; a GT piano roll plotted over its spectrogram looks right; tests pass |
| **2** | Optimisation loop; E0 sanity check; note extraction; NMF baseline; threshold tuning on validation | E0 onset F1 ≥ 0.95 on C0; ours and NMF produce note CSVs for C1 |
| **3** | Basic Pitch + ByteDance runs; C3 real audio; E1/E2; ablations A1–A4; metrics + statistics | `metrics_per_excerpt.csv` and `summary.csv` complete for the planned cells |
| **4** | Figures, intervention demo, own-recording demo, `RESULTS.md`, README results section, CV lines | All files in §8 exist; the repository runs end to end from a clean clone with `smoke.yaml` |

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Magnitude additivity is violated (beating, phase cancellation) | Accepted modelling assumption; mention it as a limitation |
| Too many spurious notes (harmonics explained as notes) | Sparsity λ_sp, the inharmonicity prior and threshold tuning; report octave and harmonic errors explicitly |
| Real audio (C3) has room acoustics, mic colouration and a different piano | Learnable noise floor and envelope; report C3 separately; this is a finding, not a failure |
| Basic Pitch / TensorFlow install conflicts | Separate notebook or venv; save CSVs |
| HF mirror schema unknown or unavailable | Inspect first; fall back to dropping C3 |
| Kaggle session limits | Idempotent, resumable runs; smaller N or ablations on C1 only |
| Data leakage | Tune thresholds and hyper-parameters on validation only; the test manifest is fixed before any results |

---

## 13. Licences and attribution

- **MAESTRO v3.0.0**: CC BY-NC-SA 4.0, © Google LLC. Non-commercial; derived audio excerpts must not be redistributed commercially. Cite Hawthorne et al. (2019).
- **FluidR3_GM**: MIT. **Salamander Grand Piano**: by Alexander Holm, now public domain (originally CC BY 3.0).
- **Basic Pitch**: Apache 2.0. **piano_transcription_inference**: see its repository's licence.
- Do **not** commit large audio files to the repository. Commit only the code, small CSVs, figures and a few short example clips (with attribution).

---

## 14. References

*Verify the details of each reference before citing it in any write-up.*

- Hawthorne, C. et al. (2019). *Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset.* ICLR.
- Bittner, R. M. et al. (2022). *A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation.* ICASSP. (Basic Pitch)
- Kong, Q. et al. (2021). *High-resolution Piano Transcription with Pedals by Regressing Onset and Offset Times.* IEEE/ACM TASLP. arXiv:2010.01815
- Hawthorne, C. et al. (2018). *Onsets and Frames: Dual-Objective Piano Transcription.* ISMIR.
- Engel, J. et al. (2020). *DDSP: Differentiable Digital Signal Processing.* ICLR.
- Smaragdis, P. & Brown, J. C. (2003). *Non-negative Matrix Factorization for Polyphonic Music Transcription.* IEEE WASPAA.
- Rigaud, F., David, B. & Daudet, L. (2013). *A parametric model and estimation techniques for the inharmonicity and tuning of the piano.* JASA.
- Fletcher, H. (1964). *Normal Vibration Frequencies of a Stiff Piano String.* JASA.
- Raffel, C. et al. (2014). *mir_eval: A Transparent Implementation of Common MIR Metrics.* ISMIR.
