"""Running methods on excerpts: activations, threshold tuning on validation, predictions,
and evaluation (README §3.4–3.5, §6–§8).

Two kinds of methods:

* **activation methods** (``ours``, ``nmf_harmonic``, ``nmf_inharmonic``) produce
  an activation matrix (88 x T) per excerpt; notes are extracted with
  thresholds tuned on the validation manifest (README §3.5);
* **note methods** (``basic_pitch``, ``bytedance_hr``; M6) write note lists
  directly.

Everything is **idempotent**: an excerpt whose outputs exist *and were made with
the same configuration* is skipped, so an interrupted Kaggle session can simply
re-run the same cell. Each output stores a fingerprint (hash of the settings
that affect it, plus ``MODEL_VERSION``); outputs with a different fingerprint
are recomputed, and thresholds tuned for other activations are refused. This
prevents silently reusing stale results after a config or model change.

Threshold tuning rules (README §3.5, agreed decision B4): tuned on validation
only, per (method, variant, condition); C3 has no validation audio (the HF
mirror is test-only), so C3 reuses the thresholds tuned on C1.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pianoabs import paths
from pianoabs.audio import load_mono
from pianoabs.baselines.nmf import kl_divergence, kl_nmf_activations, nmf_dictionary
from pianoabs.config import variant_config
from pianoabs.data.groundtruth import read_notes_csv
from pianoabs.data.sampling import load_manifest
from pianoabs.eval.metrics import crop_predictions, excerpt_metrics, onset_prf
from pianoabs.model.c0 import synthesize_c0
from pianoabs.model.fit import audio_targets, fit_excerpt
from pianoabs.model.notes import extract_with_config, threshold_grid
from pianoabs.predictions import is_up_to_date, write_prediction
from pianoabs.utils import resolve_device

ACTIVATION_METHODS = ("ours", "nmf_harmonic", "nmf_inharmonic")
# Bump when code changes alter activations (2: amplitude scale s, 2026-11-12).
MODEL_VERSION = 2
TUNING_CONDITION = {"C3": "C1"}  # conditions without validation audio borrow thresholds
METRICS_COLUMNS = [
    "excerpt_id", "condition", "method", "variant", "onset_P", "onset_R", "onset_F1",
    "onoff_F1", "frame_F1", "octave_err_rate", "harmonic_err_rate", "n_ref", "n_pred",
    "polyphony_bin", "pedal_fraction", "runtime_s", "device",
]


# ---------------------------------------------------------------- inputs

def gt_for(cfg: dict, condition: str, excerpt_id: str) -> pd.DataFrame:
    return read_notes_csv(paths.gt_path(cfg, cfg["conditions"][condition]["gt"], excerpt_id))


def load_targets(cfg: dict, condition: str, excerpt_id: str, device: str) -> dict[int, torch.Tensor]:
    """Target spectrograms at cfg's resolutions: synthesised for C0, STFT of the WAV otherwise."""
    if cfg["conditions"][condition]["source"] == "model":
        specs, _ = synthesize_c0(gt_for(cfg, condition, excerpt_id), cfg)
        return {n: s.to(device) for n, s in specs.items() if n in cfg["features"]["n_ffts"]}
    audio = load_mono(paths.audio_path(cfg, condition, excerpt_id), cfg["data"]["sample_rate"])
    return audio_targets(audio, cfg, device)


def manifest_for(cfg: dict, which: str, max_excerpts: int | None = None) -> pd.DataFrame:
    return load_manifest(paths.manifest_path(cfg, which), max_excerpts)


def activation_key(method: str, variant: str) -> str:
    """Folder name under outputs/activations and outputs/params (README §8.2)."""
    return variant if method == "ours" else method


def activation_fingerprint(cfg: dict, method: str, variant: str) -> str:
    """Short hash of every setting that changes a method's activations."""
    vcfg = variant_config(cfg, variant) if method == "ours" else cfg
    relevant = {
        "version": MODEL_VERSION, "method": method, "seed": vcfg["seed"],
        "data": {k: vcfg["data"][k] for k in ("sample_rate", "excerpt_s", "tail_s", "peak_norm")},
        "features": vcfg["features"], "model": vcfg["model"], "c0": vcfg["c0"],
    }
    if method == "ours":
        relevant["fit"] = {k: v for k, v in vcfg["fit"].items() if k != "device"}
    else:
        relevant["nmf"] = vcfg["nmf"]
    return hashlib.sha1(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:12]


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ---------------------------------------------------------------- activations

def _save_activations(cfg: dict, key: str, condition: str, excerpt_id: str,
                      activations: np.ndarray, info: dict) -> None:
    act_path = paths.activation_path(cfg, key, condition, excerpt_id)
    par_path = paths.params_path(cfg, key, condition, excerpt_id)
    act_path.parent.mkdir(parents=True, exist_ok=True)
    par_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(act_path, activations.astype(np.float32))
    par_path.write_text(json.dumps(info), encoding="utf-8")


def compute_activations(cfg: dict, method: str, variant: str, condition: str,
                        excerpt_id: str, device: str, force: bool = False) -> bool:
    """Compute and save activations + parameters of one excerpt. Returns True if it ran."""
    key = activation_key(method, variant)
    fingerprint = activation_fingerprint(cfg, method, variant)
    saved = _read_json(paths.params_path(cfg, key, condition, excerpt_id))
    if (not force and saved is not None and saved.get("fingerprint") == fingerprint
            and paths.activation_path(cfg, key, condition, excerpt_id).exists()):
        return False
    base = {"excerpt_id": excerpt_id, "condition": condition, "method": method,
            "variant": variant, "seed": cfg["seed"], "fingerprint": fingerprint}
    if method == "ours":
        vcfg = variant_config(cfg, variant)
        result = fit_excerpt(load_targets(vcfg, condition, excerpt_id, device), vcfg, device)
        _save_activations(cfg, key, condition, excerpt_id, result.activations, {
            **base, "device": result.device, "runtime_s": round(result.runtime_s, 3),
            "fit": vcfg["fit"], "variant_overrides": cfg["variants"][variant],
            **result.params, "loss_history": result.loss_history})
    elif method in ("nmf_harmonic", "nmf_inharmonic"):
        n_fft = cfg["nmf"]["n_fft"]
        ncfg = {**cfg, "features": {**cfg["features"], "n_ffts": [n_fft]}}
        V = load_targets(ncfg, condition, excerpt_id, "cpu")[n_fft].numpy()
        W = nmf_dictionary(cfg, method)
        start = time.perf_counter()
        H = kl_nmf_activations(V, W, cfg["nmf"]["n_iter"], cfg["nmf"]["eps"], cfg["seed"])
        runtime = time.perf_counter() - start
        _save_activations(cfg, key, condition, excerpt_id, H, {
            **base, "device": "cpu", "runtime_s": round(runtime, 3),
            "n_iter": cfg["nmf"]["n_iter"], "n_fft": n_fft,
            "final_kl": kl_divergence(V, W @ H)})
    else:
        raise ValueError(f"{method} does not produce activations")
    return True


def activations_for_manifest(cfg: dict, method: str, variant: str, condition: str,
                             which: str, max_excerpts: int | None = None) -> None:
    device = resolve_device(cfg["fit"]["device"]) if method == "ours" else "cpu"
    manifest = manifest_for(cfg, which, max_excerpts)
    for eid in tqdm(manifest["excerpt_id"], desc=f"{method}/{variant}/{condition}/{which} [{device}]"):
        compute_activations(cfg, method, variant, condition, eid, device)


def load_activations(cfg: dict, method: str, variant: str, condition: str,
                     excerpt_id: str) -> np.ndarray:
    return np.load(paths.activation_path(cfg, activation_key(method, variant), condition, excerpt_id))


def load_params(cfg: dict, method: str, variant: str, condition: str, excerpt_id: str) -> dict:
    path = paths.params_path(cfg, activation_key(method, variant), condition, excerpt_id)
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- thresholds

def thresholds_path(cfg: dict, method: str, variant: str, condition: str) -> Path:
    return paths.results_dir(cfg) / "thresholds" / f"{method}__{variant}__{condition}.json"


def tune_thresholds(cfg: dict, method: str, variant: str, condition: str,
                    max_excerpts: int | None = None) -> dict:
    """Grid search (θ_on, θ_rise, re-onset ratio) maximising mean onset F1 on validation."""
    if condition in TUNING_CONDITION:
        raise ValueError(f"{condition} has no validation audio; it uses the thresholds of "
                         f"{TUNING_CONDITION[condition]}")
    activations_for_manifest(cfg, method, variant, condition, "val", max_excerpts)
    manifest = manifest_for(cfg, "val", max_excerpts)
    excerpt_s = cfg["data"]["excerpt_s"]
    data = [(load_activations(cfg, method, variant, condition, eid), gt_for(cfg, condition, eid))
            for eid in manifest["excerpt_id"]]
    grid = []
    for theta_on, theta_rise, ratio in threshold_grid(cfg):
        f1s = [onset_prf(gt, crop_predictions(
            extract_with_config(A, cfg, theta_on, theta_rise, ratio), excerpt_s))["onset_F1"]
            for A, gt in data]
        grid.append({"theta_on": theta_on, "theta_rise": theta_rise,
                     "reonset_rise_ratio": ratio, "mean_onset_F1": float(np.mean(f1s))})
    best = max(grid, key=lambda g: g["mean_onset_F1"])  # first maximum in grid order
    result = {"method": method, "variant": variant, "condition": condition,
              "fingerprint": activation_fingerprint(cfg, method, variant),
              "tuned_on": "val", "excerpts": list(manifest["excerpt_id"]),
              "theta_on": best["theta_on"], "theta_rise": best["theta_rise"],
              "reonset_rise_ratio": best["reonset_rise_ratio"],
              "theta_off": best["theta_on"] * cfg["notes"]["theta_off_ratio"],
              "val_mean_onset_F1": best["mean_onset_F1"], "grid": grid}
    path = thresholds_path(cfg, method, variant, condition)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def load_thresholds(cfg: dict, method: str, variant: str, condition: str) -> dict | None:
    path = thresholds_path(cfg, method, variant, TUNING_CONDITION.get(condition, condition))
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ---------------------------------------------------------------- predictions

def predict_manifest(cfg: dict, method: str, variant: str, condition: str, which: str,
                     max_excerpts: int | None = None, force: bool = False) -> int:
    """Predictions of an activation method from saved activations + tuned thresholds."""
    thr = load_thresholds(cfg, method, variant, condition)
    if thr is None:
        raise FileNotFoundError(f"no tuned thresholds for {method}/{variant}/{condition}; "
                                "run 'tune' first")
    fingerprint = activation_fingerprint(cfg, method, variant)
    if thr.get("fingerprint") != fingerprint:
        raise RuntimeError(f"thresholds for {method}/{variant} were tuned on activations from "
                           "a different configuration; re-run 'tune' first")
    setting = {"activation_fingerprint": fingerprint, "theta_on": thr["theta_on"],
               "theta_rise": thr["theta_rise"], "reonset_rise_ratio": thr["reonset_rise_ratio"],
               "thresholds_from": thr["condition"]}
    written = 0
    for eid in manifest_for(cfg, which, max_excerpts)["excerpt_id"]:
        csv = paths.prediction_path(cfg, method, variant, condition, eid, ".csv")
        if not force and is_up_to_date(csv, setting):
            continue
        notes = extract_with_config(load_activations(cfg, method, variant, condition, eid), cfg,
                                    thr["theta_on"], thr["theta_rise"], thr["reonset_rise_ratio"])
        info = load_params(cfg, method, variant, condition, eid)
        write_prediction(cfg, method, variant, condition, eid, notes, info["runtime_s"],
                         info["device"], setting)
        written += 1
    return written


def run_bytedance(cfg: dict, condition: str, which: str, max_excerpts: int | None = None) -> int:
    """ByteDance predictions for a manifest (model loaded once)."""
    import importlib.metadata

    from pianoabs.baselines.bytedance_runner import BytedanceTranscriber

    setting = {"method": "bytedance_hr",
               "package_version": importlib.metadata.version("piano_transcription_inference"),
               "apply_predicted_pedal": cfg["baselines"]["bytedance"]["apply_predicted_pedal"]}
    todo = [eid for eid in manifest_for(cfg, which, max_excerpts)["excerpt_id"]
            if not is_up_to_date(paths.prediction_path(cfg, "bytedance_hr", "default", condition,
                                                        eid, ".csv"), setting)]
    if not todo:
        return 0
    device = resolve_device(cfg["fit"]["device"])
    transcriber = BytedanceTranscriber(cfg, device)
    for eid in tqdm(todo, desc=f"bytedance_hr/{condition}/{which} [{device}]"):
        audio = load_mono(paths.audio_path(cfg, condition, eid), cfg["data"]["sample_rate"])
        notes, runtime = transcriber.transcribe(audio)
        write_prediction(cfg, "bytedance_hr", "default", condition, eid, notes, runtime,
                         device, setting)
    return len(todo)


def run_basic_pitch(cfg: dict, condition: str, which: str, max_excerpts: int | None = None) -> int:
    """Basic Pitch in its own interpreter (``baselines.basic_pitch.python``) to keep
    TensorFlow away from PyTorch; it writes predictions in the common format."""
    import subprocess
    import sys

    python = cfg["baselines"]["basic_pitch"]["python"] or sys.executable
    cmd = [python, "-m", "pianoabs.baselines.basic_pitch_runner", "--config", cfg["config_path"],
           "--condition", condition, "--manifest", which]
    if max_excerpts:
        cmd += ["--set", f"run.max_excerpts={max_excerpts}"]
    subprocess.run(cmd, check=True)
    return -1  # count printed by the subprocess


NOTE_METHODS = {"bytedance_hr": run_bytedance, "basic_pitch": run_basic_pitch}


def run_method(cfg: dict, method: str, variant: str, condition: str, which: str,
               max_excerpts: int | None = None) -> int:
    """Predictions for one cell of the experiment grid (activations first, if needed)."""
    if method in NOTE_METHODS:
        return NOTE_METHODS[method](cfg, condition, which, max_excerpts)
    if method not in ACTIVATION_METHODS:
        raise ValueError(f"unknown method '{method}'")
    activations_for_manifest(cfg, method, variant, condition, which, max_excerpts)
    return predict_manifest(cfg, method, variant, condition, which, max_excerpts)


# ---------------------------------------------------------------- evaluation

def evaluate_cell(cfg: dict, method: str, variant: str, condition: str, which: str,
                  max_excerpts: int | None = None) -> pd.DataFrame:
    """Per-excerpt metrics of one (method, variant, condition) on a manifest.

    Excerpts without a prediction file are skipped, and so are predictions of an
    activation method whose fingerprint does not match the current configuration
    (stale outputs of an earlier model or config).
    """
    current = (activation_fingerprint(cfg, method, variant)
               if method in ACTIVATION_METHODS and variant in cfg["variants"] | {"default": {}}
               else None)
    rows = []
    for ex in manifest_for(cfg, which, max_excerpts).itertuples(index=False):
        csv = paths.prediction_path(cfg, method, variant, condition, ex.excerpt_id, ".csv")
        if not csv.exists():
            continue
        meta = _read_json(csv.with_suffix(".json")) or {}
        if current is not None and meta.get("activation_fingerprint") != current:
            continue
        m = excerpt_metrics(gt_for(cfg, condition, ex.excerpt_id), read_notes_csv(csv), cfg)
        rows.append({"excerpt_id": ex.excerpt_id, "condition": condition, "method": method,
                     "variant": variant, **m, "polyphony_bin": ex.polyphony_bin,
                     "pedal_fraction": ex.pedal_fraction,
                     "runtime_s": meta.get("runtime_s", float("nan")),
                     "device": meta.get("device", "")})
    return pd.DataFrame(rows)


def discover_cells(cfg: dict) -> list[tuple[str, str, str]]:
    """All (method, variant, condition) folders that contain predictions."""
    root = Path(cfg["paths"]["outputs_dir"]) / "predictions"
    if not root.exists():
        return []
    return sorted((m.name, v.name, c.name) for m in root.iterdir() if m.is_dir()
                  for v in m.iterdir() if v.is_dir() for c in v.iterdir() if c.is_dir())


def evaluate_all(cfg: dict, which: str, max_excerpts: int | None = None) -> dict[str, pd.DataFrame]:
    """Write metrics_per_excerpt.csv, metrics_extra.csv, summary.csv and stats.csv (README §8.3)."""
    from pianoabs.eval.stats import planned_tests, summarise

    frames = [evaluate_cell(cfg, *cell, which, max_excerpts) for cell in discover_cells(cfg)]
    per = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) \
        else pd.DataFrame(columns=METRICS_COLUMNS)
    out_dir = paths.results_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    per[METRICS_COLUMNS].to_csv(out_dir / "metrics_per_excerpt.csv", index=False,
                                float_format="%.6f", lineterminator="\n")
    extra_cols = ["excerpt_id", "condition", "method", "variant", "n_fp", "n_octave_fp",
                  "n_harmonic_fp", "mean_polyphony_manifest"]
    manifest = manifest_for(cfg, which, max_excerpts).set_index("excerpt_id")
    if len(per):
        per["mean_polyphony_manifest"] = per["excerpt_id"].map(manifest["mean_polyphony"])
        per[extra_cols].to_csv(out_dir / "metrics_extra.csv", index=False,
                               float_format="%.6f", lineterminator="\n")
    summary = summarise(per, cfg["eval"]["bootstrap_n"], cfg["eval"]["bootstrap_seed"]) \
        if len(per) else pd.DataFrame()
    summary.to_csv(out_dir / "summary.csv", index=False, float_format="%.6f", lineterminator="\n")
    tests = planned_tests(per) if len(per) else pd.DataFrame()
    tests.to_csv(out_dir / "stats.csv", index=False, float_format="%.6g", lineterminator="\n")
    return {"per_excerpt": per, "summary": summary, "stats": tests}
