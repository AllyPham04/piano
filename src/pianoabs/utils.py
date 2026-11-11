"""Reproducibility helpers: seeding, device choice and run logging.

Every CLI command calls :func:`seed_everything` and :func:`write_run_info`, so each
result can be traced back to its config, code version (git hash), packages and device.
"""

from __future__ import annotations

import datetime as _dt
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

# Packages whose versions are logged in run_info.json.
LOGGED_PACKAGES = [
    "numpy", "scipy", "pandas", "torch", "librosa", "soundfile", "pretty_midi",
    "mir_eval", "matplotlib", "pyyaml", "datasets", "basic-pitch",
    "piano_transcription_inference",
]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch on CPU and all GPUs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # e.g. an isolated Basic Pitch environment
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> str:
    """Map ``auto`` to ``cuda`` if a GPU is visible, else ``cpu``."""
    if name != "auto":
        return name
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def git_info(repo_dir: str | Path = ".") -> dict:
    """Return the current commit hash and whether the working tree has uncommitted changes."""
    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=repo_dir, capture_output=True,
                                 text=True, check=True, timeout=10)
            return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def package_versions(names: list[str] = LOGGED_PACKAGES) -> dict:
    """Installed version of each package, or None if it is not installed."""
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect_run_info(cfg: dict, command: str, device: str | None = None) -> dict:
    """Everything needed to reproduce a run: config, versions, device, git hash, time."""
    info = {
        "command": command,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": device or resolve_device(cfg.get("fit", {}).get("device", "auto")),
        "git": git_info(),
        "packages": package_versions(),
        "config": cfg,
    }
    try:
        import torch
        info["cuda_device_name"] = (torch.cuda.get_device_name(0)
                                    if torch.cuda.is_available() else None)
    except ImportError:
        info["cuda_device_name"] = None
    return info


def write_run_info(cfg: dict, command: str, device: str | None = None) -> Path:
    """Write ``results/run_info.json`` (latest run) and append to ``results/run_log.jsonl``.

    The JSONL file keeps the history of every command, so an earlier run's
    settings are never lost when run_info.json is overwritten.
    """
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    info = collect_run_info(cfg, command, device)
    path = results_dir / "run_info.json"
    path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    with open(results_dir / "run_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(info, default=str) + "\n")
    return path
