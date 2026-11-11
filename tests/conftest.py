"""Shared pytest fixtures. Tests run from the repository root."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pianoabs.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _run_from_repo_root(monkeypatch):
    """Config paths are relative to the repo root, so tests always run from there."""
    monkeypatch.chdir(REPO_ROOT)


@pytest.fixture
def default_cfg() -> dict:
    return load_config(REPO_ROOT / "configs" / "default.yaml")


@pytest.fixture
def smoke_cfg() -> dict:
    return load_config(REPO_ROOT / "configs" / "smoke.yaml")


@pytest.fixture
def tmp_results_cfg(default_cfg, tmp_path) -> dict:
    """Default config whose results/outputs go to a temporary folder."""
    cfg = dict(default_cfg)
    cfg["paths"] = dict(default_cfg["paths"],
                        results_dir=os.fspath(tmp_path / "results"),
                        outputs_dir=os.fspath(tmp_path / "outputs"))
    return cfg
