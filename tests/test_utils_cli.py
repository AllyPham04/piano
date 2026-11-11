"""Seeding, run logging, path layout and the CLI stub."""

from __future__ import annotations

import json
import random

import numpy as np
import torch

from pianoabs import paths
from pianoabs.cli import main
from pianoabs.utils import resolve_device, seed_everything, write_run_info


def test_seed_everything_is_reproducible():
    def draw():
        return random.random(), np.random.rand(), torch.rand(1).item()

    seed_everything(123)
    first = draw()
    seed_everything(123)
    assert draw() == first


def test_resolve_device():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in ("cpu", "cuda")


def test_write_run_info(tmp_results_cfg):
    path = write_run_info(tmp_results_cfg, command="test")
    info = json.loads(path.read_text(encoding="utf-8"))
    for key in ("command", "timestamp_utc", "python", "device", "git", "packages", "config"):
        assert key in info
    assert info["config"]["fit"]["lambda_sp"] == tmp_results_cfg["fit"]["lambda_sp"]
    assert info["packages"]["numpy"] == np.__version__
    # history file gets one line per call
    write_run_info(tmp_results_cfg, command="test2")
    log = (path.parent / "run_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["command"] for line in log] == ["test", "test2"]


def test_paths_layout(default_cfg):
    p = paths.prediction_path(default_cfg, "ours", "B0", "C1", "test_001")
    assert p.as_posix() == "outputs/predictions/ours/B0/C1/test_001.csv"
    assert paths.manifest_path(default_cfg, "val").as_posix() == "data/manifests/excerpts_val.csv"
    assert paths.gt_path(default_cfg, "real", "x").as_posix() == "data/gt/real/x.csv"
    assert paths.activation_path(default_cfg, "full", "C0", "x").suffix == ".npy"


def test_cli_info_writes_run_info(tmp_path, capsys):
    rc = main(["info", "--config", "configs/smoke.yaml",
               "--set", f"paths.results_dir={tmp_path.as_posix()}"])
    assert rc == 0
    info = json.loads((tmp_path / "run_info.json").read_text(encoding="utf-8"))
    assert info["config"]["config_path"].endswith("configs/smoke.yaml")
    assert info["device"] == "cpu"


def test_cli_stub_reports_milestone(capsys):
    rc = main(["figures", "--config", "configs/smoke.yaml"])
    assert rc == 2
    assert "M9" in capsys.readouterr().err
