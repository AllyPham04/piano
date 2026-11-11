"""Config loading, inheritance, overrides and the values of README §3."""

from __future__ import annotations

import math

import pytest

from pianoabs.config import (apply_overrides, deep_merge, parse_set_args,
                             variant_config)


def test_deep_merge_nested_and_lists():
    base = {"a": {"x": 1, "y": 2}, "l": [1, 2]}
    out = deep_merge(base, {"a": {"y": 3}, "l": [9]})
    assert out == {"a": {"x": 1, "y": 3}, "l": [9]}
    assert base["a"]["y"] == 2  # input not mutated


def test_default_matches_readme(default_cfg):
    f, m, fit = default_cfg["features"], default_cfg["model"], default_cfg["fit"]
    assert f["n_ffts"] == [1024, 2048, 4096] and f["hop"] == 256 and f["gamma"] == 100.0
    assert default_cfg["data"]["sample_rate"] == 16000
    assert f["hop"] / default_cfg["data"]["sample_rate"] == pytest.approx(0.016)
    assert (m["pitch_min"], m["pitch_max"], m["k_max"]) == (21, 108, 40)
    assert m["B_clamp"] == [1.0e-6, 5.0e-2] and math.isclose(m["B_init"], 3e-4)
    assert (fit["stage1_steps"], fit["stage2_steps"]) == (300, 1200)
    assert (fit["lambda_sp"], fit["lambda_tv"], fit["lambda_B"]) == (1e-2, 1e-3, 1e-3)
    assert (fit["lr_Z"], fit["lr_physical"]) == (0.05, 0.01)
    s = default_cfg["data"]["sampling"]["manifests"]
    assert s["test"] == {"split": "test", "n_excerpts": 30, "seed": 0}
    assert s["val"] == {"split": "validation", "n_excerpts": 6, "seed": 1}


def _all_leaves(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _all_leaves(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}", v


def test_no_float_parsed_as_string(default_cfg):
    """PyYAML reads '1e-3' (no decimal point) as a string; catch that mistake."""
    for key, value in _all_leaves(default_cfg):
        if isinstance(value, str):
            try:
                float(value)
            except ValueError:
                continue
            pytest.fail(f"{key} = {value!r} is a numeric string; write it as 1.0e-3")


def test_smoke_inherits_and_overrides(default_cfg, smoke_cfg):
    assert smoke_cfg["fit"]["stage1_steps"] + smoke_cfg["fit"]["stage2_steps"] == 100
    assert smoke_cfg["fit"]["device"] == "cpu"
    assert smoke_cfg["run"] == {"manifest": "val", "max_excerpts": 2}
    assert smoke_cfg["paths"]["outputs_dir"] != default_cfg["paths"]["outputs_dir"]
    assert smoke_cfg["paths"]["results_dir"] != default_cfg["paths"]["results_dir"]
    # untouched values come from default.yaml
    assert smoke_cfg["fit"]["lr_Z"] == default_cfg["fit"]["lr_Z"]
    assert smoke_cfg["model"] == default_cfg["model"]


def test_set_overrides_parse_yaml_values(default_cfg):
    cfg = apply_overrides(default_cfg, parse_set_args(["fit.lambda_sp=0.1",
                                                       "features.n_ffts=[2048]"]))
    assert cfg["fit"]["lambda_sp"] == 0.1
    assert cfg["features"]["n_ffts"] == [2048]
    assert default_cfg["fit"]["lambda_sp"] == 1e-2  # original untouched


def test_unknown_override_key_fails(default_cfg):
    with pytest.raises(KeyError):
        apply_overrides(default_cfg, {"fit.lamda_sp": 0.1})  # typo


@pytest.mark.parametrize("variant, key, expected", [
    ("full", "model.B_mode", "learned"),
    ("B0", "model.B_mode", "zero"),
    ("rho_fixed", "model.learn_rho", False),
    ("lsp_1e-1", "fit.lambda_sp", 0.1),
    ("single_res", "features.n_ffts", [2048]),
])
def test_variants(default_cfg, variant, key, expected):
    cfg = variant_config(default_cfg, variant)
    section, name = key.split(".")
    assert cfg[section][name] == expected
    assert cfg["variant"] == variant


def test_all_variants_apply(default_cfg):
    for name in default_cfg["variants"]:
        variant_config(default_cfg, name)  # raises on unknown keys
