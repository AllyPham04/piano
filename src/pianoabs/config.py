"""Configuration loading.

A config is a plain nested ``dict`` read from YAML. Plain dicts are easy to print,
save to JSON (``run_info.json``) and reason about, so no config class is used.

Two mechanisms keep configs short:

* ``inherit: other.yaml`` at the top of a file: the file is deep-merged on top of
  its parent, so ``smoke.yaml`` lists only what differs from ``default.yaml``.
* dotted-key overrides such as ``{"fit.lambda_sp": 0.1}``: used by the ``--set``
  CLI flag and by the ``variants`` section (ablations A1–A4).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """Return a copy of ``base`` with ``override`` merged in recursively.

    Nested dicts are merged key by key; any other value (including lists) in
    ``override`` replaces the value in ``base``.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict:
    """Load a YAML config, resolving ``inherit:`` relative to the file's folder."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent is not None:
        cfg = deep_merge(load_config(path.parent / parent), cfg)
    cfg["config_path"] = str(path).replace("\\", "/")
    return cfg


def set_dotted(cfg: dict, dotted_key: str, value: Any) -> None:
    """Set ``cfg["a"]["b"]["c"] = value`` for ``dotted_key = "a.b.c"`` (in place).

    The key must already exist, so a typo in an override fails loudly instead of
    silently adding an unused setting.
    """
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise KeyError(f"Unknown config section '{part}' in '{dotted_key}'")
        node = node[part]
    if parts[-1] not in node:
        raise KeyError(f"Unknown config key '{dotted_key}'")
    node[parts[-1]] = copy.deepcopy(value)


def get_dotted(cfg: dict, dotted_key: str) -> Any:
    """Return ``cfg["a"]["b"]["c"]`` for ``dotted_key = "a.b.c"``."""
    node = cfg
    for part in dotted_key.split("."):
        node = node[part]
    return node


def apply_overrides(cfg: dict, overrides: dict[str, Any]) -> dict:
    """Return a copy of ``cfg`` with dotted-key ``overrides`` applied."""
    out = copy.deepcopy(cfg)
    for key, value in overrides.items():
        set_dotted(out, key, value)
    return out


def parse_set_args(items: list[str] | None) -> dict[str, Any]:
    """Parse CLI ``--set key=value`` items; values are parsed as YAML (``0.1`` -> float)."""
    overrides: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got '{item}'")
        key, raw = item.split("=", 1)
        overrides[key.strip()] = yaml.safe_load(raw)
    return overrides


def variant_config(cfg: dict, variant: str) -> dict:
    """Return the config for a model variant (e.g. ``"B0"``), see ``variants:`` in the YAML."""
    variants = cfg.get("variants", {})
    if variant not in variants:
        raise KeyError(f"Unknown variant '{variant}'. Known: {sorted(variants)}")
    out = apply_overrides(cfg, variants[variant] or {})
    out["variant"] = variant
    return out
