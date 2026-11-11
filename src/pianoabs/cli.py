"""Command-line entry points (README §9).

Usage::

    python -m pianoabs.cli <command> --config configs/default.yaml [--set key=value ...]

Every command: loads the config, applies ``--set`` overrides, seeds all RNGs and
writes ``results/run_info.json``. Commands that are not built yet say which
milestone implements them.
"""

from __future__ import annotations

import argparse
import json
import sys

from pianoabs.config import apply_overrides, load_config, parse_set_args
from pianoabs.utils import seed_everything, write_run_info


def _not_implemented(milestone: str):
    def handler(cfg: dict, args: argparse.Namespace) -> int:
        print(f"'{args.command}' is not implemented yet (milestone {milestone}).",
              file=sys.stderr)
        return 2
    return handler


def cmd_info(cfg: dict, args: argparse.Namespace) -> int:
    """Print and log the environment (config, versions, device, git hash)."""
    path = write_run_info(cfg, command="info")
    info = json.loads(path.read_text(encoding="utf-8"))
    summary = {k: info[k] for k in ("python", "platform", "device", "git", "cuda_device_name")}
    summary["packages"] = {k: v for k, v in info["packages"].items() if v is not None}
    print(json.dumps(summary, indent=2))
    print(f"wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pianoabs", description=__doc__.split("\n")[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="configs/default.yaml", help="YAML config file")
    common.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="override a config value, e.g. --set fit.lambda_sp=0.1")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", parents=[common], help="log environment to results/run_info.json")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("prepare-data", parents=[common],
                       help="download MIDI, sample excerpts, build GT, render C1/C2")
    p.set_defaults(func=_not_implemented("M2"))

    p = sub.add_parser("fetch-real", parents=[common], help="C3 audio from the HF mirror")
    p.set_defaults(func=_not_implemented("M7"))

    p = sub.add_parser("tune", parents=[common], help="tune note thresholds on validation")
    p.add_argument("--method", default="ours")
    p.add_argument("--variant", default="full")
    p.add_argument("--condition", default="C1")
    p.add_argument("--split", default="val", choices=["val"],
                   help="thresholds are only ever tuned on validation")
    p.set_defaults(func=_not_implemented("M4/M5"))

    p = sub.add_parser("run", parents=[common], help="transcribe all excerpts of a condition")
    p.add_argument("--method", default="ours")
    p.add_argument("--variant", default="full")
    p.add_argument("--condition", default="C1")
    p.set_defaults(func=_not_implemented("M4"))

    p = sub.add_parser("evaluate", parents=[common], help="metrics, summary, statistics")
    p.add_argument("--all", action="store_true")
    p.set_defaults(func=_not_implemented("M5"))

    p = sub.add_parser("figures", parents=[common], help="figures of README §8.4")
    p.set_defaults(func=_not_implemented("M9"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = apply_overrides(load_config(args.config), parse_set_args(args.set))
    seed_everything(cfg["seed"])
    return args.func(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
