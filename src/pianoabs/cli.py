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


def cmd_prepare_data(cfg: dict, args: argparse.Namespace) -> int:
    """Download MIDI, freeze/verify manifests, build GT, render the requested conditions."""
    from pianoabs.data.prepare import prepare_data

    write_run_info(cfg, command="prepare-data")
    max_excerpts = args.max_excerpts if args.max_excerpts is not None else cfg["run"]["max_excerpts"]
    status = prepare_data(cfg, args.manifests, args.conditions, max_excerpts,
                          verify=not args.no_verify)
    for which, s in status.items():
        print(f"manifest {which}: {s}")
    return 0


def cmd_sanity_plot(cfg: dict, args: argparse.Namespace) -> int:
    """GT over spectrogram for one excerpt (README §11, Day 1 check)."""
    from pianoabs import paths
    from pianoabs.audio import load_mono
    from pianoabs.data.groundtruth import read_notes_csv
    from pianoabs.viz.plots import plot_gt_over_spectrogram

    sr = cfg["data"]["sample_rate"]
    kind = cfg["conditions"][args.condition]["gt"]
    audio = load_mono(paths.audio_path(cfg, args.condition, args.excerpt), sr)
    gt = read_notes_csv(paths.gt_path(cfg, kind, args.excerpt))
    out = paths.figures_dir(cfg) / "sanity" / f"gt_over_spec_{args.condition}_{args.excerpt}.png"
    plot_gt_over_spectrogram(audio, sr, gt, out,
                             title=f"{args.excerpt} ({args.condition}): {kind} GT over |STFT| 4096",
                             hop=cfg["features"]["hop"], gamma=cfg["features"]["gamma"],
                             excerpt_s=cfg["data"]["excerpt_s"])
    print(f"wrote {out}")
    return 0


def cmd_tune(cfg: dict, args: argparse.Namespace) -> int:
    """Fit the validation excerpts (if needed) and grid-search the note thresholds."""
    from pianoabs.experiments import tune_thresholds

    variant = _variant_for(args)
    if args.method in ("basic_pitch", "bytedance_hr"):
        print(f"{args.method} has no thresholds to tune (default settings, README §4)")
        return 0
    write_run_info(cfg, command=f"tune {args.method}/{variant}/{args.condition}")
    res = tune_thresholds(cfg, args.method, variant, args.condition,
                          cfg["run"]["max_excerpts"])
    print(f"best on validation ({len(res['excerpts'])} excerpts): theta_on={res['theta_on']} "
          f"theta_rise={res['theta_rise']} reonset_ratio={res['reonset_rise_ratio']} "
          f"mean onset F1={res['val_mean_onset_F1']:.4f}")
    return 0


def _variant_for(args: argparse.Namespace) -> str:
    """Baselines have a single variant called 'default'."""
    return args.variant if args.method == "ours" else "default"


def cmd_run(cfg: dict, args: argparse.Namespace) -> int:
    """Activations + predictions for all excerpts of a manifest; print onset scores."""
    from pianoabs.experiments import evaluate_cell, run_method

    which = args.manifest or cfg["run"]["manifest"]
    max_excerpts = cfg["run"]["max_excerpts"]
    variant = _variant_for(args)
    write_run_info(cfg, command=f"run {args.method}/{variant}/{args.condition}/{which}")
    n = run_method(cfg, args.method, variant, args.condition, which, max_excerpts)
    scores = evaluate_cell(cfg, args.method, variant, args.condition, which, max_excerpts)
    if n >= 0:
        print(f"wrote {n} new prediction files")
    cols = ["excerpt_id", "onset_P", "onset_R", "onset_F1", "frame_F1", "n_ref", "n_pred"]
    print(scores[cols].round(4).to_string(index=False))
    print(f"mean onset F1 ({args.method}/{variant}/{args.condition}, {which}, "
          f"{len(scores)} excerpts): {scores['onset_F1'].mean():.4f}")
    return 0


def cmd_evaluate(cfg: dict, args: argparse.Namespace) -> int:
    """Metrics for every prediction folder + summary (bootstrap CIs) + Wilcoxon tests."""
    from pianoabs.experiments import evaluate_all

    which = args.manifest or cfg["run"]["manifest"]
    write_run_info(cfg, command=f"evaluate {which}")
    res = evaluate_all(cfg, which, cfg["run"]["max_excerpts"])
    per, summary = res["per_excerpt"], res["summary"]
    print(f"{len(per)} rows in metrics_per_excerpt.csv")
    if len(summary):
        f1 = summary[summary.metric == "onset_F1"]
        print(f1[["condition", "method", "variant", "n", "mean", "ci_low", "ci_high"]]
              .round(4).to_string(index=False))
    if len(res["stats"]):
        print(res["stats"][["comparison", "condition", "metric", "n_pairs", "median_diff",
                            "p_value"]].to_string(index=False))
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
    p.add_argument("--manifests", nargs="+", default=["val", "test"], choices=["val", "test"])
    p.add_argument("--conditions", nargs="*", default=["C1"], choices=["C1", "C2"],
                   help="FluidSynth conditions to render (none: GT and MIDI only)")
    p.add_argument("--max-excerpts", type=int, default=None,
                   help="only build files for the first N excerpts (default: run.max_excerpts)")
    p.add_argument("--no-verify", action="store_true",
                   help="skip re-sampling to verify existing (frozen) manifests")
    p.set_defaults(func=cmd_prepare_data)

    p = sub.add_parser("sanity-plot", parents=[common], help="GT over spectrogram for one excerpt")
    p.add_argument("--excerpt", default="val_000")
    p.add_argument("--condition", default="C1")
    p.set_defaults(func=cmd_sanity_plot)

    p = sub.add_parser("fetch-real", parents=[common], help="C3 audio from the HF mirror")
    p.set_defaults(func=_not_implemented("M7"))

    p = sub.add_parser("tune", parents=[common], help="tune note thresholds on validation")
    p.add_argument("--method", default="ours")
    p.add_argument("--variant", default="full")
    p.add_argument("--condition", default="C1")
    p.add_argument("--split", default="val", choices=["val"],
                   help="thresholds are only ever tuned on validation")
    p.set_defaults(func=cmd_tune)

    p = sub.add_parser("run", parents=[common], help="transcribe all excerpts of a condition")
    p.add_argument("--method", default="ours")
    p.add_argument("--variant", default="full")
    p.add_argument("--condition", default="C1")
    p.add_argument("--manifest", choices=["val", "test"], default=None,
                   help="default: run.manifest from the config")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("evaluate", parents=[common], help="metrics, summary, statistics")
    p.add_argument("--all", action="store_true", help="(default) evaluate every prediction folder")
    p.add_argument("--manifest", choices=["val", "test"], default=None,
                   help="default: run.manifest from the config")
    p.set_defaults(func=cmd_evaluate)

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
