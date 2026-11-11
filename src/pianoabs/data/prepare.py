"""``prepare-data``: download MIDI, freeze manifests, build GT, render audio (README §5).

Idempotent: existing GT files, excerpt MIDIs and WAVs are skipped, so an
interrupted run can simply be restarted. Manifests are created once and
afterwards only verified (see :mod:`pianoabs.data.sampling`).
"""

from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from pianoabs import paths
from pianoabs.data.groundtruth import (ground_truth, make_excerpt_midi, performance_notes,
                                       write_notes_csv)
from pianoabs.data.maestro import ensure_maestro, load_performance
from pianoabs.data.render import find_soundfont, render_midi
from pianoabs.data.sampling import build_manifest, load_manifest, save_or_verify_manifest


def freeze_manifests(cfg: dict, which_list: list[str], verify: bool = True) -> dict[str, str]:
    """Create each manifest if missing; otherwise re-sample and verify (if ``verify``)."""
    meta, midi_root = ensure_maestro(cfg)
    status = {}
    for which in which_list:
        path = paths.manifest_path(cfg, which)
        if path.exists() and not verify:
            status[which] = "exists (not verified)"
            continue
        status[which] = save_or_verify_manifest(build_manifest(meta, midi_root, which, cfg), path)
    return status


def build_excerpt_files(cfg: dict, manifest: pd.DataFrame, conditions: list[str]) -> None:
    """GT CSVs (synthetic + real), excerpt MIDI and rendered WAVs for each excerpt."""
    data = cfg["data"]
    _, midi_root = ensure_maestro(cfg)
    soundfonts = {c: find_soundfont(cfg, cfg["conditions"][c]["soundfont"]) for c in conditions}

    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="excerpts"):
        eid, t0 = row.excerpt_id, float(row.start_s)
        gt_files = {kind: paths.gt_path(cfg, kind, eid) for kind in ("synthetic", "real")}
        midi_file = paths.excerpt_midi_path(cfg, eid)
        if not all(p.exists() for p in gt_files.values()) or not midi_file.exists():
            pm = load_performance(midi_root, row.midi_filename)
            notes, _ = performance_notes(pm, data["pedal_threshold"])
            for kind, df in ground_truth(notes, t0, data["excerpt_s"]).items():
                write_notes_csv(df, gt_files[kind])
            excerpt = make_excerpt_midi(pm, t0, data["excerpt_s"], data["tail_s"],
                                        data["pedal_threshold"], data["render"]["program"])
            midi_file.parent.mkdir(parents=True, exist_ok=True)
            excerpt.write(str(midi_file))
        for condition in conditions:
            wav = paths.audio_path(cfg, condition, eid)
            if not wav.exists():
                render_midi(midi_file, wav, soundfonts[condition], cfg)


def prepare_data(cfg: dict, which_list: list[str], conditions: list[str],
                 max_excerpts: int | None = None, verify: bool = True) -> dict[str, str]:
    status = freeze_manifests(cfg, which_list, verify)
    for which in which_list:
        manifest = load_manifest(paths.manifest_path(cfg, which), max_excerpts)
        build_excerpt_files(cfg, manifest, conditions)
    return status
