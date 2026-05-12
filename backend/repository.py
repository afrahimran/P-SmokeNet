from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from config import settings


def safe_relpath(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path.resolve())


def list_runs() -> list[str]:
    return sorted([p.name for p in settings.indices_dir.iterdir() if p.is_dir() and (p / "config.json").exists()])


def get_run_info(run_name: str) -> dict[str, Any]:
    run_dir = settings.indices_dir / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run folder: {run_dir}")
    out: dict[str, Any] = {"run_name": run_name}
    for name in ["config.json", "splits.json", "class_balance.json"]:
        p = run_dir / name
        if p.exists():
            out[name.replace(".json", "")] = json.loads(p.read_text())
    return out


def list_experiments(run_name: str) -> list[dict[str, Any]]:
    result_root = settings.results_dir / run_name
    model_root = settings.models_dir / run_name
    if not result_root.exists():
        return []
    exps: list[dict[str, Any]] = []
    for p in sorted([x for x in result_root.iterdir() if x.is_dir()]):
        summary_path = p / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
        ckpt_exists = (model_root / p.name / "best_checkpoint.pt").exists()
        exps.append(
            {
                "exp_name": p.name,
                "summary": summary,
                "checkpoint_exists": ckpt_exists,
                "result_dir": str(p),
                "model_dir": str(model_root / p.name),
            }
        )
    return exps


def get_experiment_summary(run_name: str, exp_name: str) -> dict[str, Any]:
    p = settings.results_dir / run_name / exp_name / "summary.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing summary: {p}")
    return json.loads(p.read_text())


def list_experiment_assets(run_name: str, exp_name: str) -> dict[str, list[str]]:
    base = settings.results_dir / run_name / exp_name
    if not base.exists():
        raise FileNotFoundError(f"Missing experiment directory: {base}")
    images: list[str] = []
    csvs: list[str] = []
    jsons: list[str] = []
    others: list[str] = []
    for p in sorted(base.rglob("*")):
        if p.is_dir():
            continue
        rel = safe_relpath(p, settings.artifacts_dir)
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            images.append(rel)
        elif p.suffix.lower() == ".csv":
            csvs.append(rel)
        elif p.suffix.lower() == ".json":
            jsons.append(rel)
        else:
            others.append(rel)
    return {"images": images, "csvs": csvs, "jsons": jsons, "others": others}


def _comparison_label_from_name(file_name: str) -> str:
    stem = Path(file_name).stem.replace("_", " ").strip()
    return " ".join(word.capitalize() for word in stem.split())


def list_comparison_files() -> list[dict[str, str]]:
    base = settings.results_dir / "comparisons"
    if not base.exists():
        return []
    preferred_order = [
        "ablations.csv",
        "backbone_experiments.csv",
        "temporal_experiments.csv",
        "horizon_experiments.csv",
    ]
    present = {p.name: p for p in base.glob("*.csv")}
    files: list[Path] = []
    for name in preferred_order:
        if name in present:
            files.append(present.pop(name))
    files.extend(sorted(present.values(), key=lambda p: p.name))
    return [
        {
            "file_name": p.name,
            "label": _comparison_label_from_name(p.name),
            "path": str(p),
        }
        for p in files
    ]


def load_comparison_file(file_name: str) -> tuple[Path, pd.DataFrame]:
    base = settings.results_dir / "comparisons"
    p = (base / file_name).resolve()
    if p.parent != base.resolve() or not p.exists() or p.suffix.lower() != ".csv":
        raise FileNotFoundError(f"Missing comparison CSV: {base / file_name}")
    return p, pd.read_csv(p)


def load_latest_comparison() -> tuple[Path, pd.DataFrame]:
    files = list_comparison_files()
    if not files:
        raise FileNotFoundError(f"Missing comparison CSV directory: {settings.results_dir / 'comparisons'}")
    return load_comparison_file(files[0]["file_name"])


def load_events_df(run_name: str) -> pd.DataFrame:
    p = settings.results_dir / run_name / "events_df.csv"
    if p.exists():
        return pd.read_csv(p)
    smoke_seg_csv = settings.common_dir / "smoke_segment_inventory.csv"
    fps = float(json.loads((settings.indices_dir / run_name / "config.json").read_text())["fps"])
    smoke_segments_df = pd.read_csv(smoke_seg_csv)
    smoke_only = smoke_segments_df[smoke_segments_df["label"] == 1].copy().sort_values(["video_id", "frame_start"])
    rows = []
    for r in smoke_only.itertuples(index=False):
        rows.append({
            "video_id": r.video_id,
            "event_start": int(r.frame_start),
            "event_end": int(r.frame_end),
            "event_length_frames": int(r.frame_end - r.frame_start + 1),
            "event_length_seconds": float((r.frame_end - r.frame_start + 1) / fps),
        })
    return pd.DataFrame(rows)


def load_video_inventory() -> pd.DataFrame:
    p = settings.common_dir / "video_inventory.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing video inventory: {p}")
    return pd.read_csv(p)
