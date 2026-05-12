from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config import settings
from modeling import (
    DEVICE,
    build_features_from_raw_video,
    get_backbone_dir,
    load_experiment,
    predict_single_video_timeline,
    predict_timeline_from_arrays,
)
from repository import (
    get_experiment_summary,
    get_run_info,
    list_experiment_assets,
    list_experiments,
    list_runs,
    load_events_df,
    list_comparison_files,
    load_comparison_file,
    load_latest_comparison,
    load_video_inventory,
)
from schemas import (
    AssetsListing,
    ComparisonResponse,
    DemoVideoInfo,
    ExperimentInfo,
    FrontendMetaResponse,
    HealthResponse,
    RunInfo,
    TimelineResponse,
    UploadInferenceResponse,
)

app = FastAPI(title="P-SmokeNet Backend", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static/artifacts", StaticFiles(directory=str(settings.artifacts_dir)), name="static-artifacts")
app.mount("/static/demo-videos", StaticFiles(directory=str(settings.demo_videos_dir)), name="static-demo-videos")


def _list_demo_video_files() -> list[Path]:
    allowed = {".mp4", ".mov", ".avi", ".mkv"}
    return sorted([p for p in settings.demo_videos_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed])


def _trim_rows_to_demo_limit(rows: list[dict], fps: float) -> list[dict]:
    max_frames = int(settings.demo_max_seconds * fps)
    return [r for r in rows if int(r["t"]) <= max_frames]


def _trim_events_to_demo_limit(event_starts_sec: list[float]) -> list[float]:
    return [x for x in event_starts_sec if float(x) <= settings.demo_max_seconds]


def _has_demo_cache(video_id: str, backbone_name: str) -> bool:
    try:
        backbone_dir, _ = get_backbone_dir(backbone_name)
    except FileNotFoundError:
        return False

    emb_path = backbone_dir / f"{video_id}.npy"
    phys_path = settings.features_dir / "physics" / f"{video_id}.npy"
    return emb_path.exists() and phys_path.exists()


def _predict_demo_timeline_from_cache(video_id: str, experiment) -> list[dict]:
    backbone_dir, _ = get_backbone_dir(experiment.backbone_name)
    emb_path = backbone_dir / f"{video_id}.npy"
    phys_path = settings.features_dir / "physics" / f"{video_id}.npy"

    if not emb_path.exists():
        raise FileNotFoundError(f"Missing demo embedding cache: {emb_path}")
    if not phys_path.exists():
        raise FileNotFoundError(f"Missing demo physics cache: {phys_path}")

    emb_arr = np.load(emb_path, mmap_mode="r")
    phys_arr = np.load(phys_path, mmap_mode="r")
    return predict_timeline_from_arrays(emb_arr, phys_arr, experiment, video_id=video_id)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=str(DEVICE),
        project_root=str(settings.project_root),
        selected_run_name=settings.selected_run_name,
        final_experiment_name=settings.final_experiment_name,
    )


@app.get("/api/meta", response_model=FrontendMetaResponse)
def frontend_meta() -> FrontendMetaResponse:
    runs = list_runs()
    exps = [x["exp_name"] for x in list_experiments(settings.selected_run_name)] if settings.selected_run_name in runs else []
    return FrontendMetaResponse(
        selected_run_name=settings.selected_run_name,
        final_experiment_name=settings.final_experiment_name,
        available_runs=runs,
        available_experiments_for_selected_run=exps,
        static_base_url="/static/artifacts",
    )


@app.get("/api/demo-videos", response_model=list[DemoVideoInfo])
def api_demo_videos() -> list[DemoVideoInfo]:
    inventory_ids = set(load_video_inventory()["video_id"].astype(str).tolist())
    rows: list[DemoVideoInfo] = []
    for path in _list_demo_video_files():
        vid = path.stem
        rows.append(
            DemoVideoInfo(
                video_id=vid,
                filename=path.name,
                url=f"/static/demo-videos/{path.name}",
                exists_in_inventory=vid in inventory_ids,
            )
        )
    return rows


@app.get("/api/runs", response_model=list[str])
def api_list_runs() -> list[str]:
    return list_runs()


@app.get("/api/runs/{run_name}", response_model=RunInfo)
def api_get_run(run_name: str) -> RunInfo:
    try:
        info = get_run_info(run_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return RunInfo(
        run_name=run_name,
        config=info.get("config", {}),
        split_counts=info.get("splits", {}).get("counts") if isinstance(info.get("splits"), dict) else None,
        class_balance=info.get("class_balance"),
    )


@app.get("/api/runs/{run_name}/experiments", response_model=list[ExperimentInfo])
def api_list_experiments(run_name: str) -> list[ExperimentInfo]:
    return [ExperimentInfo(**x) for x in list_experiments(run_name)]


@app.get("/api/comparisons")
def api_list_comparisons() -> dict:
    return {"files": list_comparison_files()}


@app.get("/api/comparisons/{file_name}")
def api_comparison_file(file_name: str) -> dict:
    try:
        path, df = load_comparison_file(file_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "file_name": file_name,
        "label": Path(file_name).stem.replace("_", " ").title(),
        "path": str(path),
        "rows": df.to_dict(orient="records"),
    }


@app.get("/api/comparisons/latest", response_model=ComparisonResponse)
def api_latest_comparison() -> ComparisonResponse:
    try:
        path, df = load_latest_comparison()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ComparisonResponse(path=str(path), rows=df.to_dict(orient="records"))


@app.get("/api/experiments/{run_name}/{exp_name}/summary")
def api_experiment_summary(run_name: str, exp_name: str) -> dict:
    try:
        return get_experiment_summary(run_name, exp_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/experiments/{run_name}/{exp_name}/assets", response_model=AssetsListing)
def api_experiment_assets(run_name: str, exp_name: str) -> AssetsListing:
    try:
        assets = list_experiment_assets(run_name, exp_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return AssetsListing(**assets)


@app.get("/api/videos")
def api_videos() -> list[str]:
    df = load_video_inventory()
    return sorted(df["video_id"].astype(str).tolist())


@app.get("/api/videos/{video_id}/events")
def api_video_events(video_id: str, run_name: str = Query(default=None)) -> list[dict]:
    run_name = run_name or settings.selected_run_name
    events_df = load_events_df(run_name)
    rows = events_df[events_df["video_id"] == video_id].copy().sort_values("event_start")
    if "event_start" in rows.columns:
        rows = rows[rows["event_start"].astype(float) <= settings.demo_max_seconds * 25_000] if False else rows
    return rows.to_dict(orient="records")


@app.get("/api/inference/timeline/{video_id}", response_model=TimelineResponse)
def api_predict_video_timeline(
    video_id: str,
    run_name: str = Query(default=None),
    exp_name: str = Query(default=None),
) -> TimelineResponse:
    run_name = run_name or settings.selected_run_name
    exp_name = exp_name or settings.final_experiment_name

    try:
        experiment = load_experiment(run_name, exp_name)

        # Prefer cropped demo caches when they exist.
        if _has_demo_cache(video_id, experiment.backbone_name):
            rows = _predict_demo_timeline_from_cache(video_id, experiment)
        else:
            rows = predict_single_video_timeline(video_id, experiment)

        events_df = load_events_df(run_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    rows = _trim_rows_to_demo_limit(rows, experiment.run_context.fps)

    event_starts = events_df[events_df["video_id"] == video_id]["event_start"].astype(float).tolist()
    event_starts_sec = [x / experiment.run_context.fps for x in event_starts]
    event_starts_sec = _trim_events_to_demo_limit(event_starts_sec)

    return TimelineResponse(
        video_id=video_id,
        run_name=run_name,
        exp_name=exp_name,
        threshold=experiment.threshold,
        fps=experiment.run_context.fps,
        horizon_frames=experiment.run_context.w_frames,
        clip_len_frames=experiment.run_context.clip_len_frames,
        rows=rows,
        event_starts_sec=event_starts_sec,
    )


@app.post("/api/inference/upload-video", response_model=UploadInferenceResponse)
async def api_upload_video_inference(
    file: UploadFile = File(...),
    run_name: str | None = Query(default=None),
    exp_name: str | None = Query(default=None),
) -> UploadInferenceResponse:
    run_name = run_name or settings.selected_run_name
    exp_name = exp_name or settings.final_experiment_name

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise HTTPException(status_code=400, detail="Only video files are supported")

    temp_dir = Path(tempfile.mkdtemp(prefix="psmokenet_", dir=settings.temp_dir))
    temp_path = temp_dir / file.filename

    try:
        content = await file.read()
        if len(content) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Uploaded file is too large")

        temp_path.write_bytes(content)
        experiment = load_experiment(run_name, exp_name)
        emb_arr, phys_arr = build_features_from_raw_video(temp_path, experiment.backbone_name)
        rows = predict_timeline_from_arrays(emb_arr, phys_arr, experiment, video_id=file.filename)
        rows = _trim_rows_to_demo_limit(rows, experiment.run_context.fps)

        return UploadInferenceResponse(
            filename=file.filename,
            run_name=run_name,
            exp_name=exp_name,
            threshold=experiment.threshold,
            fps=experiment.run_context.fps,
            clip_len_frames=experiment.run_context.clip_len_frames,
            stride_frames=experiment.run_context.stride_frames,
            rows=rows,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/")
def root() -> dict:
    return {
        "name": "P-SmokeNet Backend",
        "message": "Backend is running.",
        "docs": "/docs",
    }