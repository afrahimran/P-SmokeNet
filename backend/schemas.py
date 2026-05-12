from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    device: str
    project_root: str
    selected_run_name: str
    final_experiment_name: str


class RunInfo(BaseModel):
    run_name: str
    config: dict[str, Any]
    split_counts: dict[str, Any] | None = None
    class_balance: dict[str, Any] | None = None


class ExperimentInfo(BaseModel):
    exp_name: str
    summary: dict[str, Any] | None = None
    checkpoint_exists: bool
    result_dir: str
    model_dir: str


class PredictionRow(BaseModel):
    video_id: str
    t: int
    clip_start: int
    clip_end: int
    prob: float
    pred: int
    target: int | None = None


class TimelineResponse(BaseModel):
    video_id: str
    run_name: str
    exp_name: str
    threshold: float
    fps: float
    horizon_frames: int
    clip_len_frames: int
    rows: list[PredictionRow]
    event_starts_sec: list[float] = Field(default_factory=list)


class UploadInferenceResponse(BaseModel):
    filename: str
    run_name: str
    exp_name: str
    threshold: float
    fps: float
    clip_len_frames: int
    stride_frames: int
    rows: list[PredictionRow]


class AssetsListing(BaseModel):
    images: list[str]
    csvs: list[str]
    jsons: list[str]
    others: list[str]


class ComparisonResponse(BaseModel):
    path: str
    rows: list[dict[str, Any]]


class FrontendMetaResponse(BaseModel):
    selected_run_name: str
    final_experiment_name: str
    available_runs: list[str]
    available_experiments_for_selected_run: list[str]
    static_base_url: str


class DemoVideoInfo(BaseModel):
    video_id: str
    filename: str
    url: str
    exists_in_inventory: bool
