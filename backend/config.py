from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel, Field


class Settings(BaseModel):
    project_root: Path = Field(default_factory=lambda: Path(os.getenv("PSMOKENET_PROJECT_ROOT", Path(__file__).resolve().parents[1])))
    selected_run_name: str = os.getenv("PSMOKENET_RUN_NAME", "w2p0s_l2p0s_s0p2s")
    final_experiment_name: str = os.getenv("PSMOKENET_FINAL_EXPERIMENT_NAME", "abl_w2p0s_l2p0s_s0p2s_fusion")
    allow_origins: list[str] = Field(default_factory=lambda: os.getenv("PSMOKENET_ALLOW_ORIGINS", "*").split(","))
    max_upload_mb: int = int(os.getenv("PSMOKENET_MAX_UPLOAD_MB", "512"))
    temp_dir_name: str = os.getenv("PSMOKENET_TEMP_DIR", "temp")
    demo_videos_dir_name: str = os.getenv("PSMOKENET_DEMO_VIDEOS_DIR", "demo_videos")
    demo_max_seconds: int = int(os.getenv("PSMOKENET_DEMO_MAX_SECONDS", "60"))

    @property
    def datasets_dir(self) -> Path:
        return self.project_root / "datasets"

    @property
    def cholec80_root(self) -> Path:
        return self.datasets_dir / "Cholec80"

    @property
    def videos_dir(self) -> Path:
        return self.cholec80_root / "videos"

    @property
    def smoke_root(self) -> Path:
        return self.datasets_dir / "Smoke_Cholec80"

    @property
    def smoke_map_dir(self) -> Path:
        return self.smoke_root / "cholec80_mappings"

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts"

    @property
    def common_dir(self) -> Path:
        return self.artifacts_dir / "common"

    @property
    def indices_dir(self) -> Path:
        return self.artifacts_dir / "indices"

    @property
    def features_dir(self) -> Path:
        return self.artifacts_dir / "features"

    @property
    def embeddings_dir(self) -> Path:
        return self.artifacts_dir / "embeddings"

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def results_dir(self) -> Path:
        return self.artifacts_dir / "results"

    @property
    def figures_dir(self) -> Path:
        return self.artifacts_dir / "figures"

    @property
    def logs_dir(self) -> Path:
        return self.artifacts_dir / "logs"

    @property
    def temp_dir(self) -> Path:
        return self.artifacts_dir / self.temp_dir_name

    @property
    def backend_dir(self) -> Path:
        return self.project_root / "backend"

    @property
    def demo_videos_dir(self) -> Path:
        return self.backend_dir / self.demo_videos_dir_name


settings = Settings()
for p in [
    settings.artifacts_dir,
    settings.common_dir,
    settings.indices_dir,
    settings.features_dir,
    settings.embeddings_dir,
    settings.models_dir,
    settings.results_dir,
    settings.figures_dir,
    settings.logs_dir,
    settings.temp_dir,
    settings.demo_videos_dir,
]:
    p.mkdir(parents=True, exist_ok=True)
