from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    EfficientNet_B0_Weights,
    ConvNeXt_Tiny_Weights,
    convnext_tiny,
    efficientnet_b0,
)

from config import settings

PHYS_DIM = 6
PHYS_FEATURE_NAMES = [
    "gray_mean",
    "gray_std",
    "lap_var",
    "diff_energy",
    "bright_ratio",
    "edge_ratio",
]


@dataclass
class RunContext:
    run_name: str
    config: dict
    phys_mean: np.ndarray
    phys_std: np.ndarray
    fps: float
    clip_len_frames: int
    stride_frames: int
    w_frames: int


@dataclass
class LoadedExperiment:
    exp_name: str
    run_context: RunContext
    checkpoint: dict
    model: nn.Module
    threshold: float
    backbone_name: str
    embedding_dim: int
    use_physics: bool


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        return x + residual


class TemporalAttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn = torch.softmax(self.score(x).squeeze(-1), dim=1)
        pooled = torch.sum(x * attn.unsqueeze(-1), dim=1)
        return pooled, attn


class EmbeddingOnlyTCN(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int = 192, dropout: float = 0.2) -> None:
        super().__init__()
        self.emb_proj = nn.Sequential(nn.Linear(emb_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.tcn = nn.Sequential(
            TCNBlock(hidden_dim, dilation=1, dropout=dropout),
            TCNBlock(hidden_dim, dilation=2, dropout=dropout),
            TCNBlock(hidden_dim, dilation=4, dropout=dropout),
        )
        self.pool = TemporalAttentionPool(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_emb: torch.Tensor, x_phys: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.emb_proj(x_emb)
        x = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        pooled, attn = self.pool(x)
        logits = self.head(pooled).squeeze(-1)
        return logits, attn


class PhysicsOnlyTCN(nn.Module):
    def __init__(self, phys_dim: int = PHYS_DIM, hidden_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.phys_proj = nn.Sequential(nn.Linear(phys_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.tcn = nn.Sequential(
            TCNBlock(hidden_dim, dilation=1, dropout=dropout),
            TCNBlock(hidden_dim, dilation=2, dropout=dropout),
            TCNBlock(hidden_dim, dilation=4, dropout=dropout),
        )
        self.pool = TemporalAttentionPool(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_emb: torch.Tensor, x_phys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.phys_proj(x_phys)
        x = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        pooled, attn = self.pool(x)
        logits = self.head(pooled).squeeze(-1)
        return logits, attn


class FusionTCN(nn.Module):
    def __init__(self, emb_dim: int, phys_dim: int = PHYS_DIM, hidden_dim: int = 192, phys_hidden: int = 48, dropout: float = 0.2) -> None:
        super().__init__()
        self.emb_proj = nn.Sequential(nn.Linear(emb_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.phys_proj = nn.Sequential(nn.Linear(phys_dim, phys_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.fusion_proj = nn.Sequential(nn.Linear(hidden_dim + phys_hidden, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        fusion_dim = hidden_dim
        self.tcn = nn.Sequential(
            TCNBlock(fusion_dim, dilation=1, dropout=dropout),
            TCNBlock(fusion_dim, dilation=2, dropout=dropout),
            TCNBlock(fusion_dim, dilation=4, dropout=dropout),
        )
        self.pool = TemporalAttentionPool(fusion_dim)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )

    def forward(self, x_emb: torch.Tensor, x_phys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.emb_proj(x_emb)
        phys = self.phys_proj(x_phys)
        fused = self.fusion_proj(torch.cat([emb, phys], dim=-1))
        x = self.tcn(fused.transpose(1, 2)).transpose(1, 2)
        pooled, attn = self.pool(x)
        logits = self.head(pooled).squeeze(-1)
        return logits, attn

class GatedFusionTCN(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        phys_dim: int = PHYS_DIM,
        hidden_dim: int = 192,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.emb_proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.phys_proj = nn.Sequential(
            nn.Linear(phys_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.fusion_refine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.tcn = nn.Sequential(
            TCNBlock(hidden_dim, dilation=1, dropout=dropout),
            TCNBlock(hidden_dim, dilation=2, dropout=dropout),
            TCNBlock(hidden_dim, dilation=4, dropout=dropout),
        )

        self.pool = TemporalAttentionPool(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )

    def forward(self, x_emb: torch.Tensor, x_phys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.emb_proj(x_emb)
        phys = self.phys_proj(x_phys)

        gate_input = torch.cat([emb, phys], dim=-1)
        gate = self.gate_net(gate_input)

        fused = gate * emb + (1.0 - gate) * phys
        fused = self.fusion_refine(fused)

        x = self.tcn(fused.transpose(1, 2)).transpose(1, 2)
        pooled, attn = self.pool(x)
        logits = self.head(pooled).squeeze(-1)
        return logits, attn
    
class FusionMeanPool(nn.Module):
    def __init__(self, emb_dim: int, phys_dim: int = PHYS_DIM,
                 hidden_dim: int = 192, phys_hidden: int = 48, dropout: float = 0.2) -> None:
        super().__init__()
        self.emb_proj = nn.Sequential(nn.Linear(emb_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.phys_proj = nn.Sequential(nn.Linear(phys_dim, phys_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim + phys_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )

    def forward(self, x_emb: torch.Tensor, x_phys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.emb_proj(x_emb)
        phys = self.phys_proj(x_phys)
        fused = self.fusion_proj(torch.cat([emb, phys], dim=-1))
        pooled = fused.mean(dim=1)
        attn = torch.full(
            (fused.shape[0], fused.shape[1]),
            1.0 / fused.shape[1],
            dtype=fused.dtype,
            device=fused.device,
        )
        logits = self.head(pooled).squeeze(-1)
        return logits, attn

class FusionGRU(nn.Module):
    def __init__(self, emb_dim: int, phys_dim: int = PHYS_DIM,
                 hidden_dim: int = 192, phys_hidden: int = 48, dropout: float = 0.2) -> None:
        super().__init__()
        self.emb_proj = nn.Sequential(nn.Linear(emb_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.phys_proj = nn.Sequential(nn.Linear(phys_dim, phys_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim + phys_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 96),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(96, 1),
        )

    def forward(self, x_emb: torch.Tensor, x_phys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.emb_proj(x_emb)
        phys = self.phys_proj(x_phys)
        fused = self.fusion_proj(torch.cat([emb, phys], dim=-1))
        out, _ = self.gru(fused)
        pooled = out[:, -1]
        attn = torch.zeros(out.shape[0], out.shape[1], dtype=out.dtype, device=out.device)
        attn[:, -1] = 1.0
        logits = self.head(pooled).squeeze(-1)
        return logits, attn
      
class EffB0Embedder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
        self.features = model.features
        self.avgpool = model.avgpool
        self.mean = np.array(weights.transforms().mean, dtype=np.float32)
        self.std = np.array(weights.transforms().std, dtype=np.float32)
        self.embedding_dim = 1280

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.avgpool(self.features(x)), 1)


class ConvNeXtTinyEmbedder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        model = convnext_tiny(weights=weights)
        self.features = model.features
        self.avgpool = model.avgpool
        self.mean = np.array(weights.transforms().mean, dtype=np.float32)
        self.std = np.array(weights.transforms().std, dtype=np.float32)
        self.embedding_dim = 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.avgpool(self.features(x)), 1)

_BACKBONE_CACHE: dict[str, nn.Module] = {}


def get_backbone_embedder(backbone_name: str) -> nn.Module:
    if backbone_name not in _BACKBONE_CACHE:
        if backbone_name == "effb0":
            model = EffB0Embedder()
        elif backbone_name == "convnext_tiny":
            model = ConvNeXtTinyEmbedder()
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        model.to(DEVICE)
        model.eval()
        _BACKBONE_CACHE[backbone_name] = model
    return _BACKBONE_CACHE[backbone_name]


def bgr_batch_to_torch_rgb(batch_bgr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    x = batch_bgr[..., ::-1].copy().astype(np.float32) / 255.0
    x = np.transpose(x, (0, 3, 1, 2))
    mean = mean.reshape(1, 3, 1, 1)
    std = std.reshape(1, 3, 1, 1)
    x = (x - mean) / std
    return torch.from_numpy(x)


def compute_frame_physics(gray_u8: np.ndarray, prev_gray_u8: np.ndarray | None = None) -> np.ndarray:
    gray = gray_u8.astype(np.float32)
    gray_mean = float(gray.mean())
    gray_std = float(gray.std())
    lap = cv2.Laplacian(gray_u8, cv2.CV_64F)
    lap_var = float(lap.var())
    if prev_gray_u8 is None:
        diff_energy = 0.0
    else:
        diff_energy = float(cv2.absdiff(gray_u8, prev_gray_u8).astype(np.float32).mean())
    bright_ratio = float((gray_u8 > 220).mean())
    edges = cv2.Canny(gray_u8, 50, 150)
    edge_ratio = float((edges > 0).mean())
    return np.array([gray_mean, gray_std, lap_var, diff_energy, bright_ratio, edge_ratio], dtype=np.float32)


def _load_run_context(run_name: str) -> RunContext:
    run_dir = settings.indices_dir / run_name
    config = json.loads((run_dir / "config.json").read_text())
    norm_path = settings.results_dir / run_name / "physics_norm_stats.json"
    norm = json.loads(norm_path.read_text()) if norm_path.exists() else {"mean": [0.0] * PHYS_DIM, "std": [1.0] * PHYS_DIM}
    return RunContext(
        run_name=run_name,
        config=config,
        phys_mean=np.array(norm["mean"], dtype=np.float32),
        phys_std=np.array(norm["std"], dtype=np.float32),
        fps=float(config["fps"]),
        clip_len_frames=int(config["clip_len_frames"]),
        stride_frames=int(config["stride_frames"]),
        w_frames=int(config["w_frames"]),
    )

def build_model_from_checkpoint(checkpoint: dict) -> nn.Module:
    model_type = checkpoint["model_type"]
    embedding_dim = int(checkpoint["embedding_dim"])

    if model_type == "embedding_only":
        model = EmbeddingOnlyTCN(emb_dim=embedding_dim)
    elif model_type == "physics_only":
        model = PhysicsOnlyTCN(phys_dim=PHYS_DIM)
    elif model_type == "fusion":
        model = FusionTCN(emb_dim=embedding_dim, phys_dim=PHYS_DIM)
    elif model_type == "fusion_gated":
        model = GatedFusionTCN(emb_dim=embedding_dim, phys_dim=PHYS_DIM)
    elif model_type == "fusion_meanpool":
        model = FusionMeanPool(emb_dim=embedding_dim, phys_dim=PHYS_DIM)
    elif model_type == "fusion_gru":
        model = FusionGRU(emb_dim=embedding_dim, phys_dim=PHYS_DIM)
    else:
        raise ValueError(f"Unknown model_type in checkpoint: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model

def load_experiment(run_name: str, exp_name: str) -> LoadedExperiment:
    ckpt_path = settings.models_dir / run_name / exp_name / "best_checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model = build_model_from_checkpoint(checkpoint)
    run_context = _load_run_context(run_name)
    return LoadedExperiment(
        exp_name=exp_name,
        run_context=run_context,
        checkpoint=checkpoint,
        model=model,
        threshold=float(checkpoint["threshold"]),
        backbone_name=str(checkpoint["backbone_name"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
        use_physics=bool(checkpoint.get("use_physics", True)),
    )


def get_backbone_dir(backbone_name: str) -> tuple[Path, int]:
    candidates = [settings.features_dir / backbone_name, settings.embeddings_dir / backbone_name]
    for d in candidates:
        if d.exists() and any(d.glob("*.npy")):
            meta_path = d / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                return d, int(meta["embedding_dim"])
            if backbone_name == "effb0":
                return d, 1280
            if backbone_name == "convnext_tiny":
                return d, 768
    raise FileNotFoundError(f"No cache found for backbone {backbone_name}")


def predict_single_video_timeline(video_id: str, experiment: LoadedExperiment) -> list[dict]:
    run_dir = settings.indices_dir / experiment.run_context.run_name
    clips_df = np.array([])
    import pandas as pd
    clips_df = pd.read_csv(run_dir / "clips_index.csv")
    video_clips = clips_df[clips_df["video_id"] == video_id].copy().sort_values("t").reset_index(drop=True)
    if len(video_clips) == 0:
        raise ValueError(f"No clips found for video_id={video_id}")

    backbone_dir, _ = get_backbone_dir(experiment.backbone_name)
    emb = np.load(backbone_dir / f"{video_id}.npy", mmap_mode="r")
    phys = np.load(settings.features_dir / "physics" / f"{video_id}.npy", mmap_mode="r")
    rows: list[dict] = []
    with torch.no_grad():
        for r in video_clips.itertuples(index=False):
            s = int(r.clip_start)
            e = int(r.clip_end)
            x_emb = torch.from_numpy(np.asarray(emb[s:e + 1], dtype=np.float32)).unsqueeze(0).to(DEVICE)
            x_phys = np.asarray(phys[s:e + 1], dtype=np.float32)
            x_phys = (x_phys - experiment.run_context.phys_mean) / experiment.run_context.phys_std
            x_phys_t = torch.from_numpy(x_phys).unsqueeze(0).to(DEVICE)
            logits, _ = experiment.model(x_emb, x_phys_t)
            prob = float(torch.sigmoid(logits).cpu().item())
            rows.append({
                "video_id": video_id,
                "t": int(r.t),
                "clip_start": s,
                "clip_end": e,
                "prob": prob,
                "pred": int(prob >= experiment.threshold),
                "target": int(r.target),
            })
    return rows


def read_video_all_frames(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from: {video_path}")
    return np.stack(frames, axis=0)


def build_features_from_raw_video(video_path: Path, backbone_name: str) -> tuple[np.ndarray, np.ndarray]:
    frames = read_video_all_frames(video_path)
    resized = np.stack([cv2.resize(f, (224, 224), interpolation=cv2.INTER_AREA) for f in frames], axis=0)
    backbone = get_backbone_embedder(backbone_name)
    mean = backbone.mean
    std = backbone.std
    batch_size = 64
    embs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(resized), batch_size):
            batch = resized[i:i + batch_size]
            x = bgr_batch_to_torch_rgb(batch, mean, std).to(DEVICE)
            embs.append(backbone(x).cpu().numpy().astype(np.float16))
    emb_arr = np.concatenate(embs, axis=0)

    phys_rows: list[np.ndarray] = []
    prev_gray = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        phys_rows.append(compute_frame_physics(gray, prev_gray))
        prev_gray = gray
    phys_arr = np.stack(phys_rows, axis=0).astype(np.float32)
    return emb_arr, phys_arr


def predict_timeline_from_arrays(emb_arr: np.ndarray, phys_arr: np.ndarray, experiment: LoadedExperiment, video_id: str = "uploaded_video") -> list[dict]:
    ctx = experiment.run_context
    rows: list[dict] = []
    n = emb_arr.shape[0]
    t_min = ctx.clip_len_frames - 1
    t_max = n - 1
    with torch.no_grad():
        for t in range(t_min, t_max + 1, ctx.stride_frames):
            s = t - ctx.clip_len_frames + 1
            e = t
            x_emb = torch.from_numpy(np.asarray(emb_arr[s:e + 1], dtype=np.float32)).unsqueeze(0).to(DEVICE)
            x_phys = np.asarray(phys_arr[s:e + 1], dtype=np.float32)
            x_phys = (x_phys - ctx.phys_mean) / ctx.phys_std
            x_phys_t = torch.from_numpy(x_phys).unsqueeze(0).to(DEVICE)
            logits, _ = experiment.model(x_emb, x_phys_t)
            prob = float(torch.sigmoid(logits).cpu().item())
            rows.append({
                "video_id": video_id,
                "t": int(t),
                "clip_start": int(s),
                "clip_end": int(e),
                "prob": prob,
                "pred": int(prob >= experiment.threshold),
                "target": None,
            })
    return rows
