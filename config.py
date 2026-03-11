from __future__ import annotations
import os, random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch

def seed_everything(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def pick_root(preferred_roots: List[Path]) -> Path:
    for r in preferred_roots:
        if (r / "processed").exists():
            return r
    for r in preferred_roots:
        if r.exists():
            return r
    return Path.cwd()

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_vram_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    p = torch.cuda.get_device_properties(0)
    return float(p.total_memory / (1024**3))

@dataclass(frozen=True)
class PathsConfig:
    """
    Mirrors the PDF's directory layout:
      ROOT/processed/{melody_tokens,mert_embeddings,mert_clusters,posttrain_*}
    """
    root: Path

    @staticmethod
    def auto(preferred_roots: Optional[List[Path]] = None) -> "PathsConfig":
        cands: List[Path] = []
        if "POP909_ROOT" in os.environ and os.environ["POP909_ROOT"].strip():
            cands.append(Path(os.environ["POP909_ROOT"]).expanduser())
        cands += [
            Path("/autodl-fs/data/pop909_mert_diversity"),
            Path("/root/pop909_mert_diversity"),
        ]
        if preferred_roots:
            cands = list(preferred_roots) + cands
        root = pick_root(cands)
        return PathsConfig(root=root)

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def tok_dir(self) -> Path:
        return self.processed / "melody_tokens"

    @property
    def emb_dir(self) -> Path:
        return self.processed / "mert_embeddings"

    @property
    def clu_dir(self) -> Path:
        return self.processed / "mert_clusters"

    @property
    def eval_dir(self) -> Path:
        return self.processed / "eval_runs" / "eval_online_az_mcts_spark"

    # artifacts
    @property
    def data_npz(self) -> Path:
        return self.tok_dir / "dataset.npz"

    @property
    def vocab_json(self) -> Path:
        return self.tok_dir / "vocab.json"

    @property
    def emb_file(self) -> Path:
        return self.emb_dir / "mert_embeds.npz"

    @property
    def resp_file(self) -> Path:
        return self.clu_dir / "resp.npy"

    @property
    def p_data_file(self) -> Path:
        return self.clu_dir / "p_data.npy"

    @property
    def centroids_file(self) -> Path:
        return self.clu_dir / "centroids.npy"

    @property
    def sample_w_file(self) -> Path:
        return self.clu_dir / "sample_weights.npy"

    # checkpoints (same names as PDF)
    @property
    def ckpt_baseline(self) -> Path:
        return self.processed / "posttrain_melodylm_baseline_longform"

    @property
    def ckpt_softrew(self) -> Path:
        return self.processed / "posttrain_melodylm_softrew_kl_ch_longform"

    @property
    def ckpt_value_head_dir(self) -> Path:
        return self.processed / "posttrain_valuehead_prefix_cluster_longform"

    @property
    def value_head_pt(self) -> Path:
        return self.ckpt_value_head_dir / "value_head.pt"

@dataclass
class TrainArgs:
    batch_size: int = 4
    grad_accum: int = 8
    lr: float = 2e-4
    weight_decay: float = 0.01
    epochs: int = 5
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    fp16: bool = True
    num_workers: int = 2
    log_every: int = 50

@dataclass
class ValueTrainArgs:
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 0.0
    epochs: int = 6
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    fp16: bool = True
    num_workers: int = 2
    log_every: int = 100
    min_prefix_len: int = 8

@dataclass(frozen=True)
class ModelSize:
    """For GPT-scale experiments."""
    n_embd: int
    n_layer: int
    n_head: int

SCALE_SPECS = {
    "small": ModelSize(256, 6, 8),
    "mid":   ModelSize(512, 8, 8),
    "large": ModelSize(768, 12, 12),
}
