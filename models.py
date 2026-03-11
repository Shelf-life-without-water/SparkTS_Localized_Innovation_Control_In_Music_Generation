from __future__ import annotations
import json
from pathlib import Path
from typing import Dict
import torch
from transformers import GPT2LMHeadModel, GPT2Config

class GPT2WithClusterHead(torch.nn.Module):
    """
    Policy LM + auxiliary cluster head (predicts MERT cluster distribution from last hidden).
    """
    def __init__(self, cfg: GPT2Config, num_clusters: int):
        super().__init__()
        self.lm = GPT2LMHeadModel(cfg)
        self.cluster_head = torch.nn.Linear(cfg.n_embd, int(num_clusters))
        self.num_clusters = int(num_clusters)
        self.cfg = cfg
        if cfg.n_embd >= 512:
            try:
                self.lm.gradient_checkpointing_enable()
            except Exception:
                pass
        self.lm.config.use_cache = False

    def forward(self, input_ids, attention_mask=None, labels=None, output_hidden_states=True):
        out = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        h = out.hidden_states[-1]  # [B,L,H]
        if attention_mask is None:
            last = h[:, -1, :]
        else:
            lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
            last = h[torch.arange(h.size(0), device=h.device), lengths, :]
        cluster_logits = self.cluster_head(last)
        return out, cluster_logits

def save_ckpt(model: GPT2WithClusterHead, ckpt_dir: Path, meta: Dict):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.lm.save_pretrained(ckpt_dir)
    torch.save(model.cluster_head.state_dict(), ckpt_dir / "cluster_head.pt")
    (ckpt_dir / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def load_ckpt(ckpt_dir: Path, num_clusters: int) -> GPT2WithClusterHead:
    lm = GPT2LMHeadModel.from_pretrained(ckpt_dir)
    cfg = lm.config
    model = GPT2WithClusterHead(cfg, num_clusters=int(num_clusters))
    model.lm = lm
    ch_path = ckpt_dir / "cluster_head.pt"
    if ch_path.exists():
        model.cluster_head.load_state_dict(torch.load(ch_path, map_location="cpu"))
    return model

class ValueHead(torch.nn.Module):
    """Prefix -> cluster logits (K)."""
    def __init__(self, hidden_size: int, num_clusters: int, mlp_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(int(hidden_size), int(mlp_hidden)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(int(mlp_hidden), int(num_clusters)),
        )

    def forward(self, h_last: torch.Tensor) -> torch.Tensor:
        return self.net(h_last)

class GPT2WithValueHead(torch.nn.Module):
    """Frozen LM backbone + trainable value head."""
    def __init__(self, lm: GPT2LMHeadModel, value_head: ValueHead):
        super().__init__()
        self.lm = lm
        self.value_head = value_head

    def forward(self, input_ids, attention_mask=None):
        out = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            output_hidden_states=True,
            return_dict=True,
        )
        h = out.hidden_states[-1]
        if attention_mask is None:
            last = h[:, -1, :]
        else:
            lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
            last = h[torch.arange(h.size(0), device=h.device), lengths, :]
        v_logits = self.value_head(last)
        return v_logits
