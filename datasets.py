from __future__ import annotations
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import Dataset

class MelodyDS(Dataset):
    """Stage-1 LM training dataset."""
    def __init__(self, seqs: List[List[int]], w_soft: np.ndarray, R_soft: np.ndarray):
        self.seqs = seqs
        self.w = np.array(w_soft, dtype=np.float32)
        self.R = np.array(R_soft, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.seqs[i], dtype=torch.long),
            "w": torch.tensor(self.w[i], dtype=torch.float32),
            "R": torch.tensor(self.R[i], dtype=torch.float32),
        }

def collate_lm(batch: List[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    ids = [b["input_ids"] for b in batch]
    B = len(ids)
    L = max(x.numel() for x in ids)
    input_ids = torch.full((B, L), int(pad_id), dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    for i, x in enumerate(ids):
        l = x.numel()
        input_ids[i, :l] = x
        attn[i, :l] = 1
    labels = input_ids.clone()
    labels[input_ids == int(pad_id)] = -100
    w = torch.stack([b["w"] for b in batch], dim=0)  # [B]
    R = torch.stack([b["R"] for b in batch], dim=0)  # [B,K]
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels, "w": w, "R": R}

class PrefixValueDS(Dataset):
    """Value-head training dataset (random cut prefixes)."""
    def __init__(self, seqs: List[List[int]], R_soft: np.ndarray, min_prefix_len: int = 8):
        self.seqs = seqs
        self.R = np.array(R_soft, dtype=np.float32)
        self.min_prefix_len = int(min_prefix_len)

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        full = self.seqs[i]
        L = len(full)
        max_cut = max(self.min_prefix_len, L - 1)
        max_cut = min(max_cut, L - 1)
        if max_cut <= 2:
            cut = min(L, 2)
        else:
            import random
            cut = random.randint(2, max_cut)
        prefix = full[:cut]
        return {"input_ids": torch.tensor(prefix, dtype=torch.long), "R": torch.tensor(self.R[i], dtype=torch.float32)}

def collate_value(batch: List[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    ids = [b["input_ids"] for b in batch]
    B = len(ids)
    L = max(x.numel() for x in ids)
    input_ids = torch.full((B, L), int(pad_id), dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    for i, x in enumerate(ids):
        l = x.numel()
        input_ids[i, :l] = x
        attn[i, :l] = 1
    R = torch.stack([b["R"] for b in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attn, "R": R}
