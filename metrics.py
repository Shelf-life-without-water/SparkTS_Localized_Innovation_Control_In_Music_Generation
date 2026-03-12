from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

def compute_surprise_threshold(
    surpr_lists: Sequence[Sequence[float]],
    *,
    q: float = 0.99,
    clip_min: float = 1.0,
    clip_max: float = 30.0,
) -> float:
    vals = []
    for s in surpr_lists:
        vals.extend([float(x) for x in s])
    if not vals:
        return float(clip_max)
    th = float(np.quantile(np.array(vals, dtype=np.float32), q))
    return float(np.clip(th, clip_min, clip_max))

@dataclass
class SparkStats:
    spark_rate: float
    spark_in_span_ratio: float
    spark_outside_span_rate: float
    recovery_drop: float
    recovery_steps: float

def spark_metrics(
    surpr: Sequence[float],
    spans: Sequence[Tuple[int,int]],
    *,
    spark_th: float,
    recovery_K: int,
) -> SparkStats:
    T = len(surpr)
    is_spark = np.array([float(x) >= float(spark_th) for x in surpr], dtype=np.bool_)
    total_sparks = int(is_spark.sum())
    spark_rate = float(total_sparks / max(1, T))

    # in-span mask
    in_span = np.zeros((T,), dtype=np.bool_)
    for a,b in spans:
        a = max(0,int(a)); b = min(T,int(b))
        if b> a:
            in_span[a:b] = True
    in_span_sparks = int((is_spark & in_span).sum())
    spark_in_span_ratio = float(in_span_sparks / max(1, total_sparks))

    outT = int((~in_span).sum())
    spark_outside_span_rate = float(int((is_spark & (~in_span)).sum()) / max(1, outT))

    # recovery: after each spark, find minimum surpr within K steps; report avg drop and steps to get below th
    drops = []
    steps = []
    for i in np.where(is_spark)[0].tolist():
        jmax = min(T, i + 1 + int(recovery_K))
        if jmax <= i+1:
            continue
        window = np.array(surpr[i+1:jmax], dtype=np.float32)
        if window.size == 0:
            continue
        minv = float(window.min())
        drops.append(float(surpr[i]) - minv)
        below = np.where(window < float(spark_th))[0]
        if below.size > 0:
            steps.append(float(below[0] + 1))
        else:
            steps.append(float(recovery_K))
    recovery_drop = float(np.mean(drops)) if drops else 0.0
    recovery_steps = float(np.mean(steps)) if steps else float(recovery_K)
    return SparkStats(
        spark_rate=spark_rate,
        spark_in_span_ratio=spark_in_span_ratio,
        spark_outside_span_rate=spark_outside_span_rate,
        recovery_drop=recovery_drop,
        recovery_steps=recovery_steps,
    )

def distinct_n(seq: Sequence[int], n: int, pad_id: int) -> float:
    s = [int(x) for x in seq if int(x) != int(pad_id)]
    if len(s) < n:
        return 0.0
    grams = [tuple(s[i:i+n]) for i in range(len(s)-n+1)]
    return float(len(set(grams)) / max(1, len(grams)))

def novelty_n(seq: Sequence[int], n: int, pad_id: int, train_ngrams: set) -> float:
    s = [int(x) for x in seq if int(x) != int(pad_id)]
    if len(s) < n:
        return 0.0
    grams = [tuple(s[i:i+n]) for i in range(len(s)-n+1)]
    nov = sum(1 for g in grams if g not in train_ngrams)
    return float(nov / max(1, len(grams)))

def token_entropy(seq: Sequence[int], pad_id: int) -> float:
    s = [int(x) for x in seq if int(x) != int(pad_id)]
    if not s:
        return 0.0
    vals = np.array(s, dtype=np.int64)
    hist = np.bincount(vals)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log(p + 1e-12)).sum())

def uniq_ratio(seq: Sequence[int], pad_id: int) -> float:
    s = [int(x) for x in seq if int(x) != int(pad_id)]
    if not s:
        return 0.0
    return float(len(set(s)) / len(s))

def repeat_rate(seq: Sequence[int], n: int, pad_id: int) -> float:
    s = [int(x) for x in seq if int(x) != int(pad_id)]
    if len(s) < n:
        return 0.0
    grams = [tuple(s[i:i+n]) for i in range(len(s)-n+1)]
    return float(1.0 - (len(set(grams)) / max(1, len(grams))))

@torch.no_grad()
def nll_per_seq(lm, seqs: List[List[int]], pad_id: int, device: str, max_ctx: int = 512, batch_size: int = 8) -> np.ndarray:
    """
    Returns per-seq average NLL under lm for next-token prediction, ignoring pad.
    Uses cropping to max_ctx.
    """
    lm.eval()
    out = []
    for i in range(0, len(seqs), int(batch_size)):
        batch = seqs[i:i+int(batch_size)]
        # crop & pad
        batch_ctx = [s[-max_ctx:] for s in batch]
        L = max(len(s) for s in batch_ctx)
        x = torch.full((len(batch_ctx), L), int(pad_id), dtype=torch.long, device=device)
        attn = torch.zeros((len(batch_ctx), L), dtype=torch.long, device=device)
        for j, s in enumerate(batch_ctx):
            x[j,:len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            attn[j,:len(s)] = 1
        logits = lm(input_ids=x, attention_mask=attn, labels=None, return_dict=True).logits  # [B,L,V]
        # shift
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = x[:, 1:].contiguous()
        shift_mask = (shift_labels != int(pad_id)).float()
        logp = F.log_softmax(shift_logits, dim=-1)
        nll = -logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1) * shift_mask
        per = nll.sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1.0)
        out.append(per.detach().cpu().numpy())
    return np.concatenate(out, axis=0)

@torch.no_grad()
def cluster_probs_per_seq(lm_with_cluster_head, seqs: List[List[int]], pad_id: int, device: str, max_ctx: int = 512, batch_size: int = 16) -> np.ndarray:
    """
    Returns p(cluster|seq) per sequence using model.cluster_head(last_hidden).
    Expects lm_with_cluster_head is GPT2WithClusterHead or equivalent.
    """
    lm_with_cluster_head.eval()
    probs = []
    for i in range(0, len(seqs), int(batch_size)):
        batch = seqs[i:i+int(batch_size)]
        batch_ctx = [s[-max_ctx:] for s in batch]
        L = max(len(s) for s in batch_ctx)
        x = torch.full((len(batch_ctx), L), int(pad_id), dtype=torch.long, device=device)
        attn = torch.zeros((len(batch_ctx), L), dtype=torch.long, device=device)
        for j, s in enumerate(batch_ctx):
            x[j,:len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            attn[j,:len(s)] = 1
        out, c_logits = lm_with_cluster_head(x, attention_mask=attn, labels=None, output_hidden_states=True)
        p = torch.softmax(c_logits, dim=-1).detach().cpu().numpy().astype(np.float32)
        probs.append(p)
    return np.concatenate(probs, axis=0)

def entropy_from_probs(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum(axis=-1)
