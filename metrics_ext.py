from __future__ import annotations
from dataclasses import asdict
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import torch

from .metrics import (
    SparkStats, spark_metrics, distinct_n, novelty_n, token_entropy, uniq_ratio, repeat_rate,
    nll_per_seq, cluster_probs_per_seq, entropy_from_probs
)

def spans_to_pairs(spans) -> List[Tuple[int,int]]:
    out = []
    for s in spans:
        if isinstance(s, (tuple, list)) and len(s) == 2:
            out.append((int(s[0]), int(s[1])))
        else:
            out.append((int(s.start), int(s.end)))
    return out

def per_seq_metrics(
    *,
    seqs: List[List[int]],
    surpr_lists: List[List[float]],
    spans_list: List[List[Tuple[int,int]]],
    pad_id: int,
    spark_th: float,
    recovery_K: int,
    train_ng2: set,
    train_ng3: set,
    lm_quality,                 # GPT2LMHeadModel
    model_cluster,              # GPT2WithClusterHead
    device: str,
    max_ctx: int = 512,
    nll_batch: int = 8,
    cluster_batch: int = 16,
) -> Tuple[pd.DataFrame, np.ndarray]:
    assert len(seqs) == len(surpr_lists) == len(spans_list)

    # per-seq NLL and cluster probs (batched)
    nll = nll_per_seq(lm_quality, seqs, pad_id=pad_id, device=device, max_ctx=max_ctx, batch_size=nll_batch)
    clu_p = cluster_probs_per_seq(model_cluster, seqs, pad_id=pad_id, device=device, max_ctx=max_ctx, batch_size=cluster_batch)
    clu_ent = entropy_from_probs(clu_p)

    rows: List[Dict] = []
    for i in range(len(seqs)):
        s = seqs[i]
        surpr = surpr_lists[i]
        spans = spans_list[i]

        sp = spark_metrics(surpr, spans, spark_th=spark_th, recovery_K=recovery_K)

        rows.append({
            "spark_rate": float(sp.spark_rate),
            "spark_in_span_ratio": float(sp.spark_in_span_ratio),
            "spark_outside_span_rate": float(sp.spark_outside_span_rate),
            "recovery_drop": float(sp.recovery_drop),
            "recovery_steps": float(sp.recovery_steps),

            "nll_quality": float(nll[i]),
            "surpr_mean": float(np.mean(surpr) if len(surpr)>0 else 0.0),
            "surpr_p99": float(np.quantile(np.array(surpr, dtype=np.float32), 0.99) if len(surpr)>0 else 0.0),
            "distinct2": float(distinct_n(s, 2, pad_id)),
            "distinct3": float(distinct_n(s, 3, pad_id)),
            "novelty2": float(novelty_n(s, 2, pad_id, train_ng2)),
            "novelty3": float(novelty_n(s, 3, pad_id, train_ng3)),
            "token_entropy": float(token_entropy(s, pad_id)),
            "uniq_ratio": float(uniq_ratio(s, pad_id)),
            "repeat1": float(repeat_rate(s, 1, pad_id)),
            "repeat3": float(repeat_rate(s, 3, pad_id)),
            "cluster_entropy": float(clu_ent[i]),
        })

    df = pd.DataFrame(rows)
    return df, clu_p

def summarize_metrics(
    *,
    df_perseq: pd.DataFrame,
    cluster_probs: np.ndarray,
    spark_rate_target: float,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for col in [
        "spark_rate","spark_in_span_ratio","spark_outside_span_rate","recovery_drop","recovery_steps",
        "nll_quality","surpr_mean","surpr_p99","distinct2","distinct3","novelty2","novelty3",
        "token_entropy","uniq_ratio","repeat1","repeat3","cluster_entropy",
    ]:
        out[f"{col}_mean"] = float(df_perseq[col].mean())
        out[f"{col}_std"] = float(df_perseq[col].std(ddof=1)) if len(df_perseq) > 1 else float("nan")

    out["spark_rate_target"] = float(spark_rate_target)
    out["spark_rate_gap"] = float(out["spark_rate_mean"] - float(spark_rate_target))

    # set-level cluster histogram entropy + coverage
    hist = cluster_probs.mean(axis=0)  # [K]
    hist = hist / (hist.sum() + 1e-12)
    ent = float(-(hist * np.log(hist + 1e-12)).sum())
    K = int(hist.shape[0])
    thr = float(1.0 / (10.0 * max(1, K)))
    coverage = int((hist > thr).sum())
    out["cluster_entropy_hist"] = float(ent)
    out["cluster_coverage"] = float(coverage)
    return out

def required_columns() -> List[str]:
    return [
        "spark_rate_mean","spark_in_span_ratio_mean","spark_outside_span_rate_mean","recovery_drop_mean","recovery_steps_mean",
        "spark_rate_std","spark_in_span_ratio_std","spark_outside_span_rate_std","recovery_drop_std","recovery_steps_std",
        "spark_rate_target","spark_rate_gap",
        "nll_quality_mean","surpr_mean_mean","surpr_p99_mean","distinct2_mean","distinct3_mean","novelty2_mean","novelty3_mean",
        "token_entropy_mean","uniq_ratio_mean","repeat1_mean","repeat3_mean","cluster_entropy_mean",
        "nll_quality_std","surpr_mean_std","surpr_p99_std","distinct2_std","distinct3_std","novelty2_std","novelty3_std",
        "token_entropy_std","uniq_ratio_std","repeat1_std","repeat3_std","cluster_entropy_std",
        "cluster_entropy_hist","cluster_coverage",
    ]
