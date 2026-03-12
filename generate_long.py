from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .az_mcts import AZMCTS, AZNode, az_mcts_pick_action
from .models import ValueHead

@dataclass
class InnovationSpan:
    start: int
    end: int
    target_mode: str = "tilted"  # "tilted" | "uniform" | "contrast"

def make_default_spans(
    *,
    total_new: int,
    n_spans: int = 1,
    span_len: int = 128,
    warmup: int = 400,
    seed: int = 1234,
    target_mode: str = "tilted",
) -> List[InnovationSpan]:
    rng = np.random.default_rng(seed)
    spans: List[InnovationSpan] = []
    for _ in range(int(n_spans)):
        if total_new <= warmup + span_len + 1:
            start = max(0, total_new // 2 - span_len // 2)
        else:
            start = int(rng.integers(warmup, total_new - span_len))
        spans.append(InnovationSpan(start=start, end=start + int(span_len), target_mode=target_mode))
    spans.sort(key=lambda s: s.start)
    return spans

def _sample_from_probs(probs: torch.Tensor) -> int:
    return int(torch.multinomial(probs, num_samples=1).item())

@torch.no_grad()
def _lm_next_logits(lm, seq: List[int], device: str, max_ctx: int) -> torch.Tensor:
    seq_ctx = seq[-max_ctx:]
    x = torch.tensor(seq_ctx, dtype=torch.long, device=device)[None, :]
    out = lm(input_ids=x, attention_mask=None, labels=None, return_dict=True)
    return out.logits[0, -1]

@torch.no_grad()
def _lm_last_hidden(lm, seq: List[int], device: str, max_ctx: int) -> torch.Tensor:
    seq_ctx = seq[-max_ctx:]
    x = torch.tensor(seq_ctx, dtype=torch.long, device=device)[None, :]
    out = lm(
        input_ids=x,
        attention_mask=None,
        labels=None,
        output_hidden_states=True,
        return_dict=True,
    )
    return out.hidden_states[-1][0, -1]

def _pick_target_cluster(
    *,
    K: int,
    p_tilt: np.ndarray,
    mode: str,
    baseline_cluster: Optional[int],
    rng: np.random.Generator,
) -> int:
    if mode == "uniform":
        return int(rng.integers(0, K))
    if mode == "tilted":
        return int(rng.choice(np.arange(K), p=p_tilt))
    if mode == "contrast":
        if baseline_cluster is None:
            return int(rng.choice(np.arange(K), p=p_tilt))
        for _ in range(10):
            c = int(rng.choice(np.arange(K), p=p_tilt))
            if c != int(baseline_cluster):
                return c
        return int((baseline_cluster + 1) % K)
    raise ValueError(f"Unknown target_mode: {mode}")

@torch.no_grad()
def estimate_context_cluster(value_head: ValueHead, lm, seq: List[int], device: str, max_ctx: int) -> int:
    h = _lm_last_hidden(lm, seq, device, max_ctx=max_ctx)
    logits = value_head(h[None, :])[0]
    return int(torch.argmax(logits).item())

@torch.no_grad()
def sample_next_token_with_surprise(
    *,
    lm,
    device: str,
    seq: List[int],
    temperature: float,
    top_k: int,
    max_ctx: int,
) -> Tuple[int, float]:
    logits = _lm_next_logits(lm, seq, device, max_ctx=max_ctx)
    logp_raw = F.log_softmax(logits, dim=-1)
    logits_s = logits / max(1e-6, float(temperature))
    probs = torch.softmax(logits_s, dim=-1)
    if 0 < int(top_k) < probs.numel():
        p, idx = torch.topk(probs, k=int(top_k))
        p = p / (p.sum() + 1e-12)
        tok = int(idx[_sample_from_probs(p)].item())
    else:
        tok = _sample_from_probs(probs)
    surpr = float(-float(logp_raw[tok].item()))
    return tok, surpr

@torch.no_grad()
def generate_long_sampling(
    *,
    lm,
    device: str,
    bos_id: int,
    eos_id: int,
    spans: Optional[Sequence[InnovationSpan]] = None,
    total_new: int = 1600,
    temperature: float = 1.05,
    top_k: int = 48,
    max_ctx: int = 512,
    seed: int = 1234,
) -> Tuple[List[int], List[float]]:
    _ = np.random.default_rng(seed)  # reserved
    seq: List[int] = [int(bos_id)]
    surprs: List[float] = []
    for _t in range(int(total_new)):
        tok, surpr = sample_next_token_with_surprise(
            lm=lm,
            device=device,
            seq=seq,
            temperature=temperature,
            top_k=top_k,
            max_ctx=max_ctx,
        )
        seq.append(int(tok))
        surprs.append(float(surpr))
        if int(eos_id) >= 0 and int(tok) == int(eos_id):
            break
    return seq, surprs

@torch.no_grad()
def generate_long_online_mcts_spark(
    *,
    lm,
    device: str,
    bos_id: int,
    eos_id: int,
    p_tilt: np.ndarray,
    spans: Sequence[InnovationSpan],
    total_new: int = 1600,
    temperature: float = 1.05,
    top_k: int = 48,
    max_ctx: int = 512,
    seed: int = 1234,
    mcts: AZMCTS,
    value_head_for_target: Optional[ValueHead] = None,
) -> Tuple[List[int], List[float], List[int]]:
    rng = np.random.default_rng(seed)
    spans = sorted(list(spans or []), key=lambda s: s.start)
    K = int(len(p_tilt))

    seq: List[int] = [int(bos_id)]
    surprs: List[float] = []
    targets: List[int] = []

    span_i = 0
    cur_span: Optional[InnovationSpan] = None
    cur_target_cluster: Optional[int] = None
    cur_root: Optional[AZNode] = None

    for t in range(int(total_new)):
        if cur_span is None and span_i < len(spans) and t >= spans[span_i].start:
            cur_span = spans[span_i]
            span_i += 1
            cur_target_cluster = None
            cur_root = None

        if cur_span is not None and t >= cur_span.end:
            cur_span = None
            cur_target_cluster = None
            cur_root = None

        in_span = cur_span is not None
        if not in_span:
            tok, surpr = sample_next_token_with_surprise(
                lm=lm,
                device=device,
                seq=seq,
                temperature=temperature,
                top_k=top_k,
                max_ctx=max_ctx,
            )
            seq.append(int(tok))
            surprs.append(float(surpr))
            targets.append(-1)
            if int(eos_id) >= 0 and int(tok) == int(eos_id):
                break
            continue

        if cur_target_cluster is None:
            baseline_cluster = None
            if cur_span.target_mode == "contrast" and value_head_for_target is not None:
                baseline_cluster = estimate_context_cluster(value_head_for_target, lm, seq, device, max_ctx=max_ctx)
            cur_target_cluster = _pick_target_cluster(
                K=K,
                p_tilt=p_tilt,
                mode=cur_span.target_mode,
                baseline_cluster=baseline_cluster,
                rng=rng,
            )
            span_len = int(cur_span.end) - int(cur_span.start)
            mcts.set_span_len(span_len)
            cur_root = AZNode(list(seq), parent=None, prior=1.0, step=0)

        assert cur_root is not None and cur_target_cluster is not None
        tok, surpr = az_mcts_pick_action(mcts, cur_root, target_cluster=int(cur_target_cluster))

        if int(tok) in cur_root.children:
            nxt = cur_root.children[int(tok)]
            nxt.parent = None
            cur_root = nxt
        else:
            cur_root = AZNode(cur_root.seq + [int(tok)], parent=None, prior=1.0, step=int(cur_root.step) + 1, incoming_surpr=float(surpr))

        seq = list(cur_root.seq)
        surprs.append(float(surpr))
        targets.append(int(cur_target_cluster))
        if int(eos_id) >= 0 and int(tok) == int(eos_id):
            break

    return seq, surprs, targets
