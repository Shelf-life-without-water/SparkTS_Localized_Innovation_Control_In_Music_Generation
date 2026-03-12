from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Callable, Dict

import numpy as np
import torch
import torch.nn.functional as F

from .generate_long import InnovationSpan, _pick_target_cluster
from .az_mcts import build_ngram_set, _kl_divergence  # type: ignore
from .models import ValueHead

@torch.no_grad()
def _forward_last(lm, seq: List[int], device: str, max_ctx: int, output_hidden: bool = False):
    seq_ctx = seq[-max_ctx:]
    x = torch.tensor(seq_ctx, dtype=torch.long, device=device)[None, :]
    out = lm(
        input_ids=x,
        attention_mask=None,
        labels=None,
        output_hidden_states=bool(output_hidden),
        return_dict=True,
    )
    logits_last = out.logits[0, -1]
    logp_raw = F.log_softmax(logits_last, dim=-1)
    hidden_last = out.hidden_states[-1][0, -1] if output_hidden else None
    return logits_last, logp_raw, hidden_last

def _sample_from_scores(scores: torch.Tensor, top_k: int, temperature: float, rng: np.random.Generator) -> int:
    scores = scores / max(1e-6, float(temperature))
    if 0 < int(top_k) < scores.numel():
        v, idx = torch.topk(scores, k=int(top_k))
        p = torch.softmax(v, dim=-1)
        j = int(rng.choice(np.arange(int(top_k)), p=p.detach().cpu().numpy()))
        return int(idx[j].item())
    p = torch.softmax(scores, dim=-1)
    j = int(rng.choice(np.arange(scores.numel()), p=p.detach().cpu().numpy()))
    return int(j)

@torch.no_grad()
def sample_step(
    *,
    lm_sample,
    lm_surpr,
    seq: List[int],
    device: str,
    max_ctx: int,
    temperature: float,
    top_k: int,
    rng: np.random.Generator,
) -> Tuple[int, float]:
    logits_s, _logp_raw_s, _ = _forward_last(lm_sample, seq, device, max_ctx, output_hidden=False)
    # sampling distribution
    logits_s = logits_s / max(1e-6, float(temperature))
    tok = _sample_from_scores(logits_s, top_k=top_k, temperature=1.0, rng=rng)

    # surpr always under lm_surpr (guided)
    _logits_u, logp_raw_u, _ = _forward_last(lm_surpr, seq, device, max_ctx, output_hidden=False)
    surpr = float(-float(logp_raw_u[int(tok)].item()))
    return int(tok), float(surpr)

@torch.no_grad()
def contrastive_step(
    *,
    lm_expert,
    lm_amateur,
    lm_surpr,
    seq: List[int],
    device: str,
    max_ctx: int,
    alpha: float,
    temperature: float,
    top_k: int,
    rng: np.random.Generator,
) -> Tuple[int, float]:
    logits_e, logp_raw_e, _ = _forward_last(lm_expert, seq, device, max_ctx, output_hidden=False)
    logits_a, logp_raw_a, _ = _forward_last(lm_amateur, seq, device, max_ctx, output_hidden=False)

    # candidate set: top-k from expert temperatured logits
    scores_e = logits_e / max(1e-6, float(temperature))
    k = min(int(top_k), int(scores_e.numel()))
    v, idx = torch.topk(scores_e, k=k)

    # DExperts-style combination on log-probs (raw)
    comb = logp_raw_e[idx] - float(alpha) * logp_raw_a[idx]
    tok = int(idx[int(torch.argmax(comb).item())].item())  # deterministic by default

    # Optional stochasticity: uncomment if you want
    # tok = int(idx[_sample_from_scores(comb, top_k=k, temperature=1.0, rng=rng)].item())

    # surpr always under guided (lm_surpr)
    if lm_surpr is lm_expert:
        surpr = float(-float(logp_raw_e[tok].item()))
    else:
        _logits_u, logp_raw_u, _ = _forward_last(lm_surpr, seq, device, max_ctx, output_hidden=False)
        surpr = float(-float(logp_raw_u[tok].item()))
    return tok, surpr

def _repeat_penalty(seq: List[int], w_repeat1: float, w_repeat3: float) -> float:
    pen = 0.0
    if len(seq) >= 2 and seq[-1] == seq[-2]:
        pen += float(w_repeat1)
    if len(seq) >= 3:
        last3 = tuple(seq[-3:])
        for i in range(0, len(seq) - 3):
            if tuple(seq[i : i + 3]) == last3:
                pen += float(w_repeat3)
                break
    return float(pen)

def _novelty_bonus(seq: List[int], ng2: set, ng3: set, w_novel2: float, w_novel3: float) -> float:
    bonus = 0.0
    if len(seq) >= 2 and tuple(seq[-2:]) not in ng2:
        bonus += float(w_novel2)
    if len(seq) >= 3 and tuple(seq[-3:]) not in ng3:
        bonus += float(w_novel3)
    return float(bonus)

@torch.no_grad()
def greedy_reward_rescore_step(
    *,
    lm_guided,
    value_head: ValueHead,
    p_data: np.ndarray,
    kl_max: float,
    ng2: set,
    ng3: set,
    seq: List[int],
    device: str,
    max_ctx: int,
    target_cluster: int,
    # reward weights
    w_value: float,
    w_novel2: float,
    w_novel3: float,
    w_repeat1: float,
    w_repeat3: float,
    w_kl_barrier: float,
    # candidate selection
    cand_topk: int = 48,
) -> Tuple[int, float]:
    # 1) get candidate tokens from guided logits
    logits, logp_raw, _ = _forward_last(lm_guided, seq, device, max_ctx, output_hidden=False)
    k = min(int(cand_topk), int(logits.numel()))
    _, cand = torch.topk(logits, k=k)

    # 2) batch forward for each candidate to get hidden_last AFTER appending cand
    seq_ctx = seq[-max_ctx:]
    # build batch: each is ctx + [cand]
    B = int(cand.numel())
    maxL = len(seq_ctx) + 1
    x = torch.full((B, maxL), int(seq_ctx[0] if len(seq_ctx)>0 else 0), dtype=torch.long, device=device)
    attn = torch.zeros((B, maxL), dtype=torch.long, device=device)
    for i in range(B):
        toks = seq_ctx + [int(cand[i].item())]
        x[i, :len(toks)] = torch.tensor(toks, dtype=torch.long, device=device)
        attn[i, :len(toks)] = 1

    out = lm_guided(
        input_ids=x,
        attention_mask=attn,
        labels=None,
        output_hidden_states=True,
        return_dict=True,
    )
    hs = out.hidden_states[-1]  # [B,T,H]
    idx_last = attn.sum(dim=1).clamp(min=1) - 1
    h_last = hs[torch.arange(B, device=device), idx_last]  # [B,H]

    v_logits = value_head(h_last)  # [B,K]
    p = torch.softmax(v_logits, dim=-1)  # [B,K]
    v = p[:, int(target_cluster)]  # [B]

    # KL barrier term
    p_np = p.detach().cpu().numpy().astype(np.float32)
    klv = np.array([_kl_divergence(p_np[i], p_data) for i in range(B)], dtype=np.float32)
    kl_pen = float(w_kl_barrier) * np.maximum(0.0, klv - float(kl_max))  # [B]

    # novelty / repeat (cheap)
    seq_base = list(seq)
    bonus = np.zeros((B,), dtype=np.float32)
    rep = np.zeros((B,), dtype=np.float32)
    for i in range(B):
        s2 = seq_base + [int(cand[i].item())]
        bonus[i] = float(_novelty_bonus(s2, ng2, ng3, w_novel2, w_novel3))
        rep[i] = float(_repeat_penalty(s2, w_repeat1, w_repeat3))

    # combine score: logp_raw (quality) + reward terms
    score = logp_raw[cand] + float(w_value) * v.detach() + torch.tensor(bonus, device=device) - torch.tensor(rep, device=device) - torch.tensor(kl_pen, device=device)
    best_i = int(torch.argmax(score).item())
    tok = int(cand[best_i].item())

    surpr = float(-float(logp_raw[int(tok)].item()))  # guided surpr
    return tok, surpr

@torch.no_grad()
def beam_search_window(
    *,
    lm,
    device: str,
    prefix: List[int],
    n_new: int,
    max_ctx: int,
    beam_size: int = 4,
    cand_topk: int = 32,
    length_penalty: float = 1.0,
) -> Tuple[List[int], List[float]]:
    """
    Beam search for a fixed-length window.
    Returns (new_tokens, new_surprs) relative to the prefix.
    """
    beams: List[Tuple[List[int], float, List[float]]] = [(list(prefix), 0.0, [])]  # (seq, logp, surprs)
    for _t in range(int(n_new)):
        all_cand = []
        for seq, logp_acc, surprs in beams:
            logits, logp_raw, _ = _forward_last(lm, seq, device, max_ctx, output_hidden=False)
            k = min(int(cand_topk), int(logits.numel()))
            top_logp, idx = torch.topk(logp_raw, k=k)  # use raw logp for beam
            for j in range(k):
                tok = int(idx[j].item())
                lp = float(top_logp[j].item())
                new_seq = seq + [tok]
                new_logp = logp_acc + lp
                new_surprs = surprs + [float(-lp)]
                all_cand.append((new_seq, new_logp, new_surprs))

        # rank beams
        def score(item):
            seq, logp_sum, _sur = item
            L = max(1, len(seq) - len(prefix))
            return logp_sum / (L ** float(length_penalty))
        all_cand.sort(key=score, reverse=True)
        beams = all_cand[: int(beam_size)]

    best_seq, _best_logp, best_surprs = beams[0]
    new_tokens = best_seq[len(prefix):]
    return new_tokens, best_surprs

@torch.no_grad()
def generate_long_with_spans(
    *,
    lm_outside,
    lm_inside,  # kept for symmetry (strategy can close over it)
    lm_surpr,
    device: str,
    bos_id: int,
    eos_id: int,
    spans: Sequence[InnovationSpan],
    total_new: int,
    temperature: float,
    top_k: int,
    max_ctx: int,
    seed: int,
    inside_step_fn: Callable[..., Tuple[int, float]],
    inside_kwargs: Dict,
) -> Tuple[List[int], List[float]]:
    """
    Generic loop: outside spans uses sampling on lm_outside, inside spans uses `inside_step_fn`.
    Surpr always measured under `lm_surpr` (you can also choose to use a different surpr model via your inside_step_fn).
    """
    rng = np.random.default_rng(seed)
    spans = sorted(list(spans or []), key=lambda s: s.start)
    span_i = 0
    cur_span: Optional[InnovationSpan] = None

    seq: List[int] = [int(bos_id)]
    surprs: List[float] = []

    for t in range(int(total_new)):
        if cur_span is None and span_i < len(spans) and t >= spans[span_i].start:
            cur_span = spans[span_i]
            span_i += 1
        if cur_span is not None and t >= cur_span.end:
            cur_span = None

        in_span = cur_span is not None
        if not in_span:
            tok, surpr = sample_step(
                lm_sample=lm_outside,
                lm_surpr=lm_surpr,
                seq=seq,
                device=device,
                max_ctx=max_ctx,
                temperature=temperature,
                top_k=top_k,
                rng=rng,
            )
            seq.append(int(tok))
            surprs.append(float(surpr))
        else:
            tok, surpr = inside_step_fn(
                seq=seq,
                device=device,
                max_ctx=max_ctx,
                rng=rng,
                **inside_kwargs,
            )
            seq.append(int(tok))
            surprs.append(float(surpr))

        if int(eos_id) >= 0 and int(seq[-1]) == int(eos_id):
            break

    return seq, surprs

