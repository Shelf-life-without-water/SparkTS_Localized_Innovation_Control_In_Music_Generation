from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

DEBUG_LEAF: bool = False  # prints first 3 leaf calls when True

@dataclass
class AZMCTSConfig:
    # budget / pruning
    sims: int = 192
    max_depth: int = 64
    max_children: int = 64

    # priors / sampling
    prior_topk: int = 160
    lm_temp: float = 1.05

    # PUCT
    c_puct: float = 1.5

    # progressive widening
    pw_k: float = 2.0
    pw_alpha: float = 0.55
    entropy_coef: float = 1.25

    # baseline reward shaping
    w_value: float = 1.0
    w_novel2: float = 0.02
    w_novel3: float = 0.15
    w_repeat1: float = 0.03
    w_repeat3: float = 0.04
    w_kl_barrier: float = 0.25

    # action selection temperature
    act_temp: float = 1.0
    stop_on_eos: bool = True

    # surprise/spark/recovery
    nll_hard_max: float = 8.0
    nll_soft_max: float = 6.0
    spark_th: float = 6.6
    target_spark_rate: float = 0.03
    w_surpr_soft_barrier: float = 0.12
    w_spark_bonus: float = 0.40
    w_spark_sparsity: float = 0.30
    recovery_K: int = 16
    w_recovery_drop: float = 0.08
    w_recovery_still_high_pen: float = 0.10
    w_recovery_low_surpr_bonus: float = 0.05

class AZNode:
    __slots__ = (
        "seq","parent","children","N","W","P",
        "prior_tok","prior_p","prior_entropy","prior_logp_raw_topk",
        "value_logits",
        "step","incoming_surpr","spark_count","last_spark_step","last_spark_surpr",
    )
    def __init__(
        self,
        seq: List[int],
        parent=None,
        prior: float = 1.0,
        *,
        step: int = 0,
        incoming_surpr: float = 0.0,
        spark_count: int = 0,
        last_spark_step: int = -10**9,
        last_spark_surpr: float = 0.0,
    ):
        self.seq = seq
        self.parent = parent
        self.children: Dict[int, "AZNode"] = {}
        self.N = 0
        self.W = 0.0
        self.P = float(prior)

        self.prior_tok = None
        self.prior_p = None
        self.prior_entropy = None
        self.prior_logp_raw_topk = None
        self.value_logits = None

        self.step = int(step)
        self.incoming_surpr = float(incoming_surpr)
        self.spark_count = int(spark_count)
        self.last_spark_step = int(last_spark_step)
        self.last_spark_surpr = float(last_spark_surpr)

    @property
    def Q(self) -> float:
        return self.W / (self.N + 1e-9)

def build_ngram_set(seqs: List[List[int]], n: int, pad_id: int) -> set:
    S = set()
    for s in seqs:
        s = [int(x) for x in s if int(x) != int(pad_id)]
        if len(s) < n:
            continue
        for i in range(len(s) - n + 1):
            S.add(tuple(s[i : i + n]))
    return S

def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))

def _puct_score(child: AZNode, parent_visits: int, c_puct: float) -> float:
    return child.Q + c_puct * child.P * math.sqrt(parent_visits + 1e-9) / (1.0 + child.N)

class AZMCTS:
    """
    Online AZ-MCTS for token generation (span-level).
    """
    def __init__(
        self,
        *,
        lm,  # GPT2LMHeadModel
        value_head,  # ValueHead
        device: str,
        cfg: Optional[AZMCTSConfig],
        pad_id: int,
        eos_id: int,
        p_data: np.ndarray,
        kl_max: float,
        ng2: set,
        ng3: set,
        max_ctx: int = 512,
        span_len: int = 256,
    ):
        self.lm = lm
        self.value_head = value_head
        self.device = device
        self.cfg = cfg or AZMCTSConfig()
        self.PAD_ID = int(pad_id)
        self.EOS_ID = int(eos_id)
        self.p_data = np.array(p_data, dtype=np.float32)
        self.kl_max = float(kl_max)
        self.ng2 = ng2
        self.ng3 = ng3
        self.max_ctx = int(max_ctx)
        self._span_len = int(span_len)
        self.lm.eval()
        self.value_head.eval()
        self._leaf_calls = 0

    def set_span_len(self, span_len: int) -> None:
        self._span_len = int(max(1, span_len))

    @property
    def span_len(self) -> int:
        return int(self._span_len)

    def _spark_zone_end(self) -> int:
        return max(1, int(self._span_len) - int(self.cfg.recovery_K))

    @torch.no_grad()
    def _node_forward(self, seq: List[int]):
        seq_ctx = seq[-self.max_ctx:]
        x = torch.tensor(seq_ctx, dtype=torch.long, device=self.device)[None, :]
        out = self.lm(
            input_ids=x,
            attention_mask=None,
            labels=None,
            output_hidden_states=True,
            return_dict=True,
        )
        logits_last = out.logits[0, -1]
        hidden_last = out.hidden_states[-1][0, -1]
        return logits_last, hidden_last

    @torch.no_grad()
    def _ensure_prior_and_value(self, node: AZNode):
        if (
            node.prior_tok is not None
            and node.value_logits is not None
            and node.prior_entropy is not None
            and node.prior_logp_raw_topk is not None
        ):
            return

        logits_last, hidden_last = self._node_forward(node.seq)
        logp_raw = F.log_softmax(logits_last, dim=-1)

        logits_temp = logits_last / max(1e-6, float(self.cfg.lm_temp))
        logp_temp = F.log_softmax(logits_temp, dim=-1)
        k = min(int(self.cfg.prior_topk), int(logp_temp.numel()))
        top_logp_temp, idx = torch.topk(logp_temp, k=k)
        p = torch.softmax(top_logp_temp, dim=-1)
        ent = float(-(p * torch.log(p + 1e-12)).sum().item())
        ent_norm = ent / max(1e-9, math.log(float(k) + 1e-9))

        node.prior_tok = idx.detach().cpu().numpy().astype(np.int64)
        node.prior_p = p.detach().cpu().numpy().astype(np.float32)
        node.prior_entropy = float(ent_norm)
        node.prior_logp_raw_topk = logp_raw[idx].detach().cpu().numpy().astype(np.float32)

        node.value_logits = self.value_head(hidden_last[None, :])[0]

    def _allowed_children(self, node: AZNode) -> int:
        H = node.prior_entropy if node.prior_entropy is not None else 0.0
        base = float(self.cfg.pw_k) * ((node.N + 1.0) ** float(self.cfg.pw_alpha))
        m = int(base * (1.0 + float(self.cfg.entropy_coef) * float(H)))
        m = max(1, min(m, int(self.cfg.max_children)))
        return m

    @torch.no_grad()
    def _maybe_expand(self, node: AZNode):
        self._ensure_prior_and_value(node)
        m_allow = self._allowed_children(node)
        if len(node.children) >= m_allow:
            return
        spark_zone_end = self._spark_zone_end()
        for tok, pi, logp_raw in zip(node.prior_tok.tolist(), node.prior_p.tolist(), node.prior_logp_raw_topk.tolist()):
            tok_i = int(tok)
            if tok_i in node.children:
                continue
            surpr = float(-float(logp_raw))
            if surpr > float(self.cfg.nll_hard_max):
                continue

            child_step = int(node.step) + 1
            child_spark_count = int(node.spark_count)
            child_last_spark_step = int(node.last_spark_step)
            child_last_spark_surpr = float(node.last_spark_surpr)

            is_spark = (child_step <= spark_zone_end) and (surpr >= float(self.cfg.spark_th))
            if is_spark:
                child_spark_count += 1
                child_last_spark_step = child_step
                child_last_spark_surpr = surpr

            node.children[tok_i] = AZNode(
                node.seq + [tok_i],
                parent=node,
                prior=float(pi),
                step=child_step,
                incoming_surpr=surpr,
                spark_count=child_spark_count,
                last_spark_step=child_last_spark_step,
                last_spark_surpr=child_last_spark_surpr,
            )
            if len(node.children) >= m_allow:
                break

    def _repeat_penalty(self, seq: List[int]) -> float:
        pen = 0.0
        if len(seq) >= 2 and seq[-1] == seq[-2]:
            pen += float(self.cfg.w_repeat1)
        if len(seq) >= 3:
            last3 = tuple(seq[-3:])
            for i in range(0, len(seq) - 3):
                if tuple(seq[i : i + 3]) == last3:
                    pen += float(self.cfg.w_repeat3)
                    break
        return float(pen)

    def _novelty_bonus(self, seq: List[int]) -> float:
        bonus = 0.0
        if len(seq) >= 2 and tuple(seq[-2:]) not in self.ng2:
            bonus += float(self.cfg.w_novel2)
        if len(seq) >= 3 and tuple(seq[-3:]) not in self.ng3:
            bonus += float(self.cfg.w_novel3)
        return float(bonus)

    @torch.no_grad()
    def _leaf_value_baseline(self, node: AZNode, target_cluster: int) -> float:
        self._ensure_prior_and_value(node)
        p = torch.softmax(node.value_logits, dim=-1).detach().cpu().numpy()
        v = float(p[int(target_cluster)])
        klv = _kl_divergence(p, self.p_data)
        kl_pen = float(self.cfg.w_kl_barrier) * max(0.0, float(klv) - float(self.kl_max))
        bonus = self._novelty_bonus(node.seq)
        rep_pen = self._repeat_penalty(node.seq)
        return float(self.cfg.w_value) * v + bonus - rep_pen - kl_pen

    @torch.no_grad()
    def _leaf_value(self, node: AZNode, target_cluster: int) -> float:
        global DEBUG_LEAF
        if DEBUG_LEAF and self._leaf_calls < 3:
            print(f"[DEBUG_LEAF] _leaf_value called #{self._leaf_calls+1} | step={node.step} incoming_surpr={node.incoming_surpr:.3f} spark_count={node.spark_count}")
        self._leaf_calls += 1

        base = self._leaf_value_baseline(node, target_cluster)
        surpr = float(node.incoming_surpr)

        soft_pen = float(self.cfg.w_surpr_soft_barrier) * max(0.0, surpr - float(self.cfg.nll_soft_max))

        spark_zone_end = self._spark_zone_end()
        in_spark_zone = int(node.step) <= int(spark_zone_end)
        spark_bonus = 0.0
        if in_spark_zone and surpr >= float(self.cfg.spark_th):
            denom = max(1e-6, float(self.cfg.nll_hard_max) - float(self.cfg.spark_th))
            intensity = min(1.0, max(0.0, (surpr - float(self.cfg.spark_th)) / denom))
            spark_bonus = float(self.cfg.w_spark_bonus) * intensity

        sparsity_pen = 0.0
        if in_spark_zone and node.step > 0:
            target = float(self.cfg.target_spark_rate) * float(node.step)
            excess = max(0.0, float(node.spark_count) - target)
            sparsity_pen = float(self.cfg.w_spark_sparsity) * (excess ** 2)

        in_recovery_zone = int(node.step) > int(spark_zone_end)
        recovery_bonus = 0.0
        recovery_pen = 0.0
        if in_recovery_zone:
            recovery_bonus += float(self.cfg.w_recovery_low_surpr_bonus)
            recovery_pen += float(self.cfg.w_recovery_still_high_pen) * max(0.0, surpr - float(self.cfg.nll_soft_max))

        if int(node.last_spark_step) > -10**8:
            dt = int(node.step) - int(node.last_spark_step)
            if 1 <= dt <= int(self.cfg.recovery_K):
                drop = max(0.0, float(node.last_spark_surpr) - surpr)
                recovery_bonus += float(self.cfg.w_recovery_drop) * drop
                if surpr >= float(self.cfg.spark_th):
                    recovery_pen += float(self.cfg.w_recovery_still_high_pen)

        return float(base) + float(spark_bonus) - float(soft_pen) - float(sparsity_pen) + float(recovery_bonus) - float(recovery_pen)

    @torch.no_grad()
    def search(self, root: AZNode, target_cluster: int) -> None:
        self._maybe_expand(root)
        for _ in range(int(self.cfg.sims)):
            node = root
            depth = 0
            while True:
                if self.cfg.stop_on_eos and node.seq and int(node.seq[-1]) == int(self.EOS_ID) and self.EOS_ID >= 0:
                    break
                if depth >= int(self.cfg.max_depth):
                    break
                self._maybe_expand(node)
                if not node.children:
                    break
                parentN = node.N
                best_child = None
                best_sc = -1e18
                for ch in node.children.values():
                    sc = _puct_score(ch, parentN, float(self.cfg.c_puct))
                    if sc > best_sc:
                        best_sc, best_child = sc, ch
                node = best_child
                depth += 1
                if node.N == 0:
                    break

            v = self._leaf_value(node, target_cluster)
            while node is not None:
                node.N += 1
                node.W += v
                node = node.parent

    def policy_from_visits(self, root: AZNode) -> Tuple[np.ndarray, np.ndarray]:
        items = list(root.children.items())
        if not items:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        toks = np.array([t for t, _ in items], dtype=np.int64)
        visits = np.array([ch.N for _, ch in items], dtype=np.float64) + 1e-12
        probs = visits / (visits.sum() + 1e-12)
        return toks, probs.astype(np.float32)

    def pick_action(self, root: AZNode, target_cluster: int) -> int:
        self.search(root, target_cluster)
        toks, probs = self.policy_from_visits(root)
        if toks.size == 0:
            self._maybe_expand(root)
            toks, probs = self.policy_from_visits(root)
            if toks.size == 0:
                return self.EOS_ID if self.EOS_ID >= 0 else int(0)

        if float(self.cfg.act_temp) <= 1e-6:
            return int(toks[int(probs.argmax())])

        w = probs.astype(np.float64) ** (1.0 / float(self.cfg.act_temp))
        w = w / (w.sum() + 1e-12)
        return int(np.random.choice(toks, p=w))

    def surpr_for_child_token(self, root: AZNode, tok: int) -> float:
        self._ensure_prior_and_value(root)
        if root.prior_tok is None or root.prior_logp_raw_topk is None:
            return float("inf")
        for t, logp in zip(root.prior_tok.tolist(), root.prior_logp_raw_topk.tolist()):
            if int(t) == int(tok):
                return float(-float(logp))
        logits_last, _ = self._node_forward(root.seq)
        logp_raw = F.log_softmax(logits_last, dim=-1)
        return float(-float(logp_raw[int(tok)].item()))

def az_mcts_pick_action(mcts: AZMCTS, root: AZNode, target_cluster: int) -> Tuple[int, float]:
    tok = int(mcts.pick_action(root, target_cluster=target_cluster))
    surpr = float(mcts.surpr_for_child_token(root, tok))
    return tok, surpr
