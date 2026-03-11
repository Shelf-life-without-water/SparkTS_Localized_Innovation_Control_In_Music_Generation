from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from .config import PathsConfig

def _softmax_np(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + eps)

def kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))

def find_tau_with_kl_guardrail(
    p_data: np.ndarray,
    q_target: np.ndarray,
    kl_max: float = 0.35,
    tau_init: float = 2.0,
    max_iter: int = 40,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Exponential tilting with KL guardrail:
      w_c(tau) = exp(tau * log(q_c / p_c))
      p_tau(c) ∝ p_data(c) * w_c(tau)
    Pick the largest tau such that KL(p_tau || p_data) <= kl_max.
    """
    p = np.clip(p_data.astype(np.float64), 1e-12, 1.0)
    q = np.clip(q_target.astype(np.float64), 1e-12, 1.0)
    log_ratio = np.log(q) - np.log(p)

    def p_tau(tau: float):
        w = np.exp(tau * log_ratio)
        pt = p * w
        pt = pt / (pt.sum() + 1e-12)
        return pt, w

    pt0, w0 = p_tau(tau_init)
    if kl_div(pt0, p) <= kl_max:
        return float(tau_init), w0.astype(np.float32), pt0.astype(np.float32)

    lo, hi = 0.0, float(tau_init)
    best_tau, best_w, best_pt = 0.0, np.ones_like(p), p.copy()
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        pt, w = p_tau(mid)
        if kl_div(pt, p) <= kl_max:
            best_tau, best_w, best_pt = mid, w, pt
            lo = mid
        else:
            hi = mid
    return float(best_tau), best_w.astype(np.float32), best_pt.astype(np.float32)

def load_dataset_npz(path: Path) -> Tuple[List[str], List[List[int]], Optional[np.ndarray], str]:
    d = np.load(path, allow_pickle=True)
    keys = list(d.files)

    stem_key = None
    for k in ["stems", "stem", "names", "ids", "files"]:
        if k in keys:
            stem_key = k
            break

    tok_key = None
    for k in ["input_ids", "X", "tokens", "seqs", "arr_0", "data"]:
        if k in keys:
            tok_key = k
            break
    if tok_key is None:
        raise RuntimeError(f"Token key not found in {keys}")

    emb_idx_key = "emb_idx" if "emb_idx" in keys else None

    stems = (
        [str(x) for x in d[stem_key].tolist()] if stem_key
        else [f"item_{i:06d}" for i in range(len(d[tok_key]))]
    )

    toks_arr = d[tok_key]
    seqs: List[List[int]] = []
    if getattr(toks_arr, "dtype", None) == object:
        for s in toks_arr:
            s = np.array(s).astype(np.int64).tolist()
            seqs.append([int(x) for x in s])
    else:
        toks_arr = np.array(toks_arr).astype(np.int64)
        for row in toks_arr:
            seqs.append([int(x) for x in row.tolist()])

    emb_idx = None
    if emb_idx_key:
        emb_idx = np.array(d[emb_idx_key]).astype(np.int64)
        if len(emb_idx) != len(seqs):
            emb_idx = None

    return stems, seqs, emb_idx, tok_key

def load_mert_embeds(npz_path: Path) -> Tuple[Optional[List[str]], np.ndarray, str]:
    d = np.load(npz_path, allow_pickle=True)
    keys = list(d.files)
    stems = None
    if "stems" in keys:
        stems = [str(x) for x in d["stems"].tolist()]

    emb_key = None
    for k in ["X", "embeds", "embeddings", "feat", "features"]:
        if k in keys:
            emb_key = k
            break
    if emb_key is None:
        for k in keys:
            arr = d[k]
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                emb_key = k
                break
    if emb_key is None:
        raise RuntimeError(f"Cannot find 2D embedding matrix in {keys}")

    X = np.array(d[emb_key]).astype(np.float32)
    return stems, X, emb_key

def rebuild_clusters_kmeans(X: np.ndarray, K: int = 32, seed: int = 1234) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback if cluster artifacts missing."""
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=K, random_state=seed, n_init="auto")
        labels = km.fit_predict(X)
        centroids = km.cluster_centers_.astype(np.float32)
    except Exception:
        rng = np.random.default_rng(seed)
        centroids = X[rng.choice(len(X), size=K, replace=False)].copy()
        labels = np.zeros(len(X), dtype=np.int64)
        for _ in range(10):
            d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
            labels = d2.argmin(-1)
            for k in range(K):
                idx = np.where(labels == k)[0]
                if len(idx) > 0:
                    centroids[k] = X[idx].mean(0)
    resp = np.zeros((len(X), K), dtype=np.float32)
    resp[np.arange(len(X)), labels] = 1.0
    p_data = resp.mean(0).astype(np.float32)
    p_data = p_data / (p_data.sum() + 1e-9)
    return resp, p_data, centroids

def parse_vocab_json(path: Path) -> Dict[str, int]:
    """
    Robust to common wrappers:
      {"stoi": {...}} or {"token2id": {...}} or directly {token: id}
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "stoi" in raw and isinstance(raw["stoi"], dict):
        raw = raw["stoi"]
    if isinstance(raw, dict) and "token2id" in raw and isinstance(raw["token2id"], dict):
        raw = raw["token2id"]
    if isinstance(raw, dict):
        ok = True
        for v in raw.values():
            if not isinstance(v, (int, np.integer)):
                ok = False
                break
        if ok:
            return {str(k): int(v) for k, v in raw.items()}
    return {}

def infer_special_ids_from_vocab_or_data(token2id: Dict[str, int], seqs: List[List[int]]) -> Tuple[int, int, int, int]:
    def get_id(names):
        for n in names:
            if n in token2id:
                return int(token2id[n])
        return None

    def mode_first_token(seqs_):
        vals = [s[0] for s in seqs_ if len(s) > 0]
        if not vals:
            return 1
        vals = np.array(vals, dtype=np.int64)
        return int(np.bincount(vals).argmax())

    def mode_last_token(seqs_):
        vals = [s[-1] for s in seqs_ if len(s) > 0]
        if not vals:
            return 2
        vals = np.array(vals, dtype=np.int64)
        return int(np.bincount(vals).argmax())

    PAD_ID = get_id(["<pad>", "[PAD]", "PAD"])
    BOS_ID = get_id(["<bos>", "[BOS]", "BOS"])
    EOS_ID = get_id(["<eos>", "[EOS]", "EOS"])
    UNK_ID = get_id(["<unk>", "[UNK]", "UNK"])

    if BOS_ID is None:
        BOS_ID = mode_first_token(seqs)
    if EOS_ID is None:
        EOS_ID = mode_last_token(seqs)
    if PAD_ID is None:
        PAD_ID = 0
    if UNK_ID is None:
        UNK_ID = 3 if 3 not in [PAD_ID, BOS_ID, EOS_ID] else max(PAD_ID, BOS_ID, EOS_ID) + 1
    return int(PAD_ID), int(BOS_ID), int(EOS_ID), int(UNK_ID)

def infer_vocab_size_from_seqs(seqs: List[List[int]]) -> int:
    mx = 0
    for s in seqs:
        if len(s) == 0:
            continue
        mx = max(mx, int(np.max(np.array(s, dtype=np.int64))))
    return int(mx + 1)

@dataclass
class Module8Input:
    paths: PathsConfig
    stems_all: List[str]
    seqs_all: List[List[int]]
    emb_idx_all: np.ndarray
    X_emb: np.ndarray
    valid_emb_mask: np.ndarray
    K: int
    R_sample: np.ndarray
    p_data: np.ndarray
    KL_MAX: float
    tau: float
    p_tilt: np.ndarray
    w_soft: np.ndarray
    token2id: Dict[str, int]
    vocab_size: int
    PAD_ID: int
    BOS_ID: int
    EOS_ID: int
    UNK_ID: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    seqs_train: List[List[int]]
    seqs_val: List[List[int]]
    w_train: np.ndarray
    w_val: np.ndarray
    R_train: np.ndarray
    R_val: np.ndarray

def load_module8_input(
    paths: PathsConfig,
    *,
    K_default: int = 32,
    val_ratio: float = 0.1,
    max_len_for_training: int = 512,
    KL_MAX: float = 0.35,
    TAU_INIT: float = 2.0,
    CLIP_W: Tuple[float, float] = (0.2, 5.0),
    seed: int = 1234,
) -> Module8Input:
    for d in [paths.tok_dir, paths.emb_dir, paths.clu_dir, paths.ckpt_baseline, paths.ckpt_softrew]:
        d.mkdir(parents=True, exist_ok=True)

    assert paths.data_npz.exists(), f"Missing dataset.npz: {paths.data_npz}"
    assert paths.emb_file.exists(), f"Missing mert embeds: {paths.emb_file}"

    stems_all, seqs_all, emb_idx_all, _ = load_dataset_npz(paths.data_npz)
    N = len(seqs_all)

    _, X_emb, _ = load_mert_embeds(paths.emb_file)
    N_emb = X_emb.shape[0]

    if emb_idx_all is None:
        emb_idx_all = np.arange(N, dtype=np.int64)
    emb_idx_all = np.array(emb_idx_all, dtype=np.int64)
    valid_emb_mask = (emb_idx_all >= 0) & (emb_idx_all < N_emb)

    # clusters
    if paths.resp_file.exists() and paths.p_data_file.exists():
        R = np.load(paths.resp_file).astype(np.float32)
        p_data = np.load(paths.p_data_file).astype(np.float32)
        p_data = p_data / (p_data.sum() + 1e-9)
        K = int(R.shape[1])
    else:
        K = int(K_default)
        R, p_data, centroids = rebuild_clusters_kmeans(X_emb, K=K, seed=seed)
        np.save(paths.resp_file, R)
        np.save(paths.p_data_file, p_data)
        np.save(paths.centroids_file, centroids)

    R_sample = np.zeros((N, K), dtype=np.float32)
    R_sample[valid_emb_mask] = R[emb_idx_all[valid_emb_mask]]
    R_sample[~valid_emb_mask] = (np.ones(K, dtype=np.float32) / K)

    # KL-guarded tilt
    q_target = np.ones(K, dtype=np.float32) / K
    tau, w_cluster, p_tilt = find_tau_with_kl_guardrail(p_data=p_data, q_target=q_target, kl_max=KL_MAX, tau_init=TAU_INIT)
    w_cluster = np.clip(w_cluster, CLIP_W[0], CLIP_W[1]).astype(np.float32)
    w_soft = (R_sample * w_cluster[None, :]).sum(-1).astype(np.float32)

    if paths.sample_w_file.exists():
        w_raw = np.load(paths.sample_w_file).astype(np.float32)
        if len(w_raw) == N:
            w_aligned = w_raw
        elif len(w_raw) == N_emb:
            w_aligned = np.ones(N, dtype=np.float32)
            w_aligned[valid_emb_mask] = w_raw[emb_idx_all[valid_emb_mask]]
        else:
            w_aligned = None
        if w_aligned is not None:
            w_soft = 0.5 * w_soft + 0.5 * np.clip(w_aligned, CLIP_W[0], CLIP_W[1])

    w_soft = w_soft / (w_soft.mean() + 1e-9)
    w_soft = np.clip(w_soft, CLIP_W[0], CLIP_W[1]).astype(np.float32)

    # vocab
    token2id: Dict[str, int] = {}
    if paths.vocab_json.exists():
        token2id = parse_vocab_json(paths.vocab_json)

    PAD_ID, BOS_ID, EOS_ID, UNK_ID = infer_special_ids_from_vocab_or_data(token2id, seqs_all)
    vocab_size = infer_vocab_size_from_seqs(seqs_all)
    vocab_size = max(vocab_size, PAD_ID + 1, BOS_ID + 1, EOS_ID + 1, UNK_ID + 1)

    def sanitize_seq(seq: List[int]) -> List[int]:
        out = []
        for x in seq:
            xi = int(x)
            if xi < 0:
                out.append(PAD_ID)
            elif xi >= vocab_size:
                out.append(UNK_ID)
            else:
                out.append(xi)
        return out

    def normalize_seq(seq: List[int]) -> List[int]:
        seq = sanitize_seq(seq)[:max_len_for_training]
        if len(seq) == 0 or seq[0] != BOS_ID:
            seq = [BOS_ID] + seq
        seq = seq[:max_len_for_training]
        if len(seq) == 0:
            seq = [BOS_ID, EOS_ID]
        if seq[-1] != EOS_ID:
            if len(seq) < max_len_for_training:
                seq = seq + [EOS_ID]
            else:
                seq[-1] = EOS_ID
        return seq

    seqs_all = [normalize_seq(s) for s in seqs_all]

    # split
    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = max(1, int(N * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    seqs_train = [seqs_all[i] for i in train_idx]
    seqs_val = [seqs_all[i] for i in val_idx]
    w_train = w_soft[train_idx]
    w_val = np.ones(len(val_idx), dtype=np.float32)
    R_train = R_sample[train_idx]
    R_val = R_sample[val_idx]

    return Module8Input(
        paths=paths,
        stems_all=stems_all,
        seqs_all=seqs_all,
        emb_idx_all=emb_idx_all,
        X_emb=X_emb,
        valid_emb_mask=valid_emb_mask,
        K=int(K),
        R_sample=R_sample,
        p_data=p_data.astype(np.float32),
        KL_MAX=float(KL_MAX),
        tau=float(tau),
        p_tilt=p_tilt.astype(np.float32),
        w_soft=w_soft.astype(np.float32),
        token2id=token2id,
        vocab_size=int(vocab_size),
        PAD_ID=int(PAD_ID),
        BOS_ID=int(BOS_ID),
        EOS_ID=int(EOS_ID),
        UNK_ID=int(UNK_ID),
        train_idx=train_idx,
        val_idx=val_idx,
        seqs_train=seqs_train,
        seqs_val=seqs_val,
        w_train=w_train.astype(np.float32),
        w_val=w_val.astype(np.float32),
        R_train=R_train.astype(np.float32),
        R_val=R_val.astype(np.float32),
    )
