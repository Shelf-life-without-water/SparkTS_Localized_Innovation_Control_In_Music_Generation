from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import GPT2Config
from transformers.optimization import get_linear_schedule_with_warmup

from .config import TrainArgs, ValueTrainArgs, get_device, get_vram_gb, ModelSize
from .datasets import MelodyDS, PrefixValueDS, collate_lm, collate_value
from .models import GPT2WithClusterHead, GPT2WithValueHead, ValueHead, save_ckpt
from .module8_input import Module8Input

def auto_model_size_by_vram(device: str) -> Tuple[int, int, int]:
    if device != "cuda":
        return 256, 6, 8
    vram = get_vram_gb()
    if vram < 14:
        return 256, 6, 8
    if vram < 20:
        return 512, 8, 8
    return 768, 12, 12

@torch.no_grad()
def eval_nll(model: GPT2WithClusterHead, loader: DataLoader, device: str) -> float:
    model.eval()
    losses = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out, _ = model(batch["input_ids"], batch["attention_mask"], batch["labels"])
        losses.append(float(out.loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")

def train_stage1_lm(
    m8: Module8Input,
    *,
    run_name: str,
    ckpt_dir: Path,
    use_softrew: bool,
    lambda_cluster: float,
    max_len: int = 512,
    args: Optional[TrainArgs] = None,
    force: bool = False,
    seed: int = 1234,
    model_size: Optional[ModelSize] = None,  # NEW: for GPT-scale exp
) -> None:
    device = get_device()
    args = args or TrainArgs()
    args.fp16 = bool(args.fp16 and device == "cuda")

    if (ckpt_dir / "config.json").exists() and (ckpt_dir / "cluster_head.pt").exists() and not force:
        print(f"[stage1:{run_name}] ckpt exists -> skip: {ckpt_dir}")
        return

    if model_size is None:
        MODEL_DIM, N_LAYER, N_HEAD = auto_model_size_by_vram(device)
    else:
        MODEL_DIM, N_LAYER, N_HEAD = int(model_size.n_embd), int(model_size.n_layer), int(model_size.n_head)

    gpt_cfg = GPT2Config(
        vocab_size=m8.vocab_size,
        n_positions=max_len,
        n_ctx=max_len,
        n_embd=MODEL_DIM,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        bos_token_id=m8.BOS_ID,
        eos_token_id=m8.EOS_ID,
        pad_token_id=m8.PAD_ID,
        use_cache=False,
    )

    train_ds = MelodyDS(m8.seqs_train, m8.w_train, m8.R_train)
    val_ds = MelodyDS(m8.seqs_val, m8.w_val, m8.R_val)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=lambda b: collate_lm(b, m8.PAD_ID),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=lambda b: collate_lm(b, m8.PAD_ID),
    )

    model = GPT2WithClusterHead(gpt_cfg, num_clusters=m8.K).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, (len(train_loader) * args.epochs) // max(1, args.grad_accum))
    warmup = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    ce_none = torch.nn.CrossEntropyLoss(reduction="none")
    kld = torch.nn.KLDivLoss(reduction="batchmean")

    best = float("inf")
    global_step = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[stage1:{run_name}] ep{ep}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(pbar, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            w = batch["w"]
            R = batch["R"]
            with torch.cuda.amp.autocast(enabled=args.fp16):
                out, cluster_logits = model(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])
                logits = out.logits
                labels = batch["labels"]

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                B, Lm1, V = shift_logits.shape
                loss_tok = ce_none(shift_logits.view(B * Lm1, V), shift_labels.view(B * Lm1)).view(B, Lm1)
                mask = (shift_labels != -100).float()
                per_ex = (loss_tok * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1.0))
                lm_loss = (per_ex * w).mean() if use_softrew else per_ex.mean()

                logp = torch.log_softmax(cluster_logits, dim=-1)
                cl_loss = kld(logp, R)

                loss = (lm_loss + float(lambda_cluster) * cl_loss) / max(1, args.grad_accum)

            scaler.scale(loss).backward()
            if step % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if global_step % args.log_every == 0:
                    val = eval_nll(model, val_loader, device)
                    pbar.set_postfix({"lm": float(lm_loss.detach().cpu()),
                                      "cl": float(cl_loss.detach().cpu()),
                                      "val": float(val),
                                      "lr": float(scheduler.get_last_lr()[0])})

        val = eval_nll(model, val_loader, device)
        print(f"[stage1:{run_name}] epoch {ep} val_nll={val:.4f}")
        if val < best:
            best = val
            meta = {
                "run": run_name,
                "best_val_nll": best,
                "use_softrew": use_softrew,
                "lambda_cluster": float(lambda_cluster),
                "KL_MAX": float(m8.KL_MAX),
                "tau": float(m8.tau),
                "model_dim": int(MODEL_DIM),
                "n_layer": int(N_LAYER),
                "n_head": int(N_HEAD),
                "max_len": int(max_len),
                "seed": int(seed),
                "train_args": asdict(args),
            }
            save_ckpt(model, ckpt_dir, meta)
            print(f"[stage1:{run_name}] saved best -> {ckpt_dir} (val={best:.4f})")

@torch.no_grad()
def eval_value(model_v: GPT2WithValueHead, loader: DataLoader, device: str):
    model_v.eval()
    losses = []
    accs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        v_logits = model_v(batch["input_ids"], batch["attention_mask"])
        logp = F.log_softmax(v_logits, dim=-1)
        loss = F.kl_div(logp, batch["R"], reduction="batchmean")
        losses.append(float(loss.detach().cpu()))
        pred = v_logits.argmax(dim=-1)
        gold = batch["R"].argmax(dim=-1)
        accs.append(float((pred == gold).float().mean().detach().cpu()))
    return float(np.mean(losses)), float(np.mean(accs))

def train_value_head(
    m8: Module8Input,
    *,
    backbone_ckpt_dir: Path,
    out_dir: Path,
    value_head_pt: Path,
    args: Optional[ValueTrainArgs] = None,
    force: bool = False,
    seed: int = 1234,
) -> None:
    device = get_device()
    args = args or ValueTrainArgs()
    args.fp16 = bool(args.fp16 and device == "cuda")

    out_dir.mkdir(parents=True, exist_ok=True)
    if value_head_pt.exists() and not force:
        print(f"[value] exists -> skip: {value_head_pt}")
        return

    from transformers import GPT2LMHeadModel
    lm = GPT2LMHeadModel.from_pretrained(backbone_ckpt_dir).to(device)
    lm.eval()
    for p in lm.parameters():
        p.requires_grad = False

    v_head = ValueHead(hidden_size=lm.config.n_embd, num_clusters=m8.K, mlp_hidden=256, dropout=0.1).to(device)
    model_v = GPT2WithValueHead(lm, v_head).to(device)

    ds_tr = PrefixValueDS(m8.seqs_train, m8.R_train, min_prefix_len=args.min_prefix_len)
    ds_va = PrefixValueDS(m8.seqs_val, m8.R_val, min_prefix_len=args.min_prefix_len)

    tr_loader = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=lambda b: collate_value(b, m8.PAD_ID),
        drop_last=False,
    )
    va_loader = DataLoader(
        ds_va,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=lambda b: collate_value(b, m8.PAD_ID),
        drop_last=False,
    )

    opt = AdamW(v_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(tr_loader) * args.epochs)
    warmup = int(total_steps * args.warmup_ratio)
    sch = get_linear_schedule_with_warmup(opt, warmup, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        model_v.train()
        pbar = tqdm(tr_loader, desc=f"[value] ep{ep}/{args.epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=args.fp16):
                v_logits = model_v(batch["input_ids"], batch["attention_mask"])
                logp = F.log_softmax(v_logits, dim=-1)
                loss = F.kl_div(logp, batch["R"], reduction="batchmean")
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(v_head.parameters(), args.max_grad_norm)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sch.step()

        va_loss, va_acc = eval_value(model_v, va_loader, device)
        print(f"[value] epoch {ep} val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
        if va_loss < best:
            best = va_loss
            torch.save(v_head.state_dict(), value_head_pt)
            meta = {"best_val_loss": best, "val_acc": va_acc, "K": int(m8.K), "hidden": int(lm.config.n_embd), "seed": int(seed)}
            (out_dir / "value_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[value] saved -> {value_head_pt}")
