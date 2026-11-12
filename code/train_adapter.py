"""在冻结的 Chinese-CLIP 特征上训练线性探针 / 轻量 Adapter，做图文检索改进实验。

实验设计（对应 PPT「改进建议」与「对比实验」）:
  baseline : zero-shot（主干直接输出）
  method-1 : 线性探针  f' = normalize(W f)，W 可训练，主干冻结
  method-2 : 轻量 Adapter f' = normalize(f + W2·GELU(W1 f))，残差结构，主干冻结
  method-3 : 部分参数微调（LoRA）→ 见 train_lora.py，需要端到端前向

因为主干冻结、特征已缓存（results/feats），这里只训很小的映射层，
单个 epoch 在秒级，可以低成本扫「训练样本量」曲线（1%/10%/50%/100%）。

用法:
  python code/train_adapter.py --method linear --fraction 0.01 --epochs 10
  python code/train_adapter.py --method linear --fraction 1.0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from muge_eval import RESULT

FEATS = RESULT / "feats"


class LinearProbe(nn.Module):
    """线性探针: f' = normalize(W f)。

    关键：W 必须初始化为单位矩阵。随机初始化会把预训练学到的跨模态对齐结构
    直接打散（实测 1% 数据下 R@1 从 52 掉到 0.68），恒等初始化则从 zero-shot
    水平出发，训练只做"微调式"线性重映射。
    """

    def __init__(self, dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=True)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(dim))
            self.proj.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)


class ResidualAdapter(nn.Module):
    """轻量 Adapter: f' = f + W2·GELU(W1 f)，瓶颈维可调。

    up 层零初始化 → 初始输出等于输入，训练从 zero-shot 水平起步，只学增量。
    """

    def __init__(self, dim: int = 512, hidden: int = 128, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Linear(dim, hidden)
        self.up = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
        with torch.no_grad():
            self.up.weight.zero_()
            self.up.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up(F.gelu(self.down(x)))
        return F.normalize(x + self.drop(h), dim=-1)


def load_split(split: str):
    imgs = np.load(FEATS / f"{split}_imgs.npz")
    txts = np.load(FEATS / f"{split}_texts.npz")
    pairs = [json.loads(l) for l in open(FEATS / f"{split}_pairs.jsonl", encoding="utf-8")]
    return (imgs["ids"], imgs["feats"].astype(np.float32),
            txts["ids"], txts["feats"].astype(np.float32), pairs)


def build_pairs(pairs, img_ids, txt_ids):
    """把 text_id -> [image_ids] 展开成 (img_row, txt_row) 对。"""
    img_pos = {int(i): k for k, i in enumerate(img_ids)}
    txt_pos = {int(t): k for k, t in enumerate(txt_ids)}
    ii, tt = [], []
    for rec in pairs:
        ti = txt_pos.get(int(rec["text_id"]))
        if ti is None:
            continue
        for g in rec["image_ids"]:
            gi = img_pos.get(int(g))
            if gi is not None:
                ii.append(gi)
                tt.append(ti)
    return np.array(ii, dtype=np.int64), np.array(tt, dtype=np.int64)


@torch.no_grad()
def eval_retrieval(img_adapter, txt_adapter, v_img_f, v_txt_f, v_img_ids, v_txt_ids, v_pairs, ks=(1, 5, 10)):
    device = v_img_f.device
    img_e = img_adapter(v_img_f)
    txt_e = txt_adapter(v_txt_f)

    img_pos = {int(i): k for k, i in enumerate(v_img_ids)}
    gt_sets = []
    for rec in v_pairs:
        gt_sets.append({img_pos[int(g)] for g in rec["image_ids"] if int(g) in img_pos})
    valid_q = sum(1 for g in gt_sets if g)

    hits = {k: 0 for k in ks}
    for s in range(0, txt_e.shape[0], 512):
        sim = txt_e[s:s + 512] @ img_e.T
        topk = torch.topk(sim, k=max(ks), dim=1).indices.cpu().numpy()
        for bi in range(topk.shape[0]):
            qi = s + bi
            if not gt_sets[qi]:
                continue
            pred = topk[bi]
            for k in ks:
                if any(p in gt_sets[qi] for p in pred[:k]):
                    hits[k] += 1
    t2i = {k: hits[k] / max(valid_q, 1) * 100 for k in ks}

    # I2T：某图的正样本文本 = 所有把该图列入 image_ids 的文本
    txt_pos = {int(t): k for k, t in enumerate(v_txt_ids)}
    gt_by_img = {}
    for rec in v_pairs:
        ti = txt_pos.get(int(rec["text_id"]))
        if ti is None:
            continue
        for g in rec["image_ids"]:
            gp = img_pos.get(int(g))
            if gp is not None:
                gt_by_img.setdefault(gp, set()).add(ti)
    hits_i = {k: 0 for k in ks}
    valid_i = 0
    for s in range(0, img_e.shape[0], 512):
        sim = img_e[s:s + 512] @ txt_e.T
        topk = torch.topk(sim, k=max(ks), dim=1).indices.cpu().numpy()
        for bi in range(topk.shape[0]):
            gi = s + bi
            g = gt_by_img.get(gi)
            if not g:
                continue
            valid_i += 1
            for k in ks:
                if any(int(p) in g for p in topk[bi][:k]):
                    hits_i[k] += 1
    i2t = {k: hits_i[k] / max(valid_i, 1) * 100 for k in ks}
    return t2i, i2t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["linear", "adapter"], default="linear")
    ap.add_argument("--fraction", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--logit-scale", type=float, default=100.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("loading cached features ...", flush=True)
    tr_img_ids, tr_img_f, tr_txt_ids, tr_txt_f, tr_pairs = load_split("train")
    va_img_ids, va_img_f, va_txt_ids, va_txt_f, va_pairs = load_split("valid")
    print(f"train imgs {tr_img_f.shape} texts {tr_txt_f.shape}", flush=True)

    ii, tt = build_pairs(tr_pairs, tr_img_ids, tr_txt_ids)
    n_total = len(ii)
    n_use = max(64, int(n_total * args.fraction))
    perm = np.random.permutation(n_total)[:n_use]
    ii, tt = ii[perm], tt[perm]
    print(f"pairs total {n_total} | 使用 {n_use} ({args.fraction:.0%})", flush=True)

    tr_img = torch.from_numpy(tr_img_f).to(device)
    tr_txt = torch.from_numpy(tr_txt_f).to(device)
    va_img = torch.from_numpy(va_img_f).to(device)
    va_txt = torch.from_numpy(va_txt_f).to(device)
    ii_t = torch.from_numpy(ii).to(device)
    tt_t = torch.from_numpy(tt).to(device)

    make = (lambda: LinearProbe()) if args.method == "linear" else \
           (lambda: ResidualAdapter(hidden=args.hidden))
    img_adapter = make().to(device)
    txt_adapter = make().to(device)
    params = list(img_adapter.parameters()) + list(txt_adapter.parameters())
    n_params = sum(p.numel() for p in params)
    print(f"method={args.method} trainable params={n_params/1e6:.3f}M", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps_per_epoch = max(1, n_use // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.epochs * steps_per_epoch)
    logit_scale = args.logit_scale

    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(n_use, device=device)
        img_adapter.train(); txt_adapter.train()
        tot = 0.0
        for s in range(0, n_use - args.bs + 1, args.bs):
            sel = perm[s:s + args.bs]
            bi, bt = ii_t[sel], tt_t[sel]
            ie = img_adapter(tr_img[bi])
            te = txt_adapter(tr_txt[bt])
            logits = ie @ te.T * logit_scale
            labels = torch.arange(ie.shape[0], device=device)
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            tot += loss.item()
        el = time.time() - t0
        print(f"  epoch {ep+1}/{args.epochs} loss={tot/max(steps_per_epoch,1):.4f} "
              f"elapsed={el:.0f}s", flush=True)

    img_adapter.eval(); txt_adapter.eval()
    torch.cuda.reset_peak_memory_stats()
    t2i, i2t = eval_retrieval(img_adapter, txt_adapter, va_img, va_txt,
                              va_img_ids, va_txt_ids, va_pairs)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    train_time = time.time() - t0

    print(f"\n[T2I] R@1={t2i[1]:.2f} R@5={t2i[5]:.2f} R@10={t2i[10]:.2f} "
          f"MR={(t2i[1]+t2i[5]+t2i[10])/3:.2f}")
    print(f"[I2T] R@1={i2t[1]:.2f} R@5={i2t[5]:.2f} R@10={i2t[10]:.2f} "
          f"MR={(i2t[1]+i2t[5]+i2t[10])/3:.2f}")
    print(f"train {train_time:.0f}s | peak VRAM {peak:.2f} GB | pairs {n_use}")

    summary = {
        "method": args.method, "fraction": args.fraction, "epochs": args.epochs,
        "bs": args.bs, "lr": args.lr, "hidden": args.hidden, "seed": args.seed,
        "trainable_params": n_params,
        "pairs_total": int(n_total), "pairs_used": int(n_use),
        "t2i": {f"R@{k}": round(v, 2) for k, v in t2i.items()},
        "t2i_MR": round(sum(t2i.values()) / 3, 2),
        "i2t": {f"R@{k}": round(v, 2) for k, v in i2t.items()},
        "i2t_MR": round(sum(i2t.values()) / 3, 2),
        "train_sec": round(train_time, 1),
        "peak_vram_gb": round(peak, 2),
    }
    tag = args.out or f"adapter_{args.method}_frac{args.fraction}_ep{args.epochs}_seed{args.seed}"
    with open(RESULT / f"{tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    torch.save({"img": img_adapter.state_dict(), "txt": txt_adapter.state_dict(),
                "args": vars(args)}, RESULT / f"{tag}.pt")
    print("saved:", RESULT / f"{tag}.json")


if __name__ == "__main__":
    main()
