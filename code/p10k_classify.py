"""Products-10K 商品细粒度分类：线性探针 + 长尾分析 + 少样本曲线。

背景：Products-10K 只有 SKU 编号、没有可读类目名称，因此**无法做零样本分类**
（CLIP 零样本需要文本类名）；本实验定位是方向五选题示例中「视觉语言商品分类」
的补充任务，做的是「冻结特征 + 线性探针」与「少样本可训练性」分析。

实验:
  1. 全量线性探针：141,931 训练图 → 9,691 类，测试集 top-1 / top-5
  2. 长尾分析：按每类图片数分 5 组，比较各组准确率
  3. 少样本曲线：每类 K=1/5/10/50 张训练，观察 top-1 变化

用法:
  python code/p10k_classify.py
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
FIG = RESULT / "figs"


def load(split: str):
    d = np.load(FEATS / f"p10k_{split}.npz")
    feats = d["feats"].astype(np.float32)
    labels = d["labels"].astype(np.int64)
    return feats, labels


def train_probe(tr_x, tr_y, n_classes, device, epochs=20, lr=2e-3, bs=2048, seed=0):
    torch.manual_seed(seed)
    model = nn.Linear(tr_x.shape[1], n_classes).to(device)
    nn.init.zeros_(model.bias)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    x = torch.from_numpy(tr_x).to(device)
    y = torch.from_numpy(tr_y).to(device)
    n = len(y)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * max(1, n // bs))
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n - bs + 1, bs):
            idx = perm[s:s + bs]
            logits = model(x[idx])
            loss = F.cross_entropy(logits, y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return model


@torch.no_grad()
def evaluate(model, te_x, te_y, device, bs=4096):
    x = torch.from_numpy(te_x).to(device)
    y = torch.from_numpy(te_y).to(device)
    top1 = top5 = 0
    per_class_hit = {}
    per_class_tot = {}
    for s in range(0, len(y), bs):
        logits = model(x[s:s + bs])
        k = min(5, logits.shape[1])
        idx = logits.topk(k, dim=1).indices
        hit1 = (idx[:, 0] == y[s:s + bs])
        hit5 = (idx == y[s:s + bs].unsqueeze(1)).any(dim=1)
        top1 += hit1.sum().item()
        top5 += hit5.sum().item()
        for lab, h in zip(y[s:s + bs].tolist(), hit1.tolist()):
            per_class_tot[lab] = per_class_tot.get(lab, 0) + 1
            per_class_hit[lab] = per_class_hit.get(lab, 0) + int(h)
    acc1 = top1 / len(y) * 100
    acc5 = top5 / len(y) * 100
    per_acc = {c: per_class_hit[c] / per_class_tot[c] * 100 for c in per_class_tot}
    return acc1, acc5, per_acc, per_class_tot


def subsample_k_per_class(feats, labels, k, rng):
    """每类取 K 张；返回样本索引与该 K 下可用的类别集合。"""
    idx_all = []
    used = []
    for c in np.unique(labels):
        pos = np.where(labels == c)[0]
        if len(pos) < k:
            continue
        idx_all.append(rng.choice(pos, size=k, replace=False))
        used.append(c)
    idx = np.concatenate(idx_all)
    return feats[idx], labels[idx], set(used)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda"
    FIG.mkdir(parents=True, exist_ok=True)

    tr_x, tr_y = load("train")
    te_x, te_y = load("test")
    classes = np.unique(np.concatenate([tr_y, te_y]))
    print(f"train {tr_x.shape} test {te_x.shape} classes {len(classes)}", flush=True)

    # 保证类别索引从 0 连续（本数据集 sku 已是 0..9690，但仍做一次映射以防未来变化）
    remap = {c: i for i, c in enumerate(classes)}
    tr_y_m = np.array([remap[c] for c in tr_y])
    te_y_m = np.array([remap[c] for c in te_y])
    n_classes = len(classes)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = train_probe(tr_x, tr_y_m, n_classes, device, epochs=args.epochs, seed=args.seed)
    train_sec = time.time() - t0
    acc1, acc5, per_acc, per_tot = evaluate(model, te_x, te_y_m, device)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"[full] top1={acc1:.2f} top5={acc5:.2f} train={train_sec:.0f}s vram={peak:.2f}GB", flush=True)

    # 长尾分析：按训练集每类样本数分 5 组
    tr_counts = {c: int((tr_y_m == c).sum()) for c in range(n_classes)}
    class_sizes = np.array([tr_counts[c] for c in range(n_classes)])
    quint = np.quantile(class_sizes, [0.2, 0.4, 0.6, 0.8])
    groups = {"Q1(最少20%)": [], "Q2": [], "Q3": [], "Q4": [], "Q5(最多20%)": []}
    names = list(groups.keys())
    for c in range(n_classes):
        gi = int(np.searchsorted(quint, class_sizes[c], side="right"))
        groups[names[gi]].append(per_acc.get(c, 0.0))
    tail = {k: {"n_classes": len(v), "mean_acc": round(float(np.mean(v)), 2)} for k, v in groups.items()}
    print("长尾分组准确率:", json.dumps(tail, ensure_ascii=False), flush=True)

    # 少样本曲线：只在该 K 下可用（样本数 >= K）的类子集上训练与评测，保证可比
    rng = np.random.default_rng(args.seed)
    few = {}
    for k in (1, 5, 10, 20):
        fx, fy, used = subsample_k_per_class(tr_x, tr_y_m, k, rng)
        mask = np.isin(te_y_m, list(used))
        ex, ey = te_x[mask], te_y_m[mask]
        if len(used) < 2:
            continue
        m = train_probe(fx, fy, n_classes, device, epochs=20, seed=args.seed)
        a1, a5, _, _ = evaluate(m, ex, ey, device)
        few[k] = {"train_samples": int(len(fy)), "n_classes": len(used),
                  "test_samples": int(len(ey)), "top1": round(a1, 2), "top5": round(a5, 2)}
        print(f"  K={k}: classes={len(used)} train={len(fy)} test={len(ey)} "
              f"top1={a1:.2f} top5={a5:.2f}", flush=True)

    summary = {
        "dataset": "Products-10K (nyris/products-10k-v1)",
        "n_train": int(len(tr_y_m)), "n_test": int(len(te_y_m)), "n_classes": n_classes,
        "method": "frozen Chinese-CLIP ViT-B/16 features + linear probe",
        "full": {"top1": round(acc1, 2), "top5": round(acc5, 2),
                 "train_sec": round(train_sec, 1), "peak_vram_gb": round(peak, 2),
                 "trainable_params": int(n_classes * tr_x.shape[1] + n_classes)},
        "long_tail": tail,
        "few_shot": few,
        "note": "无类目名称 → 无法做零样本分类，本表仅反映特征可分性与可训练性",
    }
    with open(RESULT / "p10k_classify.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("saved:", RESULT / "p10k_classify.json")

    # 图：长尾散点 + 少样本曲线
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    # 左：长尾分组柱状图（比散点更直观，避免每类测试样本少导致的离散）
    labels_g = names
    vals = [tail[k]["mean_acc"] for k in labels_g]
    ns = [tail[k]["n_classes"] for k in labels_g]
    bars = axes[0].bar(range(len(labels_g)), vals, color="#4C72B0")
    for i, (v, n) in enumerate(zip(vals, ns)):
        axes[0].text(i, v + 1.5, f"{v:.1f}%\n(n={n})", ha="center", fontsize=8)
    axes[0].set_xticks(range(len(labels_g)))
    axes[0].set_xticklabels(["最少20%", "Q2", "Q3", "Q4", "最多20%"], fontsize=9)
    axes[0].set_ylabel("该类测试准确率 (%)")
    axes[0].set_ylim(0, max(vals) * 1.25 + 2)
    axes[0].set_title(f"长尾：按训练样本数分组（总体 top-1 = {acc1:.1f}%）")
    axes[0].grid(axis="y", alpha=0.3)

    # 右：少样本曲线
    ks = sorted(few.keys())
    axes[1].plot(ks, [few[k]["top1"] for k in ks], "o-", color="#E8712C", label="top-1")
    axes[1].plot(ks, [few[k]["top5"] for k in ks], "s--", color="#4C72B0", label="top-5")
    axes[1].set_xticks(ks)
    axes[1].set_xticklabels([f"K={k}" for k in ks])
    axes[1].set_xlabel("每类训练样本数 K（仅统计可用类）")
    axes[1].set_ylabel("测试准确率 (%)")
    axes[1].set_title("少样本曲线（类子集随 K 变化）")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "p10k_classify.png", dpi=150)
    print("figure saved:", FIG / "p10k_classify.png")


if __name__ == "__main__":
    main()
