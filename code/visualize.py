"""可视化：成功案例 / 失败案例 + 正样本排名分布（对应 Level 3「可视化展示」）。

输出:
  results/figs/success_cases.png   6 个检索正确案例（GT 排第 1）
  results/figs/failure_cases.png   6 个失败案例（GT 排名 > 20），用于局限性分析
  results/figs/rank_hist.png       5,008 条查询的正样本排名分布
  results/rank_analysis.json       定量结论（Top1/Top5/Top10 命中、中位排名等）
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from muge_eval import ROOT, DATA, RESULT, as_tensor, load_texts_jsonl
from retrieval_eval import encode_images, encode_texts

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

FIG = RESULT / "figs"
FIG.mkdir(parents=True, exist_ok=True)


def main() -> None:
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor
    from peft import PeftModel

    device = "cuda"
    dtype = torch.float16
    model_dir = str(ROOT / "models" / "chinese-clip-vit-b16")

    img_ids, img_b64 = [], []
    with open(DATA / "valid_imgs.tsv", encoding="utf-8") as f:
        for line in f:
            iid, b64 = line.rstrip("\n").split("\t", 1)
            img_ids.append(int(iid))
            img_b64.append(b64)
    txt_ids, texts, gt = load_texts_jsonl(DATA / "valid_texts.jsonl")
    print(f"{len(img_ids)} imgs / {len(texts)} texts", flush=True)

    model = ChineseCLIPModel.from_pretrained(model_dir, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, str(RESULT / "lora_1200step_ckpt")).to(device).eval()
    processor = ChineseCLIPProcessor.from_pretrained(model_dir)

    img_f = encode_images(model, img_ids, img_b64, device, 128, dtype)
    txt_f = encode_texts(model, processor, texts, device)

    img_pos = {i: k for k, i in enumerate(img_ids)}
    ranks, hits = [], {1: 0, 5: 0, 10: 0}
    best_idx = []
    sims_all = []
    for s in range(0, len(txt_f), 512):
        sim = txt_f[s:s + 512] @ img_f.T
        sims_all.append(sim.cpu())
    sims = torch.cat(sims_all, dim=0)

    for i, g in enumerate(gt):
        cand = [img_pos[x] for x in g if x in img_pos]
        if not cand:
            ranks.append(None)
            continue
        order = torch.argsort(sims[i], descending=True)
        pos = {int(c): r for r, c in enumerate(order[:200].tolist())}
        r = min(pos.get(c, 9999) for c in cand)
        ranks.append(r)
        for k in hits:
            if r < k:
                hits[k] += 1
        best_idx.append((i, cand[0], r))

    valid = [r for r in ranks if r is not None]
    stats = {
        "n_queries": len(valid),
        "R@1": round(hits[1] / len(valid) * 100, 2),
        "R@5": round(hits[5] / len(valid) * 100, 2),
        "R@10": round(hits[10] / len(valid) * 100, 2),
        "median_rank": float(np.median(valid)),
        "mean_rank": round(float(np.mean([min(r, 200) for r in valid])), 1),
        "frac_rank_gt20": round(sum(1 for r in valid if r > 20) / len(valid) * 100, 2),
    }
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    with open(RESULT / "rank_analysis.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 排名分布直方图（截断到 200）
    plt.figure(figsize=(8, 4.5))
    plt.hist([min(r, 200) for r in valid], bins=50, color="#4C72B0")
    plt.xlabel("正样本图像排名（>200 截断为 200）")
    plt.ylabel("查询数量")
    plt.title(f"MUGE valid 正样本排名分布（LoRA） 中位排名={stats['median_rank']:.0f}")
    plt.tight_layout()
    plt.savefig(FIG / "rank_hist.png", dpi=150)
    plt.close()

    def draw(cases, title, fname):
        n = len(cases)
        fig, axes = plt.subplots(n, 2, figsize=(9, 2.6 * n))
        for row, (qi, gt_i, rank) in enumerate(cases):
            raw = base64.urlsafe_b64decode(img_b64[gt_i]) if "-" in img_b64[gt_i] or "_" in img_b64[gt_i] else base64.b64decode(img_b64[gt_i])
            ax = axes[row, 0]
            ax.imshow(Image.open(io.BytesIO(raw)))
            ax.axis("off")
            ax.set_title(f"真实匹配图片 (id={img_ids[gt_i]})", fontsize=9)
            # 模型 top1
            top1 = int(torch.argmax(sims[qi]))
            raw2 = base64.urlsafe_b64decode(img_b64[top1]) if "-" in img_b64[top1] or "_" in img_b64[top1] else base64.b64decode(img_b64[top1])
            ax2 = axes[row, 1]
            ax2.imshow(Image.open(io.BytesIO(raw2)))
            ax2.axis("off")
            ax2.set_title(f"模型 Top1 (id={img_ids[top1]})", fontsize=9)
            fig.text(0.02, 1 - (row + 0.5) / n, f"查询：{texts[qi]}\nGT 排名：{rank}",
                     fontsize=9, va="center")
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=[0.18, 0, 1, 0.97])
        fig.savefig(FIG / fname, dpi=130)
        plt.close()

    succ = [(i, g, r) for i, g, r in best_idx if r == 0][:6]
    fail = [(i, g, r) for i, g, r in best_idx if r > 20][:6]
    if succ:
        draw(succ, "成功案例：查询文本与 Top1 检索结果", "success_cases.png")
    if fail:
        draw(fail, "失败案例：正样本未能进入前 20 名", "failure_cases.png")
    print("figures saved to", FIG)


if __name__ == "__main__":
    main()
