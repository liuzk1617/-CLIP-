"""通用中文图文检索评测器：支持 MUGE / Flickr30k-CN 与 LoRA 权重。

用途（对应 PPT 第 8 页"跨场景泛化"对比实验）:
  - MUGE 域内评测：验证基线复现
  - Flickr30k-CN 跨域评测：MUGE 上训练的 LoRA 直接迁移到开放域照片，看掉多少

数据格式（两个数据集一致）:
  {split}_imgs.tsv    image_id \t base64(jpeg)
  {split}_texts.jsonl {"text_id", "text", "image_ids": [...]}

用法:
  python code/retrieval_eval.py --data-dir data/raw/MUGE --split valid
  python code/retrieval_eval.py --data-dir data/raw/Flickr30k-CN --split test \
         --lora results/lora_3000step_ckpt
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

from muge_eval import ROOT, RESULT, as_tensor, official_transform, load_texts_jsonl
from train_adapter import eval_retrieval


def load_imgs_tsv(path: Path, limit=None):
    ids, b64s = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            iid, b64 = line.split("\t", 1)
            ids.append(int(iid))
            b64s.append(b64)
            if limit and len(ids) >= limit:
                break
    return ids, b64s


@torch.no_grad()
def encode_images(model, ids, b64s, device, batch_size=128, dtype=torch.float16):
    tf = official_transform(224)
    feats = []
    t0 = time.time()
    for s in range(0, len(b64s), batch_size):
        tensors = []
        for b64 in b64s[s:s + batch_size]:
            raw = base64.urlsafe_b64decode(b64) if ("-" in b64 or "_" in b64) else base64.b64decode(b64)
            tensors.append(tf(Image.open(io.BytesIO(raw))))
        px = torch.stack(tensors).to(device, dtype=dtype)
        f = as_tensor(model.get_image_features(pixel_values=px))
        feats.append((f / f.norm(dim=-1, keepdim=True)).float())
        if (s // batch_size) % 20 == 0:
            print(f"  imgs {min(s+batch_size, len(b64s))}/{len(b64s)} {time.time()-t0:.0f}s", flush=True)
    return torch.cat(feats, dim=0)


@torch.no_grad()
def encode_texts(model, processor, texts, device, batch_size=512, dtype=torch.float16, max_len=48):
    feats = []
    for s in range(0, len(texts), batch_size):
        inp = processor(text=texts[s:s + batch_size], return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len)
        inp = {k: v.to(device) for k, v in inp.items()}
        f = as_tensor(model.get_text_features(**inp))
        feats.append((f / f.norm(dim=-1, keepdim=True)).float())
    return torch.cat(feats, dim=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="valid")
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "chinese-clip-vit-b16"))
    ap.add_argument("--lora", default=None, help="LoRA checkpoint 目录")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--limit-imgs", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda"
    dtype = torch.float16
    data_dir = Path(args.data_dir)
    print(f"data={data_dir.name} split={args.split} lora={args.lora}", flush=True)

    img_ids, img_b64 = load_imgs_tsv(data_dir / f"{args.split}_imgs.tsv", args.limit_imgs)
    txt_ids, texts, gt = load_texts_jsonl(data_dir / f"{args.split}_texts.jsonl")
    pairs = [{"text_id": int(t), "image_ids": g} for t, g in zip(txt_ids, gt)]
    print(f"loaded {len(img_ids)} imgs / {len(texts)} texts", flush=True)

    model = ChineseCLIPModel.from_pretrained(args.model_dir, torch_dtype=dtype).to(device)
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora).to(device)
    model.eval()
    processor = ChineseCLIPProcessor.from_pretrained(args.model_dir)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    img_f = encode_images(model, img_ids, img_b64, device, args.batch_size, dtype)
    txt_f = encode_texts(model, processor, texts, device, 512, dtype)
    enc = time.time() - t0

    ident = nn.Identity().to(device)
    t2i, i2t = eval_retrieval(ident, ident, img_f, txt_f, img_ids, txt_ids, pairs)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"\n[T2I] R@1={t2i[1]:.2f} R@5={t2i[5]:.2f} R@10={t2i[10]:.2f} MR={sum(t2i.values())/3:.2f}")
    print(f"[I2T] R@1={i2t[1]:.2f} R@5={i2t[5]:.2f} R@10={i2t[10]:.2f} MR={sum(i2t.values())/3:.2f}")
    print(f"encode {enc:.0f}s | peak VRAM {peak:.2f} GB")

    summary = {
        "dataset": data_dir.name, "split": args.split, "lora": args.lora,
        "n_images": len(img_ids), "n_texts": len(texts),
        "t2i": {f"R@{k}": round(v, 2) for k, v in t2i.items()},
        "t2i_MR": round(sum(t2i.values()) / 3, 2),
        "i2t": {f"R@{k}": round(v, 2) for k, v in i2t.items()},
        "i2t_MR": round(sum(i2t.values()) / 3, 2),
        "encode_sec": round(enc, 1), "peak_vram_gb": round(peak, 2),
    }
    RESULT.mkdir(parents=True, exist_ok=True)
    tag = args.out or f"eval_{data_dir.name}_{args.split}" + ("_lora" if args.lora else "_base")
    with open(RESULT / f"{tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("saved:", RESULT / f"{tag}.json")


if __name__ == "__main__":
    main()
