"""把 MUGE 某个 split 的冻结 CLIP 特征缓存成 npz，供线性探针/Adapter 训练复用。

为什么先缓存：主干完全冻结时，特征只需算一次。之后训练 Adapter、扫样本量曲线
（1%/10%/100%）都只在小矩阵上做，单次实验从分钟级降到秒级——这直接决定
Level 3 的"不同训练样本量"实验能不能在答辩前扫完。

流式读取 tsv，避免把 1.27GB 的 base64 全塞进内存。

用法:
  python code/build_feats.py --split train            # 129k 图 / 248k 文本，约 7 分钟
  python code/build_feats.py --split valid            # 复用已有结果
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
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

from muge_eval import ROOT, DATA, RESULT, as_tensor, official_transform

CHUNK_IMGS_TSV = 1_000_000  # 每 1MB 文本块解析一次，控制内存


def iter_images_tsv(path: Path):
    """流式产出 (image_id, base64_str)。"""
    buf = ""
    with open(path, encoding="utf-8") as f:
        while True:
            chunk = f.read(CHUNK_IMGS_TSV)
            if not chunk:
                break
            buf += chunk
            lines = buf.split("\n")
            buf = lines.pop()
            for line in lines:
                if not line:
                    continue
                iid, b64 = line.split("\t", 1)
                yield int(iid), b64
        if buf:
            iid, b64 = buf.split("\t", 1)
            yield int(iid), b64


def _decode(b64: str) -> Image.Image:
    raw = base64.urlsafe_b64decode(b64) if ("-" in b64 or "_" in b64) else base64.b64decode(b64)
    return Image.open(io.BytesIO(raw))


@torch.no_grad()
def cache_images(model, transform, path, device, batch_size, dtype, max_images=None):
    ids, feats = [], []
    batch_t, batch_id = [], []
    t0 = time.time()
    n = 0
    for iid, b64 in iter_images_tsv(path):
        try:
            batch_t.append(transform(_decode(b64)))
            batch_id.append(iid)
        except Exception as e:  # 单张坏图不应该打断全量任务
            print(f"  skip image {iid}: {e}", flush=True)
        if len(batch_t) >= batch_size:
            px = torch.stack(batch_t).to(device, dtype=dtype)
            f = as_tensor(model.get_image_features(pixel_values=px))
            feats.append((f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy())
            ids.extend(batch_id)
            n += len(batch_t)
            batch_t, batch_id = [], []
            if n % (batch_size * 20) == 0:
                el = time.time() - t0
                print(f"  {n} imgs  {n/el:.1f} img/s  {el:.0f}s", flush=True)
        if max_images and n >= max_images:
            break
    if batch_t:
        px = torch.stack(batch_t).to(device, dtype=dtype)
        f = as_tensor(model.get_image_features(pixel_values=px))
        feats.append((f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy())
        ids.extend(batch_id)
        n += len(batch_t)
    el = time.time() - t0
    print(f"images done: {n} in {el:.1f}s ({n/max(el,1e-9):.1f} img/s)", flush=True)
    return np.array(ids), np.concatenate(feats, axis=0)


@torch.no_grad()
def cache_texts(model, processor, path, device, batch_size, dtype, max_len=24):
    ids, texts, gt = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            ids.append(int(o["text_id"]))
            texts.append(o["text"].lower())
            gt.append([int(x) for x in (o.get("image_ids") or [])])
    feats = []
    t0 = time.time()
    for s in range(0, len(texts), batch_size):
        inp = processor(text=texts[s:s + batch_size], return_tensors="pt",
                        padding=True, truncation=True, max_length=max_len)
        inp = {k: v.to(device) for k, v in inp.items()}
        f = as_tensor(model.get_text_features(**inp))
        feats.append((f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy())
    el = time.time() - t0
    print(f"texts done: {len(texts)} in {el:.1f}s", flush=True)
    return np.array(ids), np.concatenate(feats, axis=0), gt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "chinese-clip-vit-b16"))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out-dir", default=str(RESULT / "feats"))
    ap.add_argument("--max-images", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"split={args.split} device={device}", flush=True)
    model = ChineseCLIPModel.from_pretrained(args.model_dir, torch_dtype=dtype).to(device).eval()
    processor = ChineseCLIPProcessor.from_pretrained(args.model_dir)

    torch.cuda.reset_peak_memory_stats()
    img_ids, img_feats = cache_images(model, official_transform(224),
                                      DATA / f"{args.split}_imgs.tsv", device,
                                      args.batch_size, dtype, args.max_images)
    np.savez(out / f"{args.split}_imgs.npz", ids=img_ids, feats=img_feats.astype(np.float16))

    txt_ids, txt_feats, gt = cache_texts(model, processor, DATA / f"{args.split}_texts.jsonl",
                                         device, 512, dtype)
    with open(out / f"{args.split}_pairs.jsonl", "w", encoding="utf-8") as f:
        for tid, g in zip(txt_ids.tolist(), gt):
            f.write(json.dumps({"text_id": tid, "image_ids": g}, ensure_ascii=False) + "\n")
    np.savez(out / f"{args.split}_texts.npz", ids=txt_ids, feats=txt_feats.astype(np.float16))

    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"saved to {out} | peak VRAM {peak:.2f} GB", flush=True)
    print(f"  imgs {img_feats.shape} | texts {txt_feats.shape}", flush=True)


if __name__ == "__main__":
    main()
