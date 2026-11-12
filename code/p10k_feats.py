"""Products-10K 冻结特征提取：用 Chinese-CLIP 图像塔把 197,307 张商品图编码成 512 维向量。

为什么要缓存：分类实验的核心是「线性探针」，主干冻结时特征只需算一次；
之后 9,691 类分类、长尾分析、少样本曲线（每类 1/5/10/50 张）都只在小矩阵上做，
单次实验从小时级降到秒级。

数据: data/raw/products10k_v1/data/*.parquet
  字段 image(bytes) / img_file(相对路径) / sku(SKU 编号字符串)

用法:
  python code/p10k_feats.py --limit-shards 1      # 小规模测试（约 5,000 张）
  python code/p10k_feats.py                       # 全量（约 10 分钟）
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import ChineseCLIPModel

from muge_eval import ROOT, RESULT, as_tensor, official_transform

DATA_DIR = ROOT / "data" / "raw" / "products10k_v1" / "data"
OUT_DIR = RESULT / "feats"


def shard_files(split: str):
    if split == "train":
        return sorted(DATA_DIR.glob("train-*.parquet"))
    return sorted(DATA_DIR.glob("test-*.parquet"))


@torch.no_grad()
def encode_split(model, transform, files, device, dtype, batch_size=128,
                 limit_shards=None, log_every=20, workers=8):
    """8 线程并行解码（PIL 解码时释放 GIL），GPU 侧只做前向，避免 CPU 成为瓶颈。"""
    from concurrent.futures import ThreadPoolExecutor

    def decode(item):
        im, sk = item
        try:
            return transform(Image.open(io.BytesIO(im["bytes"]))), int(sk)
        except Exception:
            return None

    feats, labels = [], []
    done = 0
    t0 = time.time()
    if limit_shards:
        files = files[:limit_shards]
    pool = ThreadPoolExecutor(workers)
    for fi, path in enumerate(files):
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256, columns=["image", "sku"]):
            items = list(zip(batch.column("image").to_pylist(), batch.column("sku").to_pylist()))
            decoded = [d for d in pool.map(decode, items) if d is not None]
            tensors = [d[0] for d in decoded]
            labs = [d[1] for d in decoded]
            for s in range(0, len(tensors), batch_size):
                px = torch.stack(tensors[s:s + batch_size]).to(device, dtype=dtype)
                f = as_tensor(model.get_image_features(pixel_values=px))
                feats.append((f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy())
                labels.extend(labs[s:s + batch_size])
                done += len(px)
            if done % (log_every * 256) < 256:
                el = time.time() - t0
                print(f"  [{fi+1}/{len(files)}] {done} imgs  {done/max(el,1e-9):.1f} img/s  {el:.0f}s",
                      flush=True)
    pool.shutdown()
    el = time.time() - t0
    print(f"done: {done} imgs in {el:.0f}s ({done/max(el,1e-9):.1f} img/s)", flush=True)
    return np.concatenate(feats, axis=0), np.array(labels, dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-shards", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--tag", default=None, help="输出文件名后缀，小规模测试用 smoke")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = ChineseCLIPModel.from_pretrained(
        str(ROOT / "models" / "chinese-clip-vit-b16"), torch_dtype=dtype).to(device).eval()
    transform = official_transform(224)

    for split in ("train", "test"):
        files = shard_files(split)
        print(f"[{split}] {len(files)} shards", flush=True)
        feats, labels = encode_split(model, transform, files, device, dtype,
                                     args.batch_size, args.limit_shards)
        tag = args.tag or ""
        out = OUT_DIR / f"p10k_{split}{tag}.npz"
        np.savez(out, feats=feats.astype(np.float16), labels=labels)
        print(f"saved {out}  feats={feats.shape} labels={labels.shape} "
              f"classes={len(np.unique(labels))}", flush=True)

    print("peak VRAM %.2f GB" % (torch.cuda.max_memory_allocated() / 1024 ** 3))


if __name__ == "__main__":
    main()
