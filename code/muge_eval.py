"""MUGE 中文电商图文检索评测（对齐 Chinese-CLIP 官方口径）。

数据格式（来自 OFA-Sys/chinese-clip-eval 的 MUGE.zip）:
  {split}_imgs.tsv   : image_id \t base64(jpeg)
  {split}_texts.jsonl : {"text_id": int, "text": str, "image_ids": [int, ...]}

官方口径要点（见 docs/ref/README.md 与 cn_clip/eval/data.py）:
  - 图片预处理为 Resize((224,224)) 直接拉伸（不是短边 resize + center crop）
  - 文本小写化后 tokenize，context_length=24
  - 指标: Text->Image 与 Image->Text 的 R@1/5/10 及平均召回 MR=(R1+R5+R10)/3
  - MUGE 官方基准使用 valid 划分（test 的 image_ids 为比赛保留，为空）

用法:
  python code/muge_eval.py --split valid                 # 全量
  python code/muge_eval.py --split valid --limit-imgs 300 --limit-texts 100   # 冒烟
"""
from __future__ import annotations

import argparse
import base64
import json
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "MUGE"
RESULT = ROOT / "results"

NORMALIZE = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))


def official_transform(resolution: int = 224) -> transforms.Compose:
    """官方 _build_transform: Resize 后直接转 RGB（不裁剪），保持与官方数字可比。"""
    return transforms.Compose([
        transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(*NORMALIZE),
    ])


def as_tensor(out):
    """transformers 5.x 的 get_*_features 返回 BaseModelOutputWithPooling。"""
    if isinstance(out, torch.Tensor):
        return out
    pooled = getattr(out, "pooler_output", None)
    if pooled is not None:
        return pooled
    raise TypeError(f"unsupported output type {type(out).__name__}")


def load_images_tsv(path: Path, limit: int | None = None):
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


def load_texts_jsonl(path: Path, limit: int | None = None):
    ids, texts, gt = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ids.append(int(obj["text_id"]))
            texts.append(obj["text"].lower())
            gt.append([int(x) for x in (obj.get("image_ids") or [])])
            if limit and len(ids) >= limit:
                break
    return ids, texts, gt


@torch.no_grad()
def encode_images(model, transform, ids, b64s, device, batch_size, dtype):
    feats = []
    t0 = time.time()
    for start in range(0, len(b64s), batch_size):
        chunk = b64s[start:start + batch_size]
        tensors = []
        for s in chunk:
            raw = base64.urlsafe_b64decode(s) if "-" in s or "_" in s else base64.b64decode(s)
            img = Image.open(BytesIO(raw))
            tensors.append(transform(img))
        pixel_values = torch.stack(tensors).to(device, dtype=dtype)
        feat = as_tensor(model.get_image_features(pixel_values=pixel_values))
        feat = feat / feat.norm(dim=-1, keepdim=True)
        feats.append(feat.float().cpu().numpy())
        done = min(start + batch_size, len(b64s))
        elapsed = time.time() - t0
        print(f"  imgs {done}/{len(b64s)}  {done/elapsed:.1f} img/s  elapsed {elapsed:.0f}s", flush=True)
    return np.concatenate(feats, axis=0)


@torch.no_grad()
def encode_texts(model, processor, texts, device, batch_size, dtype, max_len=24):
    feats = []
    t0 = time.time()
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        inputs = processor(text=chunk, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_len)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        feat = as_tensor(model.get_text_features(**inputs))
        feat = feat / feat.norm(dim=-1, keepdim=True)
        feats.append(feat.float().cpu().numpy())
    elapsed = time.time() - t0
    print(f"  texts {len(texts)} in {elapsed:.1f}s ({len(texts)/elapsed:.1f} txt/s)", flush=True)
    return np.concatenate(feats, axis=0)


def recall_text_to_image(text_feats, img_feats, img_ids, gt_lists, ks=(1, 5, 10)):
    img_index = {iid: i for i, iid in enumerate(img_ids)}
    gt_sets = [set(gt_lists[i]) & set(img_ids) for i in range(len(gt_lists))]
    hits = {k: 0 for k in ks}
    valid_q = 0
    chunk = 512
    for start in range(0, len(text_feats), chunk):
        sim = text_feats[start:start + chunk] @ img_feats.T          # [b, N]
        topk = np.argpartition(-sim, kth=max(ks) - 1, axis=1)[:, :max(ks)]
        # 按分数排序 topk 内部
        rows = np.arange(sim.shape[0])[:, None]
        order = np.argsort(-sim[rows, topk], axis=1)
        topk = topk[rows, order]
        for bi in range(sim.shape[0]):
            qi = start + bi
            if not gt_sets[qi]:
                continue
            valid_q += 1
            preds = [img_ids[j] for j in topk[bi]]
            for k in ks:
                if any(p in gt_sets[qi] for p in preds[:k]):
                    hits[k] += 1
    return {k: hits[k] / valid_q * 100 for k in ks}, valid_q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid")
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "chinese-clip-vit-b16"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--text-batch-size", type=int, default=256)
    ap.add_argument("--limit-imgs", type=int, default=None)
    ap.add_argument("--limit-texts", type=int, default=None)
    ap.add_argument("--feat-dir", default=str(RESULT / "feats"))
    ap.add_argument("--save-feats", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"split={args.split} device={device} dtype={dtype}")
    img_path = DATA / f"{args.split}_imgs.tsv"
    txt_path = DATA / f"{args.split}_texts.jsonl"

    t0 = time.time()
    img_ids, img_b64 = load_images_tsv(img_path, args.limit_imgs)
    txt_ids, texts, gt = load_texts_jsonl(txt_path, args.limit_texts)
    print(f"loaded {len(img_ids)} images / {len(texts)} texts in {time.time()-t0:.1f}s")

    print("loading model ...")
    model = ChineseCLIPModel.from_pretrained(args.model_dir, torch_dtype=dtype).to(device).eval()
    processor = ChineseCLIPProcessor.from_pretrained(args.model_dir)

    torch.cuda.reset_peak_memory_stats()
    print("encoding images ...")
    img_feats = encode_images(model, official_transform(224), img_ids, img_b64, device,
                              args.batch_size, dtype)
    print("encoding texts ...")
    text_feats = encode_texts(model, processor, texts, device, args.text_batch_size, dtype)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"peak VRAM {peak:.2f} GB | img_feats {img_feats.shape} text_feats {text_feats.shape}")

    t2i, n_valid_q = recall_text_to_image(text_feats, img_feats, img_ids, gt)
    print(f"\n[T2I] valid queries (有 GT): {n_valid_q}")
    print(f"  R@1={t2i[1]:.1f}  R@5={t2i[5]:.1f}  R@10={t2i[10]:.1f}  "
          f"MR={(t2i[1]+t2i[5]+t2i[10])/3:.1f}")

    # I2T
    gt_by_img = {}
    for ti, g in enumerate(gt):
        for iid in g:
            gt_by_img.setdefault(iid, set()).add(ti)
    hits = {1: 0, 5: 0, 10: 0}
    n_valid_i = 0
    for start in range(0, len(img_feats), 512):
        sim = img_feats[start:start + 512] @ text_feats.T
        topk = np.argpartition(-sim, kth=9, axis=1)[:, :10]
        rows = np.arange(sim.shape[0])[:, None]
        topk = topk[rows, np.argsort(-sim[rows, topk], axis=1)]
        for bi in range(sim.shape[0]):
            iid = img_ids[start + bi]
            g = gt_by_img.get(iid)
            if not g:
                continue
            n_valid_i += 1
            for k in (1, 5, 10):
                if any(int(t) in g for t in topk[bi][:k]):
                    hits[k] += 1
    i2t = {k: hits[k] / n_valid_i * 100 for k in hits}
    print(f"[I2T] valid images (有 GT): {n_valid_i}")
    print(f"  R@1={i2t[1]:.1f}  R@5={i2t[5]:.1f}  R@10={i2t[10]:.1f}  "
          f"MR={(i2t[1]+i2t[5]+i2t[10])/3:.1f}")

    if args.save_feats:
        out = Path(args.feat_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / f"{args.split}_imgs.npz", ids=np.array(img_ids), feats=img_feats.astype(np.float16))
        np.savez(out / f"{args.split}_texts.npz", ids=np.array(txt_ids), feats=text_feats.astype(np.float16))
        print(f"saved feats to {out}")

    summary = {
        "split": args.split,
        "limit_imgs": args.limit_imgs,
        "limit_texts": args.limit_texts,
        "n_images": len(img_ids),
        "n_texts": len(texts),
        "t2i": {f"R@{k}": round(v, 2) for k, v in t2i.items()},
        "t2i_MR": round((t2i[1] + t2i[5] + t2i[10]) / 3, 2),
        "i2t": {f"R@{k}": round(v, 2) for k, v in i2t.items()},
        "i2t_MR": round((i2t[1] + i2t[5] + i2t[10]) / 3, 2),
        "peak_vram_gb": round(peak, 2),
    }
    RESULT.mkdir(parents=True, exist_ok=True)
    tag = f"muge_{args.split}" + ("_smoke" if (args.limit_imgs or args.limit_texts) else "")
    with open(RESULT / f"{tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nsummary:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
