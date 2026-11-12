"""LoRA 部分参数微调：唯一能在 8GB 显存上端到端更新主干的方法。

与 train_adapter.py 的区别:
  - Adapter/线性探针: 主干冻结，只在已缓存特征上做映射 → T2I 无增益（已实测）
  - LoRA: 在注意力层的 q/k/v/out 上注入低秩增量，主干参与前向，视觉表征可被真正调整

数据: 直接用官方预处理好的 LMDB
  lmdb/train/pairs  -> (image_id, text_id, text)  250,315 条
  lmdb/train/imgs   -> image_id -> base64(jpeg)

用法:
  python code/train_lora.py --max-steps 300 --bs 32          # 冒烟
  python code/train_lora.py --max-steps 3000 --bs 32 --eval-every 1000
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

from muge_eval import ROOT, DATA, RESULT, as_tensor, official_transform
from train_adapter import eval_retrieval

LMDB_DIR = DATA / "lmdb"


class PairData:
    """按索引随机读取 (image, text)；每个进程各自打开 LMDB（lmdb 对象不能跨进程 pickle）。"""

    def __init__(self, split: str, transform):
        self.split = split
        self.transform = transform
        self.pairs_dir = str(LMDB_DIR / split / "pairs")
        self.imgs_dir = str(LMDB_DIR / split / "imgs")
        self.n = lmdb.open(self.pairs_dir, readonly=True, lock=False).stat()["entries"]
        self._env_pairs = None
        self._env_imgs = None
        self._img_cache = None  # 复用同一进程内的环境

    def _open(self):
        if self._env_pairs is None:
            self._env_pairs = lmdb.open(self.pairs_dir, readonly=True, lock=False,
                                        readahead=False, meminit=False)
            self._env_imgs = lmdb.open(self.imgs_dir, readonly=True, lock=False,
                                       readahead=False, meminit=False)

    def __len__(self):
        return self.n

    def get(self, idx: int):
        self._open()
        import pickle
        with self._env_pairs.begin() as txn:
            img_id, text_id, text = pickle.loads(txn.get(str(idx).encode()))
        with self._env_imgs.begin() as txn:
            b64 = txn.get(str(img_id).encode()).decode()
        raw = base64.urlsafe_b64decode(b64) if ("-" in b64 or "_" in b64) else base64.b64decode(b64)
        img = self.transform(Image.open(io.BytesIO(raw)))
        return img, text.lower()


def build_batch(data: PairData, idxs, processor, device, max_len=24):
    imgs, texts = [], []
    for i in idxs:
        im, tx = data.get(int(i))
        imgs.append(im)
        texts.append(tx)
    px = torch.stack(imgs).to(device)
    tin = processor(text=texts, return_tensors="pt", padding=True, truncation=True,
                    max_length=max_len)
    tin = {k: v.to(device) for k, v in tin.items()}
    return px, tin


@torch.no_grad()
def eval_model(model, processor, device, img_ids, img_b64_iter, txt_ids, texts, pairs,
               batch_size=128, dtype=torch.float16):
    """在 valid 上完整评测：编码全部图像与文本后算 Recall。"""
    t0 = time.time()
    transform = official_transform(224)
    feats = []
    buf, b_ids = [], []
    for iid, b64 in img_b64_iter:
        raw = base64.urlsafe_b64decode(b64) if ("-" in b64 or "_" in b64) else base64.b64decode(b64)
        buf.append(transform(Image.open(io.BytesIO(raw))))
        b_ids.append(iid)
        if len(buf) == batch_size:
            px = torch.stack(buf).to(device, dtype=dtype)
            f = as_tensor(model.get_image_features(pixel_values=px))
            feats.append((f / f.norm(dim=-1, keepdim=True)).float())
            buf, b_ids = [], []
    if buf:
        px = torch.stack(buf).to(device, dtype=dtype)
        f = as_tensor(model.get_image_features(pixel_values=px))
        feats.append((f / f.norm(dim=-1, keepdim=True)).float())
    img_f = torch.cat(feats, dim=0)
    print(f"  eval: encoded {img_f.shape[0]} imgs in {time.time()-t0:.0f}s", flush=True)

    tfeats = []
    for s in range(0, len(texts), 512):
        inp = processor(text=texts[s:s + 512], return_tensors="pt", padding=True,
                        truncation=True, max_length=24)
        inp = {k: v.to(device) for k, v in inp.items()}
        f = as_tensor(model.get_text_features(**inp))
        tfeats.append((f / f.norm(dim=-1, keepdim=True)).float())
    txt_f = torch.cat(tfeats, dim=0)

    import torch.nn as nn
    ident = nn.Identity().to(device)
    t2i, i2t = eval_retrieval(ident, ident, img_f, txt_f, img_ids, txt_ids, pairs)
    return t2i, i2t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=0)
    ap.add_argument("--out", default="lora")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    print(f"LoRA r={args.r} alpha={args.alpha} bs={args.bs} accum={args.accum} "
          f"lr={args.lr} max_steps={args.max_steps}", flush=True)

    from peft import LoraConfig, get_peft_model

    base = ChineseCLIPModel.from_pretrained(str(ROOT / "models" / "chinese-clip-vit-b16"))
    cfg = LoraConfig(
        r=args.r, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "query", "key", "value"],
    )
    model = get_peft_model(base, cfg).to(device)
    model.print_trainable_parameters()
    processor = ChineseCLIPProcessor.from_pretrained(str(ROOT / "models" / "chinese-clip-vit-b16"))

    train = PairData("train", official_transform(224))
    print(f"train pairs: {len(train)}", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.max_steps)
    scaler = torch.amp.GradScaler("cuda")

    # valid 评测素材（只读一次）
    va = np.load(RESULT / "feats" / "valid_imgs.npz")
    va_img_ids = va["ids"].tolist()
    va_b64 = None
    va_pairs = [json.loads(l) for l in open(RESULT / "feats" / "valid_pairs.jsonl", encoding="utf-8")]
    va_txt_ids, va_texts = [], []
    with open(DATA / "valid_texts.jsonl", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            va_txt_ids.append(int(o["text_id"]))
            va_texts.append(o["text"].lower())

    def valid_img_iter():
        env = lmdb.open(str(LMDB_DIR / "valid" / "imgs"), readonly=True, lock=False,
                        readahead=False, meminit=False)
        with env.begin() as txn:
            for iid in va_img_ids:
                yield iid, txn.get(str(iid).encode()).decode()
        env.close()

    model.train()
    t0 = time.time()
    step = 0
    opt.zero_grad(set_to_none=True)
    while step < args.max_steps:
        idxs = np.random.randint(0, len(train), size=args.bs)
        px, tin = build_batch(train, idxs, processor, device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            ie = as_tensor(model.get_image_features(pixel_values=px))
            te = as_tensor(model.get_text_features(**tin))
            ie = F.normalize(ie.float(), dim=-1)
            te = F.normalize(te.float(), dim=-1)
            logits = ie @ te.T * 100.0
            labels = torch.arange(ie.shape[0], device=device)
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        scaler.scale(loss / args.accum).backward()
        if (step + 1) % args.accum == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()  # 必须在 optimizer.step() 之后，否则会跳掉首个学习率
        step += 1
        if step % 50 == 0 or step == 1:
            el = time.time() - t0
            print(f"  step {step}/{args.max_steps} loss={loss.item():.4f} "
                  f"vram={torch.cuda.max_memory_allocated()/1024**3:.2f}GB "
                  f"{step/max(el,1e-9):.2f} it/s", flush=True)
        if args.eval_every and step % args.eval_every == 0:
            model.eval()
            t2i, i2t = eval_model(model, processor, device, va_img_ids, valid_img_iter(),
                                  va_txt_ids, va_texts, va_pairs)
            print(f"  [eval@{step}] T2I R@1={t2i[1]:.2f} R@5={t2i[5]:.2f} MR={sum(t2i.values())/3:.2f} "
                  f"| I2T R@1={i2t[1]:.2f} MR={sum(i2t.values())/3:.2f}", flush=True)
            model.train()

    model.eval()
    t2i, i2t = eval_model(model, processor, device, va_img_ids, valid_img_iter(),
                          va_txt_ids, va_texts, va_pairs)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    total = time.time() - t0
    print(f"\n[LoRA final] T2I R@1={t2i[1]:.2f} R@5={t2i[5]:.2f} R@10={t2i[10]:.2f} "
          f"MR={sum(t2i.values())/3:.2f}")
    print(f"[LoRA final] I2T R@1={i2t[1]:.2f} R@5={i2t[5]:.2f} R@10={i2t[10]:.2f} "
          f"MR={sum(i2t.values())/3:.2f}")
    print(f"total {total:.0f}s | peak VRAM {peak:.2f} GB")

    summary = {
        "method": "lora", "r": args.r, "alpha": args.alpha, "bs": args.bs,
        "lr": args.lr, "max_steps": args.max_steps, "seed": args.seed,
        "t2i": {f"R@{k}": round(v, 2) for k, v in t2i.items()},
        "t2i_MR": round(sum(t2i.values()) / 3, 2),
        "i2t": {f"R@{k}": round(v, 2) for k, v in i2t.items()},
        "i2t_MR": round(sum(i2t.values()) / 3, 2),
        "train_sec": round(total, 1), "peak_vram_gb": round(peak, 2),
    }
    RESULT.mkdir(parents=True, exist_ok=True)
    with open(RESULT / f"{args.out}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    model.save_pretrained(str(RESULT / f"{args.out}_ckpt"))
    print("saved:", RESULT / f"{args.out}.json")


if __name__ == "__main__":
    main()
