"""交互式商品图文检索演示：双击 run_demo.bat 后，自己输入查询词。

用法:
  1) 双击 run_demo.bat（或运行 run_demo.bat）→ 进入交互模式，逐条输入查询词
  2) 命令行直传: python code/demo_search.py --query "手机壳"
  3) 多条查询: python code/demo_search.py --query "手机壳" "圣诞抱枕"
"""
from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from muge_eval import ROOT, DATA, RESULT, as_tensor

FEATS = RESULT / "feats"
OUT = RESULT / "demo"


# ---------------- 数据与特征 ----------------

def load_valid_images(limit: int | None = None):
    ids, b64s = [], []
    with open(DATA / "valid_imgs.tsv", encoding="utf-8") as f:
        for line in f:
            iid, b64 = line.rstrip("\n").split("\t", 1)
            ids.append(int(iid))
            b64s.append(b64)
            if limit and len(ids) >= limit:
                break
    return ids, b64s


@torch.no_grad()
def encode_images_lora(model, ids, b64s, device, batch=128):
    from muge_eval import official_transform
    tf = official_transform(224)
    feats = []
    for s in range(0, len(b64s), batch):
        tensors = []
        for b64 in b64s[s:s + batch]:
            raw = base64.urlsafe_b64decode(b64) if ("-" in b64 or "_" in b64) else base64.b64decode(b64)
            tensors.append(tf(Image.open(io.BytesIO(raw))))
        px = torch.stack(tensors).to(device, dtype=torch.float16)
        f = as_tensor(model.get_image_features(pixel_values=px))
        feats.append((f / f.norm(dim=-1, keepdim=True)).float())
    return torch.cat(feats, dim=0)


def decode(b64: str) -> Image.Image:
    raw = base64.urlsafe_b64decode(b64) if ("-" in b64 or "_" in b64) else base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def build_context(args):
    """加载模型与图像特征，返回一个 run(queries) 函数。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在准备候选图（{args.images} 张）...", flush=True)
    img_ids, img_b64 = load_valid_images(args.images)

    from transformers import ChineseCLIPModel, ChineseCLIPProcessor
    model_dir = str(ROOT / "models" / "chinese-clip-vit-b16")
    model = ChineseCLIPModel.from_pretrained(model_dir, torch_dtype=torch.float16)
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora)
    model = model.to(device).eval()
    processor = ChineseCLIPProcessor.from_pretrained(model_dir)

    cache = RESULT / f"demo_img_feats_{args.images}{'_lora' if args.lora else ''}.npz"
    if cache.exists():
        d = np.load(cache)
        # 缓存是 fp16，取回必须转 fp32，否则和 float 查询做矩阵乘会 dtype 不匹配
        img_f = torch.from_numpy(d["feats"].astype(np.float32)).to(device)
        print(f"已加载缓存特征：{cache.name}", flush=True)
    else:
        if args.lora:
            img_f = encode_images_lora(model, img_ids, img_b64, device)
        else:
            d = np.load(FEATS / "valid_imgs.npz")
            keep = np.isin(d["ids"], np.array(img_ids))
            img_f = torch.from_numpy(d["feats"][keep].astype(np.float32)).to(device)
        np.savez(cache, feats=img_f.half().cpu().numpy())

    @torch.no_grad()
    def run(queries: list[str]):
        inp = processor(text=queries, return_tensors="pt", padding=True,
                        truncation=True, max_length=24).to(device)
        q = as_tensor(model.get_text_features(**inp))
        q = q / q.norm(dim=-1, keepdim=True)
        sim = q.float() @ img_f.T
        for qi, query in enumerate(queries):
            scores, idx = sim[qi].topk(args.topk)
            print(f"\n查询「{query}」的 Top-{args.topk}：")
            imgs = []
            for rank, (sc, i) in enumerate(zip(scores.tolist(), idx.tolist()), 1):
                imgs.append(decode(img_b64[i]).resize((224, 224)))
                print(f"  Top{rank}: image_id={img_ids[i]}  相似度={sc:.4f}")
            canvas = Image.new("RGB", (224 * len(imgs), 224), "white")
            for k, im in enumerate(imgs):
                canvas.paste(im, (224 * k, 0))
            safe = "".join(c for c in query if c.isalnum())[:12] or "query"
            path = OUT / f"demo_{safe}.png"
            canvas.save(path)
            print(f"  结果图已保存：{path}")

    return run


# ---------------- 主流程 ----------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", nargs="+", default=None,
                    help="不给该参数则进入交互模式，逐条输入查询词")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--lora", default=None, help="LoRA checkpoint 目录")
    ap.add_argument("--images", type=int, default=3000, help="候选图数量")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    run = build_context(args)

    # 命令行直传模式
    if args.query:
        run(args.query)
        print(f"\n结果图目录：{OUT}")
        return

    # 交互模式（双击 bat 走这里）
    print("=" * 64)
    print("商品图文检索演示 —— 输入中文商品词，回车开始检索")
    print("  示例：手机壳 / 圣诞抱枕 / 男t丝光棉长袖 / 亚克力展示盒")
    print("  直接回车（不输入内容）退出")
    print("=" * 64)
    count = 0
    while True:
        try:
            raw = input("\n请输入要搜索的商品词：")
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break
        query = raw.strip()
        if not query:
            print(f"已退出。本次共检索 {count} 条，结果图在：{OUT}")
            break
        try:
            run([query])
            count += 1
        except Exception as e:  # 单条查询失败不应退出整个程序
            print(f"  [出错] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
