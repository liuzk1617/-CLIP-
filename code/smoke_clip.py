"""Chinese-CLIP 冒烟测试：验证 5060 上 fp16 推理链路、显存占用与图文相似度。

用途：环境验收，不作实验结论。
"""
import os
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import ChineseCLIPModel, ChineseCLIPProcessor

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "chinese-clip-vit-b16"
IMG_DIR = ROOT / "作业要求"

TEXTS = [
    "一张投影幕布上的演示幻灯片",
    "教室里的投影屏幕，上面写着中文标题",
    "一张商品照片",
    "一份报纸的扫描件",
    "一张手机拍摄的会议现场照片",
    "深度学习模型的架构示意图",
    "一份数据集的统计表格",
    "一张人物肖像照片",
]


def _as_tensor(out):
    """transformers 5.x 的 get_*_features 返回 BaseModelOutputWithPooling，取 pooler_output。"""
    if isinstance(out, torch.Tensor):
        return out
    pooled = getattr(out, "pooler_output", None)
    if pooled is not None:
        return pooled
    raise TypeError(f"无法从 {type(out).__name__} 取特征")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    t0 = time.time()
    model = ChineseCLIPModel.from_pretrained(str(MODEL_DIR), torch_dtype=torch.float16).to(device).eval()
    processor = ChineseCLIPProcessor.from_pretrained(str(MODEL_DIR))
    print(f"model loaded in {time.time() - t0:.2f}s")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params={n_params:.1f}M")

    images = sorted(IMG_DIR.glob("*.jpg"))
    print(f"found {len(images)} images: {[p.name for p in images]}")
    imgs = [Image.open(p).convert("RGB") for p in images]

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        img_inputs = processor(images=imgs, return_tensors="pt").to(device)
        text_inputs = processor(text=TEXTS, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)

        img_feat = _as_tensor(model.get_image_features(**img_inputs))
        txt_feat = _as_tensor(model.get_text_features(**text_inputs))
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        sim = (img_feat.float() @ txt_feat.float().T)
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3

    print(f"\n{len(imgs)} imgs + {len(TEXTS)} texts encoded in {dt:.3f}s | peak VRAM {peak:.2f} GB")
    print(f"image feature dim={tuple(img_feat.shape)} text feature dim={tuple(txt_feat.shape)}")

    print("\n每张图最匹配的中文文本:")
    for i, p in enumerate(images):
        best = int(sim[i].argmax())
        print(f"  {p.name:8s} -> [{sim[i, best]:.3f}] {TEXTS[best]}")
    print("\nsim matrix range:", float(sim.min()), float(sim.max()))

    # 单图批量吞吐（用于估算正式实验预处理耗时）
    batch = imgs * 4  # 28 张
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        for k in range(0, len(batch), 16):
            x = processor(images=batch[k:k + 16], return_tensors="pt").to(device)
            f = model.get_image_features(**x)
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"\n吞吐: {len(batch)} imgs in {dt:.3f}s -> {len(batch)/dt:.1f} img/s, "
          f"peak VRAM {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print(f"推算: 10万张图特征提取约 {100000/(len(batch)/dt)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
