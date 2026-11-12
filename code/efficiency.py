"""效率分析：参数量、显存、推理时延对比（Base vs LoRA）。

对应 PPT 第 9 页 Level 3 的「效率分析」。所有数字在本机实测，可直接进报告表格。
"""
from __future__ import annotations

import json
import time

import torch
from PIL import Image

from muge_eval import ROOT, RESULT

SAMPLE_IMG = ROOT / "results" / "samples" / "p10k_0.jpg"
TEXTS = ["金丝绒木床头罩", "手机壳", "男t丝光棉长袖", "圣诞抱枕", "亚克力展示盒"]


def measure(model, processor, device, batch_sizes=(1, 8, 32), warmup=3, iters=10):
    px = processor(images=[Image.open(SAMPLE_IMG).convert("RGB")], return_tensors="pt")["pixel_values"]
    px = px.repeat(max(batch_sizes), 1, 1, 1).to(device)
    tin = processor(text=TEXTS, return_tensors="pt", padding=True, truncation=True, max_length=24)
    tin = {k: v.to(device) for k, v in tin.items()}
    out = {}
    torch.cuda.reset_peak_memory_stats()
    for bs in batch_sizes:
        with torch.no_grad():
            for _ in range(warmup):
                model.get_image_features(pixel_values=px[:bs])
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(iters):
                model.get_image_features(pixel_values=px[:bs])
            torch.cuda.synchronize()
            dt = (time.time() - t0) / iters
        out[f"img_bs{bs}_ms"] = round(dt * 1000, 2)
        out[f"img_bs{bs}_ips"] = round(bs / dt, 1)
    with torch.no_grad():
        for _ in range(warmup):
            model.get_text_features(**tin)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            model.get_text_features(**tin)
        torch.cuda.synchronize()
    out["text_bs5_ms"] = round((time.time() - t0) / iters * 1000, 2)
    out["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
    return out


def main() -> None:
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor

    device = "cuda"
    dtype = torch.float16
    model_dir = str(ROOT / "models" / "chinese-clip-vit-b16")
    processor = ChineseCLIPProcessor.from_pretrained(model_dir)

    results = {}
    # base
    base = ChineseCLIPModel.from_pretrained(model_dir, torch_dtype=dtype).to(device).eval()
    base_cfg = {"n_params": sum(p.numel() for p in base.parameters()),
                "trainable": sum(p.numel() for p in base.parameters() if p.requires_grad)}
    results["base"] = {**base_cfg, **measure(base, processor, device)}
    print("base:", json.dumps(results["base"], ensure_ascii=False), flush=True)
    del base
    torch.cuda.empty_cache()

    # lora
    from peft import PeftModel
    lora_dir = RESULT / "lora_1200step_ckpt"
    m = ChineseCLIPModel.from_pretrained(model_dir, torch_dtype=dtype)
    m = PeftModel.from_pretrained(m, str(lora_dir)).to(device).eval()
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in m.parameters())
    results["lora"] = {"n_params": n_all, "trainable": n_train,
                       "overhead_params": n_all - results["base"]["n_params"],
                       **measure(m, processor, device)}
    print("lora:", json.dumps(results["lora"], ensure_ascii=False), flush=True)

    # LoRA 可合并回主干：合并后参数量与结构等价于基线，无额外推理开销
    merged = m.merge_and_unload().eval()
    results["lora_merged"] = {"n_params": sum(p.numel() for p in merged.parameters()),
                              **measure(merged, processor, device)}
    print("lora_merged:", json.dumps(results["lora_merged"], ensure_ascii=False), flush=True)

    with open(RESULT / "efficiency.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved:", RESULT / "efficiency.json")


if __name__ == "__main__":
    main()
