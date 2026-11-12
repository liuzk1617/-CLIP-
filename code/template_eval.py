"""系统性提示模板优化：只改文本侧 prompt，图像特征复用缓存，秒级出结果。

动机：Chinese-CLIP 预训练时文本是电商标题式短句（"金丝绒木床头罩"），
检索查询同样是标题；把查询包装成不同句式，可能激活不同的语义对齐模式。
这是 PPT「改进建议」里的"系统性提示模板优化"，也是成本最低的可解释改进。

用法:
  python code/template_eval.py
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn

from muge_eval import ROOT, DATA, RESULT, as_tensor, load_texts_jsonl
from train_adapter import eval_retrieval

FEATS = RESULT / "feats"

TEMPLATES = {
    "raw": "{text}",
    "a_photo_of": "一张{text}的图片",
    "product_photo": "{text}的商品图片",
    "prefix_goods": "商品：{text}",
    "sentence": "这是一张{text}的图片。",
}


def main() -> None:
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    va = np.load(FEATS / "valid_imgs.npz")
    img_ids, img_f = va["ids"], torch.from_numpy(va["feats"].astype(np.float32)).to(device)
    txt_ids, texts, gt = load_texts_jsonl(DATA / "valid_texts.jsonl")
    pairs = [{"text_id": int(t), "image_ids": g} for t, g in zip(txt_ids, gt)]

    model = ChineseCLIPModel.from_pretrained(str(ROOT / "models" / "chinese-clip-vit-b16"),
                                             torch_dtype=dtype).to(device).eval()
    processor = ChineseCLIPProcessor.from_pretrained(str(ROOT / "models" / "chinese-clip-vit-b16"))

    ident = nn.Identity().to(device)
    results = {}
    feats_by_tpl = {}

    for name, tpl in TEMPLATES.items():
        prompt_texts = [tpl.format(text=t.lower()) for t in texts]
        feats = []
        with torch.no_grad():
            for s in range(0, len(prompt_texts), 512):
                inp = processor(text=prompt_texts[s:s + 512], return_tensors="pt",
                                padding=True, truncation=True, max_length=48)
                inp = {k: v.to(device) for k, v in inp.items()}
                f = as_tensor(model.get_text_features(**inp))
                feats.append((f / f.norm(dim=-1, keepdim=True)).float())
        txt_f = torch.cat(feats, dim=0)
        feats_by_tpl[name] = txt_f
        t2i, i2t = eval_retrieval(ident, ident, img_f, txt_f, img_ids, txt_ids, pairs)
        results[name] = {"t2i": t2i, "i2t": i2t}
        print(f"[{name:14s}] T2I R@1={t2i[1]:.2f} R@5={t2i[5]:.2f} R@10={t2i[10]:.2f} "
              f"MR={sum(t2i.values())/3:.2f} | I2T R@1={i2t[1]:.2f} MR={sum(i2t.values())/3:.2f}",
              flush=True)

    # 模板集成：多模板特征取平均后再归一化
    ens = torch.stack([feats_by_tpl[k] for k in TEMPLATES], dim=0).mean(dim=0)
    ens = ens / ens.norm(dim=-1, keepdim=True)
    t2i, i2t = eval_retrieval(ident, ident, img_f, ens, img_ids, txt_ids, pairs)
    results["ensemble"] = {"t2i": t2i, "i2t": i2t}
    print(f"[{'ensemble':14s}] T2I R@1={t2i[1]:.2f} R@5={t2i[5]:.2f} R@10={t2i[10]:.2f} "
          f"MR={sum(t2i.values())/3:.2f} | I2T R@1={i2t[1]:.2f} MR={sum(i2t.values())/3:.2f}")

    out = {k: {"t2i": {f"R@{kk}": round(vv, 2) for kk, vv in v["t2i"].items()},
               "t2i_MR": round(sum(v["t2i"].values()) / 3, 2),
               "i2t": {f"R@{kk}": round(vv, 2) for kk, vv in v["i2t"].items()},
               "i2t_MR": round(sum(v["i2t"].values()) / 3, 2)} for k, v in results.items()}
    with open(RESULT / "template_eval.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("saved:", RESULT / "template_eval.json")


if __name__ == "__main__":
    main()
