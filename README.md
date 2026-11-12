# 基于中文 CLIP 的电商图文检索

人工智能导论期末大作业 · **方向五：多模态 AI 与 VLM**

在单张 8GB 显存的笔记本显卡（RTX 5060 Laptop）上，以 Chinese-CLIP ViT-B/16 为基础模型，
系统对比了四类轻量适配方法（线性探针 / 轻量 Adapter / 提示模板 / LoRA），
并在中文电商检索数据集 MUGE 的官方验证集上验证改进效果。

## 主要结果

| 方法 | 可训练参数 | T2I R@1 | I2T R@1 |
|---|---|---|---|
| Zero-shot 基线 | 0 | 51.98 | 38.69 |
| 线性探针（100%） | 0.52M | 50.10 | 38.60 |
| 轻量 Adapter（100%） | 0.26M | 51.40 | 40.14 |
| 提示模板优化 | 0 | 51.86 | 40.64 |
| **LoRA r=8（1200 步）** | **1.03M（0.55%）** | **54.71** | **43.61** |

- 三个随机种子：T2I R@1 = 55.20 ± 0.65，验证可复现
- 与官方公开数字复现误差 ≤0.12（MUGE）、≤0.40（Flickr30K-CN）
- 跨域（MUGE 训练 → Flickr30K-CN 测试）无灾难性遗忘，I2T R@1 +3.40
- LoRA 合并回主干后，参数量、时延与基线完全一致，部署零开销

**核心结论**：冻结特征上做事后映射（线性探针 / Adapter）对 T2I 无效，瓶颈在图像侧表征；
只有让主干参与前向的 LoRA 才能同时提升双向检索。

## 给同学：5 分钟了解这个项目

| 步骤 | 看什么 | 用时 |
|---|---|---|
| 1 | 上面这段「主要结果」表 —— 一句话：我们只调了 0.55% 的参数，就把检索准确率从 52.0 提到 54.7 | 1 分钟 |
| 2 | [`交付/技术报告.docx`](交付/技术报告.docx) —— 完整方法、实验、失败分析（10 页） | 5 分钟 |
| 3 | [`results/figs/`](results/figs/) —— 训练曲线、失败案例、长尾分布图 | 1 分钟 |
| 4 | 装好环境后双击 [`run_demo.bat`](run_demo.bat) —— 输入中文商品词，看真实检索结果 | 1 分钟 |

答辩/组会要看的三份材料都在 [`交付/`](交付/)：
技术报告、答辩 PPT、答辩准备手册（逐页讲稿 + 20 个问答）。

> 说明：`data/` 与 `models/` 体积过大未入库，获取方式见下方「数据与模型准备」；
> 课程原始资料（老师给的 PPT 照片）属于他人材料，也不入库。

## 目录结构

```
.
├── code/                  # 全部可运行脚本（18 个）
│   ├── muge_eval.py       # 零样本基线评测（复现官方口径）
│   ├── build_feats.py     # 缓存训练集冻结特征
│   ├── train_adapter.py   # 线性探针 / 轻量 Adapter
│   ├── template_eval.py   # 系统性提示模板优化
│   ├── train_lora.py      # LoRA 部分参数微调
│   ├── retrieval_eval.py  # 通用检索评测（支持 LoRA 权重）
│   ├── efficiency.py      # 效率分析（参数/显存/时延）
│   ├── visualize.py       # 失败案例与排名分布可视化
│   ├── p10k_feats.py      # Products-10K 特征提取
│   ├── p10k_classify.py   # 商品细粒度分类（长尾/少样本）
│   ├── demo_search.py     # 交互式检索演示
│   └── build_*.py         # 报告 / PPT / 答辩手册生成
├── docs/
│   ├── 实验记录.md         # 实验数据底稿（全部数字来源）
│   └── ref/               # Chinese-CLIP 官方评测脚本参考
├── results/
│   ├── *.json             # 各实验指标
│   ├── figs/              # 结果图（训练曲线/失败案例/排名分布）
│   ├── demo/              # 演示检索结果图
│   └── samples/           # 数据样例
├── 交付/                  # 技术报告、答辩 PPT、论文分享、答辩准备手册
├── 运行说明.md             # 详细运行文档
├── run_demo.bat           # 双击运行交互式检索演示
└── requirements.txt
```

## 快速开始

```bash
# 1. 创建环境（关键：清空 UV_DEFAULT_INDEX，否则会装成 CPU 版 torch）
uv venv --python 3.12 .venv
uv pip install -r requirements.txt

# 2. 双击 run_demo.bat 体验交互式检索（需先准备数据与权重，见下）
```

## 数据与模型准备

本仓库不包含数据与权重（体积过大），需自行下载：

| 资源 | 来源 | 放置位置 |
|---|---|---|
| MUGE 中文电商图文检索 | [OFA-Sys/chinese-clip-eval](https://huggingface.co/datasets/OFA-Sys/chinese-clip-eval) 的 `MUGE.zip` | `data/raw/MUGE/` |
| Flickr30K-CN | 同上，`Flickr30k-CN.zip` | `data/raw/Flickr30k-CN/` |
| Products-10K | [nyris/products10k-v1](https://huggingface.co/datasets/nyris/products10k-v1) | `data/raw/products10k_v1/` |
| Chinese-CLIP ViT-B/16 | [OFA-Sys/chinese-clip-vit-base-patch16](https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16) | `models/chinese-clip-vit-b16/` |

国内建议设置 `HF_ENDPOINT=https://hf-mirror.com`，并用 curl 直连下载（hf 客户端会被镜像限速）。

## 复现主要实验

```bash
# 零样本基线（约 1.5 分钟）
python code/muge_eval.py --split valid --batch-size 128 --save-feats

# 缓存训练集特征（约 7 分钟）
python code/build_feats.py --split train --batch-size 128

# 线性探针 / Adapter（秒级）
python code/train_adapter.py --method linear  --fraction 1.0 --epochs 20
python code/train_adapter.py --method adapter --fraction 1.0 --epochs 20 --hidden 128

# 提示模板优化（零成本）
python code/template_eval.py

# LoRA 训练（约 10 分钟，含评测）
python code/train_lora.py --max-steps 1200 --bs 32 --accum 1 --lr 1.5e-4 --out lora_1200step

# 跨域评测（MUGE 训练 → Flickr30K-CN 测试）
python code/retrieval_eval.py --data-dir data/raw/Flickr30k-CN --split test --lora results/lora_1200step_ckpt

# 效率分析与可视化
python code/efficiency.py
python code/visualize.py
```

## 实现中踩过的坑

1. **线性探针必须恒等初始化**：随机初始化会把预训练跨模态对齐打散，T2I R@1 从 52 掉到 0.68。
2. **transformers 5.x 的 `get_*_features` 返回 `BaseModelOutputWithPooling`**，需取 `pooler_output`。
3. **预处理必须对齐官方**：`Resize((224,224))` 直接拉伸，不是短边 resize + center crop。
4. **装 cu128 版 torch 前要清空 `UV_DEFAULT_INDEX`**，否则解析到 CPU 轮子。
5. **LoRA 推理前要 `merge_and_unload()`**，否则 PEFT 前向钩子会让吞吐降到 1/3。

## 参考文献

- Radford et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021.
- Yang et al. *Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese.* arXiv:2211.01335
- Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
- Gao et al. *CLIP-Adapter: Better Vision-Language Models with Feature Adapters.* arXiv:2110.04544
- Bai et al. *Products-10K: A Large-scale Product Recognition Dataset.* arXiv:2008.10545
