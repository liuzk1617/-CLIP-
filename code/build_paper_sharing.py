"""生成《论文分享》pptx —— 第 12 周 Paper Sharing（7 分钟），重点讲论文与本组项目的关联。

三篇论文：CLIP（原理）、Chinese-CLIP（中文适配与基线）、LoRA（本项目的改进方法）
运行：
  python code\\build_paper_sharing.py
"""
from __future__ import annotations

from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Cm, Pt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "交付" / "论文分享.pptx"

NAVY = RGBColor(0x1F, 0x3B, 0x73)
ACCENT = RGBColor(0xE8, 0x71, 0x2C)
GREY = RGBColor(0x55, 0x55, 0x55)
LIGHT = RGBColor(0xF2, 0xF4, 0xF8)
TEAL = RGBColor(0x0E, 0x7C, 0x86)
FONT = "微软雅黑"
W, H = Cm(33.867), Cm(19.05)


def _font(run, size=18, bold=False, color=NAVY):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find("{http://schemas.openxmlformats.org/drawingml/2006/main}ea")
    if ea is None:
        ea = etree.SubElement(rPr, "{http://schemas.openxmlformats.org/drawingml/2006/main}ea")
    ea.set("typeface", FONT)


def add_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def header(slide, title, sub=None, color=NAVY):
    bar = slide.shapes.add_shape(1, Cm(0), Cm(0), W, Cm(1.9))
    bar.fill.solid(); bar.fill.fore_color.rgb = color; bar.line.fill.background()
    tb = slide.shapes.add_textbox(Cm(1.2), Cm(0.35), W - Cm(2.4), Cm(1.3))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run(); _font(r, 24, True, RGBColor(0xFF, 0xFF, 0xFF)); r.text = title
    if sub:
        tb2 = slide.shapes.add_textbox(Cm(1.2), Cm(2.1), W - Cm(2.4), Cm(0.9))
        p2 = tb2.text_frame.paragraphs[0]
        r2 = p2.add_run(); _font(r2, 13, False, GREY); r2.text = sub


def bullets(slide, items, top=Cm(3.3), left=Cm(1.6), width=None, size=18, gap=8):
    width = width or (W - Cm(3.2))
    tb = slide.shapes.add_textbox(left, top, width, H - top - Cm(1.2))
    tf = tb.text_frame; tf.word_wrap = True
    for i, item in enumerate(items):
        level, text = (item if isinstance(item, tuple) else (0, item))
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap); p.level = level
        r = p.add_run()
        _font(r, size if level == 0 else size - 3, level == 0, NAVY if level == 0 else GREY)
        r.text = ("● " if level == 0 else "– ") + text


def table(slide, headers, rows, top=Cm(3.4), left=Cm(1.4), width=None, size=13,
          col_ratio=None, row_h=Cm(1.2)):
    width = width or (W - Cm(2.8))
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), left, top, width,
                                   row_h * (len(rows) + 1))
    tbl = shape.table
    if col_ratio:
        total = sum(col_ratio)
        for i, ratio in enumerate(col_ratio):
            tbl.columns[i].width = int(width * ratio / total)
    for j, h in enumerate(headers):
        cell = tbl.cell(0, j); cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); _font(r, size, True, RGBColor(0xFF, 0xFF, 0xFF)); r.text = str(h)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            cell = tbl.cell(i + 1, j); cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT if i % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            raw = str(v); bold = raw.startswith("**") and raw.endswith("**")
            _font(r, size, bold, ACCENT if bold else RGBColor(0x33, 0x33, 0x33))
            r.text = raw.strip("*") if bold else raw


def note(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # 1 封面
    s = add_slide(prs)
    bg = s.shapes.add_shape(1, Cm(0), Cm(0), W, H)
    bg.fill.solid(); bg.fill.fore_color.rgb = TEAL; bg.line.fill.background()
    tb = s.shapes.add_textbox(Cm(2.4), Cm(4.6), W - Cm(4.8), Cm(3))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run(); _font(r, 32, True, RGBColor(0xFF, 0xFF, 0xFF))
    r.text = "论文分享：从 CLIP 到 LoRA"
    tb2 = s.shapes.add_textbox(Cm(2.4), Cm(8.2), W - Cm(4.8), Cm(2))
    p2 = tb2.text_frame.paragraphs[0]
    r2 = p2.add_run(); _font(r2, 18, False, RGBColor(0xDD, 0xF0, 0xF2))
    r2.text = "多模态图文检索的预训练、中文适配与轻量微调"
    tb3 = s.shapes.add_textbox(Cm(2.4), Cm(13.6), W - Cm(4.8), Cm(2.4))
    for i, line in enumerate(["人工智能导论 · 期末大作业 · 方向五：多模态 AI 与 VLM",
                              "第 12 周 Paper Sharing · 第 __ 组 · ____"]):
        p3 = tb3.text_frame.paragraphs[0] if i == 0 else tb3.text_frame.add_paragraph()
        r3 = p3.add_run(); _font(r3, 13, False, RGBColor(0xB8, 0xD8, 0xDC)); r3.text = line
    note(s, "这次论文分享围绕三篇与本题最相关的论文：CLIP 讲原理，Chinese-CLIP 讲中文适配和我们的基线，LoRA 讲我们采用的改进方法。")

    # 2 为什么是这三篇
    s = add_slide(prs)
    header(s, "为什么是这三篇", "三篇论文恰好对应本组项目的三个环节")
    table(s,
          ["论文", "解决的问题", "在本项目中的位置"],
          [["CLIP\n(Radford, 2021)", "图文如何对齐到一个空间", "模型原理基础"],
           ["Chinese-CLIP\n(Yang, 2022)", "英文预训练模型不懂中文电商标题", "**本项目的基座模型与基线**"],
           ["LoRA\n(Hu, 2022)", "8GB 显存无法全参数微调", "**本项目的改进方法**"]],
          top=Cm(3.6), size=14, col_ratio=[4, 6, 5.4], row_h=Cm(2.2))
    bullets(s, ["分享顺序：问题 → 核心模型 → 主要实验 → 与本组项目的关联"],
            top=Cm(13.4), size=15)
    note(s, "选这三篇不是随便选的：CLIP 是原理源头，Chinese-CLIP 是我们的基座，LoRA 是我们的改进手段，三者串起来正好是本项目的技术路线。")

    # 3 CLIP 原理
    s = add_slide(prs)
    header(s, "CLIP：用自然语言监督学视觉表征", "Radford et al., ICML 2021 · arXiv:2103.00020")
    bullets(s, [
        "核心问题：传统视觉模型依赖固定类别标签，无法做开放词汇与零样本迁移",
        "做法：双塔结构（图像编码器 + 文本编码器）+ 对比学习",
        (1, "在 4 亿图文对上，把配对图文拉近、非配对推远（对称 InfoNCE）"),
        (1, "推理时把类别名写成句子，用文本嵌入当分类器权重 → 零样本分类"),
        "关键结论：规模化自然语言监督可以替代人工标注，且表征可迁移",
    ], top=Cm(3.6), size=17, gap=10)
    note(s, "CLIP 最重要的思想是：不用类别标签，用自然语言描述当监督信号。这让模型能理解开放词汇，也为后来的中文版和检索任务打下基础。")

    # 4 CLIP 实验
    s = add_slide(prs)
    header(s, "CLIP：主要实验与对我们的启示")
    bullets(s, [
        "零样本 ImageNet 准确率 76.2%（ViT-L/14@336），与有监督基线接近",
        "在 30+ 个数据集上零样本迁移，多数任务超过专门训练的 ResNet 基线",
        "启示 1：双塔 + 对比学习是图文检索的标准范式",
        "启示 2：但 CLIP 的文本塔是英文的 —— 中文标题直接输入会失效",
    ], top=Cm(3.6), size=17, gap=10)
    note(s, "这个缺口正是 Chinese-CLIP 的动机：CLIP 的文本塔用英文 BPE 词表，中文会被拆成字节级 token，语义完全丢失。我们组的查询恰恰是中文电商标题。")

    # 5 Chinese-CLIP
    s = add_slide(prs)
    header(s, "Chinese-CLIP：把双塔换成中文", "Yang et al., 2022 · arXiv:2211.01335")
    bullets(s, [
        "文本塔改为中文 RoBERTa/BERT（中文词表），图像塔沿用 CLIP ViT",
        "在约 2 亿中文图文对上预训练，天然理解「丝光棉」「金丝绒」等电商用语",
        "提供多个规模：RN50 / ViT-B/16 / ViT-L/14 / ViT-L/14@336 / ViT-H/14",
        "评测集：MUGE（电商图文检索）、Flickr30K-CN、COCO-CN",
    ], top=Cm(3.6), size=17, gap=10)
    note(s, "关键改动只有一处：文本塔换成中文预训练模型。这说明跨模态对齐的能力主要来自对比学习目标，而语言侧可以按语种替换。")

    # 6 Chinese-CLIP 结果 + 我们的复现
    s = add_slide(prs)
    header(s, "Chinese-CLIP 结果 + 我们的复现验证", "MUGE 官方验证集，Text-to-Image，%")
    table(s,
          ["模型规模", "R@1", "R@5", "R@10"],
          [["RN50", "42.6", "68.6", "77.9"],
           ["**ViT-B/16（本组基座）**", "**52.1**", "**76.7**", "**84.4**"],
           ["ViT-L/14", "56.3", "79.8", "86.2"],
           ["ViT-H/14", "63.0", "84.1", "89.2"]],
          top=Cm(3.4), size=14, col_ratio=[6, 3, 3, 3], row_h=Cm(1.35))
    bullets(s, ["我们的复现：R@1 = 51.98 / R@5 = 76.76 / R@10 = 84.29，与官方差 ≤0.12",
                "说明数据划分、预处理、tokenizer、指标口径完全对齐 → 后续改进才可比"],
            top=Cm(12.2), size=15, gap=8)
    note(s, "这张表是我们组立项的第一件事：先把官方数字复现出来。差值都在0.12以内，说明评测协议对齐了。这一步不做，后面的改进数字就不可信。")

    # 7 LoRA 原理
    s = add_slide(prs)
    header(s, "LoRA：冻结主干，只学低秩增量", "Hu et al., ICLR 2022 · arXiv:2106.09685")
    bullets(s, [
        "问题：全参数微调大模型成本高；且在小数据集上容易破坏预训练能力",
        "观察：微调时权重变化矩阵是低秩的（ΔW 的内在秩很低）",
        (1, "把 ΔW 分解为 B·A，A∈R^{r×d}、B∈R^{d×r}，只训练 A、B"),
        (1, "推理时把 BA 合并回原权重，不增加任何时延"),
        "效果：GPT-3 175B 上可训练参数减少约 10,000 倍，显存减少约 3 倍，效果与全量微调相当",
    ], top=Cm(3.6), size=16, gap=9)
    note(s, "LoRA 的关键是两句话：一是权重变化是低秩的，二是合并后零开销。这两点正好解决我们的两个约束：8GB 显存、以及要能现场演示不卡。")

    # 8 LoRA 为什么适合我们
    s = add_slide(prs)
    header(s, "LoRA 与本组项目的对应关系")
    table(s,
          ["我们的约束", "LoRA 的对应设计"],
          [["8GB 显存，无法全参数微调", "只训练 0.55% 参数（1.03M），训练峰值 3.74GB"],
           ["要能现场演示、推理不能变慢", "merge_and_unload() 合并后与基线完全等价"],
           ["改进必须可解释、可复现", "r/α 可解释，3 个种子结果稳定（±0.65）"]],
          top=Cm(3.6), size=14, col_ratio=[6.5, 8], row_h=Cm(2.4))
    bullets(s, ["我们也在 MUGE 上验证了：LoRA T2I R@1 51.98 → 54.71（+2.73），I2T +4.92"],
            top=Cm(12.6), size=15)
    note(s, "这一页是把论文和我们的项目直接对齐：论文里的三条卖点，恰好对应我们的三个约束。后面第14周 Demo 会展示这些数字。")

    # 9 从论文到实现
    s = add_slide(prs)
    header(s, "从论文到实现：我们做了什么")
    bullets(s, [
        "复现：MUGE / Flickr30K-CN 两套官方评测，误差 ≤0.5",
        "对照实验：线性探针、轻量 Adapter、提示模板、LoRA 四类方法",
        "结论：冻结特征上的事后映射对 T2I 无效；LoRA 是唯一双向提升的方法",
        "下一步（第 14 周 Demo）：补训练量曲线 + Products-10K 商品分类补充任务",
    ], top=Cm(3.6), size=17, gap=12)
    note(s, "最后汇报一下我们自己的进展：基线已经复现，四类改进方法做了对比，LoRA 效果最好。第14周会带运行结果和消融表过来。")

    # 10 结语
    s = add_slide(prs)
    header(s, "小结")
    bullets(s, [
        "CLIP 给出「自然语言监督 + 双塔对比学习」的范式",
        "Chinese-CLIP 把文本塔换成中文，使中文电商检索成为可能",
        "LoRA 用低秩增量解决算力约束，且部署零开销",
        "我们的贡献：在 8GB 单卡上把三者串成一条可复现的技术路线",
        "谢谢，欢迎提问",
    ], top=Cm(3.8), size=18, gap=14)
    note(s, "结束。预期提问：为什么不直接全量微调？（显存不够）；为什么不用 CoOp？（也做了模板实验，对T2I无效）；为什么不用更大模型？（8GB约束）。")

    prs.save(str(OUT))
    print("saved:", OUT, "| slides:", len(prs.slides))


if __name__ == "__main__":
    main()
