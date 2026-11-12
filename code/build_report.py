"""生成《技术报告》docx —— 数据来自 results/*.json 与 docs/实验记录.md 的实测值。

运行：
  python code\\build_report.py
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "交付" / "技术报告.docx"
FIGS = ROOT / "results" / "figs"
CN_FONT = "微软雅黑"
MONO_FONT = "Courier New"


def _repeat_header(table):
    """表头跨页重复，避免表格被分页切开后第二页没有表头。"""
    tr = table.rows[0]._tr
    trPr = tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    trPr.append(header)
    for row in table.rows:
        trPr2 = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        cant.set(qn("w:val"), "true")
        trPr2.append(cant)


def set_font(run, size=10.5, bold=False, name=CN_FONT, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)


def para(doc, text="", size=10.5, bold=False, align=None, space_after=6, indent=True):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    if indent:
        p.paragraph_format.first_line_indent = Cm(0.74)
    if align:
        p.alignment = align
    r = p.add_run(text)
    set_font(r, size=size, bold=bold)
    return p


def heading(doc, text, level=1):
    h = doc.add_heading(level=level)
    r = h.add_run(text)
    set_font(r, size={1: 16, 2: 13.5, 3: 12}.get(level, 11), bold=True, color=(0x1F, 0x3B, 0x73))
    return h


def add_runs(p, text, size=10.5, bold=False):
    """支持单元格内局部加粗：文本里 **xx** 的部分加粗，其余正常。"""
    parts = str(text).split("**")
    for i, part in enumerate(parts):
        if not part:
            continue
        r = p.add_run(part)
        set_font(r, size=size, bold=(bold or (i % 2 == 1)))


def add_table(doc, headers, rows, caption=None, widths=None, font_size=9):
    if caption:
        c = doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = c.add_run(caption)
        set_font(r, size=9, bold=True)
        c.paragraph_format.space_after = Pt(3)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(str(h))
        set_font(r, size=font_size, bold=True)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_runs(p, v, size=font_size)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    _repeat_header(t)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(doc, path: Path, caption: str, width_cm=13.5):
    if not path.exists():
        print(f"[warn] missing figure {path}")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(caption)
    set_font(r, size=9)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    for section in doc.sections:
        section.top_margin = Cm(2.4)
        section.bottom_margin = Cm(2.2)
        section.left_margin = Cm(2.6)
        section.right_margin = Cm(2.6)

    # ---------- 标题 ----------
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("基于中文 CLIP 的电商图文检索")
    set_font(r, size=20, bold=True, color=(0x1F, 0x3B, 0x73))
    t2 = doc.add_paragraph()
    t2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t2.add_run("——线性探针、轻量 Adapter 与 LoRA 的系统对比")
    set_font(r, size=14, bold=True, color=(0x1F, 0x3B, 0x73))
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sub.add_run("人工智能导论 · 期末大作业 · 方向五：多模态 AI 与 VLM")
    set_font(r, size=10.5, color=(0x60, 0x60, 0x60))

    # ---------- 摘要 ----------
    heading(doc, "摘要", 1)
    para(doc, "本项目面向中文电商图文检索这一真实问题，以 Chinese-CLIP ViT-B/16 为基础模型，"
              "在 MUGE 电商检索数据集（官方验证集）上系统比较了四类轻量化适配方法：零样本直接推理、"
              "线性探针、轻量 Adapter 与 LoRA 部分参数微调，并额外考察系统性提示模板优化。"
              "全部实验在单张 8GB 显存的 RTX 5060 Laptop GPU 上完成。")
    para(doc, "实验表明：冻结主干后仅对终态特征做线性或 Adapter 映射，对文本检图（T2I）指标无增益，"
              "增加容量反而过拟合；而 LoRA 在注意力层注入低秩增量、让主干参与前向，是唯一同时提升双向检索的方法。"
              "LoRA（r=8，1200 步）在三个随机种子上取得 T2I R@1 = 55.20 ± 0.65、I2T R@1 = 43.63 ± 0.17，"
              "相对零样本基线分别提升 +3.22 与 +4.94，训练仅需 6.3 分钟、可训练参数 1.03M（占全模型 0.55%）。"
              "进一步实验显示：rank 从 4 增到 16 指标变化小于 0.5，说明适配所需秩很低；"
              "在电商域训练的 LoRA 迁移到开放域 Flickr30k-CN 未出现灾难性遗忘（I2T R@1 +3.40）；"
              "推理前将 LoRA 权重合并回主干后，参数量与推理时延与基线完全一致。")

    # ---------- 1 问题 ----------
    heading(doc, "1  问题与目标", 1)
    para(doc, "中文电商场景下，用户输入的是极短的商品标题（如“金丝绒木床头罩”“男t丝光棉长袖”），"
              "系统需要在数万张商品图中找出与之匹配的图像。该任务有三个现实难点：标题信息量极低、"
              "同品牌/同系列商品视觉高度相似、长尾分布明显。")
    para(doc, "按照方向五的要求，本项目需要：① 运行图文模型并建立检索测试集；"
              "② 至少在“线性探针 / 轻量 Adapter / 部分参数微调 / 系统性提示模板优化”中实施一种可解释、可复现的改进；"
              "③ 完成 Zero-shot 与改进方法的公平对比、不同训练样本量对比与跨场景泛化对比。")

    # ---------- 2 原理 ----------
    heading(doc, "2  模型原理与选择依据", 1)
    heading(doc, "2.1  为什么选 Chinese-CLIP", 2)
    para(doc, "CLIP 采用双塔结构：图像编码器与文本编码器分别把图像和文本映射到同一 512 维嵌入空间，"
              "并用对称 InfoNCE 损失拉近配对样本、推远非配对样本。Chinese-CLIP 在中文图文对上预训练，"
              "文本侧使用中文 RoBERTa/BERT 词表，因此能直接理解“金丝绒”“丝光棉”这类中文电商用语。"
              "这是本项目选择它而非英文 CLIP 的核心依据：英文 CLIP 面对中文标题需要先翻译，"
              "会引入额外的语义损失，而 MUGE 的查询恰恰是原生中文短标题。")
    heading(doc, "2.2  为什么改进策略是“轻量适配”", 2)
    para(doc, "8GB 显存无法支撑 ViT-B/16 全参数微调，且全量微调在小数据集上容易破坏预训练对齐。"
              "因此本项目把改进限定在低秩、低参数量、可合并的范围：线性探针与 Adapter 只训练末端映射，"
              "LoRA 只在注意力层的 q/k/v/out 投影上注入低秩增量。三类方法都满足“可解释、可复现”的要求。")

    # ---------- 3 数据 ----------
    heading(doc, "3  数据与评测协议", 1)
    add_table(doc,
              ["数据集", "划分", "规模", "用途"],
              [["MUGE（中文电商图文）", "train", "129,380 图 / 248,786 文本 / 250,314 图文对", "训练改进方法"],
               ["MUGE", "valid（官方评测集）", "29,806 图 / 5,008 文本", "主评测"],
               ["Flickr30k-CN", "test", "1,000 图 / 5,000 文本", "跨场景泛化"],
               ["Products-10K", "train + test", "197,307 图 / 9,691 SKU", "商品细粒度分类（后续）"]],
              caption="表 1  实验数据规模", widths=[4.2, 3.2, 5.6, 3.2])
    para(doc, "评测指标为检索常用的 Recall@K（K=1/5/10）与平均召回 MR=(R@1+R@5+R@10)/3，"
              "同时报告文本检图（T2I）与图检文本（I2T）两个方向。图像预处理严格对齐官方实现："
              "Resize 到 224×224 直接拉伸后归一化；文本小写化后按最大长度 24 截断。")
    para(doc, "为验证评测口径正确，本项目先复现官方公开数字，结果见表 2。两个数据集均落在官方值 ±0.5 以内，"
              "说明数据划分、预处理、tokenizer 与指标统计方式完全一致，后续对比具备可比性。")
    add_table(doc,
              ["数据集 / 任务", "指标", "官方 CN-CLIP ViT-B/16", "本项目复现", "差异"],
              [["MUGE T2I", "R@1", "52.1", "51.98", "−0.12"],
               ["MUGE T2I", "R@5", "76.7", "76.76", "+0.06"],
               ["MUGE T2I", "R@10", "84.4", "84.29", "−0.11"],
               ["Flickr30k-CN T2I", "R@1", "62.7", "62.30", "−0.40"],
               ["Flickr30k-CN I2T", "R@1", "74.6", "74.30", "−0.30"]],
              caption="表 2  基线复现校验", widths=[4.2, 2.2, 4.4, 3.4, 2.0])

    # ---------- 4 方法 ----------
    heading(doc, "4  方法", 1)
    heading(doc, "4.1  零样本基线", 2)
    para(doc, "直接使用预训练 Chinese-CLIP 编码图像与文本，按余弦相似度排序，不做任何训练。")
    heading(doc, "4.2  线性探针（Linear Probe）", 2)
    para(doc, "冻结主干，在两侧特征后各接一层 512×512 线性映射再 L2 归一化。关键实现细节：权重必须初始化为单位矩阵。"
              "若随机初始化，线性映射会把预训练学到的跨模态对齐结构整体打散——实测在 1% 数据下 T2I R@1 从 52 直接跌到 0.68；"
              "改为恒等初始化后，模型从零样本水平出发，训练只做微调式重映射。")
    heading(doc, "4.3  轻量 Adapter", 2)
    para(doc, "在特征上接瓶颈结构 f′=f+W₂·GELU(W₁f)，隐藏维 128。上投影层 W₂ 零初始化，使初始输出等于输入，"
              "训练只学习增量。这与 CLIP-Adapter 的残差设计同理，目的是避免训练初期破坏原有表征。")
    heading(doc, "4.4  系统性提示模板优化", 2)
    para(doc, "不训练任何参数，只改变查询文本的包装方式：raw（原文）、一张{text}的图片、{text}的商品图片、"
              "商品：{text}、这是一张{text}的图片。以及多模板特征平均的集成方式。")
    heading(doc, "4.5  LoRA 部分参数微调", 2)
    para(doc, "在视觉塔与文本塔的注意力 q/k/v/out 投影上注入低秩增量（r=8，α=16，dropout=0.05），"
              "可训练参数 1.03M（0.55%）。损失沿用对称 InfoNCE（温度尺度 100），"
              "AdamW 优化器，学习率 1.5e-4，OneCycle 调度，batch 32，训练 1200 步。")

    # ---------- 5 设置 ----------
    heading(doc, "5  实验设置", 1)
    add_table(doc,
              ["项", "配置"],
              [["GPU", "NVIDIA RTX 5060 Laptop，8GB，sm_120"],
               ["软件环境", "Python 3.12.9，PyTorch 2.11.0+cu128，transformers 5.21.0，peft 0.21.2"],
               ["基座模型", "OFA-Sys/chinese-clip-vit-base-patch16（188.26M 参数）"],
               ["LoRA 超参", "r=8，α=16，dropout=0.05，lr=1.5e-4，batch=32，steps=1200"],
               ["随机种子", "0 / 1 / 2（用于复现性检验）"],
               ["峰值显存", "训练 3.74GB，评测 0.99GB"]],
              caption="表 3  实验设置与计算资源", widths=[3.6, 12.4])

    # ---------- 6 结果 ----------
    heading(doc, "6  结果与分析", 1)
    heading(doc, "6.1  主结果：四类方法对比", 2)
    add_table(doc,
              ["方法", "可训练参数", "T2I R@1", "T2I R@5", "T2I MR", "I2T R@1", "I2T MR"],
              [["零样本基线", "0", "51.98", "76.76", "71.01", "38.69", "59.69"],
               ["线性探针 1%", "0.52M", "52.04", "76.58", "71.05", "39.92", "61.60"],
               ["线性探针 100%", "0.52M", "50.10", "76.60", "70.49", "38.60", "61.49"],
               ["轻量 Adapter 100%", "0.26M", "51.40", "77.96", "71.60", "40.14", "62.94"],
               ["提示模板（商品图片）", "0", "51.86", "77.00", "71.05", "40.64", "62.37"],
               ["**LoRA（1200 步）**", "**1.03M**", "**54.71**", "**80.21**", "**73.98**", "**43.61**", "**65.81**"],
               ["LoRA（3000 步）", "1.03M", "54.89", "79.99", "74.19", "43.76", "66.01"]],
              caption="表 4  各改进方法在 MUGE valid 上的结果（%）", widths=[4.4, 2.6, 2.4, 2.4, 2.2, 2.4, 2.2])
    para(doc, "结论有三点。第一，冻结特征上做事后映射对 T2I 无效，甚至因过拟合而下降（线性探针 100% 降至 50.10），"
              "原因是 T2I 的瓶颈在图像侧视觉表征，末端线性变换无法改变已固化的特征。"
              "第二，LoRA 是唯一同时提升双向检索的方法，T2I R@1 提升 2.73、I2T R@1 提升 4.92。"
              "第三，提示模板对 I2T 有效（MR +2.68）而对 T2I 无效，且句式越接近电商标题分布效果越好。")
    heading(doc, "6.2  可复现性：随机种子", 2)
    add_table(doc,
              ["seed", "T2I R@1", "T2I MR", "I2T R@1", "I2T MR"],
              [["0", "54.71", "73.98", "43.61", "65.81"],
               ["1", "55.93", "74.33", "43.80", "65.86"],
               ["2", "54.95", "74.01", "43.47", "65.67"],
               ["**均值 ± 标准差**", "**55.20 ± 0.65**", "**74.11 ± 0.19**", "**43.63 ± 0.17**", "**65.78 ± 0.10**"]],
              caption="表 5  LoRA 三个随机种子的结果（%）", widths=[3.4, 3.6, 3.4, 3.4, 3.4])
    para(doc, "三个种子间波动小于 0.7，相对基线的提升（T2I +3.22、I2T +4.94）远大于波动，说明改进可复现，并非偶然。")
    heading(doc, "6.3  消融：LoRA 秩", 2)
    add_table(doc,
              ["配置", "可训练参数", "T2I R@1", "T2I MR", "I2T R@1"],
              [["r=4, α=8", "0.52M", "**55.15**", "74.12", "43.74"],
               ["r=8, α=16", "1.03M", "54.71", "73.98", "43.61"],
               ["r=16, α=32", "2.06M", "54.91", "73.92", "43.51"]],
              caption="表 6  LoRA 低秩维度消融（seed=0，1200 步）", widths=[3.6, 3.4, 3.4, 3.4, 3.4])
    para(doc, "rank 从 4 增到 16，指标变化小于 0.5 且无单调趋势。这说明该任务的适配所需秩很低，"
              "收益来自“端到端更新主干”这一机制本身，而非参数规模堆叠；r=4 用一半参数即可达到同等效果。")
    heading(doc, "6.4  跨场景泛化", 2)
    add_table(doc,
              ["模型", "测试集", "T2I R@1", "T2I MR", "I2T R@1", "I2T MR"],
              [["零样本基线", "Flickr30k-CN test", "62.30", "80.51", "74.30", "88.10"],
               ["**MUGE 训练的 LoRA**", "Flickr30k-CN test", "**62.94**", "**80.86**", "**77.70**", "**89.70**"]],
              caption="表 7  跨域迁移结果：电商域训练 → 开放域测试（%）", widths=[4.6, 4.0, 2.4, 2.2, 2.4, 2.2])
    para(doc, "在电商域上训练的 LoRA 迁移到开放域照片时没有出现灾难性遗忘，T2I 基本持平略升，I2T R@1 提升 3.40。"
              "这说明窄域适配并未破坏模型的通用图文对齐能力。")
    heading(doc, "6.5  效率分析", 2)
    add_table(doc,
              ["项", "基线", "LoRA 未合并", "LoRA 合并回主干"],
              [["参数量", "188,262,913", "189,295,105（+0.55%）", "**188,262,913（一致）**"],
               ["图像编码 bs=8", "603.6 img/s", "238.9 img/s", "**629.7 img/s**"],
               ["图像编码 bs=32", "486.4 img/s", "368.9 img/s", "**533.6 img/s**"],
               ["峰值显存", "0.54 GB", "0.54 GB", "0.54 GB"]],
              caption="表 8  推理效率对比", widths=[3.6, 4.2, 4.4, 4.4])
    para(doc, "LoRA 训练只增加 0.55% 的参数；部署前执行 merge_and_unload() 合并后，参数量、时延与显存全部回到基线水平，"
              "没有额外推理代价。未合并时 PEFT 的前向钩子会使吞吐下降，这属于训练态现象，不构成部署开销。")

    heading(doc, "6.6  训练量曲线", 2)
    add_table(doc,
              ["训练步数", "T2I R@1", "T2I MR", "I2T R@1", "I2T MR", "训练耗时"],
              [["300", "54.83", "73.36", "43.28", "65.17", "326s"],
               ["600", "**55.31**", "73.99", "43.47", "65.54", "414s"],
               ["1200", "54.71", "73.98", "43.61", "65.81", "592s"],
               ["3000", "54.89", "**74.19**", "**43.76**", "**66.01**", "1160s"]],
              caption="表 9  LoRA 训练量曲线（seed=0，bs=32）", widths=[3.0, 2.8, 2.8, 2.8, 2.8, 2.6])
    figure(doc, FIGS / "lora_curve.png", "图 1  训练步数与检索指标（虚线为零样本基线）", width_cm=14.5)
    para(doc, "300 步（约 5 分钟）就已拿到几乎全部收益，之后基本饱和。600 步的 T2I R@1 略高于 1200 步，"
              "属于随机种子级波动（种子标准差 0.65），不构成真实趋势。这说明该任务上“端到端更新主干”的收益是低成本可得的，"
              "训练量不是瓶颈——这对课程作业的算力约束是个有利结论。")

    # ---------- 7 局限 ----------
    heading(doc, "7  局限性与失败分析", 1)
    para(doc, "人工检查失败案例后归纳出三类典型模式：① 同品牌/同系列近重复，例如查询“希雅茜”检索到同品牌另一款护肤品；"
              "② 超短标题歧义，例如“荣光二”信息量过低；③ 同款不同色，例如“方块笑脸卫衣”正样本为粉色款而 Top1 为深色款。")
    figure(doc, FIGS / "failure_cases.png", "图 2  失败案例：左侧为真实匹配图，右侧为模型 Top1 结果", width_cm=8.2)
    para(doc, "这三类说明 R@1 存在系统性低估：MUGE 每条查询平均约 6 张正样本，同品牌其他商品一律判为错误。"
              "因此本项目报告的绝对数值偏保守，改进幅度的方向性结论不受影响，但绝对水平不宜与其他标注口径不同的工作直接比较。")
    figure(doc, FIGS / "rank_hist.png", "图 3  正样本排名分布（中位排名为第 1 名，7.91% 的查询掉出前 20）", width_cm=11)

    # ---------- 8 结论 ----------
    heading(doc, "8  结论与后续工作", 1)
    para(doc, "本项目在中文电商图文检索任务上系统比较了四类轻量适配方法。核心结论是："
              "在 8GB 显存约束下，LoRA 部分参数微调以 0.55% 的可训练参数取得了 T2I R@1 +3.22、I2T R@1 +4.94 的稳定增益，"
              "且推理前可无损合并；而线性探针与轻量 Adapter 在冻结特征上对 T2I 无效，提示模板优化只对 I2T 有效。"
              "跨域实验证明该适配不会破坏通用表征。")
    para(doc, "后续工作包括：LoRA 作用位置消融（仅视觉塔 / 仅文本塔）、"
              "在 Products-10K 上补充商品细粒度分类实验（9,691 类，用于覆盖方向五“视觉语言商品分类”选题），"
              "以及多正样本标注下的评测口径改进。")

    # ---------- 9 补充任务 ----------
    heading(doc, "9  补充任务：Products-10K 商品细粒度分类", 1)
    para(doc, "为覆盖方向五选题示例中的“视觉语言商品分类”，项目额外在 Products-10K 上做了细粒度分类实验"
              "（197,307 图 / 9,691 个 SKU）。需要说明的是：该数据集只发布 SKU 编号、没有可读类目名称，"
              "因此无法构造文本类名做零样本分类，本任务只能做“冻结特征 + 线性探针”的可分性与可训练性分析。")
    add_table(doc,
              ["项", "结果"],
              [["数据规模", "train 141,931 图 / test 55,376 图 / 9,691 类"],
               ["方法", "Chinese-CLIP ViT-B/16 冻结特征（512 维）+ 线性探针"],
               ["全量结果", "**top-1 = 20.69%**，top-5 = 35.99%（训练 12 秒）"],
               ["可训练参数", "9,691 × 512 ≈ 4.96M"]],
              caption="表 10  Products-10K 线性探针结果", widths=[3.6, 12.4])
    add_table(doc,
              ["分组（按训练样本数）", "类别数", "平均准确率"],
              [["最少 20%", "1,784", "**0.0%**"],
               ["Q2", "2,088", "0.4%"],
               ["Q3", "1,683", "7.1%"],
               ["Q4", "2,017", "25.6%"],
               ["最多 20%", "2,119", "**59.7%**"]],
              caption="表 11  长尾分组准确率", widths=[6.0, 4.0, 4.0])
    para(doc, "长尾效应极强：尾部 40% 的类几乎完全学不会（准确率 <0.5%），头部 20% 可达 59.7%。"
              "这与该数据集“每个 SKU 图片数 5~87、中位 18”的分布直接对应，也说明在真实电商场景中，"
              "细粒度分类的瓶颈是长尾而非模型容量。")
    figure(doc, FIGS / "p10k_classify.png", "图 4  左：长尾分组准确率；右：少样本曲线（类子集随 K 变化）", width_cm=15)
    para(doc, "补充说明：少样本曲线中 K 越大可用类别越少（K=1 覆盖 9,691 类，K=20 只剩 2,743 类），"
              "因此曲线偏高——它同时反映样本量增加与类别子集变简单两个因素，不能当作纯粹的样本量效应。")

    # ---------- 参考 ----------
    heading(doc, "参考文献", 1)
    refs = [
        "[1] Radford A, et al. Learning Transferable Visual Models From Natural Language Supervision. ICML 2021. arXiv:2103.00020",
        "[2] Yang A, et al. Chinese CLIP: Contrastive Vision-Language Pretraining in Chinese. arXiv:2211.01335, 2022.",
        "[3] Hu E J, et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022. arXiv:2106.09685",
        "[4] Gao P, et al. CLIP-Adapter: Better Vision-Language Models with Feature Adapters. arXiv:2110.04544, 2021.",
        "[5] Zhou K, et al. Learning to Prompt for Vision-Language Models (CoOp). IJCV 2022. arXiv:2109.01134",
        "[6] Bai Y, et al. Products-10K: A Large-scale Product Recognition Dataset. arXiv:2008.10545, 2020.",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.left_indent = Cm(0.8)
        p.paragraph_format.first_line_indent = Cm(-0.8)
        r = p.add_run(ref)
        set_font(r, size=9)

    # ---------- 附录 ----------
    heading(doc, "附录 A  复现命令", 1)
    cmds = [
        "# 零样本基线",
        "python code/muge_eval.py --split valid --batch-size 128 --save-feats",
        "# 缓存训练集特征（线性探针/Adapter 用）",
        "python code/build_feats.py --split train --batch-size 128",
        "# 线性探针 / Adapter",
        "python code/train_adapter.py --method linear  --fraction 1.0 --epochs 20",
        "python code/train_adapter.py --method adapter --fraction 1.0 --epochs 20 --hidden 128",
        "# 提示模板优化",
        "python code/template_eval.py",
        "# LoRA 训练",
        "python code/train_lora.py --max-steps 1200 --bs 32 --accum 1 --lr 1.5e-4 --out lora_1200step",
        "# 跨域评测",
        "python code/retrieval_eval.py --data-dir data/raw/Flickr30k-CN --split test --lora results/lora_1200step_ckpt",
        "# 效率分析与可视化",
        "python code/efficiency.py && python code/visualize.py",
    ]
    for c in cmds:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(c)
        r.font.name = MONO_FONT
        r.font.size = Pt(8.5)

    heading(doc, "附录 B  项目文件结构", 1)
    tree = [
        "期末大作业/",
        "├── 作业要求/            # 教师提供的 PPT 照片（原始资料）",
        "├── code/                # 全部可运行脚本（IDE 可直接打开）",
        "├── data/raw/            # MUGE、Flickr30k-CN、Products-10K 原始数据",
        "├── models/              # Chinese-CLIP 权重",
        "├── results/             # 实验结果 JSON、特征缓存、图表、样例图",
        "├── docs/实验记录.md      # 实验数据底稿与复现说明",
        "└── 交付/                # 技术报告与答辩材料（最终交付物）",
    ]
    for line in tree:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(line)
        r.font.name = MONO_FONT
        r.font.size = Pt(9)

    doc.save(str(OUT))
    print("saved:", OUT)


if __name__ == "__main__":
    main()
