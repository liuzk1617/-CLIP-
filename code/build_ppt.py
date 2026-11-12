"""生成《答辩PPT》pptx —— 按 PPT 第 14 页建议结构：问题 1 / 方法 3 / 实验 3 / 演示 1 分钟。

运行：
  python code\\build_ppt.py
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Cm, Pt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "交付" / "答辩PPT.pptx"
FIGS = ROOT / "results" / "figs"
SAMPLES = ROOT / "results" / "samples"

NAVY = RGBColor(0x1F, 0x3B, 0x73)
ACCENT = RGBColor(0xE8, 0x71, 0x2C)
GREY = RGBColor(0x55, 0x55, 0x55)
LIGHT = RGBColor(0xF2, 0xF4, 0xF8)
FONT = "微软雅黑"

W, H = Cm(33.867), Cm(19.05)  # 16:9


def _font(run, size=18, bold=False, color=NAVY):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    # 中文字体
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find("{http://schemas.openxmlformats.org/drawingml/2006/main}ea")
    if ea is None:
        from lxml import etree
        ea = etree.SubElement(rPr, "{http://schemas.openxmlformats.org/drawingml/2006/main}ea")
    ea.set("typeface", FONT)


def add_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def header(slide, title, sub=None):
    bar = slide.shapes.add_shape(1, Cm(0), Cm(0), W, Cm(1.9))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    tb = slide.shapes.add_textbox(Cm(1.2), Cm(0.35), W - Cm(2.4), Cm(1.3))
    p = tb.text_frame.paragraphs[0]
    _font(p.add_run(), 24, True, RGBColor(0xFF, 0xFF, 0xFF))
    p.runs[0].text = title
    if sub:
        tb2 = slide.shapes.add_textbox(Cm(1.2), Cm(2.1), W - Cm(2.4), Cm(0.9))
        p2 = tb2.text_frame.paragraphs[0]
        r = p2.add_run()
        _font(r, 13, False, GREY)
        r.text = sub


def bullets(slide, items, top=Cm(3.3), left=Cm(1.6), width=None, size=18, gap=6):
    width = width or (W - Cm(3.2))
    tb = slide.shapes.add_textbox(left, top, width, H - top - Cm(1.2))
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for item in items:
        level = 0
        text = item
        if isinstance(item, tuple):
            level, text = item
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(gap)
        p.level = level
        r = p.add_run()
        _font(r, size if level == 0 else size - 3, level == 0,
              NAVY if level == 0 else GREY)
        r.text = ("● " if level == 0 else "– ") + text


def table(slide, headers, rows, top=Cm(3.4), left=Cm(1.4), width=None, size=13,
          col_ratio=None, row_h=Cm(1.1)):
    width = width or (W - Cm(2.8))
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), left, top, width,
                                   row_h * (len(rows) + 1))
    tbl = shape.table
    if col_ratio:
        total = sum(col_ratio)
        for i, ratio in enumerate(col_ratio):
            tbl.columns[i].width = int(width * ratio / total)
    for j, h in enumerate(headers):
        cell = tbl.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        _font(r, size, True, RGBColor(0xFF, 0xFF, 0xFF))
        r.text = str(h)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            cell = tbl.cell(i + 1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT if i % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            raw = str(v)
            bold = raw.startswith("**") and raw.endswith("**")
            _font(r, size, bold, ACCENT if bold else RGBColor(0x33, 0x33, 0x33))
            r.text = raw.strip("*") if bold else raw
    return tbl


def picture(slide, path: Path, left, top, width=None, height=None):
    if not path.exists():
        print(f"[warn] missing {path}")
        return None
    kw = {}
    if width:
        kw["width"] = width
    if height:
        kw["height"] = height
    return slide.shapes.add_picture(str(path), left, top, **kw)


def note(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H

    # 1 封面
    s = add_slide(prs)
    bg = s.shapes.add_shape(1, Cm(0), Cm(0), W, H)
    bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
    tb = s.shapes.add_textbox(Cm(2.4), Cm(5.4), W - Cm(4.8), Cm(3))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run(); _font(r, 34, True, RGBColor(0xFF, 0xFF, 0xFF))
    r.text = "基于中文 CLIP 的电商图文检索"
    tb2 = s.shapes.add_textbox(Cm(2.4), Cm(8.4), W - Cm(4.8), Cm(2))
    p2 = tb2.text_frame.paragraphs[0]
    r2 = p2.add_run(); _font(r2, 18, False, RGBColor(0xDD, 0xE3, 0xEF))
    r2.text = "线性探针 · 轻量 Adapter · LoRA 的系统对比"
    tb3 = s.shapes.add_textbox(Cm(2.4), Cm(13.4), W - Cm(4.8), Cm(2.6))
    for i, line in enumerate(["人工智能导论 · 期末大作业 · 方向五：多模态 AI 与 VLM",
                              "汇报组别 / 组员：____    日期：____"]):
        p3 = tb3.text_frame.paragraphs[0] if i == 0 else tb3.text_frame.add_paragraph()
        r3 = p3.add_run(); _font(r3, 13, False, RGBColor(0xAA, 0xB6, 0xCC)); r3.text = line
    note(s, "各位老师同学好，我们组的题目是《基于中文CLIP的电商图文检索》。8分钟汇报分四部分：问题1分钟、方法3分钟、实验3分钟、演示1分钟。")

    # 2 汇报结构
    s = add_slide(prs)
    header(s, "汇报结构（8 分钟）")
    bullets(s, [
        "① 问题：中文电商短标题检索难在哪（1 分钟）",
        "② 方法：四类轻量适配，为什么选 LoRA（3 分钟）",
        "③ 实验：主结果 / 消融 / 跨域 / 效率（3 分钟）",
        "④ 演示：真实检索结果与代码复现（1 分钟）",
    ], top=Cm(4.2), size=20, gap=14)
    note(s, "严格按建议的8分钟结构：问题1分钟、方法3分钟、实验3分钟、演示1分钟。")

    # 3 问题
    s = add_slide(prs)
    header(s, "问题：中文电商短标题检索", "用户只输入极短标题，要在数万商品图中找对图")
    bullets(s, [
        "查询极短且信息量低：「金丝绒木床头罩」「男t丝光棉长袖」",
        "同品牌/同系列商品视觉高度相似，极易混淆",
        "长尾明显：每个 SKU 图片数 5 ~ 87 张不等",
        "方向五要求：跑通图文模型 + 建测试集 + 可复现改进",
    ], top=Cm(3.4), width=Cm(17), size=16, gap=10)
    picture(s, SAMPLES / "p10k_0.jpg", Cm(19.2), Cm(4.2), height=Cm(9))
    note(s, "这是真实的电商商品图（京东数据）。商品图常带促销水印，同一 SKU 多张图仅有文案差异，这给检索带来近重复干扰。")

    # 4 原理
    s = add_slide(prs)
    header(s, "模型原理：为什么是 Chinese-CLIP", "双塔对比学习，把图像与文本映射到同一 512 维空间")
    bullets(s, [
        "CLIP：图像塔 + 文本塔，对称 InfoNCE 拉近正对、推远负对",
        "Chinese-CLIP：在中文图文对上预训练，原生理解「丝光棉」「金丝绒」等电商用语",
        "对比英文 CLIP：需先翻译标题，引入额外语义损失",
        "8GB 显存约束 → 不能全参数微调 → 只能做「轻量适配」",
    ], top=Cm(3.4), size=16, gap=10)

    # 5 方法
    s = add_slide(prs)
    header(s, "四类改进方法（均满足可解释、可复现）")
    table(s,
          ["方法", "改什么", "可训练参数", "动机"],
          [["线性探针", "末端 512×512 映射", "0.52M", "最小改动，检验特征是否够好"],
           ["轻量 Adapter", "瓶颈残差 f+W₂GELU(W₁f)", "0.26M", "非线性增量，避免破坏原表征"],
           ["提示模板", "只改查询文本包装", "0", "零成本，检验文本侧分布影响"],
           ["**LoRA**", "注意力 q/k/v/out 低秩增量", "**1.03M**", "**端到端更新主干，唯一改视觉表征**"]],
          top=Cm(3.2), size=13, col_ratio=[3, 5.4, 2.6, 4.4], row_h=Cm(1.5))
    bullets(s, ["关键实现细节：线性探针必须恒等初始化，否则随机映射会把预训练对齐打散（实测 R@1 52→0.68）"],
            top=Cm(12.4), size=14, gap=4)
    note(s, "四种方法都实现了。这里强调一个实现坑：线性探针随机初始化会崩溃，必须初始化为单位矩阵；Adapter 的上投影零初始化同理。")

    # 6 LoRA 细节
    s = add_slide(prs)
    header(s, "LoRA：为什么它能改视觉表征", "低秩增量注入注意力层，主干参与前向")
    bullets(s, [
        "在视觉塔与文本塔的 q/k/v/out 投影上注入低秩增量（r=8, α=16）",
        "可训练参数 1.03M，占全模型 0.55%，显存峰值 3.74GB",
        "训练：对称 InfoNCE，lr=1.5e-4，batch=32，1200 步，6.3 分钟",
        "部署：merge_and_unload() 合并后与基线完全等价",
    ], top=Cm(3.4), size=16, gap=10)
    note(s, "LoRA 的收益来自让主干参与前向，所以它能改图像侧表征；而线性探针和 Adapter 只是在终态特征上做变换，改不动视觉特征。")

    # 7 实验设置
    s = add_slide(prs)
    header(s, "实验设置：数据、指标与资源")
    table(s,
          ["数据集 / 划分", "规模", "用途"],
          [["MUGE train", "129,380 图 / 248,786 文本 / 250,314 对", "训练改进方法"],
           ["MUGE valid（官方评测集）", "29,806 图 / 5,008 文本", "主评测"],
           ["Flickr30k-CN test", "1,000 图 / 5,000 文本", "跨场景泛化"],
           ["Products-10K", "197,307 图 / 9,691 SKU", "商品细粒度分类（后续）"]],
          top=Cm(3.2), size=13, col_ratio=[4, 6.4, 4], row_h=Cm(1.4))
    bullets(s, ["指标：Recall@1/5/10 与 MR=(R1+R5+R10)/3，T2I 与 I2T 双向",
                "资源：RTX 5060 Laptop 8GB，PyTorch 2.11+cu128，峰值显存 3.74GB"],
            top=Cm(11.6), size=14, gap=6)

    # 8 基线复现
    s = add_slide(prs)
    header(s, "先验证口径：复现官方数字", "与官方公开值差 ≤0.5，说明评测协议一致")
    table(s,
          ["数据集 / 任务", "指标", "官方", "本项目", "差异"],
          [["MUGE T2I", "R@1", "52.1", "51.98", "−0.12"],
           ["MUGE T2I", "R@5", "76.7", "76.76", "+0.06"],
           ["MUGE T2I", "R@10", "84.4", "84.29", "−0.11"],
           ["Flickr30k-CN T2I", "R@1", "62.7", "62.30", "−0.40"],
           ["Flickr30k-CN I2T", "R@1", "74.6", "74.30", "−0.30"]],
          top=Cm(3.4), size=15, col_ratio=[5, 2.4, 2.6, 2.6, 2.4], row_h=Cm(1.25))
    note(s, "这一页是可信度基础：两个数据集都复现到官方±0.5以内，说明数据划分、预处理、tokenizer、指标口径全部对齐，后面的对比才有意义。")

    # 9 主结果
    s = add_slide(prs)
    header(s, "主结果：只有 LoRA 同时提升双向检索", "MUGE valid，%")
    table(s,
          ["方法", "参数", "T2I R@1", "T2I MR", "I2T R@1", "I2T MR"],
          [["零样本基线", "0", "51.98", "71.01", "38.69", "59.69"],
           ["线性探针 100%", "0.52M", "50.10", "70.49", "38.60", "61.49"],
           ["轻量 Adapter 100%", "0.26M", "51.40", "71.60", "40.14", "62.94"],
           ["提示模板", "0", "51.86", "71.05", "40.64", "62.37"],
           ["**LoRA（1200 步）**", "**1.03M**", "**54.71**", "**73.98**", "**43.61**", "**65.81**"]],
          top=Cm(3.2), size=14, col_ratio=[5, 2.6, 2.8, 2.8, 2.8, 2.8], row_h=Cm(1.3))
    bullets(s, ["冻结特征上做事后映射对 T2I 无效；LoRA 是唯一同时提升 T2I 与 I2T 的方法",
                "相对基线：T2I R@1 +2.73，I2T R@1 +4.92"],
            top=Cm(12.6), size=14, gap=6)
    note(s, "核心结论：线性探针和Adapter对T2I无效甚至下降，说明T2I瓶颈在图像侧表征；LoRA让主干参与前向，所以两个方向都涨。")

    # 10 种子与消融
    s = add_slide(prs)
    header(s, "可复现性与消融：收益来自机制而非参数量")
    bullets(s, [
        "3 个随机种子：T2I R@1 = 55.20 ± 0.65，I2T R@1 = 43.63 ± 0.17（波动远小于提升幅度）",
        "rank 消融：r=4→55.15、r=8→54.71、r=16→54.91，变化 <0.5 且无单调趋势",
        "训练量：300 步（约 5 分钟）即达 54.83，之后饱和 —— 收益低成本可得",
        "结论：该任务适配所需秩很低，r=4 用一半参数即可达到同等效果",
    ], top=Cm(3.4), size=15, gap=10)
    table(s,
          ["seed", "T2I R@1", "T2I MR", "I2T R@1"],
          [["0", "54.71", "73.98", "43.61"],
           ["1", "55.93", "74.33", "43.80"],
           ["2", "54.95", "74.01", "43.47"],
           ["**均值±标准差**", "**55.20 ± 0.65**", "**74.11 ± 0.19**", "**43.63 ± 0.17**"]],
          top=Cm(10.4), size=13, col_ratio=[4, 4, 4, 4], row_h=Cm(1.15))

    # 11 跨域
    s = add_slide(prs)
    header(s, "跨场景泛化：电商域训练 → 开放域测试", "Flickr30k-CN test")
    table(s,
          ["模型", "T2I R@1", "T2I MR", "I2T R@1", "I2T MR"],
          [["零样本基线", "62.30", "80.51", "74.30", "88.10"],
           ["**MUGE 训练的 LoRA**", "**62.94**", "**80.86**", "**77.70**", "**89.70**"]],
          top=Cm(3.6), size=15, col_ratio=[5, 2.8, 2.8, 2.8, 2.8], row_h=Cm(1.5))
    bullets(s, ["没有灾难性遗忘：T2I 持平略升，I2T R@1 提升 3.40",
                "说明窄域适配并未破坏模型的通用图文对齐能力"],
            top=Cm(10.2), size=16, gap=10)

    # 12 效率
    s = add_slide(prs)
    header(s, "效率：训练只加 0.55% 参数，部署零开销")
    table(s,
          ["项", "基线", "LoRA 未合并", "LoRA 合并回主干"],
          [["参数量", "188,262,913", "189,295,105", "**188,262,913（一致）**"],
           ["图像编码 bs=8", "603.6 img/s", "238.9 img/s", "**629.7 img/s**"],
           ["峰值显存", "0.54 GB", "0.54 GB", "0.54 GB"]],
          top=Cm(3.6), size=14, col_ratio=[4, 4, 4, 5], row_h=Cm(1.5))
    bullets(s, ["未合并时的吞吐下降是 PEFT 前向钩子造成，属训练态，不是部署代价",
                "merge_and_unload() 后参数量、时延、显存均回到基线"],
            top=Cm(11.2), size=15, gap=8)

    # 13 失败案例
    s = add_slide(prs)
    header(s, "失败案例与局限性", "三类典型失败模式，说明 R@1 被系统性低估")
    bullets(s, [
        "同品牌/同系列近重复（如「希雅茜」检索到同品牌另一款）",
        "超短标题歧义（如「荣光二」信息量过低）",
        "同款不同色（如「方块笑脸卫衣」粉→深）",
        "MUGE 每条查询平均约 6 张正样本，同品牌其他商品判错 → 绝对数值偏保守",
    ], top=Cm(3.4), width=Cm(15.5), size=14, gap=10)
    picture(s, FIGS / "failure_cases.png", Cm(18.6), Cm(2.3), height=Cm(13.4))
    note(s, "这页是局限性的诚实交代：失败案例里很多其实是同品牌不同商品，标注口径导致被判错，所以我们的R@1是低估的。")

    # 13.5 补充任务：Products-10K
    s = add_slide(prs)
    header(s, "补充任务：Products-10K 商品细粒度分类", "197,307 图 / 9,691 类 · 冻结特征 + 线性探针")
    bullets(s, [
        "全量线性探针：top-1 20.69% / top-5 35.99%（训练 12 秒）",
        "长尾：尾部 20% 的类 0%，头部 20% 达 59.7%",
        "少样本：K=1 → 15.1%，K=20 → 64.1%（曲线偏高）",
        "无类目名 → 无法零样本，仅做可分性分析",
    ], top=Cm(3.4), width=Cm(15.2), size=13, gap=8)
    picture(s, FIGS / "p10k_classify.png", Cm(16.6), Cm(4.6), width=Cm(16.2))
    note(s, "这是补充任务，用来覆盖方向五里的『视觉语言商品分类』。核心发现是长尾：尾部类几乎学不会，头部类能到60%。")

    # 14 结论
    s = add_slide(prs)
    header(s, "结论与后续工作")
    bullets(s, [
        "在 8GB 显存约束下，LoRA 以 0.55% 参数取得 T2I R@1 +3.22、I2T R@1 +4.94 的稳定增益",
        "线性探针/Adapter 在冻结特征上对 T2I 无效；提示模板只对 I2T 有效",
        "跨域迁移无灾难性遗忘；合并后部署零开销",
        "后续：训练量曲线、Products-10K 商品细粒度分类、多正样本评测口径",
    ], top=Cm(3.6), size=17, gap=14)

    # 15 演示与复现
    s = add_slide(prs)
    header(s, "演示与复现：一条命令跑通", "代码结构清晰，可在 IDE 中直接打开")
    bullets(s, [
        "code/muge_eval.py：零样本基线（复现官方）",
        "code/train_adapter.py：线性探针 / Adapter（秒级）",
        "code/train_lora.py：LoRA 训练（6.3 分钟）",
        "code/retrieval_eval.py：跨域评测；code/visualize.py：案例可视化",
        "演示预案：现场跑一条零样本查询 + 展示 LoRA 前后 Top-1 变化",
    ], top=Cm(3.4), size=15, gap=10)
    note(s, "演示环节：如果现场网络/GPU可用，直接跑一条真实检索；否则播放提前录好的 30 秒短片。")

    # 17 备用页 1：关键数字速查
    s = add_slide(prs)
    header(s, "备用页 1：关键数字速查（不计入 8 分钟）")
    table(s,
          ["常被问到的", "回答用数字"],
          [["提升多少", "T2I R@1 51.98 → 54.71（+2.73）；I2T R@1 38.69 → 43.61（+4.92）"],
           ["成本多少", "可训练参数 1.03M（0.55%）；训练显存 3.74GB；1200 步 6.3 分钟"],
           ["稳不稳", "3 个种子：T2I R@1 55.20 ± 0.65；rank 4/8/16 变化 <0.5"],
           ["可信吗", "复现官方误差 ≤0.12（MUGE）与 ≤0.4（Flickr30K-CN）"],
           ["部署代价", "合并后参数量 188,262,913，与基线完全一致，吞吐 629.7 img/s"]],
          top=Cm(3.4), size=13, col_ratio=[3, 9], row_h=Cm(1.55))
    note(s, "备用页，正常汇报不讲。被问到具体数字时切过来指给学生看。")

    # 18 备用页 2：为什么只有 LoRA 有效
    s = add_slide(prs)
    header(s, "备用页 2：为什么只有 LoRA 有效（原理对比）")
    table(s,
          ["方法", "作用位置", "能否改变图像特征", "实测结果"],
          [["线性探针", "模型输出的 512 维特征之后", "不能", "T2I 51.98 → 50.10（下降）"],
           ["轻量 Adapter", "同上（末端残差）", "不能", "T2I 51.40，I2T 略升"],
           ["提示模板", "文本侧输入", "不能", "只对 I2T 有效"],
           ["**LoRA**", "**视觉/文本塔注意力层内部**", "**能**", "**T2I +2.73、I2T +4.92**"]],
          top=Cm(3.4), size=13, col_ratio=[2.6, 5.2, 3.4, 4.4], row_h=Cm(1.7))
    bullets(s, ["结论：T2I 的瓶颈在图像侧表征，只有让主干参与前向（LoRA）才能改到它"],
            top=Cm(13.2), size=14)
    note(s, "备用页。如果老师追问『为什么其他方法没用』，切到这一页，重点讲『作用位置』这一列。")

    prs.save(str(OUT))
    print("saved:", OUT, "| slides:", len(prs.slides))


if __name__ == "__main__":
    main()
