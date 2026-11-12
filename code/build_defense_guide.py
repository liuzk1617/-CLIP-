"""生成《答辩准备手册》docx —— 写给完全不懂的人，逐页讲稿 + 大白话问答。

运行：
  python code\\build_defense_guide.py
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
OUT = ROOT / "交付" / "答辩准备手册.docx"
FONT = "微软雅黑"

NAVY = (0x1F, 0x3B, 0x73)
ORANGE = (0xC0, 0x50, 0x2A)
GREY = (0x55, 0x55, 0x55)


def set_font(run, size=10.5, bold=False, color=None, name=FONT):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)


def shade(paragraph, fill="FFF3E0"):
    pPr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    pPr.append(shd)


def add_runs(p, text, size=10.5, bold=False, color=None):
    """支持 **加粗** 标记：按 ** 切分，奇数段加粗。"""
    parts = str(text).split("**")
    for i, part in enumerate(parts):
        if not part:
            continue
        r = p.add_run(part)
        set_font(r, size=size, bold=(bold or (i % 2 == 1)), color=color)

def para(doc, text, size=10.5, bold=False, color=None, indent=True, space=6, align=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space)
    if indent:
        p.paragraph_format.first_line_indent = Cm(0.74)
    if align:
        p.alignment = align
    add_runs(p, text, size=size, bold=bold, color=color)
    return p


def say(doc, text):
    """讲稿框：橙色底，代表"你要念出来的话"。"""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.left_indent = Cm(0.3)
    r = p.add_run("你说：")
    set_font(r, size=10, bold=True, color=ORANGE)
    r2 = p.add_run(text)
    set_font(r2, size=10.5)
    shade(p)
    return p


def tip(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.left_indent = Cm(0.3)
    r = p.add_run("忘词就说：")
    set_font(r, size=9, bold=True, color=GREY)
    r2 = p.add_run(text)
    set_font(r2, size=9.5, color=GREY)
    return p


def heading(doc, text, level=1):
    h = doc.add_heading(level=level)
    r = h.add_run(text)
    size = {1: 16, 2: 13, 3: 11.5}.get(level, 11)
    set_font(r, size=size, bold=True, color=NAVY)
    return h


def table(doc, headers, rows, caption=None, widths=None, size=9):
    if caption:
        c = doc.add_paragraph()
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = c.add_run(caption)
        set_font(r, size=9, bold=True)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(str(h))
        set_font(r, size=size, bold=True)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_runs(p, v, size=size)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    tr = t.rows[0]._tr
    trPr = tr.get_or_add_trPr()
    hdr = OxmlElement("w:tblHeader")
    hdr.set(qn("w:val"), "true")
    trPr.append(hdr)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    for s in doc.sections:
        s.top_margin = Cm(2.2)
        s.bottom_margin = Cm(2.0)
        s.left_margin = Cm(2.4)
        s.right_margin = Cm(2.4)

    # ===== 标题 =====
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("答辩准备手册")
    set_font(r, size=26, bold=True, color=NAVY)
    t2 = doc.add_paragraph()
    t2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = t2.add_run("从零开始，照着念就能讲完 8 分钟")
    set_font(r2, size=13, color=GREY)
    t3 = doc.add_paragraph()
    t3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r3 = t3.add_run("人工智能导论 · 期末大作业 · 方向五：多模态 AI 与 VLM")
    set_font(r3, size=10, color=GREY)

    # ===== 0 速查 =====
    heading(doc, "0  一页速查（打印这一页带进教室）", 1)
    para(doc, "你只需要记住下面这些东西。看不懂没关系，后面每一节都会解释。", size=10)
    table(doc,
          ["要记的", "内容"],
          [["8 分钟怎么分", "问题 1 分钟 → 方法 3 分钟 → 实验 3 分钟 → 演示 1 分钟"],
           ["必须背下来的 5 个数", "51.98 → 54.71（提升 2.73）；I2T 提升 4.92；\n只用 0.55% 参数（103 万）；显存 3.74GB、训练 6.3 分钟；复现误差 ≤0.12"],
           ["一句话讲完整个项目", "我们让一个中文图文模型更懂电商商品，只用千分之五的参数，就把「用文字找图」的准确率从 52% 提到 54.7%。"],
           ["被问「有什么创新」怎么答", "① 系统对比了四种轻量方法，得出「只有端到端微调才有效」的结论；② 给出了可复现的完整流水线；③ 诚实报告了负结果和评测局限。"],
           ["最容易被问倒的问题", "「为什么线性探针没用？」答案在 8.6：它只改最后一层，改不动图片那边的特征。"]],
          caption="表 0  速查卡", widths=[4.6, 11.8], size=9.5)

    # ===== 1 大白话讲项目 =====
    heading(doc, "1  我们到底做了什么（大白话版）", 1)
    para(doc, "想象你在一个巨大的仓库里买东西。仓库里有几万件商品，每件商品都有照片。"
              "顾客不会说“我要 SKU 编号 10086 的那件”，顾客只会说一句话，比如“金丝绒木床头罩”。"
              "我们要做的事情就是：**听到这句话，从几万张照片里把对的那张挑出来**。")
    para(doc, "这就是“图文检索”：用文字找图片。淘宝、京东的“拍照搜同款”“文字搜商品”背后都是这个技术。")
    para(doc, "难点在哪？三个：")
    para(doc, "① 顾客说的话太短。“男t丝光棉长袖”只有 7 个字，信息很少。", indent=True)
    para(doc, "② 长得像的东西太多。同一个牌子的两个商品，照片几乎一样。", indent=True)
    para(doc, "③ 冷门商品没人买、没几张图，模型学不到。这叫“长尾”。", indent=True)
    para(doc, "我们的做法：拿一个已经很聪明的模型（Chinese-CLIP），教它更懂我们的商品。"
              "“教”的方式很讲究——因为我们的电脑只有 8GB 显存，不能把整个模型重训一遍。", size=10.5)

    # ===== 2 三个核心概念 =====
    heading(doc, "2  三个核心概念，只要记三句话", 1)
    table(doc,
          ["概念", "一句话解释", "打个比方"],
          [["图文检索",
            "把图片和文字都变成一个“坐标点”，离得近就说明是一对",
            "把每张图、每句话都放到同一张地图上，找最近的邻居"],
           ["CLIP",
            "一个会“看图”和“读句子”的双胞胎模型，把两者放进同一张地图",
            "两个翻译官，都说同一种“地图语言”，所以能比较谁像谁"],
           ["Chinese-CLIP",
            "CLIP 的中文版，读得懂“丝光棉”“金丝绒”这种中文词",
            "会说中文的翻译官 —— 英文翻译官听不懂你的话"]],
          caption="表 1  三个概念", widths=[3.4, 7.4, 5.6], size=9.5)
    para(doc, "为什么要有“同一张地图”？因为只有在同一张地图上，我们才能量“这句话”和“这张图”离多远。"
              "这个距离越小，我们就认为越匹配。这就是检索的全部原理。")

    # ===== 3 我们试了什么 =====
    heading(doc, "3  我们试了五种方法（用五个比喻记住）", 1)
    table(doc,
          ["方法", "做法", "比喻", "效果"],
          [["零样本（基线）", "模型原样用，不训练", "不改任何东西，直接问翻译官", "52.0（起点）"],
           ["线性探针", "在最后加一层线性变换", "给翻译官配一个口音校正器", "没用（反而降到 50.1）"],
           ["轻量 Adapter", "在最后加一个小网络", "给翻译官加一个可调的小配件", "只对一半任务有用"],
           ["提示模板", "改提问的说法", "换一种问话方式，比如“这是…的图”", "只对一半任务有用"],
           ["**LoRA**", "在模型内部贴“便利贴”", "不重写整本书，只在页边贴小纸条", "**有效：54.71（+2.73）**"]],
          caption="表 2  五种方法速记", widths=[3.2, 4.8, 5.4, 4.2], size=9)
    para(doc, "为什么前三种没用？因为它们都只在“最后一步”动手，没有碰模型内部看图片的那部分。"
              "而真正的问题出在“看图片”这一侧——所以只有 LoRA（能改到内部）才有效。这是本项目最重要的一个发现。")

    # ===== 4 数字怎么读 =====
    heading(doc, "4  数字怎么读（每个数一句话）", 1)
    table(doc,
          ["数字", "什么意思", "怎么讲给老师听"],
          [["51.98 → 54.71", "R@1：用文字找图，第一名就对的概率", "从 52% 提到 54.7%，多对了 2.73 个点"],
           ["39.69 → 43.61", "I2T R@1：反过来，用图找文字，第一名就对的概率", "反方向提升更大，+4.92 个点"],
           ["0.55% / 1.03M", "只训练了 103 万个参数", "整个模型有 1.88 亿参数，我们只动了千分之五"],
           ["3.74GB / 6.3 分钟", "训练时占的显存和时间", "普通笔记本显卡就能跑，不需要服务器"],
           ["≤0.12", "和官方公布数字的差距", "说明我们的实验设置和官方一致，数字可信"]],
          caption="表 3  关键数字对照", widths=[3.4, 6.6, 6.4], size=9)

    # ===== 5 逐页讲稿 =====
    heading(doc, "5  8 分钟逐页讲稿（照着念就行）", 1)
    para(doc, "下面每一页都给了“你说：”的原文。括号里是大概用时。先对着 PPT 练三遍，"
              "再脱稿讲一遍，就够了。**不要背全文，记住每页第一句和最后一句**。")

    script = [
        (1, "封面（10 秒）",
         "各位老师好，我们组做的是方向五：多模态 AI 与 VLM。题目是《基于中文 CLIP 的电商图文检索》。"
         "8 分钟我按四部分讲：问题、方法、实验、演示。",
         "各位老师好，我们的题目是基于中文 CLIP 的电商图文检索。"),
        (2, "汇报结构（10 秒）",
         "先说问题——中文电商标题那么短，模型怎么找对图；再说方法——我们试了四种轻量办法；"
         "然后说实验——哪一种真的有效；最后花一分钟现场演示。",
         "我们按问题、方法、实验、演示四部分讲。"),
        (3, "问题（40 秒）",
         "大家看这个例子：顾客只输入“金丝绒木床头罩”七个字，我们要从几万张商品图里找对那一张。"
         "难点有三个：标题太短、同款太多、冷门商品没几张图。而且我们的设备只有一张 8GB 显存的笔记本显卡，"
         "不可能把整个模型重训一遍。",
         "用户只说几个字，我们要从几万张图里找对，而且设备很弱。"),
        (4, "原理（40 秒）",
         "我们的基础模型是 CLIP 的中文版。它的原理是：把图片和文字都变成同一张地图上的一个点，"
         "配对的两个点离得近，不配对的离得远。为什么用中文版而不是原版？因为原版的文字部分是英文的，"
         "它读不懂“丝光棉”这种中文词。",
         "CLIP 把图文放到同一张地图上，中文版才读得懂中文标题。"),
        (5, "四类方法（50 秒）",
         "因为显存不够，我们只做“轻量适配”，试了四种：线性探针，只在最后加一层；轻量 Adapter，"
         "在最后加一个小网络；提示模板，只改提问的说法；LoRA，在模型内部贴低秩的“便利贴”。"
         "四种都能解释、都能复现。这里有个坑：线性探针必须初始化为单位矩阵，否则会把模型原有的能力打散，"
         "我们实测准确率从 52 直接掉到 0.68。",
         "四种轻量方法，其中线性探针要特别小心初始化。"),
        (6, "LoRA 细节（40 秒）",
         "LoRA 是唯一改到模型内部的。它不重写整个权重，只训练一个很小的增量，"
         "所以只占 0.55% 的参数。训练完可以把这个增量“合并”回原模型，合并后和原模型完全一样，"
         "推理速度一点不慢。",
         "LoRA 只学一个很小的补丁，用完能贴回原模型里。"),
        (7, "实验设置（25 秒）",
         "数据用的是 MUGE，是国内电商场景的中文图文检索数据集，训练集 25 万图文对，验证集 5008 条查询、"
         "29806 张图。指标是 Recall@1/5/10，也就是正确答案排在前几名的概率。",
         "25 万训练对，5008 条查询，看正确答案排第几。"),
        (8, "基线复现（30 秒）",
         "在改进之前，我们先做了一件很多人会跳过的事：把官方的数字复现出来。"
         "我们得到 51.98，官方是 52.1，相差 0.12；换一个数据集 Flickr30K-CN 也一致。"
         "这说明我们的数据划分、预处理、指标统计都是对的，后面的比较才有意义。",
         "我们先复现了官方数字，误差 0.12，说明评测是对的。"),
        (9, "主结果（60 秒）",
         "这张表是核心。零样本是 51.98。线性探针掉到 50.1——它没用。Adapter 和模板只对一个方向有用。"
         "只有 LoRA 两个方向都涨：文字找图从 51.98 到 54.71，图找文字从 38.69 到 43.61。"
         "为什么？因为前三种只在最后一步动手，改不动“看图片”的那部分特征；LoRA 能改到内部。",
         "只有 LoRA 两个方向都涨，因为它改到了模型内部。"),
        (10, "种子与消融（45 秒）",
         "有人会问这 2.73 个点是不是运气。我们换了 3 个随机种子，结果是 55.20 加减 0.65，稳定。"
         "又试了不同的秩：r=4 是 55.15，r=8 是 54.71，r=16 是 54.91，几乎没有差别——"
         "说明收益来自“端到端更新”这个机制，不是参数堆出来的。训练 300 步就到 54.83，后面基本饱和。",
         "三个种子稳定，秩大小无关，300 步就饱和。"),
        (11, "跨域（30 秒）",
         "我们还担心一件事：只在电商数据上训练，会不会把模型的通用能力弄坏？"
         "于是拿开放域照片数据集 Flickr30K-CN 测试，结果没有下降，图找文字还提升了 3.4 个点。"
         "说明这种轻量适配不会造成灾难性遗忘。",
         "换到开放域考试，成绩没掉，反而略升。"),
        (12, "效率（30 秒）",
         "部署层面：LoRA 只多 0.55% 参数，合并回主干之后，参数量、显存、推理速度和原模型完全一样。"
         "换句话说，效果变好了，成本没变。",
         "效果变好，成本不变。"),
        (13, "失败案例（35 秒）",
         "我们也认真看了失败的例子。发现很多“错”其实是标注的问题：比如查“希雅茜”，模型找回了同品牌另一款产品，"
         "它抓对了品牌；查“方块笑脸卫衣”，正样本是粉色款，模型给了深色款。"
         "这说明我们的 R@1 是被低估的，也说明这个任务的评测本身有局限。",
         "失败案例里很多是同品牌近重复，说明指标被低估。"),
        (14, "Products-10K（30 秒）",
         "为了覆盖“商品分类”这个方向，我们额外做了 9691 类的商品细粒度分类。"
         "结果是整体 20.69%，但拆开看：最少的 20% 类别准确率是 0，最多的 20% 到 59.7%。"
         "瓶颈不是模型能力，是长尾数据太少。",
         "分类的瓶颈是长尾，不是模型。"),
        (15, "结论（30 秒）",
         "总结三点：第一，8GB 显卡上，用 0.55% 的参数就能拿到稳定提升；第二，冻结特征上做事后映射没用，"
         "必须改到模型内部；第三，这种适配不损害通用能力。",
         "三点结论：轻量有效、必须改内部、不伤通用能力。"),
        (16, "演示与复现（30 秒）",
         "最后是现场演示。代码都在仓库里，一条命令就能跑：输入一句中文，返回最像的五张图。"
         "汇报到这里，谢谢各位老师。",
         "一条命令就能跑，现在给大家演示一下。"),
    ]
    for idx, title, words, back in script:
        heading(doc, f"第 {idx} 页｜{title}", 2)
        say(doc, words)
        tip(doc, back)
    para(doc, "合计约 8 分钟。如果时间超了，优先砍第 7 页和第 14 页（各 25~30 秒）。"
              "如果时间不够、老师催，就跳过第 13 页的失败案例，直接讲结论。", size=9.5, color=GREY)

    # ===== 6 现场演示 =====
    heading(doc, "6  现场演示怎么做（含兜底方案）", 1)
    para(doc, "演示只有 1 分钟，务必要稳。三种方案，按可靠性从高到低准备：")
    table(doc,
          ["方案", "做法", "风险"],
          [["A. 提前录屏（推荐）", "提前用命令跑一遍，录 30 秒屏幕", "零风险，但不算“真跑”"],
           ["B. 现场跑缓存版", "python code/demo_search.py --query \"手机壳\" --images 3000", "约 5 秒出结果，风险低"],
           ["C. 现场跑 LoRA 版", "同上加 --lora results/lora_1200step_ckpt", "首次编码约 3.5 分钟，来不及"]],
          caption="表 4  演示三档方案", widths=[4.2, 8.6, 3.6], size=9)
    para(doc, "演示时说的话：“我输入‘手机壳’，系统在 3000 张商品图里返回最像的 5 张，这 5 张都是手机壳类商品。”"
              "如果结果不理想，就补一句：“这里只用了 3000 张候选图，完整验证集是 29806 张，指标是在全量上算的。”")

    # ===== 7 问答 =====
    heading(doc, "7  可能被问到的 20 个问题（大白话答案）", 1)
    para(doc, "答辩是“随机提问任一成员”，所以每个人都要能答。下面每条答案都只给 2~3 句，先背关键词，别背全文。")

    qa = [
        ("Q1 为什么选这个方向？",
         "因为多模态是现在最热的方向，而图文检索是它最落地的一个任务——淘宝、京东的搜索都在用。而且它在一个 8GB 的笔记本上就能做完，适合课程作业。"),
        ("Q2 什么是 CLIP？",
         "CLIP 是 OpenAI 2021 年的模型。它同时训练一个看图的和一个读句子的，把两者变成同一空间的向量，配对的距离近、不配对的远。这样就能用文字去找图。"),
        ("Q3 什么是对比学习？",
         "就是不告诉模型“这是什么类别”，只告诉它“这两张是不是一对”。是就拉近，不是就推远。靠大量配对数据自己学出结构。"),
        ("Q4 为什么不用英文 CLIP？",
         "英文 CLIP 的文字部分是英文词表，中文会被拆成看不懂的碎片。而我们的查询是中文标题，所以必须用中文预训练的版本。"),
        ("Q5 为什么不做全量微调？",
         "全量微调要把 1.88 亿参数全部更新，8GB 显存装不下，而且小数据集上容易把预训练能力学坏。所以只做轻量适配。"),
        ("Q6 LoRA 为什么有效？",
         "因为微调时权重的变化其实是低秩的，可以用两个小矩阵表示。它只学这个小增量，但改到了模型内部，所以能改变图像特征。"),
        ("Q7 为什么线性探针没用？",
         "因为它只作用在模型最后输出的特征上，相当于给已经算好的结果做一次线性变换，改不动“看图片”那部分的特征。而 T2I 的瓶颈恰恰在图像侧。"),
        ("Q8 为什么提示模板只对 I2T 有效？",
         "模板改的是文本侧的表达，文本作为“被检索对象”时，统一句式让文本特征更集中，所以图找文变好了；但文字找图的瓶颈在图片侧，改文本没用。"),
        ("Q9 提升 2.73 个点算多吗？",
         "在这个数据集上算明显：官方各规模模型的差距只有几个点，而我们只用 0.55% 的参数就拿到 2.73，并且三个种子稳定。"),
        ("Q10 你们怎么保证结果可信？",
         "三件事：先复现官方数字，误差 0.12 以内；三个随机种子取均值方差；所有配置、种子、算力都写进报告。"),
        ("Q11 MUGE 是什么？为什么用 valid 不用 test？",
         "MUGE 是国内电商场景的中文图文检索数据集。test 的标注是天池比赛保留的、没有公开，官方论文也是报 valid 的结果，所以我们用 valid 才能和官方对比。"),
        ("Q12 R@1 和 MR 是什么？",
         "R@1 是正确答案排在第一名就得分；MR 是 R@1、R@5、R@10 的平均，衡量整体排序质量。"),
        ("Q13 为什么 I2T 比 T2I 低？",
         "因为两个方向难度不同：文字找图时，图库有几万张相似的图；图找文字时，候选文本只有几千条。而且一条图对应多条描述，统计方式也不一样。"),
        ("Q14 跨域为什么没掉？",
         "因为我们只训练了很小的低秩增量，主干的通用能力基本没动。这也说明“轻量适配”是安全的。"),
        ("Q15 失败案例分析出什么？",
         "三类：同品牌近重复、超短标题歧义、同款不同色。其中很多是标注口径造成的误判，说明我们的 R@1 被低估了。"),
        ("Q16 你们的工作有什么创新？",
         "严格说不算方法创新，但有三点贡献：一是系统对比了四种轻量方法，得出“必须端到端更新主干才有效”的明确结论；二是给出了完整可复现的流水线；三是诚实报告了负结果和评测局限。"),
        ("Q17 训练用了多少资源？",
         "一张 RTX 5060 Laptop（8GB），训练峰值显存 3.74GB，1200 步 6.3 分钟；评测 0.99GB。全部在本地完成。"),
        ("Q18 合并是什么意思？",
         "LoRA 学的是一个增量矩阵，可以加回原来的权重里。合并后模型结构和原模型完全一样，所以推理不会变慢。"),
        ("Q19 Products-10K 为什么不能零样本？",
         "因为它只给了 SKU 编号，没有商品名称，没法写成文本类名。零样本分类需要文本标签当分类器权重，所以做不了。"),
        ("Q20 如果给你更多算力，你会做什么？",
         "① 做 LoRA 作用位置消融（只训视觉塔或只训文本塔）；② 上更大规模模型；③ 用多正样本标注重做评测，把 R@1 的低估问题解决掉。"),
    ]
    for q, a in qa:
        heading(doc, q, 3)
        para(doc, a)

    heading(doc, "7.1  如果被问到完全不会的问题，怎么救场", 2)
    para(doc, "不要硬编。用下面三句话之一，然后诚实收尾：")
    para(doc, "① “这个问题我们没有直接做实验，但从我们的结果推测，可能是因为……”（说一个和已有结论一致的推理）", indent=True)
    para(doc, "② “这部分不是我负责的模块，我们组的同学更清楚，可以请他补充。”", indent=True)
    para(doc, "③ “我们确实没考虑这个角度，这是下一步可以补的实验，谢谢老师的建议。”", indent=True)
    para(doc, "关键：**先承认，再给推理，不要瞎猜数据**。老师更在意你的思维，不是你能不能背出答案。")

    # ===== 8 分工 =====
    heading(doc, "8  组员分工与随机提问怎么准备", 1)
    table(doc,
          ["角色", "负责内容", "必须能回答"],
          [["数据与评测", "数据准备、测试集、指标统计", "Q11 Q12 Q13"],
           ["改进训练", "LoRA 实现、超参、消融", "Q5 Q6 Q7 Q17 Q18"],
           ["实验分析", "结果表、跨域、效率、失败案例", "Q9 Q14 Q15"],
           ["报告与展示", "报告、PPT、讲稿、演示", "Q1 Q2 Q3 Q4 Q16"]],
          caption="表 5  分工建议（按实际人数调整）", widths=[3.6, 7.0, 5.8], size=9)
    para(doc, "随机提问的应对原则：**先确认问题是问谁的模块，属于自己就答，不属于就主动把话递给负责的同学**，"
              "但不要一句都不说——老师要看到每个人都参与了。")

    # ===== 9 checklist =====
    heading(doc, "9  答辩当天清单", 1)
    items = [
        "提前 30 分钟到教室，把 PPT 拷到教室电脑并打开一次（字体、图片是否正常）",
        "U 盘里带一份 PPT + 一份 PDF 备份（PDF 不会掉字体）",
        "带充电器；笔记本电源模式设为“高性能”，避免 GPU 掉频",
        "演示脚本提前试跑，确认缓存特征文件存在（results/demo_img_feats_3000.npz）",
        "打印第 0 页速查卡和第 5 页讲稿，讲的时候可以放在手边",
        "手机静音；计时器设 8 分钟，讲第 9 页（主结果）时应该还剩 5 分钟",
        "答辩前一晚对着 PPT 完整讲两遍，每遍计时",
    ]
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        r = p.add_run(it)
        set_font(r, size=10)

    # ===== 10 术语表 =====
    heading(doc, "10  术语表（被问到英文缩写时用）", 1)
    table(doc,
          ["术语", "中文", "一句话"],
          [["CLIP", "图文对比预训练模型", "把图文放进同一空间的模型"],
           ["VLM", "视觉语言模型", "同时处理图像和文字的模型"],
           ["LoRA", "低秩适配", "只训练很小增量的微调方法"],
           ["Adapter", "适配器", "插在模型末端的小网络"],
           ["T2I / I2T", "文字检图 / 图检文字", "两个检索方向"],
           ["R@K", "Recall@K", "正确答案排前 K 名的比例"],
           ["MR", "Mean Recall", "R@1、R@5、R@10 的平均"],
           ["zero-shot", "零样本", "不训练直接用"],
           ["fine-tune", "微调", "在预训练模型上继续训练"],
           ["InfoNCE", "对比损失", "把正样本拉近、负样本推远的损失函数"]],
          caption="表 6  术语对照", widths=[3.0, 5.0, 8.2], size=9)

    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("最后一句话：老师不期待你完美，只期待你诚实、清楚、有自己的判断。")
    set_font(r, size=11, bold=True, color=ORANGE)

    doc.save(str(OUT))
    print("saved:", OUT)


if __name__ == "__main__":
    main()
