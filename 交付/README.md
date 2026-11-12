# 交付说明

本目录是最终交付物。生成脚本在 `code/` 下，改数据或改文案后重新运行即可再生成：

```bash
# 技术报告（Word，10 页）
python code/build_report.py

# 答辩 PPT（18 页，含 2 页备用页）
python code/build_ppt.py

# 第 12 周论文分享 PPT（10 页）
python code/build_paper_sharing.py

# 答辩准备手册（10 页，大白话版）
python code/build_defense_guide.py
```

脚本需用带 python-docx / python-pptx 的解释器运行：
`python`

## 交付物

| 文件 | 内容 | 用途 |
|---|---|---|
| 技术报告.docx | 10 页，11 张数据表 + 4 张图 + 参考文献 + 复现命令 | 第 15 周统一提交 |
| 答辩PPT.pptx | 18 页（16 页正文 + 2 页备用），每页带演讲备注 | 第 16—18 周现场答辩 |
| 论文分享.pptx | 10 页（CLIP / Chinese-CLIP / LoRA 三篇） | 第 12 周 Paper Sharing（7 分钟） |
| 答辩准备手册.docx | 10 页，逐页讲稿 + 20 个问答 + 救场话术 + 当天清单 | 自己准备、组员人手一份 |

## 时间线

| 周次 | 交什么 | 用哪份 |
|---|---|---|
| 第 12 周 | Paper Sharing（7 分钟） | 论文分享.pptx |
| 第 14 周 | Progress Demo（真实运行结果 + 基线指标 + 已改进 + 下一步） | 答辩PPT 的「主结果 / 消融 / 跨域」三页 + 现场演示 |
| 第 15 周 | 最终成果统一提交 | 技术报告.docx + 代码仓库 |
| 第 16—18 周 | 现场答辩（8 分钟汇报 + 2 分钟提问） | 答辩PPT.pptx + 答辩准备手册.docx |

## 现场演示（第 14 周 / 答辩用）

```bash
# 秒开版（用缓存特征，约 5 秒）
python code/demo_search.py --query "手机壳" --images 3000

# 多条查询
python code/demo_search.py --query "手机壳" "圣诞节抱枕" --topk 5

# LoRA 版（首次需重新编码约 3.5 分钟，演示不建议现场跑）
python code/demo_search.py --query "手机壳" --lora results/lora_1200step_ckpt
```

结果图输出在 `results/demo/`。**建议提前跑一遍并录屏 30 秒作为兜底**。

## 提交前需要你补的信息

1. **封面占位**：论文分享.pptx 第 1 页的「第 __ 组」、答辩PPT 第 1 页的「组别 / 组员 / 日期」。
2. 如需学院统一模板（封面、页眉、格式要求），套用后重新生成即可。
3. 答辩前按手册第 5 节排练两遍并计时。

## 数字来源

所有数字来自 `results/*.json`，汇总见 `docs/实验记录.md`。
报告与 PPT 的表格由脚本硬编码，若重跑实验需同步更新脚本里的数值。
