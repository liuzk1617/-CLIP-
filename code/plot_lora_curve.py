"""训练量曲线：LoRA 步数对检索指标的影响（对应 Level 3「不同训练量」实验）。

数据点来自 results/lora_{300,600,1200,3000}step.json（均为 seed=0、bs=32、lr=1.5e-4）
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

from muge_eval import RESULT

FIG = RESULT / "figs"
STEPS = [300, 600, 1200, 3000]
BASELINE_T2I_R1 = 51.98


def main() -> None:
    rows = []
    for s in STEPS:
        p = RESULT / f"lora_{s}step.json"
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        rows.append({
            "steps": s,
            "t2i_r1": d["t2i"]["R@1"],
            "t2i_mr": d["t2i_MR"],
            "i2t_r1": d["i2t"]["R@1"],
            "i2t_mr": d["i2t_MR"],
            "sec": d["train_sec"],
        })

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    xs = [r["steps"] for r in rows]
    axes[0].plot(xs, [r["t2i_r1"] for r in rows], "o-", color="#E8712C", label="T2I R@1")
    axes[0].plot(xs, [r["i2t_r1"] for r in rows], "s-", color="#4C72B0", label="I2T R@1")
    axes[0].axhline(BASELINE_T2I_R1, ls="--", color="#999999", label="T2I 零样本基线 51.98")
    axes[0].set_xscale("log")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([str(x) for x in xs])
    axes[0].xaxis.set_minor_formatter(mticker.NullFormatter())
    axes[0].set_xlabel("训练步数（log）")
    axes[0].set_ylabel("R@1 (%)")
    axes[0].set_title("训练量曲线：R@1")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].plot(xs, [r["t2i_mr"] for r in rows], "o-", color="#E8712C", label="T2I MR")
    axes[1].plot(xs, [r["i2t_mr"] for r in rows], "s-", color="#4C72B0", label="I2T MR")
    axes[1].axhline(71.01, ls="--", color="#999999", label="T2I 基线 MR 71.01")
    axes[1].set_xscale("log")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([str(x) for x in xs])
    axes[1].xaxis.set_minor_formatter(mticker.NullFormatter())
    axes[1].set_xlabel("训练步数（log）")
    axes[1].set_ylabel("MR (%)")
    axes[1].set_title("训练量曲线：平均召回 MR")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "lora_curve.png"
    fig.savefig(out, dpi=150)
    print("saved:", out)
    for r in rows:
        print(f"  {r['steps']:5d} 步  T2I R@1={r['t2i_r1']:.2f}  T2I MR={r['t2i_mr']:.2f}  "
              f"I2T R@1={r['i2t_r1']:.2f}  I2T MR={r['i2t_mr']:.2f}  {r['sec']:.0f}s")


if __name__ == "__main__":
    main()
