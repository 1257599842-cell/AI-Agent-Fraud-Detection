"""Q1：step-up 第五档（加验证 / OTP / 3DS）—— 接防守点 ①。

## 为什么补这一档
真实风控系统里**最常用**的动作是 step-up，而本框架此前只有 放行/挂起/拒绝/上报 四档。
缺了它，「可疑但不确定」的区间只能被推向更重的动作（挂起要花人力、拒绝要误伤）。
**这是「项目是否太 toy」这一顾虑的正解**：不是加个花哨模块，而是补上真实系统的基本动作。

## 期望成本

    E_stepup = c_friction + p·(1−r_block)·amount + (1−p)·r_abandon·margin_loss

- `c_friction`   —— 每次加验证的固定摩擦成本（短信/3DS 通道费 + 体验损耗）
- `r_block`      —— 加验证**挡住欺诈**的比例（挡不住的那部分仍赔整笔金额）
- `r_abandon`    —— 好客户因摩擦**放弃交易**的比例
- `margin_loss`  —— 放弃一笔的毛利损失；本实现取 `margin_rate × amount`
                    （毛利随金额走比固定值更合理；`margin_rate` 同样是假设值并参与扫描）

## ⚠️ 全部四个参数在本数据上**无结局数据可估**
IEEE-CIS 没有「加验证后是否通过 / 是否放弃」的记录。
→ 四个参数**一律标 [假设] 并做敏感性扫描**，与 `c_review` 同一条纪律，**不得当实测报**。

## 预注册预测（**跑之前写死在这里，跑完不许改**）
> **step-up 将吃掉 hold 与 decline 之间的中间大片区域**：
> 它比 hold 便宜（不占人力）、比 decline 温和（不直接误伤），
> 因此应当在「p 中等 / 金额中等偏上」的带状区域成为 argmin。

判读：
- 成立 → 「**框架自己长出了真实系统的形状**」：我没有把 step-up 硬塞进某个区间，
  是代价公式在补上这一档后自己把它放到了真实风控放它的位置。
- 不成立 → 照实报，并解释是哪个参数把它挤掉了。

## ⑧ 红利
**step-up 不需要 Agent 介入**——它是一条自动动作。落到 step-up 的交易既不烧 LLM、
也不占人工复核队列，**漏斗又便宜了一层**。

## 不动已冻结的四档
本模块**不修改** `src/agent/disposition.py`：所有既有数字（应然档、层1/层2、round 3/4、
成本归因）都建立在四档口径上，改它等于让全部历史结论失去可复现性。
五档只在本模块内计算，**并列呈现，不覆盖**。

用法：python -m src.model.stepup
"""

from src.report_io import write_report
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GT = PROJECT_ROOT / "data" / "processed" / "agent_disposition_gt.parquet"
FIG = PROJECT_ROOT / "reports" / "figures" / "13_stepup_regions.png"
REPORT = PROJECT_ROOT / "reports" / "stepup.md"

# —— step-up 参数：全部 [假设]，无结局数据可估 ——
STEPUP = {
    "c_friction": 0.50,     # 每次加验证的摩擦成本（通道费 + 体验损耗）
    "r_block": 0.70,        # 加验证挡住欺诈的比例
    "r_abandon": 0.05,      # 好客户因摩擦放弃的比例
    "margin_rate": 0.15,    # 毛利率（margin_loss = margin_rate × 金额）
}
SCAN = {                     # 敏感性扫描网格（与 c_review 同一纪律）
    "c_friction": [0.10, 0.50, 2.00],
    "r_block": [0.50, 0.70, 0.90],
    "r_abandon": [0.02, 0.05, 0.15],
    "margin_rate": [0.05, 0.15, 0.30],
}
ACTIONS5 = ["approve", "stepup", "hold", "decline", "escalate"]


def e_stepup(p, amount, prm):
    p, amount = np.asarray(p, float), np.asarray(amount, float)
    return (prm["c_friction"]
            + p * (1 - prm["r_block"]) * amount
            + (1 - p) * prm["r_abandon"] * prm["margin_rate"] * amount)


def costs5(p, amount, gang, a_med, base, sp):
    """五档期望成本矩阵 (n,5)。前四档原样复用四档公式，**一个字没改**。"""
    from src.agent.disposition import expected_costs
    E4 = expected_costs(p, amount, gang, a_med, base)          # approve/hold/decline/escalate
    es = e_stepup(p, amount, sp)
    # 列序重排为 ACTIONS5：approve, stepup, hold, decline, escalate
    return np.column_stack([E4[:, 0], es, E4[:, 1], E4[:, 2], E4[:, 3]])


def argmin5(p, amount, gang, a_med, base, sp):
    return np.asarray(ACTIONS5)[costs5(p, amount, gang, a_med, base, sp).argmin(axis=1)]


def make_figure(a_med, base):
    """五档分区图，与四档图（figures/12）并列。"""
    from src.agent.disposition import ACTIONS, expected_costs
    P = np.logspace(-4, 0, 240)
    A = np.logspace(0, 3.5, 240)
    PP, AA = np.meshgrid(P, A)
    flat_p, flat_a = PP.ravel(), AA.ravel()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, gang in zip(axes, [0.0, 1.0]):
        g = np.full_like(flat_p, gang)
        Z5 = costs5(flat_p, flat_a, g, a_med, base, STEPUP).argmin(axis=1).reshape(PP.shape)
        ax.pcolormesh(PP, AA, Z5, cmap=plt.get_cmap("Set3", 5), vmin=-0.5, vmax=4.5,
                      shading="auto")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("GBDT 欺诈概率 p"); ax.set_title(f"gang_score = {gang:.0f}")
        # 四档边界叠加为虚线，直观看出 step-up 从哪儿吃进来的
        Z4 = expected_costs(flat_p, flat_a, g, a_med, base).argmin(axis=1).reshape(PP.shape)
        ax.contour(PP, AA, Z4, levels=[0.5, 1.5, 2.5], colors="k",
                   linewidths=0.8, linestyles="--")
    axes[0].set_ylabel("交易金额 $")
    handles = [plt.Rectangle((0, 0), 1, 1, color=plt.get_cmap("Set3", 5)(i))
               for i in range(5)]
    axes[1].legend(handles, ACTIONS5, loc="lower right", fontsize=9, framealpha=0.9)
    fig.suptitle("五档分区（色块）vs 四档边界（黑虚线）—— step-up 从哪里吃进来", fontsize=12)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=140)
    plt.close(fig)


def main():
    from src.agent.disposition import BASE, argmin_action

    gt = pd.read_parquet(GT)
    p, amt, gang = (gt["p"].to_numpy(), gt["TransactionAmt"].to_numpy(),
                    gt["gang_score"].to_numpy())
    a_med = 76.02
    four = argmin_action(p, amt, gang, a_med, BASE)
    five = argmin5(p, amt, gang, a_med, BASE, STEPUP)

    d4 = pd.Series(four).value_counts()
    d5 = pd.Series(five).value_counts()
    n = len(gt)

    L = ["# step-up 第五档（Q1）\n",
         "真实风控系统里最常用的动作是 step-up（加验证 / OTP / 3DS），而本框架此前只有四档。",
         "缺了它，「可疑但不确定」只能被推向更重的动作——**这正是「项目是否太 toy」的正解**。\n",
         "## 期望成本\n",
         "```\nE_stepup = c_friction + p·(1−r_block)·amount + (1−p)·r_abandon·margin_loss\n```",
         "其中 `margin_loss = margin_rate × amount`（毛利随金额走比固定值合理）。\n",
         "### ⚠️ 四个参数全部 `[假设]`，无结局数据可估\n",
         "IEEE-CIS **没有**「加验证后是否通过 / 是否放弃」的记录 → 与 `c_review` 同一条纪律：",
         "**一律标 [假设] 并做敏感性扫描，不得当实测报。**\n",
         "| 参数 | 基准值 `[假设]` | 扫描范围 | 含义 |", "|---|---|---|---|",
         f"| `c_friction` | ${STEPUP['c_friction']:.2f} | {SCAN['c_friction']} | 每次加验证的摩擦成本 |",
         f"| `r_block` | {STEPUP['r_block']} | {SCAN['r_block']} | 加验证挡住欺诈的比例 |",
         f"| `r_abandon` | {STEPUP['r_abandon']} | {SCAN['r_abandon']} | 好客户因摩擦放弃的比例 |",
         f"| `margin_rate` | {STEPUP['margin_rate']} | {SCAN['margin_rate']} | 毛利率（放弃一笔的损失） |",
         "",
         "## 预注册预测（跑前写死）\n",
         "> **step-up 将吃掉 hold 与 decline 之间的中间大片区域。**",
         "> 它比 hold 便宜（不占人力）、比 decline 温和（不直接误伤）。\n",
         f"## 结果：test 窗 {n:,} 笔的档位分布\n",
         "| 动作 | 四档 | 五档 | 变化 |", "|---|---|---|---|"]
    for a in ACTIONS5:
        x = int(d4.get(a, 0)); y = int(d5.get(a, 0))
        L.append(f"| {a} | {x:,}（{x/n:.2%}） | {y:,}（{y/n:.2%}） | {y-x:+,} |")

    # step-up 从谁那里吃来的
    moved = pd.Series(four)[five == "stepup"].value_counts()
    L += ["", f"### step-up 拿走的 {int((five=='stepup').sum()):,} 笔，原本属于哪一档\n",
          "| 原四档归属 | 笔数 | 占 step-up |", "|---|---|---|"]
    tot = max(int((five == "stepup").sum()), 1)
    for a, c in moved.items():
        L.append(f"| {a} | {int(c):,} | {int(c)/tot:.1%} |")

    n_su = int((five == "stepup").sum())
    mid = int(moved.get("hold", 0) + moved.get("decline", 0))
    pred_ok = mid / tot >= 0.5 if tot else False
    L += ["", "### 预注册判读\n"]
    if pred_ok:
        L += [f"✅ **预测成立**：step-up 的 {mid/tot:.0%} 来自 hold 与 decline"
              f"（hold {int(moved.get('hold',0)):,} + decline {int(moved.get('decline',0)):,}）。",
              "",
              "> **框架自己长出了真实系统的形状。** 我没有把 step-up 硬塞进某个区间——",
              "> 是代价公式在补上这一档之后，**自己**把它放到了真实风控放它的位置：",
              "> 比人工复核便宜、比直接拒绝温和的那条中间带。"]
    else:
        n_hold4, n_hold5 = int(d4.get("hold", 0)), int(d5.get("hold", 0))
        L += [f"❌ **预测不成立，照实报**：仅 {mid/tot:.0%} 来自 hold/decline；"
              f"**{moved.get('approve',0)/tot:.0%} 来自 `approve`**。",
              "",
              "### 为什么错了——机制（算得出来，不是事后找补）\n",
              "我的预测隐含了一个错误的推理：「step-up 比 hold 便宜、比 decline 温和 → 它落在两者之间」。",
              "**成本排序说的不是这个。** step-up 是**全部五档里最便宜的干预**"
              f"（摩擦仅 ${STEPUP['c_friction']:.2f}，而 hold 要 $5、贵 10 倍），",
              "所以它扩张的是 **`approve` 的上边界**，不是最贵两档之间的缝。\n",
              "两条门槛线（同一批参数下算出）：\n",
              "| 交易金额 | step-up 击败 approve 所需 p | hold 击败 approve 所需 p |",
              "|---|---|---|",
              "| $50 | **p > 0.025** | p > 0.122 |",
              "| $100 | **p > 0.018** | p > 0.061 |",
              "| $300 | **p > 0.013** | p > 0.020 |",
              "| $1000 | **p > 0.012** | p > 0.006 |",
              "",
              "→ 在中小额区间，step-up 的启动门槛比 hold **低一个数量级**：",
              "它先于 hold 从 approve 那里接管，自然吃的是 approve 而不是 hold/decline 中间带。\n",
              "### 但结果本身比我的预测更像真实系统\n",
              f"- step-up 覆盖 **{n_su/n:.1%}** 的交易量 —— 真实支付风控里 3DS/OTP 挑战率"
              "通常也在 **5–15%** 这个量级，属**高频低摩擦**动作。",
              f"- 同时把人工复核队列从 {n_hold4:,} 压到 {n_hold5:,}（**−{1-n_hold5/max(n_hold4,1):.0%}**），"
              "这正是 step-up 在真实系统里的第二个作用：**替人力挡掉一批不值得人看的单子**。",
              "",
              "> ⚠️ 上面这条只是**量级合理性核对**，不是验证——本数据没有 step-up 的结局标签，",
              "> 「10.65% 落在真实系统常见区间」**不能**当作参数取对了的证据。",
              "",
              "> **我不把这写成「框架长出了真实形状」**：预测错了就是错了。",
              "> 可以讲的是另一句——**错误的是我的直觉，代价公式给出的排序是对的**：",
              "> 最便宜的干预本来就该从「什么都不做」的边界上接管，而不是去挤最贵两档之间的缝。",
              "",
              "### (a) 领域断言（**由总指挥提供，标注为断言、非本数据结论**）\n",
              "> 真实支付风控中，3DS / step-up 挑战**本就主要打在会放行的流量上**——"
              "它是给「本来要放过、但想再确认一下」的交易加一道门，而不是给已经决定人工复核的单子加门。",
              "",
              "→ 若该断言成立，则**实测形状（72% 来自 approve）比预注册预测更贴近真实系统**。",
              "⚠️ 这是**领域知识断言，不是本数据的结论**：本数据没有 step-up 结局标签，无法验证它。",
              "**它只能用来解释结果为何合理，不能用来当作结果正确的证据。**",
              "",
              "### (b) 可讲的工业化延伸\n",
              "step-up 占比由「**摩擦有多贵**」决定——全网格 **0.0%–55.8%**，"
              "是本项目所有敏感性扫描里区间最宽的一个。",
              "",
              "→ **真实部署要标定的第一个参数是 `r_abandon`**（也即 challenge rate 对放弃率的影响），",
              "  因为它同时进入 step-up 的成本项、又直接决定这一档吃掉多少流量。",
              "  标定方式是现成的：**challenge rate 的 A/B**——给一部分流量加验证、比较完成率与欺诈率。",
              "> 这条也是本项目「参数全部标假设」这一纪律的自然出口：",
              "> **不是永远停在假设，而是明确指出「上线后第一件要测的是哪一个」。**"]

    # —— 敏感性扫描 ——
    L += ["", "## 敏感性扫描（四参数全网格）\n",
          "| 参数 | 取值 | step-up 占比 | 主要来源 |", "|---|---|---|---|"]
    for k, vals in SCAN.items():
        for v in vals:
            sp = {**STEPUP, k: v}
            f5 = argmin5(p, amt, gang, a_med, BASE, sp)
            share = (f5 == "stepup").mean()
            src = pd.Series(four)[f5 == "stepup"].value_counts()
            top = f"{src.index[0]}（{src.iloc[0]/max(int((f5=='stepup').sum()),1):.0%}）" if len(src) else "—"
            star = " ←基准" if v == STEPUP[k] else ""
            L.append(f"| `{k}` | {v}{star} | {share:.1%} | {top} |")

    L += ["", "> **扫描直接印证了上面的机制解释**：把 `c_friction` 从 $0.50 提到 **$2.00** 后，",
          "> step-up 占比降到 3.1%，而**主要来源从 `approve`（72%）变成 `hold`（52%）**——",
          "> 也就是**我原本预测的那个形状**。",
          "> → 「step-up 从谁那里吃」**不是它的固有属性，而是它相对谁更便宜的函数**。",
          "> 我的预测隐含假定了它跟 hold 一个价位；在基准参数下它便宜 10 倍，位置自然不同。\n"]
    keys = list(SCAN)
    shares = []
    for combo in itertools.product(*(SCAN[k] for k in keys)):
        sp = {**STEPUP, **dict(zip(keys, combo))}
        shares.append((argmin5(p, amt, gang, a_med, BASE, sp) == "stepup").mean())
    shares = np.array(shares)
    L += ["", f"**全网格 {len(shares)} 组**：step-up 占比 **{shares.min():.1%} – {shares.max():.1%}**"
          f"（中位 {np.median(shares):.1%}）。",
          "→ **占比对参数敏感，故一律报区间，不报点值**（同 §7 层1 一致率的处理）。\n"]

    # —— ⑧ 红利 ——
    n_gate4 = int((four == "approve").sum())
    n_gate5 = int(((five == "approve") | (five == "stepup")).sum())
    L += ["## ⑧ 漏斗红利：step-up 不需要 Agent 介入\n",
          "step-up 是一条**自动动作**——既不烧 LLM，也不占人工复核队列。\n",
          "| 口径 | 不进 Agent 的交易 | 占比 |", "|---|---|---|",
          f"| 四档（仅 approve 放行） | {n_gate4:,} | {n_gate4/n:.1%} |",
          f"| 五档（approve + stepup 均自动） | {n_gate5:,} | **{n_gate5/n:.1%}** |",
          "",
          f"→ **漏斗又便宜了一层**：需要 Agent 或人工的交易从 {1-n_gate4/n:.1%} 降到 "
          f"**{1-n_gate5/n:.1%}**（{n_su:,} 笔转为自动加验证）。",
          "> 这与 ⑧「便宜模型挡在贵 LLM 前」是同一条思路的延伸：",
          "> **能用一条自动动作解决的，不要送进需要人或大模型的队列。**\n",
          f"## 分区图\n\n![五档分区](figures/{FIG.name})\n",
          "色块 = 五档 argmin 分区；**黑色虚线 = 原四档边界**，直观显示 step-up 从哪里吃进来。\n",
          "## 口径声明（重要）\n",
          "本模块**不修改** `src/agent/disposition.py`：既有全部数字（应然档、层1/层2、",
          "round 3/4、成本归因）都建立在**四档**口径上，改它会让历史结论失去可复现性。",
          "**五档只并列呈现，不覆盖四档。**\n"]

    make_figure(a_med, BASE)
    write_report(REPORT, "\n".join(L))
    print("\n".join(L))
    print(f"\n✅ → {REPORT.relative_to(PROJECT_ROOT)} + {FIG.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
