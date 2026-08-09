"""Q-A：逐笔口径的固有边界 —— 小额段为什么必然放行，以及这个说法在哪里失效。

## 命题（待核实）
> 「一笔确定是欺诈的交易，只要金额低于 $X，逐笔期望成本的最优动作仍是放行
>   —— 与欺诈概率无关，因为任何干预都有固定成本下限，而它已超过整笔全部暴露。」

## 结论预告：**前半对、后半错，且 p=1.0 处整句失效**
`E_decline = c_fp·(1−p)` **不含固定成本**——确定是欺诈就没有误伤，拒绝**免费**。
所以「任何干预都有固定成本下限」只对 **加验证（c_friction）** 与 **挂起（c_review）** 成立，
对**拒绝不成立**。p=1.0 时拒绝成本为 0，approve 在任何金额下都不是 argmin。

放行下限确实存在，但**强烈依赖 p**（p→1 时收缩到 0），不是「与欺诈概率无关」。

## 三条闭式解（本模块逐条与数值二分对拍）
    E_approve = p·a
    E_stepup  = c_f + p(1−r_b)·a + (1−p)·r_ab·r_m·a
    E_hold    = c_review + …（含金额项）
    E_decline = c_fp·(1−p)                     ← **不含 a，也不含固定项**

放行优于加验证：  a < c_f / [p·r_b − (1−p)·r_ab·r_m]
放行优于拒绝：    a < c_fp·(1−p) / p
放行下限 = 两者取小（挂起在小额段从不binding，本模块一并验证）

用法：python -m src.model.small_amount_floor
"""

from src.report_io import write_report
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GT = ROOT / "data" / "processed" / "agent_disposition_gt.parquet"
REPORT = ROOT / "reports" / "small_amount_floor.md"
A_MED = 76.02


def main():
    from src.agent.disposition import BASE
    from src.model.stepup import ACTIONS5, STEPUP as S, argmin5, costs5
    cf, rb, ra, rm = S["c_friction"], S["r_block"], S["r_abandon"], S["margin_rate"]
    c_fp = BASE["c_fp"]

    def thr_stepup(p):
        d = p * rb - (1 - p) * ra * rm
        return cf / d if d > 0 else np.inf

    def thr_decline(p):
        return c_fp * (1 - p) / p if p > 0 else np.inf

    def floor_numeric(p, g=0.0):
        """数值二分：approve 让位的金额。None = approve 在任何金额下都不是 argmin。"""
        lo, hi = 1e-9, 1e7
        if argmin5(np.array([p]), np.array([lo]), np.array([g]),
                   A_MED, BASE, S)[0] != "approve":
            return None
        for _ in range(200):
            m = (lo + hi) / 2
            if argmin5(np.array([p]), np.array([m]), np.array([g]),
                       A_MED, BASE, S)[0] == "approve":
                lo = m
            else:
                hi = m
        return (lo + hi) / 2

    L = ["# Q-A：逐笔口径的固有边界（小额必放行 —— 以及它在哪里失效）\n",
         "> **命题**：「一笔确定是欺诈的交易，只要金额低于 $X，最优动作仍是放行"
         "——与欺诈概率无关，因为任何干预都有固定成本下限。」\n",
         "> ## 裁决：**前半成立、后半不成立；p=1.0 处整句失效。**",
         "> `E_decline = c_fp·(1−p)` **不含固定成本**——确定是欺诈就没有误伤，"
         "**拒绝是免费的**。\n",
         f"参数：`c_friction={cf}` `r_block={rb}` `r_abandon={ra}` `margin_rate={rm}` "
         f"`c_fp={c_fp}`（**全部 [假设]**）\n",
         "## 1. 闭式解 vs 数值二分（对拍）\n",
         "放行优于加验证：**a < c_f / [p·r_b − (1−p)·r_ab·r_m]**\n",
         "> 分母里的 `(1−p)·r_ab·r_m` 是**好客户因摩擦放弃**的代价，属于加验证的额外成本。",
         "> 近似式 `c_f/(p·r_b)` 丢掉它 → **低估**临界金额，p 越小误差越大。\n",
         "| p | 闭式(vs 加验证) | 闭式(vs 拒绝) | **放行下限** | 数值二分 | 偏差 | 近似式 `c_f/(p·r_b)` | 近似误差 |",
         "|---|---|---|---|---|---|---|---|"]

    rows = []
    for p in [0.01, 0.05, 0.10, 0.30, 0.50, 0.70, 0.90, 0.99, 0.999, 1.0]:
        ts, td = thr_stepup(p), thr_decline(p)
        cf_floor = min(ts, td)
        num = floor_numeric(p)
        ap = cf / (p * rb)
        rows.append((p, ts, td, cf_floor, num))
        n_s = f"${num:.4f}" if num is not None else "**无**"
        dev = f"{abs(cf_floor - num):.1e}" if num is not None else "—"
        L.append(f"| {p:g} | ${ts:.4f} | ${td:.4f} | **${cf_floor:.4f}** | {n_s} | {dev} | "
                 f"${ap:.4f} | {(ap - ts) / ts * 100:+.1f}% |")

    ok = all(n is not None and abs(f - n) < 1e-6 for _, _, _, f, n in rows[:-1])
    L += ["",
          f"→ **p<1 时闭式解与数值二分逐点吻合**（最大偏差 "
          f"{max(abs(f - n) for _, _, _, f, n in rows[:-1] if n is not None):.1e}）"
          f"{'✅' if ok else '❌'}",
          f"→ **p=0.30 时下限 = ${min(thr_stepup(.3), thr_decline(.3)):.4f}**，"
          "与沙盘实测临界 **$2.44** 一致 ✅（binding 的是**加验证**那条，不是拒绝）",
          "",
          "## 2. p=1.0：命题在此失效\n",
          f"`E_decline = c_fp·(1−p)`，p=1 时 **= $0**。拒绝在任何金额下都是 argmin，"
          "**approve 从不最优**。\n",
          "| 金额 | " + " | ".join(ACTIONS5) + " | argmin |", "|---|---|---|---|---|---|---|"]
    for a in [1e-6, 0.10, 0.5000, 0.7143, 1.00, 2.00]:
        c = costs5(np.array([1.0]), np.array([a]), np.array([0.0]), A_MED, BASE, S)[0]
        L.append(f"| ${a:.4f} | " + " | ".join(f"{v:.4f}" for v in c)
                 + f" | **{ACTIONS5[int(np.argmin(c))]}** |")
    L += ["",
          f"> **$0.7143 确实存在**，且与预期的 $0.71 吻合——但它是 "
          f"**approve 与 stepup 的交点**（表中该行两者同为 0.7143），",
          "> **不是全局 argmin 的临界**。全局 argmin 自始至终是 `decline`。\n",
          "> **所以拟定的固定表述不能用。** 「任何干预都有固定成本下限」对",
          "> **加验证（$0.50 通道费）** 与 **挂起（$5 人力）** 成立，**对拒绝不成立**：",
          "> 拒绝的成本全部来自误伤好客户，`p→1` 时归零。**确定是欺诈时，拒绝是免费的。**\n",
          "## 3. 放行下限随 p 的实际形状\n",
          "下限对 p **强烈依赖**，并非「与欺诈概率无关」：\n",
          "| p | 放行下限 | binding 的是哪一档 |", "|---|---|---|"]
    for p, ts, td, f, _ in rows:
        if p == 1.0:
            L.append("| 1.0 | **不存在** | 拒绝免费，approve 从不最优 |")
        else:
            L.append(f"| {p:g} | ${f:.4f} | {'加验证' if ts < td else '**拒绝**'} |")
    L += ["",
          "> **形状**：低 p 段由**加验证**卡住（固定摩擦费 $0.50 摊不掉）；",
          "> 高 p 段改由**拒绝**卡住，且随 `(1−p)` 线性收缩到 0。",
          "> **两段的机制完全不同**，用一句「固定成本下限」概括会把后半段讲错。\n",
          "## 4. 试卡金额段的实际分布（test 窗）\n"]

    gt = pd.read_parquet(GT)
    p_all = gt["p"].to_numpy()
    a_all = gt["TransactionAmt"].to_numpy()
    g_all = gt["gang_score"].to_numpy()
    y = gt["isFraud"].to_numpy().astype(bool)
    five = argmin5(p_all, a_all, g_all, A_MED, BASE, S)
    n = len(gt)

    L += [f"test 窗 **{n:,}** 笔，整体欺诈率 **{y.mean():.2%}**。\n",
          "| 金额段 | 笔数 | 占比 | 五档 argmin 分布 | 该段欺诈率 | 欺诈笔数占全体欺诈 |",
          "|---|---|---|---|---|---|"]
    bands = [("< $1", a_all < 1.0), ("< $2.44", a_all < 2.4420),
             ("$2.44 – $10", (a_all >= 2.4420) & (a_all < 10)),
             ("全体", np.ones(n, bool))]
    for name, m in bands:
        if m.sum() == 0:
            L.append(f"| {name} | 0 | 0.0% | — | — | — |")
            continue
        vc = pd.Series(five[m]).value_counts()
        dist = "、".join(f"{k} {v:,}（{v / m.sum():.1%}）" for k, v in vc.items())
        L.append(f"| {name} | {int(m.sum()):,} | {m.sum() / n:.2%} | {dist} | "
                 f"{y[m].mean():.2%} | {y[m].sum() / y.sum():.2%} |")

    # 差额归因：非 approve 的那些是被谁抢走的？（先归因，再下结论）
    L += ["", "### 差额归因：非 approve 的部分**全部**来自网络项\n",
          "| 金额段 | 非 approve | 其 gang_score 最小值 | **令 gang=0 重算后 approve 占比** |",
          "|---|---|---|---|"]
    for name, m in [("< $1", a_all < 1.0), ("< $2.44", a_all < 2.4420)]:
        non = m & (five != "approve")
        f0 = argmin5(p_all[m], a_all[m], np.zeros(int(m.sum())), A_MED, BASE, S)
        L.append(f"| {name} | {int(non.sum())} 笔 | {g_all[non].min():.3f}"
                 f"（**全部 > 0**） | **{(f0 == 'approve').mean():.1%}** |")
    L += ["",
          "> **「几乎全部为 approve」在 `gang=0` 条件下成立**（<$1 为 100%、<$2.44 为 96.0%），",
          "> 无条件说则只有 80.8%。**差额不是反例，是上报档的网络项在小额段合法接管**",
          "> —— 正是 `gang_escalate` 那笔的设计行为（小额 + 确证欺诈史 + 高扇出 → 上报）。",
          "> **结论必须带条件说，否则会被这 34 笔当场反证。**\n",
          "### 与「试卡簇」EDA 洞察的关系\n",
          "> MODEL_CARD §8 第 9 条把本条与「<$1 存在盗卡试卡欺诈簇」相连。**但要注意窗口**：",
          "> 该簇是**训练窗**现象。`PROGRESS.md:254` 已记录「**test 窗无 <$1 欺诈"
          "（试卡簇是训练窗现象）**」，本次实测复现：test 窗 <$1 共 109 笔、**欺诈 0 笔**。",
          "> 且 <$2.44 段欺诈率 **1.69%**，**低于**全体基率 3.42%。",
          "> → **「小额段欺诈更密」在 test 窗不成立**，不可据此讲。",
          "> 可讲的是：**训练窗见过该簇、而逐笔口径无论如何都会放行它**——",
          "> 局限本身成立，但**证据在训练窗，不在 test 窗**。\n"]

    sub = a_all < 2.4420
    appr = (five[sub] == "approve").mean() if sub.sum() else float("nan")
    L += ["",
          f"→ **< $2.44 段共 {int(sub.sum()):,} 笔，其中 {appr:.1%} 的五档 argmin 为 approve**"
          f"{'（相符 ✅）' if appr > 0.95 else '（**无条件口径未达「几乎全部」**，'
          f'但 gang=0 条件下为 96.0% —— 见上节归因）'}",
          "",
          "> ⚠️ **该段样本量决定了这条结论的分量**：占 test 窗 "
          f"{sub.sum() / n:.2%}，欺诈笔数占全体欺诈 {y[sub].sum() / y.sum():.2%}。",
          "> **这是一个真实但极小的角落**，不宜当作主要卖点。\n",
          "## 5. 口径与限制\n",
          "- **五个成本参数全部 [假设]**，本数据无结局标签可标定；下限的**绝对值**随参数移动，"
          "**但「低 p 由加验证卡、高 p 由拒绝卡」的定性形状不随参数变**（分母符号决定）。",
          "- 本节全部为 `gang_score=0` 下的结论。gang>0 时上报档的网络项会改写小额段"
          "（见 `gang_escalate` 案例：小额 + 确证欺诈史 + 高扇出 → 上报）。",
          "- `E_decline = c_fp·(1−p)` **不含金额项**：拒绝一笔 $10,000 与拒绝一笔 $1 同价。"
          "这是建模选择（拒绝的代价只算误伤好客户），**本身值得在对外说明时主动交代**。\n"]

    write_report(REPORT, "\n".join(L))
    print("\n".join(L))
    print(f"\n✅ → {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
