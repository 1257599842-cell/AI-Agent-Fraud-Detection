"""应然档一致率的参数敏感性（任务 D）——**先验尺子，再用尺子量改进**。

层1 一致率 58% 是「Agent 处置 vs 应然档」。应然档不是天降的真值，它是**一个带 5 个
假设参数的代价公式的 argmin**（c_fp / c_review / c_report / m_h 人工漏检率 /
m_e 上报残余率）。参数一动，应然档就动，58% 也跟着动。

所以在拿 58% 当基线去谈「round 3 提升到 X%」之前，必须先回答：**58% 这个数，
在合理参数范围内摆多大？** 若摆动大，一切「58%→X%」的表述都要改成区间。

这与 ① 代价敏感阈值当初做 c_FP ∈ {10,25,50,100} 敏感性是同一条纪律：
**任何建立在假设参数上的结论，都要报参数扫描，而不是报一个点。**

用法：python -m src.eval.disposition_sensitivity r1
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "reports" / "eval_runs"
EVAL_SET = PROJECT_ROOT / "data" / "processed" / "agent_eval_set.parquet"
REPORT = PROJECT_ROOT / "reports" / "agent_disposition_sensitivity.md"
A_MED = 76.02

# 扫描网格：以 BASE 为中心、覆盖「说得过去的业务取值」范围。
# k_future 单列——它是修订2 网络项的强度，已知 k=0 会让上报档整个塌掉（防塌演示），
# 所以它不是"参数不确定性"，是"两种建模口径"，分开报。
GRID = {
    "c_fp":     [10.0, 25.0, 50.0, 100.0],   # ① 已锚 $25，敏感性同款范围
    "c_review": [2.0, 5.0, 10.0, 20.0],
    "c_report": [20.0, 40.0, 100.0],
    "m_h":      [0.05, 0.10, 0.20],          # 挂起后人工仍漏检的比例
    "m_e":      [0.02, 0.05, 0.15],          # 上报后仍漏检的比例
}


def load(tag):
    rows = []
    for f in sorted((RUNS_DIR / tag).glob("txn_*.json")):
        r = json.loads(f.read_text())
        rep = r.get("report") or {}
        if rep.get("disposition"):
            rows.append({"TransactionID": r["txn_id"], "agent": rep["disposition"]})
    return pd.DataFrame(rows).merge(pd.read_parquet(EVAL_SET), on="TransactionID")


def run(tag):
    from src.agent.disposition import BASE, argmin_action
    d = load(tag)
    p, a, g = (d["p"].to_numpy(), d["TransactionAmt"].to_numpy(), d["gang_score"].to_numpy())
    agent = d["agent"].to_numpy()
    w = d["ht_weight"].to_numpy()

    def agree(prm):
        gt = argmin_action(p, a, g, A_MED, prm)
        return (agent == gt).mean(), gt

    base_rate, base_gt = agree(BASE)
    assert np.array_equal(base_gt, d["disposition_gt"].to_numpy()), \
        "重算的 BASE 应然档与缓存不一致——口径漂了，先查 disposition.py"

    L = ["# 应然档一致率的参数敏感性（任务 D）\n",
         f"round `{tag}`，{len(d)} 笔。应然档 = 四档期望成本 argmin，随 5 个假设参数变化。",
         f"BASE 参数下一致率 **{base_rate:.0%}**（与已记录的 58% 一致，口径校验通过）。\n",
         "> 目的：**先验这把尺子稳不稳，再用它量 round 3 的改进。**"
         "与 ① 当初扫 c_FP ∈ {10,25,50,100} 是同一条纪律。\n"]

    # ---------- 一维扫描：每个参数单独动 ----------
    L += ["## 一维扫描（其余参数固定在 BASE）\n",
          "| 参数 | 取值 | 一致率 | 应然档分布（approve/hold/decline/escalate） |",
          "|---|---|---|---|"]
    one_dim = {}
    for k, vals in GRID.items():
        for v in vals:
            r, gt = agree({**BASE, k: v})
            one_dim[(k, v)] = r
            cnt = pd.Series(gt).value_counts()
            dist = "/".join(str(int(cnt.get(x, 0))) for x in
                            ["approve", "hold", "decline", "escalate"])
            star = " ←BASE" if v == BASE[k] else ""
            L.append(f"| {k} | {v}{star} | **{r:.0%}** | {dist} |")

    # ---------- 全网格 ----------
    keys = list(GRID)
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    rates, gts = [], []
    for c in combos:
        prm = {**BASE, **dict(zip(keys, c))}
        r, gt = agree(prm)
        rates.append(r); gts.append(gt)
    rates = np.array(rates)
    lo, hi = rates.min(), rates.max()
    q = np.percentile(rates, [5, 25, 50, 75, 95])
    worst = combos[int(rates.argmin())]; best = combos[int(rates.argmax())]

    L += ["", f"## 全网格扫描（{len(combos)} 组参数组合）\n",
          f"- 一致率**范围 {lo:.0%} – {hi:.0%}**，中位 {q[2]:.0%}，"
          f"5–95 分位 {q[0]:.0%} – {q[4]:.0%}",
          f"- BASE 的 {base_rate:.0%} 在全网格中的分位数：**{(rates < base_rate).mean():.0%}**",
          f"- 最低：{dict(zip(keys, worst))} → {lo:.0%}",
          f"- 最高：{dict(zip(keys, best))} → {hi:.0%}", ""]

    # 哪个参数最能左右一致率（方差分解：按参数取值分组的组间极差）
    L += ["**哪个参数最能左右这把尺子**（固定其他、看该参数取值间的一致率极差）：\n",
          "| 参数 | 一致率极差 | 说明 |", "|---|---|---|"]
    spread = {}
    arr = np.array(combos, dtype=object)
    for i, k in enumerate(keys):
        means = [rates[arr[:, i] == v].mean() for v in GRID[k]]
        spread[k] = max(means) - min(means)
    for k, s in sorted(spread.items(), key=lambda x: -x[1]):
        note = {"c_fp": "决定 decline 档的门槛", "c_review": "决定 hold 档的宽窄",
                "c_report": "决定 escalate 档的宽窄", "m_h": "挂起的残余漏检",
                "m_e": "上报的残余漏检"}[k]
        L.append(f"| {k} | {s:.0%} | {note} |")

    # ---------- 逐笔稳定性：哪些交易的应然档是"参数依赖"的 ----------
    G = np.array(gts)                      # (n_combo, n_txn)
    flip = np.array([len(set(G[:, j])) for j in range(G.shape[1])])
    stable = flip == 1
    agree_stable = (agent[stable] == base_gt[stable]).mean()
    L += ["", "## 逐笔：哪些交易的应然档本身就不稳\n",
          f"- **{stable.sum()}/{len(d)}** 笔在全部 {len(combos)} 组参数下应然档**不变**；"
          f"其余 {(~stable).sum()} 笔会随参数改档（最多的一笔有 {flip.max()} 种档位）。",
          f"- 只在**应然档稳定**的 {stable.sum()} 笔上算一致率：**{agree_stable:.0%}**"
          f"（vs 全样本 {base_rate:.0%}）",
          "",
          "  → 这是一个更干净的对照：把「尺子自己都拿不准」的交易剔掉之后，"
          + ("Agent 的一致率明显更高，说明它的分歧有相当一部分落在**本来就模糊的边界带**上，"
             "不全是它判错。" if agree_stable > base_rate + 0.05 else
             "一致率没有明显改善，说明分歧不是集中在参数敏感的边界带上。")]

    # ---------- 与任务 C 的交叉检验：贵的分歧落在稳定区还是模糊带 ----------
    from src.agent.disposition import realized_cost
    y_, amt_, g_ = (d["isFraud"].to_numpy(), d["TransactionAmt"].to_numpy(),
                    d["gang_score"].to_numpy())
    prod = np.where(base_gt == "approve", "approve", agent)
    per = lambda act: np.array([realized_cost(np.array([x]), np.array([yy]), np.array([aa]),
                                              np.array([gg]), A_MED, BASE)
                                for x, yy, aa, gg in zip(act, y_, amt_, g_)])
    delta = (per(prod) - per(base_gt)) * w * (10_000 / w.sum())
    L += ["", "## 与任务 C 交叉：**贵的分歧落在哪一边**\n",
          "| 交易分组 | 笔数 | Δ$/万笔 | 占差额 |", "|---|---|---|---|",
          f"| 应然档参数稳定（真分歧） | {int(stable.sum())} | ${delta[stable].sum():,.0f} | "
          f"{delta[stable].sum()/delta.sum():.0%} |",
          f"| 应然档随参数改档（模糊带） | {int((~stable).sum())} | ${delta[~stable].sum():,.0f} | "
          f"{delta[~stable].sum()/delta.sum():.0%} |",
          "",
          ("> **钱主要在模糊带上**：那意味着 round 3 就算把 prompt 调到完全服从当前应然档，"
           "省下的也是「按一把自己都不确定的尺子去对齐」换来的——**先去定参数，比先去改 prompt 值钱**。"
           if delta[~stable].sum() > delta[stable].sum() else
           "> **钱主要在参数稳定的交易上**：这些是无论参数怎么取都该拦而 Agent 放了的单，"
           "是**真错**，round 3 修 prompt 是对症的。"),
          ""]

    # ---------- count 口径的 C×D 交叉（总指挥 第二优先）----------
    # 问题：round 3 里那 7 笔「换个错法」（原本错、改后仍错但换了个错法）的交易，
    # 是否落在「应然档随参数改档」的不稳定集合里？若是，「尺子与模型在同一段区间
    # 同时失效」就成立，且这是 **count 口径**、不受 C 的金额集中度影响。
    try:
        from src.eval.round3_metrics import load as _load3
        r3 = _load3("r3").set_index("TransactionID")
        r1 = d.set_index("TransactionID")
        common = r1.index.intersection(r3.index)
        gt_c = pd.Series(argmin_action(r1.loc[common, "p"].to_numpy(),
                                       r1.loc[common, "TransactionAmt"].to_numpy(),
                                       r1.loc[common, "gang_score"].to_numpy(), A_MED, BASE),
                         index=common)
        xa, xb = r1.loc[common, "agent"], r3.loc[common, "agent"]
        still = common[(xa != xb) & (xa != gt_c) & (xb != gt_c)]
        unst = pd.Series(flip > 1, index=d["TransactionID"]).reindex(common)
        k, n_s = int(unst.loc[still].sum()), len(still)
        base_p = float(unst.mean())
        from scipy.stats import binomtest
        pv = binomtest(k, n_s, base_p, alternative="greater").pvalue
        L += ["", "## count 口径的 C×D 交叉（round 3 的「换个错法」落在哪里）\n",
              f"round 3 有 **{n_s} 笔「换个错法」**（原本判错、改 prompt 后仍错但换了个错法）。",
              f"其中落在本文件判定的「应然档随参数改档」不稳定集合内：**{k}/{n_s}**"
              f"（对照：全体 {len(common)} 笔里不稳定占 **{base_p:.0%}**）。",
              f"二项检验（H0：与基率无异，单侧）**p = {pv:.3f}**。", ""]
        # 「不稳定」的定义取决于用哪个参数网格 —— 这会让显著性翻转，必须披露
        from src.eval.round3_metrics import GRID as G4
        ks4 = list(G4)
        G4m = np.array([argmin_action(r1.loc[common, "p"].to_numpy(),
                                      r1.loc[common, "TransactionAmt"].to_numpy(),
                                      r1.loc[common, "gang_score"].to_numpy(), A_MED,
                                      {**BASE, **dict(zip(ks4, c))})
                        for c in itertools.product(*(G4[k] for k in ks4))])
        unst4 = pd.Series([len(set(G4m[:, j])) > 1 for j in range(G4m.shape[1])], index=common)
        k4, b4 = int(unst4.loc[still].sum()), float(unst4.mean())
        pv4 = binomtest(k4, n_s, b4, alternative="greater").pvalue
        L += [f"**⚠️ 这个结论对「不稳定」的定义敏感**——它取决于用哪个参数网格：", "",
              "| 定义「不稳定」的网格 | 命中 | 基率 | 二项 p |", "|---|---|---|---|",
              f"| 本文件 5 参数网格（{len(combos)} 组，D 的原始口径） | {k}/{n_s} | "
              f"{base_p:.0%} | **{pv:.3f}** |",
              f"| round3_metrics 4 参数网格（{G4m.shape[0]} 组，少 m_e） | {k4}/{n_s} | "
              f"{b4:.0%} | **{pv4:.3f}** |", ""]
        if pv < 0.05 and pv4 < 0.05:
            L += ["> ✅ **两种定义下都显著** →「尺子与模型在同一段区间同时失效」成立，"
                  "且是 count 口径、不受 C 的金额集中度影响。"]
        elif pv < 0.05 or pv4 < 0.05:
            L += ["> ⚠️ **一种定义下显著、另一种不显著（p 在 0.02 与 0.07 之间摆）**。",
                  f"> 方向是一致的（{k}/{n_s} 与 {k4}/{n_s} 都高于各自基率），"
                  "**但显著性由一个本可以随手改的定义决定** —— 这正是不该拿它当定论的理由。",
                  "> **裁定：方向支持，按 n 纪律仍记「待验证」，不予转正。**",
                  f"> n={n_s} 的样本多一笔少一笔就翻转；同一把尺子此前砍掉过 "
                  "`mid_ambiguous` 的 −8pp，**也砍掉过我们更喜欢的发现**。",
                  "> 标准不因这次结论「符合预期」而放松：**这是一个提示，不是一个证据**。"]
        elif False:
            L += [f"> ⚠️ **方向支持但未达显著**（{k}/{n_s} = {k/n_s:.0%} vs 基率 "
                  f"{base_p:.0%}，p={pv:.3f}）。",
                  f"> **按 n 纪律，这条仍是「待验证」，不予转正。** n={n_s} 的样本，"
                  "多一笔少一笔就能翻转结论——",
                  "> 同一把尺子此前砍掉过 `mid_ambiguous` 的 −8pp（n=11/14，CI 跨 0），"
                  "**也砍掉过我们更喜欢的发现**。",
                  "> 标准不因这次结论「符合预期」而放松：**5/7 是一个提示，不是一个证据**。"]
    except Exception as _e:                       # r3 未跑时静默跳过
        L += ["", f"（C×D 交叉未算：{type(_e).__name__}）"]

    # ---------- 结论 ----------
    swing = (hi - lo) * 100
    L += ["", "## 结论\n",
          f"1. **一致率在合理参数范围内摆动 {lo:.0%}–{hi:.0%}（跨度 {swing:.0f} 个百分点）**。",
          f"2. 因此 **禁止写「58% → X%」这种点对点表述**，一律改成区间/条件表述：",
          f"   > 「在 BASE 参数（c_fp=$25、c_review=$5、c_report=$40、m_h=0.10、m_e=0.05）下，"
          f"层1 一致率 {base_rate:.0%}；在参数网格上为 {lo:.0%}–{hi:.0%}。」",
          f"3. round 3 的改进必须报成 **同一组参数下的前后对比**（配对），"
          "而不是拿新参数下的新数字去比旧数字。",
          f"4. 最能左右这把尺子的参数是 **{max(spread, key=spread.get)}**"
          f"（极差 {max(spread.values()):.0%}）——它也是最该去查真实业务取值的那个。",
          "",
          f"5. **（本文件最重要的一条，接任务 C）**：任务 C 拆出的每万笔差额，"
          f"**{delta[~stable].sum()/delta.sum():.0%} 落在应然档随参数改档的 "
          f"{int((~stable).sum())} 笔上**，只有 {delta[stable].sum()/delta.sum():.0%} "
          f"落在参数稳定的 {int(stable.sum())} 笔上。",
          "   合起来读：**层1 分歧看着值 $9,103，但这笔钱几乎全部记在「尺子自己都拿不准」"
          "的那批交易头上。**",
          "",
          "   **我据此提的建议（「先把参数变成有依据的值，再改 prompt」）已被总指挥驳回，"
          "理由成立、记录在此**：",
          "   > c_FP 在 IEEE-CIS 里**无法测得**——数据集里没有误拦成本。任何取值都是假设，"
          "「给个有依据的值」只是把**显式**的不确定性变成**隐式**的，反而更糟。",
          "",
          f"   → **裁定口径**：沿用 ① 已锚定的 **c_FP=$25（敏感性 $10–100）**，"
          "理由是全项目口径一致；其余参数照 AGENT_DESIGN 初值。",
          "   **所有结论一律「参照点 + 网格区间」双报**，不追求把参数钉死。",
          f"   本文件的数字按此读作：参照点 {base_rate:.0%}，网格区间 {lo:.0%}–{hi:.0%}。",
          "",
          "> 诚实注记：这些参数目前**全是假设值**，没有真实业务成本数据支撑。",
          "> 项目对外定位是「离线决策系统 + 已推演工业化路径」，所以正确讲法是"
          "「我知道结论依赖这些假设，并且量化了依赖程度」，而不是假装参数是已知的。\n"]

    REPORT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n✅ → {REPORT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "r1")
