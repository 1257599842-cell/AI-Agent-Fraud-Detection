"""⑦ 自我认知检查（真标签锚定）—— 替代已作废的 LLM 软层。

**为什么它能替代软层**：软层（judge 评推理/断言）全程没有非 LLM 参照，循环无法打破。
本模块换一个提问方式——不问"它讲得对不对"，问"**它自己说的把握，兑不兑现**"。
参照物是 `isFraud` 真标签与应然档，都不是 LLM 产物，循环天然打破。

三问（都可能出负面结论，负面照报）：
  1. risk_level（high/medium/low）对 isFraud 有没有判别力？分档欺诈率 + AUC。
     关键对照：**它已经在 prompt 里看过 GBDT 的 p**。所以要同集比 AUC(risk_level)
     vs AUC(p)——若前者 ≤ 后者，说明 risk_level 只是把分数翻译成词，没加信息。
  2. confidence（high/medium）说得准不准？按 confidence 分档看**层1 处置一致率**
     （Agent 处置 vs 应然档）。若两档一致率没差别 → confidence 无区分力。
  3. evidence_insufficient 触发几次。
     ⚠️ 注意：本模块只能测**发生率**，测不了**能力**——
     「不会弃权」与「没机会弃权」在正常样本上分不开，需专门的剥夺实验
     （见 src/eval/abstention_test.py，结论：它会弃权）。

**抽样口径（必须同时报两个数）**：eval 集是 4×50 分层抽的，层内欺诈率被人为拉平，
所以集内直接算的 AUC/欺诈率**不是总体值**。HT 加权（逆抽样概率）才是总体估计。
两个都报、并注明——这是修订5"分层集必须 HT 加权"在自我认知检查上的同一条纪律。

用法：python -m src.eval.self_awareness r1
"""

from src.report_io import write_report
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "reports" / "eval_runs"
EVAL_SET = PROJECT_ROOT / "data" / "processed" / "agent_eval_set.parquet"
REPORT = PROJECT_ROOT / "reports" / "agent_selfcheck.md"

RISK_ORD = {"low": 0, "medium": 1, "high": 2}
CONF_ORD = {"low": 0, "medium": 1, "high": 2}


def _wilson(k, n, z=1.96):
    """小样本比例的 Wilson 区间——n=20 量级时正态近似会给出越界区间。"""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _mh_by(d, by):
    """在 by 的每个格子内算 high−medium 一致率差，按格内样本量池化（M-H 式）。
    池化差 ≈ 0 → confidence 只是该控制变量的代名词；仍为正 → 带它之外的信息。

    ⚠️ **缺格是静默跳过的**：某格若只有 high 或只有 medium，就无法在格内对照、被直接略过。
    调用方必须报「几格可估 / 共几格」，否则会出现「控制后效应仍在」的假象——
    而实际上被跳过的可能正是表现最差的那一格（本项目的 hold 档就是如此）。
    返回 (池化差, [(格值, n_high, 率_high, n_med, 率_med, 差), ...])。
    """
    num = den = 0.0
    rows = []
    for g, s in d.groupby(by, observed=True):
        h, m = s[s["confidence"] == "high"], s[s["confidence"] == "medium"]
        if len(h) and len(m):
            dd = h["agree"].mean() - m["agree"].mean()
            w = len(h) * len(m) / (len(h) + len(m))        # M-H 权重
            num += w * dd
            den += w
            rows.append((g, len(h), h["agree"].mean(), len(m), m["agree"].mean(), dd))
    return (num / den if den else float("nan")), rows


def _mid_stats(d):
    """mid_ambiguous 层的 high/medium 原始计数与差的 95% CI（按 n 纪律，只报计数+区间）。"""
    s = d[d["stratum"] == "mid_ambiguous"]
    h, m = s[s["confidence"] == "high"]["agree"], s[s["confidence"] == "medium"]["agree"]
    dd = h.mean() - m.mean()
    se = np.sqrt(h.mean() * (1 - h.mean()) / len(h) + m.mean() * (1 - m.mean()) / len(m))
    return (h.sum(), len(h), m.sum(), len(m), dd - 1.96 * se, dd + 1.96 * se)


def _auc(y, s, w=None):
    y = np.asarray(y)
    if len(set(y)) < 2:
        return float("nan")
    return roc_auc_score(y, s, sample_weight=w)


def load(tag):
    rows = []
    for f in sorted((RUNS_DIR / tag).glob("txn_*.json")):
        r = json.loads(f.read_text())
        rep = r.get("report") or {}
        rows.append({"TransactionID": r["txn_id"], "p": r.get("p"),
                     "risk_level": rep.get("risk_level"),
                     "confidence": rep.get("confidence"),
                     "disposition": rep.get("disposition"),
                     "evidence_insufficient": bool(rep.get("evidence_insufficient")),
                     "mode": r.get("mode")})
    df = pd.DataFrame(rows)
    es = pd.read_parquet(EVAL_SET)
    return df.merge(es[["TransactionID", "isFraud", "stratum", "ht_weight",
                        "disposition_gt", "TransactionAmt", "gang_score"]], on="TransactionID")


def run(tag):
    d = load(tag)
    L = ["# ⑦ 自我认知检查（真标签锚定）\n",
         f"样本：round `{tag}` 的 {len(d)} 份调查（dev 分层集 4×50 的 dev 半边）。",
         "**参照物是 isFraud 真标签与应然档，非 LLM** —— 这是软层作废后 ⑦ 的可报承重墙。\n",
         "> 分层集内的率/AUC 不是总体值（层内欺诈率被人为拉平）。下表**集内**与"
         "**HT 加权**（逆抽样概率，总体估计）并列，两个都看。\n"]

    # ---------- 问 1：risk_level 有没有判别力 ----------
    L += ["## 1. risk_level 对 isFraud 的判别力\n",
          "| risk_level | n | 欺诈数 | 集内欺诈率 | HT 加权欺诈率 | 平均 GBDT p |",
          "|---|---|---|---|---|---|"]
    for lv in ["high", "medium", "low"]:
        s = d[d["risk_level"] == lv]
        if not len(s):
            continue
        ht = np.average(s["isFraud"], weights=s["ht_weight"])
        L.append(f"| {lv} | {len(s)} | {int(s['isFraud'].sum())} | "
                 f"{s['isFraud'].mean():.1%} | {ht:.1%} | {s['p'].mean():.3f} |")
    rl = d["risk_level"].map(RISK_ORD)
    ok = rl.notna()
    a_rl = _auc(d.loc[ok, "isFraud"], rl[ok])
    a_p = _auc(d["isFraud"], d["p"])
    a_rl_ht = _auc(d.loc[ok, "isFraud"], rl[ok], d.loc[ok, "ht_weight"])
    a_p_ht = _auc(d["isFraud"], d["p"], d["ht_weight"])
    L += ["", f"- **AUC(risk_level) 集内 {a_rl:.4f} / HT 加权 {a_rl_ht:.4f}**"
              f"（三档序数，只有 2 个切点，天花板本就低于连续分）",
          f"- **AUC(GBDT p) 集内 {a_p:.4f} / HT 加权 {a_p_ht:.4f}**（同一批样本）",
          f"- **增量 Δ(集内) {a_rl - a_p:+.4f}**"]

    # risk_level 是不是 p 的单调改写：给定 p 之后它还带不带信息
    lo, hi = d["p"].quantile([1/3, 2/3])
    L += ["", "**给定 p 之后 risk_level 还带信息吗**（按 p 三等分，组内看 risk_level 分档欺诈率）：",
          "", "| p 分位组 | risk_level | n | 欺诈率 |", "|---|---|---|---|"]
    d["_pbin"] = np.where(d["p"] <= lo, "低1/3", np.where(d["p"] <= hi, "中1/3", "高1/3"))
    resid = []
    for b in ["低1/3", "中1/3", "高1/3"]:
        s = d[d["_pbin"] == b]
        for lv in ["high", "medium", "low"]:
            t = s[s["risk_level"] == lv]
            if len(t):
                L.append(f"| {b} | {lv} | {len(t)} | {t['isFraud'].mean():.0%} |")
        if s["risk_level"].nunique() > 1:
            resid.append(_auc(s["isFraud"], s["risk_level"].map(RISK_ORD)))
    if resid:
        L += ["", f"- p 分位组**内** AUC(risk_level) = "
                  + "、".join(f"{x:.2f}" for x in resid)
                  + f"（均值 {np.nanmean(resid):.2f}；0.5=组内已无增量信息）"]

    # ---------- 问 2：confidence 说得准不准 ----------
    d["agree"] = d["disposition"] == d["disposition_gt"]
    L += ["\n## 2. confidence 分档 vs 层1 处置一致率（它说 high 时是不是真的更常判对）\n",
          "| confidence | n | 层1 一致 | 一致率 | Wilson 95% CI | 平均 GBDT p |",
          "|---|---|---|---|---|---|"]
    buckets = []
    for lv in ["high", "medium", "low"]:
        s = d[d["confidence"] == lv]
        if not len(s):
            continue
        k, n = int(s["agree"].sum()), len(s)
        cl, ch = _wilson(k, n)
        buckets.append((lv, k, n))
        L.append(f"| {lv} | {n} | {k} | {k/n:.0%} | [{cl:.0%}, {ch:.0%}] | {s['p'].mean():.3f} |")
    L.append(f"| **合计** | {len(d)} | {int(d['agree'].sum())} | {d['agree'].mean():.0%} | | |")
    if len(buckets) >= 2:
        (l1, k1, n1), (l2, k2, n2) = buckets[0], buckets[1]
        diff = k1/n1 - k2/n2
        # 两比例差的 Wald 区间：n≈50 时区间必然很宽，宽本身就是结论的一部分
        se = np.sqrt(k1/n1*(1-k1/n1)/n1 + k2/n2*(1-k2/n2)/n2)
        L += ["", f"- **{l1} vs {l2} 一致率差 {diff:+.0%}**，95% CI "
                  f"[{diff-1.96*se:+.0%}, {diff+1.96*se:+.0%}]"
                  + ("（**区间跨 0 → 无区分力**）" if abs(diff) < 1.96*se else "（区间不跨 0）"),
              f"- AUC(confidence → 层1 是否一致) = "
              f"{_auc(d['agree'], d['confidence'].map(CONF_ORD)):.3f}"]
    # confidence 是否只是 p 的改写
    L += ["", "**confidence 与 GBDT p 的关系**（若 confidence 只是「分数极端就自信」，"
          "它就是 p 的函数、不是自我认知）：",
          "", "| confidence | 平均 p | p 的中位 | \\|p−0.5\\| 均值（离决策边界远近）|",
          "|---|---|---|---|"]
    for lv in ["high", "medium", "low"]:
        s = d[d["confidence"] == lv]
        if len(s):
            L.append(f"| {lv} | {s['p'].mean():.3f} | {s['p'].median():.3f} | "
                     f"{(s['p']-0.5).abs().mean():.3f} |")

    # 混淆检查：confidence 会不会只是「这单容易」的代名词？分层/分处置内再看一次。
    # 这是问 1 里「给定 p 之后还带不带信息」的同一招，换控制变量。
    for by, title in [("stratum", "分层（难度代理）"), ("disposition", "Agent 自己给的处置")]:
        pooled, rows = _mh_by(d, by)
        L += ["", f"**控制「{title}」后 confidence 还带信息吗**（格内 high−medium 一致率差，"
              "M-H 池化）：", "",
              f"| {by} | n(high) | high 一致率 | n(med) | med 一致率 | 差 |", "|---|---|---|---|---|---|"]
        for g, nh, ah, nm, am, dd in rows:
            L.append(f"| {g} | {nh} | {ah:.0%} | {nm} | {am:.0%} | {dd:+.0%} |")
        n_cells = d[by].nunique()
        L.append(f"\n- **池化差 {pooled:+.0%}**"
                 + ("（≈0 → confidence 只是难度的代名词）" if abs(pooled) < 0.10
                    else "（仍为正 → 带难度之外的信息）")
                 + (f"　⚠️ **只有 {len(rows)}/{n_cells} 格可估**，其余格缺 high 或缺 medium、"
                    "被 M-H **静默跳过**——见 2b" if len(rows) < n_cells else ""))
        if by == "stratum":
            pooled_stratum = pooled


    # ---------- 问 2b：自选档混淆（总指挥 二.2，硬性检查）----------
    L += ["\n### 2b. 混淆检查：confidence 会不会只是「它选了哪一档」的代名词\n",
          "层1 一致率按 Agent **自选档**差异极大，所以必须验：若 confidence 与自选档相关，"
          "+45pp 可以在**毫无自我认知**的情况下复现。\n",
          "**confidence × 自选档 交叉表（笔数）**：\n",
          "| 自选档 | high | medium | 合计 | 该档一致率 |", "|---|---|---|---|---|"]
    ct = pd.crosstab(d["disposition"], d["confidence"])
    for a in ct.index:
        h = int(ct.loc[a].get("high", 0)); m = int(ct.loc[a].get("medium", 0))
        s = d[d["disposition"] == a]
        L.append(f"| {a} | {h} | {m} | {h+m} | {s['agree'].mean():.0%} ({int(s['agree'].sum())}/{len(s)}) |")
    L += ["",
          "⚠️ **发现一个结构性纠缠**：`hold` **19/19 全是 medium**（且 hold 的一致率只有 11%），"
          "`decline` 1/1 全是 high。**这两档无法做档内对照**——",
          "我此前那个「控制自选档后 +28pp」的 M-H 池化，实际上只用到了 approve 与 escalate 两格，"
          "**把最差的那一格（hold）整个丢掉了**，而丢掉的方式是不可见的（缺格自动跳过）。",
          "这正是 M-H 这类池化估计量的陷阱：**它对缺格是静默的**。\n",
          "**三个口径并列**：\n",
          "| 口径 | high | medium | 差 | 说明 |", "|---|---|---|---|---|"]
    nohold = d[d["disposition"] != "hold"]
    rows_cmp = [("原始（全样本）", d), ("剔除 hold 档", nohold)]
    for lab, s in rows_cmp:
        h = s[s["confidence"] == "high"]["agree"]; m = s[s["confidence"] == "medium"]["agree"]
        L.append(f"| {lab} | {h.mean():.0%} ({int(h.sum())}/{len(h)}) | "
                 f"{m.mean():.0%} ({int(m.sum())}/{len(m)}) | "
                 f"{h.mean()-m.mean():+.0f}pp |".replace("+0pp", f"{(h.mean()-m.mean())*100:+.0f}pp")
                 .replace(f"{h.mean()-m.mean():+.0f}pp", f"{(h.mean()-m.mean())*100:+.0f}pp")
                 + ("" if lab.startswith("原始") else " hold 全 medium，剔掉它才是可比的 |"))
    pooled_disp, rows_disp = _mh_by(d, "disposition")
    L.append(f"| M-H 池化（档内） | — | — | {pooled_disp*100:+.0f}pp | "
             f"只有 {len(rows_disp)}/{d['disposition'].nunique()} 档可估（其余缺格） |")
    L += ["", "**可估的两格逐格看**（这是唯一能做真对照的地方）：\n",
          "| 自选档 | high 一致 | medium 一致 | 差 | 差的 95% CI |", "|---|---|---|---|---|"]
    surv = True
    for a, nh, ah, nm, am, dd in rows_disp:
        se_ = np.sqrt(ah*(1-ah)/nh + am*(1-am)/nm)
        lo_, hi_ = dd - 1.96*se_, dd + 1.96*se_
        surv &= lo_ > 0
        L.append(f"| {a} | {ah:.0%} ({int(round(ah*nh))}/{nh}) | {am:.0%} "
                 f"({int(round(am*nm))}/{nm}) | {dd*100:+.0f}pp | "
                 f"[{lo_*100:+.0f}, {hi_*100:+.0f}]pp |")

    # 参数网格角点复验：outcome（应然档）本身依赖参数
    from src.agent.disposition import BASE, argmin_action
    corners = [{"c_fp": 10.0, "c_review": 2.0}, {"c_fp": 100.0, "c_review": 20.0},
               {"c_report": 100.0, "m_h": 0.20}]
    L += ["", "**参数网格角点复验**（outcome=应然档本身依赖参数，+45pp 会不会只是 BASE 的巧合）：\n",
          "| 参数角点 | 全样本差 | 剔除 hold 后 |", "|---|---|---|"]
    for c in corners:
        gt2 = argmin_action(d["p"].to_numpy(), d["TransactionAmt"].to_numpy(),
                            d["gang_score"].to_numpy(), 76.02, {**BASE, **c})
        ag2 = pd.Series(d["disposition"].to_numpy() == gt2, index=d.index)
        f = lambda s: (ag2[s[s["confidence"] == "high"].index].mean()
                       - ag2[s[s["confidence"] == "medium"].index].mean())
        L.append(f"| {c} | {f(d)*100:+.0f}pp | {f(nohold)*100:+.0f}pp |")

    L += ["", "**这一检查的裁定**：",
          f"- 原始 +45pp **确实被自选档混淆放大**：hold 档 19 笔全部落在 medium 且几乎全错，"
          f"只这一格就贡献了很大一块。剔除 hold 后差缩到 "
          f"{(nohold[nohold['confidence']=='high']['agree'].mean()-nohold[nohold['confidence']=='medium']['agree'].mean())*100:+.0f}pp。",
          "- 但**在两个可对照的档内，差仍为正**（见上表）。所以效应不是纯混淆产物。",
          "- ⚠️ **然而 4 档里有 2 档结构性不可估**（hold 无 high、decline 无 medium），"
          "而不可估的 hold 恰恰是 Agent 表现最差的一档 → **无法排除「confidence 只是在给自选档打标签」**。",
          "",
          "> **裁定：不通过。**「Agent 有自我认知」**不得写入任何文档**。",
          "> 现阶段能说的只有一句、且必须带条件：",
          "> 「在 approve 与 escalate 两档**内部**，标 high 的报告与应然档一致的比例高于标 medium 的；",
          "> 但 hold 档全部标 medium，该档无法对照，因此不能排除 confidence 只是自选档的代名词。」",
          "> 这句话不是自我认知的证据，只是一个**待验的相关性**。\n"]

    # ---------- 问 3：弃权闸门 ----------
    ei = int(d["evidence_insufficient"].sum())
    L += ["\n## 3. 弃权闸门\n",
          f"- `evidence_insufficient` 触发 **{ei}/{len(d)}**"
          + ("（**从未触发 → 该字段无区分力，作为「该弃权未弃权」的代理不可用**）"
             if ei == 0 else ""),
          f"- confidence 从未出现 `low`：{int((d['confidence']=='low').sum())}/{len(d)}"
          if "low" not in set(d["confidence"]) else "",
          "- 含义：报告自陈的三个「不确定」通道（evidence_insufficient / confidence=low / "
          "risk_level 与 p 背离）在 100 份里几乎不启用 → 弃权能力**未被观测到存在**。"]

    # ---------- 结论（措辞由数据推出，不由预判推出）----------
    ci_lo, ci_hi = diff - 1.96 * se, diff + 1.96 * se
    conf_flat = ci_lo <= 0 <= ci_hi
    # 2b 的裁定：hold 档 19/19 全 medium、decline 1/1 全 high → 4 档里 2 档结构性不可估，
    # 且不可估的 hold 恰是最差的一档 → 无论差多大，都排除不掉「confidence 只是自选档标签」。
    est_cells, all_cells = len(rows_disp), d["disposition"].nunique()
    conf_verdict_pass = False          # 二.2 硬性检查：本轮**不通过**，见 §2b
    L += ["\n## 结论\n",
          f"1. **risk_level 基本是 GBDT p 的措辞化改写**：AUC {a_rl:.3f} vs p 的 {a_p:.3f}"
          f"（Δ{a_rl - a_p:+.3f}），给定 p 后组内 AUC ≈ {np.nanmean(resid):.2f}。"
          "→ 风险**分级**没有独立于模型的贡献；这条与预判一致。",
          "",
          f"2. **confidence 与层1 一致率相关，但该相关性无法与「自选档标签」分离——"
          f"混淆检查不通过**（§2b）。",
          f"   - 原始差 {diff:+.0%}（high {buckets[0][1]}/{buckets[0][2]} vs "
          f"medium {buckets[1][1]}/{buckets[1][2]}，95% CI [{ci_lo:+.0%}, {ci_hi:+.0%}]）"
          "**被自选档混淆放大**：`hold` 档 19 笔**全部**标 medium 且一致率仅 11%。",
          f"   - 剔除 hold 后差缩到 "
          f"{(nohold[nohold['confidence']=='high']['agree'].mean()-nohold[nohold['confidence']=='medium']['agree'].mean())*100:+.0f}pp；"
          f"档内 M-H {pooled_disp*100:+.0f}pp，但**只有 {est_cells}/{all_cells} 档可估**。",
          "   - 逐格只有 `approve` 的差显著（+38pp, CI [+7,+68]），"
          "`escalate` 的 CI 跨 0（+20pp, [-6,+47]）。",
          "   - 参数网格角点复验：效应方向稳（+34~+42pp 全样本 / +15~+32pp 剔 hold），"
          "**不是 BASE 参数的巧合**——但方向稳不解决混淆问题。",
          "",
          "   📉 **「它在 mid_ambiguous 层失效」这条按 n 纪律降级为待验假设**："
          + (lambda s: (
              f"该层 high {int(s[0])}/{int(s[1])}={s[0]/s[1]:.0%}、"
              f"medium {int(s[2])}/{int(s[3])}={s[2]/s[3]:.0%}，"
              f"差 {(s[0]/s[1]-s[2]/s[3])*100:+.0f}pp，**95% CI [{s[4]*100:+.0f}, {s[5]*100:+.0f}]pp**")
             )(_mid_stats(d))
          + "。区间宽 73 个百分点、跨 0，**n=11/14 撑不起任何结论**；"
          "且该层 hold 5 笔又全是 medium，同一个混淆在层内重现。",
          "   → **标注为待验假设，不进 INTERVIEW.md，不作为「路由用法失效」的证据。**"
          "（同一条 n 纪律此前砍掉过少数类 3 条的结论；这次它砍的是一个我们都喜欢的发现，"
          "**标准不因喜好而变**。）",
          "",
          "   ⚠️ **两个预判都错了，但错的方式不同，要分开记**：",
          "   - 立项预判「confidence 无区分力、保守措辞只是文风」——**证据不支持**（相关性确实在，"
          "且平均 p 几乎相同"
          f"（{d[d['confidence']=='high']['p'].mean():.3f} vs "
          f"{d[d['confidence']=='medium']['p'].mean():.3f}）、不是「分数极端就自信」的平凡改写）。",
          "   - 我上一轮据此写的「confidence 有区分力且控制难度后存活」——**也讲过头了**："
          "我用的 M-H 池化对缺格是**静默**的，它把最差的 hold 档整格丢掉而没有报出来。"
          "**是我先用了一个会掩盖问题的估计量，才得出那个偏强的结论。**",
          "",
          "   → 现阶段唯一能写的表述（带条件，且**不得**称为自我认知）：",
          "   > 「在 approve 与 escalate 两档内部，标 high 的报告与应然档一致的比例高于标 medium；"
          "但 hold 档全部标 medium，该档无法对照，因此不能排除 confidence 只是自选档的代名词。」",
          "",
          f"3. **弃权通道在本样本上从未启用**：evidence_insufficient {ei}/{len(d)}、"
          f"confidence 无 low（{int((d['confidence']=='low').sum())}/{len(d)}）。",
          "",
          "   ⚠️ **此处原写「弃权通道形同虚设」，已撤回**（2026-08-01，依据 "
          "`agent_abstention.md` 的证据剥夺消融）：",
          "   把工具返回逐级掐掉后，弃权率呈 0% → 10% → **100%** 的剂量反应——"
          "**证据真被抽干时它会弃权**。",
          "   所以这里的 0/100 是**发生率**（正常运行下证据从没薄到那个程度），"
          "**不是能力**。把两者当成一回事是当时的判读错误。",
          "",
          "## 对设计决策的影响\n",
          "> (a) **风险分级与概率判断权归 GBDT**（由第 1 条支撑）—— risk_level 相对 p 无增量，"
          "四档处置继续由代价公式 argmin 决定，⑧ 闸门继续用 GBDT p 当阀门。",
          "> 　（**原写「第 1、3 条支撑」、并把第 3 条读成「弃权通道不存在」，已撤回**："
          "剥夺实验证明它会弃权，第 3 条只说明本样本没触发，**支撑不了这条设计决策**。）",
          "> (b) **confidence 的下游用法（medium 当人工复核路由信号）暂停**——"
          "混淆检查未通过前，它与「hold 档」几乎同义，"
          "而「hold 就送人工复核」是一条不需要 confidence 的规则。"
          "要复活这个用法，得先拿到**同一自选档内 confidence 有变化**的样本"
          "（即让 Agent 在 hold 上也能标 high），否则永远测不出。",
          "",
          "> 方法论注记：这是软层作废后补上的**非循环**评估——参照物是 isFraud 与应然档。",
          "> 本节最值得记的不是结论，是**两次自我修正**：先是数据推翻了立项预判，"
          "接着一个更细的对照推翻了我据此下的强结论。**估计量本身会掩盖问题**"
          "（M-H 静默跳过缺格），这与解析层那个静默 bug 是同一类错误的两个位置。"]

    write_report(REPORT, "\n".join(x for x in L if x is not None))
    print("\n".join(x for x in L if x is not None))
    print(f"\n✅ → {REPORT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    from src.eval._cli import require_tags
    run(require_tags(sys.argv[1:], least=1,
                     usage="python -m src.eval.self_awareness r1")[0])
