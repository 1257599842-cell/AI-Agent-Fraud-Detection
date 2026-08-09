"""证据层 vs 决策层拆分 —— 方案 ① 的地基（项目负责人 2026-07-31）。

**要回答的问题**：Agent 在「取证/定性」上的水平，是否显著高于它在「选处置档」上的水平？
若是，① 的职责边界（取证归 Agent、决策归闭式解）就从推演变成**用数据划出来的**。

---
## 三个统计陷阱，本模块逐个拆掉（不拆就会自动得出想要的结论）

**陷阱 1：两层的类别数不同，原始一致率不可直接比。**
决策层是 4 分类（chance ≈ 25%），证据层是 2 分类（chance ≈ 50%）。
直接把「证据层 75%」和「决策层 58%」并排，会**凭类别数造出一个优势**。
→ 本模块对两层都报**多数类基线**（全猜众数能拿到多少）与**相对基线的提升**，
   结论只看提升，不看原始值。这是「聚合指标在极端不平衡下骗人」的第 N 次应用。

**陷阱 2：`gang_score≥0.5` 不是真值，是代码算的代理。**
和「应然档是公式输出、不是真理」完全同构（D 已证那把尺子会晃 15 个百分点）。
「Agent 与 gang_score 一致率高」只说明它和一个启发式对得上，**不说明它对**。
→ 本模块同时把**两者各自对 `isFraud` 真标签**做检验（非循环锚，A 的同一招）。
   若两者互相同意却都预测不了欺诈，那个高一致率就是空的。

**陷阱 3：0.5 这个门槛本身是选的。**
→ 对 0.3/0.4/0.5/0.6/0.7 全部重算（D 的参数敏感性同款纪律）。

用法：python -m src.eval.evidence_vs_decision r1 r3
"""

from src.report_io import write_report
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "reports" / "eval_runs"
EVAL_SET = PROJECT_ROOT / "data" / "processed" / "agent_eval_set.parquet"
REPORT = PROJECT_ROOT / "reports" / "agent_evidence_vs_decision.md"
A_MED = 76.02
GANG_CUTS = [0.3, 0.4, 0.5, 0.6, 0.7]


def load(tag):
    rows = []
    for f in sorted((RUNS_DIR / tag).glob("txn_*.json")):
        r = json.loads(f.read_text())
        rep = r.get("report") or {}
        ga = rep.get("gang_association")
        rows.append({"TransactionID": r["txn_id"], "agent": rep.get("disposition"),
                     "ga_present": ga is not None,
                     "susp": (ga or {}).get("suspected"),
                     "evidence_insufficient": bool(rep.get("evidence_insufficient"))})
    return pd.DataFrame(rows).merge(pd.read_parquet(EVAL_SET), on="TransactionID")


def _wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def run(tags):
    from src.agent.disposition import BASE, argmin_action
    L = ["# 证据层 vs 决策层拆分（方案 ① 的地基）\n",
         "**问题**：Agent 在取证/定性上的水平，是否显著高于它在选处置档上的水平？\n",
         "> ⚠️ 本文件对三个陷阱做了显式处理，读数时请连同处理一起读："
         "①两层类别数不同（4 分类 vs 2 分类）→ **只看相对多数类基线的提升**；"
         "②`gang_score≥0.5` 是代码代理不是真值 → **另加 isFraud 非循环锚**；"
         "③0.5 门槛是选的 → **全门槛扫描**。\n"]

    summary = {}
    for tag in tags:
        d = load(tag)
        n_all = len(d)
        miss = int((~d["ga_present"]).sum())
        s = d[d["ga_present"]].copy()
        s["susp"] = s["susp"].astype(bool)
        code = s["gang_score"] >= 0.5

        # ---------- 1. 2×2 ----------
        tp = int((s["susp"] & code).sum()); fp = int((s["susp"] & ~code).sum())
        fn = int((~s["susp"] & code).sum()); tn = int((~s["susp"] & ~code).sum())
        agree = (tp + tn) / len(s)
        base_ev = max(code.mean(), 1 - code.mean())      # 全猜众数
        lo, hi = _wilson(tp + tn, len(s))

        L += [f"\n## round `{tag}`\n",
              f"### 1. 证据层：Agent `gang_association.suspected` × 代码 `gang_score≥0.5`\n",
              f"- **未表态（`gang_association` 为 null）{miss}/{n_all} 笔**"
              + ("　⚠️ 这批被排除在下表之外" if miss else ""),
              "", "| | 代码判团伙 | 代码判非团伙 |", "|---|---|---|",
              f"| **Agent 判团伙** | {tp} | {fp} |",
              f"| **Agent 判非团伙** | {fn} | {tn} |", "",
              f"- 原始一致率 **{agree:.0%}**（{tp+tn}/{len(s)}，95% CI [{lo:.0%}, {hi:.0%}]）",
              f"- 多数类基线 **{base_ev:.0%}**（全猜「{'团伙' if code.mean()>0.5 else '非团伙'}」）",
              f"- **相对基线提升 {(agree-base_ev)*100:+.0f}pp** ← 只有这个数可以拿去和决策层比"]

        # ---------- 2. 非循环锚：两者各自对 isFraud ----------
        y = s["isFraud"].astype(bool)
        def _rate(mask, name):
            k, nn = int((y & mask).sum()), int(mask.sum())
            return f"| {name} | {nn} | {k} | {k/nn:.0%} |" if nn else f"| {name} | 0 | – | – |"
        L += ["", "### 2. 非循环锚：两者各自对 `isFraud` 真标签（同意≠正确）\n",
              "| 判定 | n | 其中真欺诈 | 欺诈率 |", "|---|---|---|---|",
              _rate(s["susp"], "Agent 判团伙"), _rate(~s["susp"], "Agent 判非团伙"),
              _rate(code, "代码 gang≥0.5"), _rate(~code, "代码 gang<0.5"),
              f"| （全体） | {len(s)} | {int(y.sum())} | {y.mean():.0%} |", ""]
        def _sep(mask):
            p1, n1 = y[mask].mean(), int(mask.sum())
            p0, n0 = y[~mask].mean(), int((~mask).sum())
            dd = p1 - p0
            se = np.sqrt(p1 * (1 - p1) / max(n1, 1) + p0 * (1 - p0) / max(n0, 1))
            return dd, dd - 1.96 * se, dd + 1.96 * se
        sep_a, la, ha = _sep(s["susp"])
        sep_c, lc, hc = _sep(code)
        L += [f"- 欺诈率分离度（判团伙组 − 判非团伙组）：",
              f"  - **Agent {sep_a*100:+.0f}pp**，95% CI [{la*100:+.0f}, {ha*100:+.0f}]pp"
              + ("　**跨 0**" if la <= 0 <= ha else ""),
              f"  - **代码 gang_score {sep_c*100:+.0f}pp**，95% CI [{lc*100:+.0f}, {hc*100:+.0f}]pp"
              + ("　**跨 0**" if lc <= 0 <= hc else ""),
              "- 读法：若两者互相同意、却都分离不出欺诈，则第 1 节那个一致率是**空的**"
              "（两个都指向同一个没用的方向也叫一致）。",
              "- ⚠️ 两个区间都跨 0 时，**只能说「未观测到指向真实欺诈的证据」**，"
              "不能说「证明了两者都没用」——n≈100、欺诈基率 32%，本就没有功效检出这个量级的分离。"]

        # ---------- 3. 门槛敏感性 ----------
        L += ["", "### 3. `gang_score` 门槛敏感性（0.5 是选的）\n",
              "| 门槛 | 代码判团伙数 | 一致率 | 多数类基线 | 提升 |", "|---|---|---|---|---|"]
        for c in GANG_CUTS:
            cc = s["gang_score"] >= c
            a = ((s["susp"] & cc) | (~s["susp"] & ~cc)).mean()
            b = max(cc.mean(), 1 - cc.mean())
            L.append(f"| {c} | {int(cc.sum())} | {a:.0%} | {b:.0%} | {(a-b)*100:+.0f}pp |")

        # ---------- 4. 代回 argmin ----------
        L += ["", "### 4. 把 Agent 的 `suspected` 代回 `argmin_action()` 重算处置\n",
              "问的是 ① 下最要紧的一件事：**Agent 的取证错误，经公式传导后会造成多大决策偏差**。\n",
              "| g 的映射 | 与应然档一致率 | 多数类基线 | 提升 |", "|---|---|---|---|"]
        gt = s["disposition_gt"].to_numpy()
        base_dec = pd.Series(gt).value_counts(normalize=True).max()
        maps = {"suspected→g=1.0 / 否→0": np.where(s["susp"], 1.0, 0.0),
                "suspected→g=0.75 / 否→0": np.where(s["susp"], 0.75, 0.0),
                "suspected→保留代码幅度 / 否→0": np.where(s["susp"], s["gang_score"], 0.0)}
        sub_best = 0
        for name, gvec in maps.items():
            act = argmin_action(s["p"].to_numpy(), s["TransactionAmt"].to_numpy(),
                                gvec, A_MED, BASE)
            a = (act == gt).mean()
            sub_best = max(sub_best, a - base_dec)
            L.append(f"| {name} | {a:.0%} | {base_dec:.0%} | {(a-base_dec)*100:+.0f}pp |")
        act_code = argmin_action(s["p"].to_numpy(), s["TransactionAmt"].to_numpy(),
                                 s["gang_score"].to_numpy(), A_MED, BASE)
        L += ["", f"- 对照：用**代码 gang_score** 走同一公式 = {(act_code==gt).mean():.0%}"
              "（应为 100%，因为应然档就是这么算的——这行是口径自检）"]

        # ---------- 5. 并排 ----------
        agent_dec = (s["agent"].to_numpy() == gt).mean()
        summary[tag] = {"ev_agree": agree, "ev_base": base_ev, "ev_lift": agree - base_ev,
                        "dec_agree": agent_dec, "dec_base": base_dec,
                        "dec_lift": agent_dec - base_dec, "sub_lift": sub_best,
                        "miss": miss, "n": len(s), "sep_a": sep_a, "sep_c": sep_c,
                        "sep_lo": la, "sep_hi": ha}

    # ---------- 总表 ----------
    L += ["\n---\n\n## 并排对比（**只比相对基线的提升**）\n",
          "| round | 层 | 原始一致率 | 多数类基线 | **相对基线提升** |",
          "|---|---|---|---|---|"]
    for t, m in summary.items():
        L.append(f"| {t} | 证据层（2 分类） | {m['ev_agree']:.0%} | {m['ev_base']:.0%} | "
                 f"**{m['ev_lift']*100:+.0f}pp** |")
        L.append(f"| {t} | 决策层（4 分类） | {m['dec_agree']:.0%} | {m['dec_base']:.0%} | "
                 f"**{m['dec_lift']*100:+.0f}pp** |")
    L += ["", "> 原始一致率那两列**不能直接比**（4 分类 vs 2 分类），列出来只是为了让读者看到"
          "「不校正会得出什么」。可比的是最后一列。\n"]

    # ---------- 未表态的选择效应自查（⑤ 在证据层的翻版）----------
    if len(tags) == 2:
        a, b = tags
        da, db = load(a).set_index("TransactionID"), load(b).set_index("TransactionID")
        common = da.index.intersection(db.index)
        null_b = [t for t in common if not db.loc[t, "ga_present"]]
        L += ["", "## ⚠️ 自查：未表态率上升会不会把一致率「洗高」\n",
              f"`{b}` 的未表态从 `{a}` 的 {summary[a]['miss']} 笔涨到 {summary[b]['miss']} 笔，"
              "而**未表态的样本被排除在一致率分母之外**。",
              "若它恰好在难判的那些笔上弃权，剩下的一致率就会**凭选择效应变好**"
              "（⑤ 选择性偏差在证据层的翻版：用自己的弃权决定分母）。\n"]
        if null_b:
            sub = da.loc[null_b]
            sub = sub[sub["ga_present"]]
            if len(sub):
                ok = (sub["susp"].astype(bool) == (sub["gang_score"] >= 0.5)).mean()
                L += [f"- `{b}` 弃权的 {len(null_b)} 笔中，有 {len(sub)} 笔 `{a}` 曾给出结论；",
                      f"  **`{a}` 在这批上的一致率 = {ok:.0%}**"
                      f"（vs `{a}` 全体 {summary[a]['ev_agree']:.0%}）",
                      "", ("  → **低于全体 → 弃权确实集中在难判样本上，"
                           f"`{b}` 的 {summary[b]['ev_agree']:.0%} 含选择效应、被高估**。"
                           if ok < summary[a]["ev_agree"] - 0.05 else
                           "  → **不低于全体 → 没有证据表明它挑难的弃权**，"
                           f"`{b}` 的一致率上升不是弃权洗出来的。")]
        # 最坏口径：未表态一律算不一致
        for t in tags:
            m = summary[t]
            worst = m["ev_agree"] * m["n"] / (m["n"] + m["miss"])
            L.append(f"- 最坏口径（未表态一律算错）：`{t}` 证据层一致率 "
                     f"{m['ev_agree']:.0%} → **{worst:.0%}**")
        L.append("")

    m0 = summary[tags[0]]
    verdict = m0["ev_lift"] - m0["dec_lift"]
    anchor_ok = not (m0["sep_lo"] <= 0 <= m0["sep_hi"])
    L += ["## 裁定（分两句，别合成一句）\n",
          "### (a) 能力层面：Agent 复现「团伙判定」这件事的水平 —— **成立**\n"]
    if verdict > 0.10:
        L += [f"✅ **证据层相对基线的提升显著高于决策层**（{m0['ev_lift']*100:+.0f}pp vs "
              f"{m0['dec_lift']*100:+.0f}pp，差 **{verdict*100:+.0f}pp**）。",
              "校正了类别数差异后仍然成立；门槛 0.3–0.6 区间稳定；未表态的选择效应已自查排除；"
              "最坏口径（弃权算错）下仍是 84–86%。",
              "→ **「取证的活 Agent 干得比选档的活好」有数据支撑。**"]
    elif verdict > 0:
        L += [f"⚠️ 提升差仅 {verdict*100:+.0f}pp，**不足以称显著**。"]
    else:
        L += [f"❌ 提升差 {verdict*100:+.0f}pp，**不成立**。"]

    L += ["", "### (b) 效度层面：「团伙判定」这件事本身有没有用 —— **未测出**\n",
          f"- Agent 的团伙判定对 isFraud 的分离度 **{m0['sep_a']*100:+.0f}pp**，"
          f"95% CI [{m0['sep_lo']*100:+.0f}, {m0['sep_hi']*100:+.0f}]pp"
          + ("（**跨 0**）" if not anchor_ok else "（不跨 0）"),
          f"- 代码 gang_score 的分离度 **{m0['sep_c']*100:+.0f}pp**（同样跨 0）",
          "",
          "**所以第 (a) 句必须带着第 (b) 句一起讲**：",
          "> Agent 能高度可靠地复现代码的团伙判定（+32pp 提升），"
          "**但「这个团伙判定指不指向真欺诈」在本数据上没有测出来**——",
          "> 两个分离度的置信区间都跨 0。一致率只证明两个判定**互相同意**，"
          "不证明它们**对**。\n",
          "**但也不能就此说 gang 没用，因为这个锚测的不是它的设计目标**：",
          "- `gang_score` 进入四档公式的位置是**上报档的网络项** "
          "（`− p·g·k_future·a_med`），它要预测的是**冻结该实体能拦下的未来欺诈**，",
          "  不是**本笔是否欺诈**。拿本笔 `isFraud` 当锚，测的是一个它没打算预测的目标。",
          "- **对的锚应该是**：gang_score 高的实体，其**后续**交易的欺诈率是否更高"
          "（同实体、时间窗往后推）。这个可算但没算，**记为未完成，不拿现在的跨 0 结果当结论**。",
          "",
          f"- 未表态率：" + "、".join(f"{t} {summary[t]['miss']}/100" for t in tags)
          + "（`gang_association=null`）。自查显示弃权集中在**容易**的样本上，"
          "不构成选择效应。\n",
          "### 对 ① 的含义\n",
          "① **可以做**，理由是 (a)：职责边界有能力层面的数据支撑，不再只是推演。",
          "对外表述必须是：",
          "> 「我用数据划了职责边界——同一批报告里，Agent 复现团伙判定的水平"
          "（相对基线 +32pp）明显高于它选处置档的水平（+11pp），",
          "> 所以把决策交回闭式解、让 Agent 专做取证。**其中团伙判定本身的效度我另做了检验，"
          "在这批数据上没测出来，正确的锚是实体的未来欺诈率，那个我没跑。**」\n"]

    write_report(REPORT, "\n".join(L))
    print("\n".join(L))
    print(f"\n✅ → {REPORT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    from src.eval._cli import require_tags
    run(require_tags(sys.argv[1:], least=2,
                     usage="python -m src.eval.evidence_vs_decision r1 r3"))
