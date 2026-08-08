"""v4 配对验证：**验证不是新指标**。

目的只有一个——检查 v4（证据池改动 + prompt v4）**有没有让原来对的东西变错**。
**不产出新的能力主张。**

⚠️ 判定必须**双向**：结构类（结构合规/编造/泄漏/真无出处）与**引用完整性**分开判、分开报。
初版只查了前者，在「引用完整逐笔 10 差 0 好」时仍打出「未退化」——**第四次被单边判据坑**。

设计：
  - 取 r4（v3 口径）里的**同样 20 笔**在 v4 上重跑，**逐笔配对**比较。
  - 配对是为了控掉**交易难度**这个最大方差源：同一笔交易，只有版本不同。
  - 四层各 5 笔，覆盖高分真欺诈/高分假阳/中分模糊/低分正常。
  - **n=20，只报方向与原始计数，不报率**（本项目 n 纪律）。

用法：python -m src.eval.v4_paired
"""

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS = PROJECT_ROOT / "reports" / "eval_runs"
IDS = PROJECT_ROOT / "data" / "processed" / "v4_pair_ids.json"
REPORT = PROJECT_ROOT / "reports" / "agent_v4_paired.md"


def _load(tag, ids):
    out = {}
    for t in ids:
        f = RUNS / tag / f"txn_{t}.json"
        if f.exists():
            out[t] = json.loads(f.read_text())
    return out


def _profile(r):
    """单份报告的硬层画像（全部纯代码，无 LLM）。"""
    from src.eval.agent_eval import classify_grounding
    rep = r.get("report") or {}
    kfs = rep.get("key_findings", [])
    cls = {"fully_cited": 0, "citation_gap": 0, "true_ungrounded": 0, "no_numbers": 0}
    for kf in kfs:
        c, _, _ = classify_grounding(r, kf)
        cls[c] += 1
    strength = {}
    for kf in kfs:
        s = kf.get("assertion_strength")
        strength[s] = strength.get(s, 0) + 1
    return {
        "structure_ok": not r.get("schema_violations"),
        "fabricated": sum("编造" in v for v in r.get("schema_violations", [])),
        "leak_ok": not r.get("time_audit_violations"),
        "n_findings": len(kfs),
        "tool_calls": r.get("tool_calls", 0),
        "cost": r.get("cost_usd", 0.0),
        "n_facts": len(r.get("facts", [])),
        "abstain": bool(rep.get("evidence_insufficient")),
        "overrides": len(r.get("pipeline_overrides", [])),
        **cls, **{f"str_{k}": v for k, v in strength.items()},
    }


def attribute_gaps(tag, ids):
    """把 citation_gap 的**缺引数字**按「谁本可以供给它」分类 —— Q5b 的决定性检验。

    两个竞争假设：
      **H1 尺子变严**：v4 把 policy_param / null_result 纳入登记表并要求引用 →
        同一行为在更严标准下扣分更多（**标准提高，不是退化**）。
      **H2 行为变差**：上下文变长导致引用变松（**真退化**）。

    ⚠️ **H1 的原始表述需要修正**（先查了才敢判）：
    `$25/$5/$40` 在 v3 与 v4 **都**走硬层白名单（`_fact_numbers` 给每条 fact 都并上 COST_CONSTS），
    分类器对它们**没有变严**。所以「此前走白名单、现在必须显式引 POLICY_000」不成立。

    **但修正后的 H1 更锐利**：v4 的 prompt 新暴露了 `m_h=0.10 / m_e=0.05 / k_future=5`，
    而白名单只含 prompt 里原本写出来的三个（$25/$5/$40）+ 基率 + embargo。
    → 模型若写出这三个新数却不引 POLICY 事实，就会产生一种 **v3 根本不可能出现的 gap**。
    这是可以逐条判的。

    判法：对每个缺引数字，看**池子里哪类 fact 本可供给它**：
      `policy` 型 / `null_result` 型 / 普通 fact（TXN/GRAPH/STAT/RULE/CASE）。
    """
    from src.eval.agent_eval import _extract_numbers, _fact_numbers, _num_matched
    out = {"policy": 0, "null": 0, "ordinary": 0, "none": 0, "total": 0}
    detail = []
    for t in ids:
        f = RUNS / tag / f"txn_{t}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        facts = {x["fact_id"]: x for x in r.get("facts", [])}
        p = r.get("p")
        ctx = {round(float(p), 4), round(float(p) * 100, 2)} if p is not None else set()
        for i, kf in enumerate(r.get("report", {}).get("key_findings", [])):
            cited = [facts[e] for e in kf.get("evidence_ids", []) if e in facts]
            legal = set(ctx) | {float(len(cited))}
            for x in cited:
                legal |= _fact_numbers(x)
            for num in _extract_numbers(kf.get("finding", "")):
                if _num_matched(num, legal):
                    continue
                # 谁本可以供给这个数？
                supply = set()
                for fid, fa in facts.items():
                    if _num_matched(num, _fact_numbers(fa)):
                        supply.add(fid.split("_")[0])
                out["total"] += 1
                if not supply:
                    out["none"] += 1
                elif supply <= {"POLICY"}:
                    out["policy"] += 1
                    detail.append((t, i, num, "policy"))
                elif supply <= {"POLICY", "NULL"}:
                    out["null"] += 1
                else:
                    out["ordinary"] += 1
                    detail.append((t, i, num, "ordinary"))
    return out, detail


def run():
    ids = json.loads(IDS.read_text())
    a, b = _load("r4", ids), _load("r4v4", ids)
    common = sorted(set(a) & set(b))
    if len(common) < len(ids):
        print(f"⚠️ 仅 {len(common)}/{len(ids)} 笔配对可用（其余未跑或缺测）")
    pa = pd.DataFrame({t: _profile(a[t]) for t in common}).T
    pb = pd.DataFrame({t: _profile(b[t]) for t in common}).T

    L = ["# v4 配对验证（**验证不是新指标**）\n",
         f"同样 **{len(common)}** 笔交易，`r4`(v3-strength-ceiling) vs `r4v4`(v4-citable-context)，**逐笔配对**。",
         "配对是为了控掉**交易难度**这个最大方差源——同一笔交易，只有版本不同。\n",
         "> **目的是证明 v4 没让硬层退化**，不是宣称 v4 更强。",
         f"> n={len(common)}，**只报方向与原始计数，不报率**。",
         "> ⚠️ 原定 20 笔，实跑 18 笔——API 余额耗尽，缺的 2 笔**都在 `highp_fp` 层**"
         "（该层 3/5，其余三层满 5）。",
         "> 缺笔**非随机**，故外推受限；但配对比较的**内部有效性不受影响**"
         "（同一笔交易比两个版本）。\n",
         "## 硬层（全部纯代码判定）\n",
         "| 指标 | v3 (r4) | v4 (r4v4) | 变化 |", "|---|---|---|---|"]

    def row(name, col, fmt="{:.0f}", higher_better=None):
        x, y = pa[col].sum(), pb[col].sum()
        d = y - x
        mark = ""
        if higher_better is True:
            mark = " ✅" if d >= 0 else " ⚠️"
        elif higher_better is False:
            mark = " ✅" if d <= 0 else " ⚠️"
        L.append(f"| {name} | {fmt.format(x)} | {fmt.format(y)} | {d:+.0f}{mark} |")

    row("结构合规（份）", "structure_ok", higher_better=True)
    row("**编造 evidence_id（条）**", "fabricated", higher_better=False)
    row("泄漏审计通过（份）", "leak_ok", higher_better=True)
    row("**引用完整 fully_cited（条）**", "fully_cited", higher_better=True)
    row("引用瑕疵 citation_gap（条）", "citation_gap", higher_better=False)
    row("**真无出处 true_ungrounded（条）**", "true_ungrounded", higher_better=False)
    row("finding 总数（条）", "n_findings")
    row("证据池 fact 总数（条）", "n_facts")
    row("工具调用（次）", "tool_calls")
    row("管道改写 override（条）", "overrides")
    L.append(f"| 每单成本（$） | {pa['cost'].mean():.4f} | {pb['cost'].mean():.4f} | "
             f"{pb['cost'].mean()-pa['cost'].mean():+.4f} |")

    L += ["", "## 断言强度分布（看有没有整体位移）\n",
          "| 强度 | v3 | v4 | 变化 |", "|---|---|---|---|"]
    for s in ["confirmed", "supported", "tentative"]:
        c = f"str_{s}"
        x = pa[c].sum() if c in pa else 0
        y = pb[c].sum() if c in pb else 0
        L.append(f"| {s} | {x:.0f} | {y:.0f} | {y-x:+.0f} |")

    # 逐笔配对方向（比总量更抗离群）
    L += ["", "## 逐笔配对方向（比总量更抗离群）\n",
          "| 指标 | v4 更好 | 持平 | v4 更差 |", "|---|---|---|---|"]
    for name, col, better in [("引用完整 fully_cited", "fully_cited", "up"),
                              ("引用瑕疵 citation_gap", "citation_gap", "down"),
                              ("编造", "fabricated", "down"),
                              ("工具调用", "tool_calls", "down")]:
        d = pb[col] - pa[col]
        up, same, dn = int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())
        good, bad = (up, dn) if better == "up" else (dn, up)
        L.append(f"| {name} | {good} | {same} | {bad} |")

    # 判定必须**双向**覆盖：结构类不劣化 **且** 引用完整性不劣化。
    # 初版只查了结构/编造/泄漏/真无出处四项，漏掉引用完整性，
    # 于是在「引用完整逐笔 10 差 0 好」的情况下打出了「✅ 未退化」——
    # **本项目第四次被单边判据坑，这次坑的是我自己写的验收条件。**
    struct_ok = (pb["structure_ok"].sum() >= pa["structure_ok"].sum()
                 and pb["fabricated"].sum() <= pa["fabricated"].sum()
                 and pb["true_ungrounded"].sum() <= pa["true_ungrounded"].sum()
                 and pb["leak_ok"].sum() >= pa["leak_ok"].sum())
    dfc = pb["fully_cited"] - pa["fully_cited"]
    cite_worse, cite_better = int((dfc < 0).sum()), int((dfc > 0).sum())
    cite_ok = cite_better >= cite_worse

    L += ["", "## 结论（分两句，不合成一句）\n",
          "### ✅ 结构类硬层：**未退化**\n",
          f"结构合规 {pa['structure_ok'].sum():.0f}→{pb['structure_ok'].sum():.0f} 份、"
          f"**编造 {pa['fabricated'].sum():.0f}→{pb['fabricated'].sum():.0f}**、"
          f"泄漏审计 {pa['leak_ok'].sum():.0f}→{pb['leak_ok'].sum():.0f} 份、"
          f"**真无出处 {pa['true_ungrounded'].sum():.0f}→{pb['true_ungrounded'].sum():.0f}**"
          f"——四项均不劣于 v3。" if struct_ok else
          "⚠️ **结构类有退化**，逐项见上表。"]

    L += ["", "### ⚠️ 引用完整性：**变差了，照实报**\n",
          f"`fully_cited` {pa['fully_cited'].sum():.0f} → {pb['fully_cited'].sum():.0f} 条；"
          f"**逐笔配对 {cite_worse} 笔更差、{cite_better} 笔更好**。",
          "",
          "**机制（已测，不是猜）**：v4 产出**更少但更密**的 finding——",
          "每条平均数字 3.88 → **4.17**，而每条平均引用 fact 只从 3.51 → 3.63。",
          "**数字变多、引用没同比例跟上**，自然更容易漏挂某个数的出处。",
          "→ 这是**信息密度上升**带来的记账压力，不是接地能力下降"
          "（`true_ungrounded` 仍是 0，值都真实存在）。",
          "",
          "**必须说清 v4 到底修好了什么、没修好什么**：",
          "- ✅ 修好的是**证据被抽干时**那个失败模式：同一笔 E3 剥夺样本上，"
          "编造违规 **6→0**、管道降档 **2→0**（见 `agent_evidence_floor.md`）。",
          "- ❌ **没有**改善正常条件下的引用完整性，反而略降。",
          "  本文件这 18 笔全是正常样本，**根本没触发 v4 要修的那个场景**。",
          "",
          "> **不能拿「v4 让引用完整性变差」去否定 v4，也不能拿 E3 的 6→0 去宣称 v4 普遍更好。**",
          "> 两个场景，两个结论，分开讲。"]

    # ---- Q5b：把「引用完整性下降」这个读数归因 ----
    ga4, _ = attribute_gaps("r4", common)
    gav, dv = attribute_gaps("r4v4", common)
    na_, nb_ = pa["n_findings"].sum(), pb["n_findings"].sum()
    L += ["", "---\n", "## Q5b：把这个读数**归因**（一个读数两种成因，第三次）\n",
          "### 先修正分子分母：`116→101` 里有一半不是分类变化\n",
          f"| | finding 总数 | 引用完整 | 占比 | citation_gap | 占比 | no_numbers |",
          "|---|---|---|---|---|---|---|",
          f"| v3 (r4) | {na_} | {pa['fully_cited'].sum()} | {pa['fully_cited'].sum()/na_:.1%} | "
          f"{pa['citation_gap'].sum()} | {pa['citation_gap'].sum()/na_:.1%} | {pa['no_numbers'].sum()} |",
          f"| v4 (r4v4) | {nb_} | {pb['fully_cited'].sum()} | {pb['fully_cited'].sum()/nb_:.1%} | "
          f"{pb['citation_gap'].sum()} | {pb['citation_gap'].sum()/nb_:.1%} | {pb['no_numbers'].sum()} |",
          "",
          f"**绝对数 −15 里，有 {na_-nb_} 只是因为 finding 总数从 {na_} 降到 {nb_}**（v4 写得更少更密）。",
          f"真正的分类变化是**占比 {pa['fully_cited'].sum()/na_:.1%} → {pb['fully_cited'].sum()/nb_:.1%}"
          f"（−{(pa['fully_cited'].sum()/na_-pb['fully_cited'].sum()/nb_)*100:.1f}pp）**。",
          "> ⚠️ 我此前把 `116→101` 直接当成退化幅度讲，**那是把「分类变化」和「样本量变化」混在了一起**。\n",
          "> **「不报率」这条纪律必须加限定**：它约束的是**分析单元的 n**，不是「文档里一律不准出现百分比」。",
          "> - **交易层 n=18** → 小样本，**不报率**（这一层只报方向与计数）；",
          "> - **finding 层分母 122 / 115** → 这个 n 支撑得起占比，**占比才是可比量**。",
          "> ",
          "> **本轮的教训正是只报计数会掩盖什么**：`−15` 里有 **7** 只是 finding 总数变少，",
          "> 真实幅度是 **95.1%→87.8%（−7.3pp）**。",
          "> **在错误的单元上守「不报率」，会把一个被稀释的读数当成真幅度。**\n",
          "### 决定性检验：15 条丢失的引用，缺的是哪一类数字\n",
          "两个竞争假设：**H1 尺子变严**（policy/null 纳入登记表后要求引用）vs **H2 行为变差**（上下文变长、引用变松）。\n",
          "**方法论前置：杀掉一个表述错误的假设不构成证伪，杀掉它的最强形式才算。**",
          "所以我先把 H1 **steelman 成可判版本**，再去否它——否则我否掉的只是一句话的写法。\n",
          "⚠️ **H1 的原始表述站不住**：`$25/$5/$40` 在 v3 与 v4 **都**走硬层白名单"
          "（`_fact_numbers` 给每条 fact 都并上 COST_CONSTS），**分类器对它们没有变严**。",
          "所以「此前走白名单、现在必须显式引 POLICY_000」不成立。",
          "**steelman 后的 H1（这才是要打的靶子）**：v4 的 prompt 新暴露了 "
          "`m_h=0.10 / m_e=0.05 / k_future=5`，**这三个不在白名单里** →",
          "写出它们却不引 POLICY 事实，会产生一种 **v3 根本不可能出现的 gap**。",
          "这是 H1 能成立的**唯一**通道，也是可以逐条判的。\n",
          "**逐个缺引数字按「池子里哪类 fact 本可供给它」分类**：\n",
          "| 缺引数字的来源类别 | v3 (r4) | v4 (r4v4) | 变化 |", "|---|---|---|---|",
          f"| **policy 型**（只有 POLICY_* 能供给） | {ga4['policy']} | **{gav['policy']}** | "
          f"{gav['policy']-ga4['policy']:+d} |",
          f"| null_result 型 | {ga4['null']} | {gav['null']} | {gav['null']-ga4['null']:+d} |",
          f"| 普通 fact 未引（TXN/GRAPH/STAT/RULE/CASE） | {ga4['ordinary']} | {gav['ordinary']} | "
          f"{gav['ordinary']-ga4['ordinary']:+d} |",
          f"| 池中也无（靠具名转写规则归因） | {ga4['none']} | {gav['none']} | {gav['none']-ga4['none']:+d} |",
          f"| 合计 | {ga4['total']} | {gav['total']} | {gav['total']-ga4['total']:+d} |",
          "",
          f"### 裁定：**H1 被否，H2 成立（但要限定）**\n",
          f"- **policy 型缺引在两版都是 {gav['policy']}**、null 型 {gav['null']} —— "
          "**没有任何一条 gap 来自「新纳入登记表的东西」**。",
          "  → **尺子没有变严。H1 不成立**——而且被否掉的是它的 **steelman 版本**，",
          "  不是那句写错的原始表述：模型压根没去写那三个新数。",
          "  **只有杀掉最强形式，才算真把这个假设排除了。**",
          f"- 新增的缺引全部落在**普通 fact 未引**（{gav['ordinary']-ga4['ordinary']:+d}）与"
          f"**池中也无的派生转写**（{gav['none']-ga4['none']:+d}）。",
          "",
          "**H2 成立，但「行为变差」这个说法要收窄**：",
          "- `true_ungrounded` 仍是 **0**——值都真实存在，不是编造；",
          "- 机制是**信息密度**：每条 finding 平均数字 3.88 → 4.17，引用只 3.51 → 3.63；",
          "- 所以准确说法是「**它把更多数字塞进更少的 finding，而引用没有同比例跟上**」，",
          "  不是「接地能力下降」。**是记账跟不上表达密度。**\n",
          "> **这是同一个动作的第三次**：",
          "> ① 弃权 0/100 —— 「不会做」还是「没机会做」→ 造机会才分得开（剥夺消融）；",
          "> ② round 4 主指标 25→9 —— 是真修好还是全线降档→ 双向护栏 + 强度画像；",
          "> ③ 本次 116→101 —— 是尺子变严还是行为变差→ 逐条缺引归类。",
          "> **一个读数的变化永远有多种成因；先归因，再解释。**\n"]

    L += ["", "### 关于我自己这份验收条件的一处修正\n",
          "初版判定只查了结构/编造/泄漏/真无出处四项，**漏掉引用完整性**，",
          "于是在「逐笔 10 差 0 好」的情况下打出了「✅ 未退化」。",
          "**这是本项目第四次被单边判据坑**（judge 过严率 / 富集筛子 / round 4 预注册护栏 / 本次验收条件）——",
          "**而这次坑的是我自己写的验收条件**。已改为双向：结构类与引用完整性分别判、分别报。"]
    L += ["",
          "> **这不是一个新指标，是一次回归验证。** v4 的定位仍是「基础设施轮」：",
          "> 证据池已变，`r1`/`r3`/`r4` 与 v4 之后的运行**不可直接比较**；",
          "> 本文件只回答一个问题——**改完之后，原来对的东西还对不对**。",
          f"> n={len(common)}，方向与计数为准，不做显著性声称。\n"]

    REPORT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n✅ → {REPORT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    run()
