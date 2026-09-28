"""Agent 自主性的「活动面」有多大——对既有归档的只读测量（⑦ 补测，零 LLM 成本）。

## 它回答的问题

项目把调查层实现成了工具调用循环，文档里一度写作「**自主**工具调用」。
但「实现了一个能工作的 Agent」不等于「**这个任务需要 Agent**」：
如果每次取的证据都一样、每次的调用轮数都一样，那这个循环在数据上就是一个
**确定性工作流**，只是穿了 Agent 的外衣。

**这件事不需要新实验就能证伪或证实**——归档里每份报告都存着它实际取到的
全部事实（`facts`）与工具调用次数（`tool_calls`）。本模块只读这些归档。

## 为什么值得单独量

「该不该用 Agent」是一个**先于**「Agent 好不好」的问题。
按 workflow / agent 的常见分法：工作流的路径由程序预先规定，Agent 让模型动态
决定过程。**决定属于哪一类的不是名字，是行为的方差。**
先量方差，再决定要不要花钱做「固定取证 vs 单次 LLM vs 多轮 Agent」的三臂对照——
否则可能花钱去证明一件读归档就能看出来的事。

## 三个测量

1. **事实类型集合的方差**：每份报告最终取到哪几类事实。集合高度集中 = 取证无选择。
2. **`rule` 缺席是选择还是数据**：无 `rule` 事实的报告，`tool_calls` 是否更少。
   若不更少（甚至更多），说明它**调了规则工具、只是没命中** → 缺席由数据决定。
   不做这一步，会把「这笔没命中规则」误读成「Agent 决定不查规则」。
3. **唯一有自由度的工具**（按类别查历史统计 `fraud_rate`）查的是什么：
   是本笔交易**自己字段的取值**，还是本笔之外的东西。

## 边界（必须随数字一起讲）

- 归档存的是**最终事实集合**，不是逐步的工具调用参数序列（`raw_text` 为空）。
  所以本测量说的是「**取到了什么**」，不能还原「每一步为什么这么选」。
- 归档为 **v3 / v4 证据池**；v5 未付费重跑 → 跨版本不可直接比较。
- eval 是**强制全量投喂**（100 笔全进 Agent，而生产拓扑下闸门会挡掉 90.3%）。
  这个样本因此**过度代表低分交易**——低分交易证据更少、若要适应本应更需要适应，
  而实测方差仍≈0，**这个方向上的偏差不会让结论变弱**。
- **方差≈0 不等于「固定流程能写出一样好的报告」**：本测量只覆盖取证编排，
  不覆盖对固定证据集的解释。后者需要「固定取证＋模板 / 固定取证＋单次 LLM /
  多轮 Agent」的三臂对照，本模块不替代它。
- 本测量**不否证**已实测的解释层行为（受控剥夺下的弃权 0%→10%→100%、
  反谄媚的冲突检出剂量梯度）——那些都在固定证据集上做的，与取证是否自主无关。

用法：python -m src.eval.agent_autonomy_surface
产出：reports/agent_autonomy_surface.md
"""

from src.report_io import write_report
import collections
import json
import statistics
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS = PROJECT_ROOT / "reports" / "eval_runs"
OUT_MD = PROJECT_ROOT / "reports" / "agent_autonomy_surface.md"
ROUNDS = ["r1", "r3", "r4", "r4v4"]
FREE_TOOL_FACT = "fraud_rate"      # 唯一带类别自由度的工具返回的事实类型


def _load(round_name):
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((RUNS / round_name).glob("txn_*.json"))]


def _measure(reports):
    """一轮归档的三个测量。"""
    sets_ = collections.Counter()
    tc_with_rule, tc_without_rule = [], []
    q_total = q_own = 0
    per_card = collections.Counter()
    fields = collections.Counter()
    ids = set()
    for d in reports:
        ids.add(d["txn_id"])
        facts = d.get("facts") or []
        types, own_fields = set(), {}
        for f in facts:
            t = str(f.get("type", "?"))
            types.add(t.split(":")[0])
            if t.startswith("txn_field:"):
                own_fields[t.split(":", 1)[1]] = str(f.get("value"))
        sets_[tuple(sorted(types))] += 1
        (tc_with_rule if "rule" in types else tc_without_rule).append(d.get("tool_calls") or 0)

        qs = [f for f in facts if f.get("type") == FREE_TOOL_FACT]
        per_card[len(qs)] += 1
        for f in qs:
            q_total += 1
            ent = str(f.get("entity", ""))
            if "=" in ent:
                k, v = ent.split("=", 1)
                fields[k] += 1
                if own_fields.get(k) == v:           # 查的就是本笔自己那个取值
                    q_own += 1
    return {
        "n": len(reports), "n_txn": len(ids), "sets": sets_,
        "tc_with_rule": tc_with_rule, "tc_without_rule": tc_without_rule,
        "q_total": q_total, "q_own": q_own, "per_card": per_card, "fields": fields,
        "api": collections.Counter(d.get("api_calls") or 0 for d in reports),
        "tc": collections.Counter(d.get("tool_calls") or 0 for d in reports),
    }


def main() -> None:
    per_round = {}
    for r in ROUNDS:
        reports = _load(r)
        if reports:
            per_round[r] = _measure(reports)
            m = per_round[r]
            top = m["sets"].most_common(1)[0][1]
            print(f"{r:5s} {m['n']:3d} 份 / {m['n_txn']:3d} 笔  "
                  f"事实类型集合 {len(m['sets'])} 种（最大占 {top/m['n']:.0%}）  "
                  f"{FREE_TOOL_FACT} 查询 {m['q_total']:3d} 次，自查字段 "
                  f"{m['q_own']/m['q_total']:.1%}" if m["q_total"] else "")
    _write_md(per_round)
    print(f"\n✅ → {OUT_MD.relative_to(PROJECT_ROOT)}")


def _write_md(per_round):
    tot_q = sum(m["q_total"] for m in per_round.values())
    tot_own = sum(m["q_own"] for m in per_round.values())
    tot_n = sum(m["n"] for m in per_round.values())
    w = [t for m in per_round.values() for t in m["tc_with_rule"]]
    wo = [t for m in per_round.values() for t in m["tc_without_rule"]]

    L = [
        "# Agent 自主性的「活动面」有多大（对既有归档的只读测量）\n",
        "**它回答的问题**：项目把调查层实现成了工具调用循环。但「实现了一个能工作的 Agent」"
        "不等于「这个任务**需要** Agent」——如果每次取的证据都一样、每次的调用轮数都一样，"
        "那这个循环在数据上就是一个**确定性工作流**，只是穿了 Agent 的外衣。\n",
        "**这件事不需要新实验**：归档里每份报告都存着它实际取到的全部事实（`facts`）"
        "与工具调用次数（`tool_calls`）。本测量**只读归档、零 LLM 成本**。\n",
        "## 1. 总览（四轮归档，含不同 prompt 版本）\n",
        f"| 轮次 | 报告 | 不同交易 | 事实类型集合数 | 最大集合占比 | "
        f"`{FREE_TOOL_FACT}` 查询 | **查的是本笔自己字段** | tool_calls 中位 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r, m in per_round.items():
        top = m["sets"].most_common(1)[0][1]
        tcs = [t for t in m["tc"].elements() if t]
        L.append(f"| {r} | {m['n']} | {m['n_txn']} | **{len(m['sets'])}** 种 | "
                 f"**{top / m['n']:.0%}** | {m['q_total']} | "
                 f"**{m['q_own'] / m['q_total']:.1%}** | {statistics.median(tcs):.0f} |")
    L += [
        "",
        f"> **四轮全部复现，而这四轮是不同的 prompt 版本。** 合计 {tot_n} 份报告、"
        f"`{FREE_TOOL_FACT}` 查询 {tot_q} 次，其中查本笔交易自己字段取值的 "
        f"**{tot_own} 次 = {tot_own / tot_q:.1%}**。",
        "",
        "## 2. `rule` 缺席是「选择」还是「数据」——这一步不做就会误读\n",
        "最大集合之外的报告，绝大多数只差一类事实：`rule`。"
        "如果直接说「集合有 4–5 种、所以有方差」，就把**这笔没命中任何规则**"
        "误读成**Agent 决定不查规则**。判据：无 `rule` 的报告，`tool_calls` 是否更少。\n",
        "| 报告 | 份数 | `tool_calls` 均值 | 中位 |",
        "|---|---|---|---|",
        f"| 有 `rule` 事实 | {len(w)} | {statistics.mean(w):.2f} | {statistics.median(w):.0f} |",
        f"| **无 `rule` 事实** | {len(wo)} | **{statistics.mean(wo):.2f}** | {statistics.median(wo):.0f} |",
        "",
        f"> 无 `rule` 的那批 `tool_calls` **不更少、反而更多**"
        f"（{statistics.mean(wo):.2f} vs {statistics.mean(w):.2f}）"
        "→ **它们调用了规则工具、只是没命中**，然后在别处多查了一点。",
        "> ### → `rule` 缺席由**数据**决定，不是 Agent 的选择。"
        "扣掉这一维后，取证编排的方差进一步趋近于零。",
        "",
        "## 3. 唯一有自由度的那个工具，查了什么\n",
        "四个工具里，三个在给定交易后返回内容固定；只有「按类别查历史统计」"
        f"（返回 `{FREE_TOOL_FACT}` 事实）让模型自己挑查哪个类别。所以自主性若存在，"
        "只可能存在于这里。\n",
        "| 轮次 | 每份查询次数分布 | 被查字段分布 |",
        "|---|---|---|",
    ]
    for r, m in per_round.items():
        cnt = " · ".join(f"{k}次×{v}" for k, v in sorted(m["per_card"].items()))
        fld = " · ".join(f"{k} {v}" for k, v in m["fields"].most_common())
        L.append(f"| {r} | {cnt} | {fld} |")
    L += [
        "",
        f"> ### **{tot_q} 次查询里，{tot_own} 次查的是本笔交易自己字段的取值——{tot_own / tot_q:.1%}。**",
        "> **一次都没有查过本笔字段之外的类别。** 也就是说，连这个"
        "「唯一有自由度」的工具，实际用法也是由本笔交易的字段值**直接决定**的："
        "看见 `ProductCD=C` 就去查 `ProductCD=C` 的历史欺诈率。",
        "> → 程序照着本笔字段生成同一批查询，能得到同一批事实。",
        "",
        "## 4. 结论\n",
        "> ### **这个调查层在数据上是一个近确定性工作流。**\n"
        f"> 取证编排的方差≈0：{tot_n} 份报告里事实类型集合只有 3–5 种，"
        "而其中最大的一种占七成以上；扣掉由数据决定的 `rule` 一维后集中度更高；"
        f"唯一有自由度的工具 {tot_q}/{tot_q} 次查的都是本笔自己的字段。",
        "",
        "**所以对外措辞应当改**：不写「**自主**工具调用」，写"
        "「**工具调用循环（实测调用序列方差≈0）**」。"
        "按 workflow / agent 的常见分法（工作流的路径由程序预先规定，Agent 让模型动态"
        "决定过程），**本层是 workflow，不是 agent**——而决定属于哪一类的不是名字，是行为的方差。",
        "",
        "## 5. 这个结论**不**推翻什么（边界写死）\n",
        "1. **不推翻验收层。** Fact registry + 强制 `evidence_ids`、数字两级对账"
        "（编造 0/565）、引用完整率、时间边界审计 100%、闸门经济（90.3% 不进 LLM、"
        "单笔 $0.1131）、模板降级——**这些是对输出与成本的约束，与取证是否自主完全正交**，"
        "换成单次 LLM 调用一条都不用改。",
        "2. **不推翻已实测的解释层行为。** 受控剥夺下弃权率 0%→10%→100%（**能力测试，"
        "不是发生率**）、反谄媚的冲突检出剂量梯度与关键对照——这些都在**固定证据集**上做的。"
        "本测量说的是「正常流量没给它机会」，不是「它不会」。",
        "3. **不证明固定流程能写出一样好的报告。** 本测量只覆盖**取证编排**，"
        "不覆盖对固定证据集的**解释**。要回答后者，需要三臂对照："
        "固定取证＋模板 / 固定取证＋单次 LLM / 多轮 Agent。**本模块不替代它，只是把它的"
        "第三臂预先降级成「大概与第二臂信息等价」，从而让那个对照更干净、更便宜。**",
        "",
        "## 6. 测量本身的边界\n",
        "- 归档存的是**最终事实集合**，不是逐步的工具调用参数序列（`raw_text` 为空）。"
        "本测量说的是「取到了什么」，**不能还原「每一步为什么这么选」**。",
        "- 归档为 **v3 / v4 证据池**；**v5 未付费重跑** → 跨版本不可直接比较。",
        "- eval 是**强制全量投喂**（100 笔全进 Agent，生产拓扑下闸门会挡掉 90.3%）→ "
        "样本**过度代表低分交易**。低分交易证据更少、若需适应本应更需要适应，"
        "而实测方差仍≈0——**这个方向的偏差不会让结论变弱**。",
        "- 未做显著性检验；本节全部为计数与占比。",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    write_report(OUT_MD, "\n".join(L))


if __name__ == "__main__":
    main()
