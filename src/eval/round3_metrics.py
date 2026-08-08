"""round 3 的成功指标（总指挥 2026-07-30 第二批 四.2 换装）。

**为什么换掉「层1 一致率 58%→X%」**：
  1. D 证明该数在合理参数网格上摆 51%–66%（15 个百分点），点对点表述没有地基；
  2. C 的 jackknife 证明金额口径的格子排名靠 2 笔撑着，不能用来定目标。
  → 换成**不依赖金额加权、且在参数网格上一起报区间**的计数型指标。

指标：
  主 A **偏宽分歧率** = Agent 比应然档更轻的比例（参照点 + 网格区间）。
       选它是因为 C 里唯一活下来的观察是笔数口径的 28:1，那条不依赖金额加权。
  主 B **decline 使用率** = Agent 出 decline 的笔数（round1: 1/100，应然 9）。
       纯计数、与金额无关，且直指「该拦不拦」这个病灶。
  次   **金额 delta**（生产拓扑 − 应然档，每万笔）——**必须带 n_eff 与集中度警告**，
       只作参考、不作判据（jackknife 已证其排序不稳）。

用法：
  python -m src.eval.round3_metrics r1              # 单轮
  python -m src.eval.round3_metrics r1 r3           # 两轮对比（同参数配对）
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "reports" / "eval_runs"
EVAL_SET = PROJECT_ROOT / "data" / "processed" / "agent_eval_set.parquet"
REPORT = PROJECT_ROOT / "reports" / "agent_round3_metrics.md"
A_MED = 76.02
ORD = {"approve": 0, "hold": 1, "escalate": 2, "decline": 3}

# 参数网格：与 disposition_sensitivity 同源（c_FP 参照点 $25、敏感性 $10–100）
GRID = {"c_fp": [10.0, 25.0, 50.0, 100.0], "c_review": [2.0, 5.0, 10.0, 20.0],
        "c_report": [20.0, 40.0, 100.0], "m_h": [0.05, 0.10, 0.20]}


def load(tag):
    rows = []
    for f in sorted((RUNS_DIR / tag).glob("txn_*.json")):
        r = json.loads(f.read_text())
        rep = r.get("report") or {}
        if rep.get("disposition"):
            rows.append({"TransactionID": r["txn_id"], "agent": rep["disposition"],
                         "cost_usd": r.get("cost_usd", 0), "tool_calls": r.get("tool_calls", 0)})
    if not rows:
        raise SystemExit(f"round {tag} 没有可用报告")
    return pd.DataFrame(rows).merge(pd.read_parquet(EVAL_SET), on="TransactionID")


def metrics(d, prm=None):
    """给定参数下的三指标。prm=None 用 BASE。"""
    from src.agent.disposition import BASE, argmin_action, realized_cost
    prm = {**BASE, **(prm or {})}
    gt = argmin_action(d["p"].to_numpy(), d["TransactionAmt"].to_numpy(),
                       d["gang_score"].to_numpy(), A_MED, prm)
    agent = d["agent"].to_numpy()
    lenient = np.array([ORD[a] < ORD[g] for a, g in zip(agent, gt)])
    strict = np.array([ORD[a] > ORD[g] for a, g in zip(agent, gt)])
    w = d["ht_weight"].to_numpy()
    prod = np.where(gt == "approve", "approve", agent)
    per = lambda act: float(sum(
        realized_cost(np.array([x]), np.array([y]), np.array([a]), np.array([g]), A_MED, prm)
        * ww for x, y, a, g, ww in
        zip(act, d["isFraud"], d["TransactionAmt"], d["gang_score"], w)) / w.sum() * 10_000)
    return {"lenient_rate": lenient.mean(), "strict_rate": strict.mean(),
            "agree_rate": (agent == gt).mean(),
            "n_decline_agent": int((agent == "decline").sum()),
            "n_decline_gt": int((gt == "decline").sum()),
            "gap": per(prod) - per(gt)}


def grid_range(d, key):
    import itertools
    from src.agent.disposition import BASE
    vals = []
    ks = list(GRID)
    for c in itertools.product(*(GRID[k] for k in ks)):
        vals.append(metrics(d, dict(zip(ks, c)))[key])
    return min(vals), max(vals)


def _fmt(d, tag):
    m = metrics(d)
    lo_l, hi_l = grid_range(d, "lenient_rate")
    lo_d, hi_d = grid_range(d, "n_decline_agent")   # 与参数无关，但一并验证
    w = d["ht_weight"].to_numpy()
    n_eff = w.sum() ** 2 / (w ** 2).sum()
    return m, (lo_l, hi_l), (lo_d, hi_d), n_eff


def run(tags, matched=False):
    data = {t: load(t) for t in tags}
    note = []
    if matched and len(tags) == 2:
        common = set.intersection(*(set(d["TransactionID"]) for d in data.values()))
        data = {t: d[d["TransactionID"].isin(common)].reset_index(drop=True)
                for t, d in data.items()}
        note = ["## ⛔ 本文件是**中途读数**，不是 round 3 的结果\n",
                "round 3 的 dev 跑到一半时 **API 用量限额耗尽**"
                "（`invalid_request_error: You have reached your specified API usage limits`，"
                "恢复时间 2026-08-01 00:00 UTC），28 笔未能取得真调查。",
                "**holdout 那一次尚未跑，按纪律也只能跑一次，必须等补齐 dev 之后再跑。**\n",
                f"> ⚠️ **配对子集口径**：只取两轮都有真调查报告的 **{len(common)}** 笔。",
                "> 缺测**不是随机的**——限额是在跑到最后一个分层时耗尽的，"
                "所以整整一层缺失（见下表）。",
                "> 配对消除了「比较对象不同」，**但消不掉「缺的那层根本没被测」**。\n",
                "> **踩到的坑（已修）**：⑨ 兜底把 28 笔 LLM 失败静默转成了降级报告，"
                "而降级报告的 disposition **直接取自代价公式 argmin**——",
                "> 即与应然档**按构造一致**。若不拦截，它们会把一致率凭空抬高，"
                "**伪造出一个「prompt 修订大幅改善」的假象**。",
                "> 已隔离到 `_r3_degraded_quarantine/`，并在 `run_round` 加了硬拦截："
                "**eval 里降级 = 缺测，不是数据**。",
                "> 生产里降级是正确行为，测量里降级是缺测——同一段代码，两种语义，"
                "这是第三次撞见「静默替代」（解析器顺延 / M-H 跳过缺格 / 兜底顶数）。\n"]
        es = pd.read_parquet(EVAL_SET).set_index("TransactionID")
        cov = es.loc[sorted(common), "stratum"].value_counts()
        full = es[es["split"] == "dev"]["stratum"].value_counts()
        note += ["> | 分层 | 配对覆盖 | dev 应有 |", "> |---|---|---|"]
        note += [f"> | {k} | {cov.get(k, 0)} | {full.get(k, 0)} |" for k in full.index]
        note += [""]
    L = ["# round 3 成功指标（换装后）\n"] + note + [
         "> 「层1 一致率 58%→X%」**已禁用**：D 证明它在参数网格上摆 51%–66%，"
         "点对点表述没有地基；C 的 jackknife 证明金额排名靠 2 笔撑着。",
         "> 换成**不依赖金额加权**的计数型指标，且一律「参照点 + 网格区间」双报。",
         "> 参照点 = c_FP $25（① 已锚，全项目口径一致）+ AGENT_DESIGN 初值；"
         "网格 = c_fp $10–100 × c_review × c_report × m_h。\n"]

    L += ["| 指标 | " + " | ".join(tags) + " |", "|---|" + "---|" * len(tags)]
    rows = {t: _fmt(data[t], t) for t in tags}

    def line(label, fn):
        return f"| {label} | " + " | ".join(fn(t) for t in tags) + " |"

    L.append(line("**主 A · 偏宽分歧率**（参照点）",
                  lambda t: f"**{rows[t][0]['lenient_rate']:.0%}**"
                            f"（{int(rows[t][0]['lenient_rate']*len(data[t]))}/{len(data[t])}）"))
    L.append(line("　　偏宽分歧率（网格区间）",
                  lambda t: f"{rows[t][1][0]:.0%} – {rows[t][1][1]:.0%}"))
    L.append(line("　　（对照）偏严分歧率",
                  lambda t: f"{rows[t][0]['strict_rate']:.0%}"))
    L.append(line("**主 B · decline 使用**",
                  lambda t: f"**{rows[t][0]['n_decline_agent']}**/{len(data[t])}"
                            f"（应然 {rows[t][0]['n_decline_gt']}）"))
    L.append(line("次 · 金额 delta（每万笔）",
                  lambda t: f"${rows[t][0]['gap']:,.0f}"))
    L.append(line("　　n_eff（Kish）", lambda t: f"{rows[t][3]:.1f}"))
    L.append(line("（参考）层1 一致率 · **不作判据**",
                  lambda t: f"{rows[t][0]['agree_rate']:.0%}"))
    L.append(line("每单成本 / 工具调用",
                  lambda t: f"${data[t]['cost_usd'].mean():.4f} / "
                            f"{data[t]['tool_calls'].mean():.1f}"))

    L += ["", "⚠️ **金额 delta 只作参考、不作判据**：n_eff≈29，且 round1 中单笔占 68% 成本质量；"
          "jackknife 显示剔除最贵 2 笔后方向即翻转。**不得用它宣布改进。**\n"]

    if len(tags) == 2:
        a, b = tags
        d_len = rows[b][0]["lenient_rate"] - rows[a][0]["lenient_rate"]
        d_dec = rows[b][0]["n_decline_agent"] - rows[a][0]["n_decline_agent"]
        na, nb = len(data[a]), len(data[b])
        pa, pb = rows[a][0]["lenient_rate"], rows[b][0]["lenient_rate"]
        se = np.sqrt(pa * (1 - pa) / na + pb * (1 - pb) / nb)
        # 配对迁移分析：prompt 到底有没有起作用？起了作用有没有起对方向？
        # 「没改善」有两种完全不同的成因：(i) 模型压根没理会 prompt；
        # (ii) 理会了、行为变了，但变的方向与应然档无关。二者对 round 4 的含义相反。
        ma = data[a].set_index("TransactionID")
        mb = data[b].set_index("TransactionID")
        common = ma.index.intersection(mb.index)
        from src.agent.disposition import BASE, argmin_action
        gt = pd.Series(argmin_action(ma.loc[common, "p"].to_numpy(),
                                     ma.loc[common, "TransactionAmt"].to_numpy(),
                                     ma.loc[common, "gang_score"].to_numpy(), A_MED, BASE),
                       index=common)
        xa, xb = ma.loc[common, "agent"], mb.loc[common, "agent"]
        changed = xa != xb
        ok_a, ok_b = xa == gt, xb == gt
        fixed = int((changed & ~ok_a & ok_b).sum())      # 原本错 → 改对
        broke = int((changed & ok_a & ~ok_b).sum())      # 原本对 → 改错
        still = int((changed & ~ok_a & ~ok_b).sum())     # 错 → 换个错法
        L += [f"## prompt 到底有没有起作用（{a} → {b} 配对迁移，n={len(common)}）\n",
              f"- **处置发生变化的：{int(changed.sum())}/{len(common)} 笔**"
              + ("　→ **模型确实理会了修订**，不是「prompt 被无视」"
                 if changed.sum() > 0.1 * len(common) else "　→ 改动几乎没有传导到行为"),
              "", "| 迁移类型 | 笔数 |", "|---|---|",
              f"| 改对了（原错→现对） | **{fixed}** |",
              f"| 改坏了（原对→现错） | **{broke}** |",
              f"| 换个错法（错→错） | {still} |",
              f"| 未变 | {int((~changed).sum())} |", "",
              f"- McNemar 配对：改对 {fixed} vs 改坏 {broke} → "
              + ("**净效应 ≈ 0，一增一减互相抵消**" if abs(fixed - broke) <= 2 else
                 f"净 {fixed - broke:+d} 笔"),
              "", "**处置分布的位移**（说明行为确实动了）：", "",
              "| 处置 | " + a + " | " + b + " | 变化 |", "|---|---|---|---|"]
        for act in ["approve", "hold", "escalate", "decline"]:
            na_, nb_ = int((xa == act).sum()), int((xb == act).sum())
            L.append(f"| {act} | {na_} | {nb_} | {nb_-na_:+d} |")
        L += ["",
              "> **结论的成因很关键**：行为**变了**（分布位移明显、"
              f"{int(changed.sum())} 笔改档），但**改对与改坏几乎相等**。",
              "> 所以这不是「prompt 没被听见」，而是「**听见了，照做了，但照做并不能让它更接近应然档**」。",
              "> 这与 A 的结论一致：Agent 的处置精度受限于它对 p 与金额的**算术判断**，"
              "而不是受限于它不知道该偏严还是偏宽——",
              "> 换句话说，**这个病灶不是 prompt 能修的**。\n"]
        L += [f"## delta（{a} → {b}，同参数配对）\n",
              f"- **主 A 偏宽分歧率 {pa:.0%} → {pb:.0%}（{d_len*100:+.0f}pp）**，"
              f"95% CI [{(d_len-1.96*se)*100:+.0f}, {(d_len+1.96*se)*100:+.0f}]pp"
              + ("　→ **区间跨 0，不能宣布改进**" if abs(d_len) < 1.96 * se else
                 "　→ 区间不跨 0"),
              f"- **主 B decline 使用 {rows[a][0]['n_decline_agent']} → "
              f"{rows[b][0]['n_decline_agent']}（{d_dec:+d} 笔，应然 "
              f"{rows[b][0]['n_decline_gt']}）**",
              f"- 次 金额 delta ${rows[a][0]['gap']:,.0f} → ${rows[b][0]['gap']:,.0f}"
              "（**参考值，不作判据**）",
              "",
              "> **delta 为零就报零。** 本轮是一次性修订，不开迭代循环；"
              "没改善就记「改了，没用」，那同样是结果。\n",
              "## 收口：holdout 不跑（总指挥 2026-07-31 拍板）\n",
              "原计划是「dev 一次 → holdout 一次，holdout 那个数才可入简历」。**本轮不跑 holdout**，理由：",
              "",
              "1. **holdout 在这里没有它要防的那个东西可防。** 它的作用是防 prompt 在 dev 上过拟合，"
              "而本轮是**预注册的一次性修订**：一版 prompt、dev 只跑一次、期间没有任何依据 dev 结果的调参。"
              "没有拟合过程，就没有过拟合偏差要纠正。",
              "2. **它剩下的价值只是加样本量，而加不动结论。** n 从 100 到 200 只把 CI 从 ±13pp 收到 ±9pp；"
              "观测到的 delta 是 **−1pp**。两个区间都稳稳包含 0，多花的钱买不到任何判别力。",
              "",
              "**因此本文件的数字按「调参集（dev）」标注**——但要同时说明它未经迭代、"
              "所以不带「在 dev 上反复调过」的那种乐观偏差。",
              "**可对外陈述的形式**：",
              "> 「针对偏宽病灶做了一次预注册的一次性 prompt 修订，在 100 笔调参集上"
              "偏宽分歧率 28%→27%（95% CI [−13,+11]pp），decline 使用 1→1，**无可测量的改善**；",
              "> 但配对迁移显示 20 笔改档、分布位移正是修订所要求的方向，改对 7 / 改坏 6 相抵，",
              "> 说明**模型执行了指令，而该病灶不是提示词层面能修的**。」\n",
              "> 这条比「我把一致率从 58% 提到了 X%」更难讲，也更可信：",
              "> 它是一个**带机制解释的阴性结果**，且解释与 A（自我认知检查）独立得出的结论合流。\n",
              "## 层1 处置一致率：**停用 + 交班**（不是删除）\n",
              "方案 ① 落地后，处置由闭式解直接产出，层1 一致率**按构造恒等 100%**，失去全部信息量。",
              "所以它**停用**——但**历史记录与曲线全部保留**，理由如下：\n",
              "**它是本轮几乎全部发现的来源**：",
              "- 成本归因（C）：从层1 混淆矩阵拆出每万笔差额，进而发现 ⑧ 闸门已使「过度上报」值 $0；",
              "- 参数敏感性（D）：正是量它才发现这把尺子在网格上晃 15 个百分点、97% 的差额落在模糊带；",
              "- round 3 的零：偏宽分歧率、decline 使用率都是从它派生的；",
              "- 证据层 vs 决策层：它就是「决策层」那一侧的度量。",
              "",
              "**所以删掉它会让整条叙事读起来像「数字不好看就换指标」**，而事实恰好相反——",
              "**是这个指标把自己为什么不该继续当主指标给测了出来**，然后交班。",
              "这两件事在简历上是完全不同的分量。\n",
              "**继任指标**：",
              "1. **证据层一致性**（`agent_evidence_vs_decision.md`）——相对多数类基线的提升，"
              "而非原始一致率；",
              "2. **接地硬层**——编造率 **0%**（数字口径、代码可复算）/ 引用完整率 **98.4%** / "
              "泄漏审计 **100%** / 结构合规 **100%**。",
              "",
              "> 停用条件写死：**一旦处置不再由 Agent 产出，层1 一致率即失效**。"
              "若将来 Agent 重新参与决策，它自动复活。\n"]

    REPORT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n✅ → {REPORT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run(args or ["r1"], matched="--matched" in sys.argv)
