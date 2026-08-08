"""DEMO-1 第一交付物：`reports/demo/demo_data.json`。

**硬约束**：全部取自现有 `reports/` 与 `data/processed/`——
**不新造任何数据、不调用任何 API、不产生任何花费**。
每个字段都能回溯到一个已存在的文件。

产出结构：
  meta      —— 生成时间、版本、口径声明、红线（离线演示/非生产）
  params    —— 四档 + step-up 成本参数，逐个标 [假设] 与扫描范围
  globals   —— 闸门放行比例、每单成本、平均工具调用、五档份额及全网格区间
  cases[]   —— 5–8 笔精选案例，每笔含交易字段 / p / 完整 fact 池 /
                Agent 报告全文 / 四档与五档期望成本明细与 argmin / 「用来展示什么」

用法：python -m src.serving.build_demo_data
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT = PROJECT_ROOT / "reports" / "demo" / "demo_data.json"
SAMPLES = PROJECT_ROOT / "reports" / "samples"
RUNS = PROJECT_ROOT / "reports" / "eval_runs"
GT = PROJECT_ROOT / "data" / "processed" / "agent_disposition_gt.parquet"
MERGED = PROJECT_ROOT / "data" / "processed" / "train_merged.parquet"

A_MED = 76.02
TXN_FIELDS = ["TransactionAmt", "ProductCD", "card1", "card4", "card6",
              "addr1", "P_emaildomain", "DeviceInfo", "DeviceType"]

# 精选案例：文件来源 + 「这一笔用来展示什么」
# 覆盖：高分真欺诈 / 高分假阳 / 低分但有团伙证据(p≈0.007 必选) / 中分模糊 / 兜底降级 / 闸门放行
PICKS = [
    dict(key="highp_fraud", src=SAMPLES / "txn_3556658_highp_fraud_no_gang.json",
         label="高分真欺诈（无团伙）",
         teaches="模型给了极高分且确实是欺诈——但 Agent 选了 hold 而非 decline。"
                 "用来讲「处置精度低于公式，所以决策权交回闭式解」。"),
    dict(key="highp_fp", src=SAMPLES / "txn_3569699_highp_false_positive.json",
         label="高分假阳（历史同类被人工洗清）",
         teaches="p=0.994 但真标签是好人。用来讲「长得像欺诈不等于欺诈」"
                 "——案例库里同时放正例与被洗清的假阳，就是为了这一格。"),
    dict(key="lowp_gang", src=RUNS / "r1" / "txn_3480336.json",
         label="低分但有团伙证据（p≈0.007）",
         teaches="GBDT 给 0.007，Agent 核出扇出 375、成熟欺诈率 5.59%，判断分与证据冲突并建议上报；"
                 "**但成本明细显示公式仍判放行——网络项在这个分数上只值约 $1.5，翻不了档**。"
                 "这正是我把处置权交回公式、只让 Agent 负责取证的现场证据："
                 "**它的证据发现是对的，算术不是。**",
         caveat="⚠️ 本笔**不用来讲网络项生效**。网络项真正生效的证据另有两处，"
                "**不得与本笔混为一谈**：①`gang_escalate` 那笔（小额+确证欺诈史+高扇出→上报）；"
                "②聚合层面（k_future=0 时上报档退化为 0%）。"
                "全 test 窗**没有任何 p<0.05 的交易应然档是 escalate**。"),
    dict(key="mid_ambiguous", src=SAMPLES / "txn_3539590_hold_zone_borderline.json",
         label="中分模糊（挂起带边缘）",
         teaches="p=0.239 落在四档边界附近，参数一动档位就变。"
                 "用来讲「应然档不是真值，是一把会晃 15 个百分点的尺子」。"),
    dict(key="gang_escalate", src=SAMPLES / "txn_3474965_escalate_gang_fraud.json",
         label="团伙上报（网络项生效）",
         teaches="小额但实体有确证欺诈史 + 高扇出 → 上报冻结实体。"
                 "用来讲修订2 网络项，以及它唯一被验证过的前提（后续 30 天欺诈率 4.6–5.0×）。"),
    dict(key="fallback", src=SAMPLES / "txn_3474965_fallback_demo.json",
         label="兜底降级（同一笔的降级版）", decision="⑨",
         teaches="与上一笔**同一交易**：LLM 不可用时走确定性成本框架出报告，"
                 "**过同一个校验器**、结构合法。用来讲「降级是产品行为，不是服务故障」。"),
    dict(key="gated", src=SAMPLES / "txn_3549323_gate_normal.json",
         label="闸门放行（零 LLM 成本）", decision="⑧",
         teaches="应然档=放行 → 根本不进 Agent。用来讲漏斗：便宜模型挡在贵 LLM 前面。"),
]


def _load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _four_and_five(p, amt, gang):
    """四档与五档各自的期望成本明细 + argmin（口径与 disposition.py / stepup.py 完全一致）。"""
    from src.agent.disposition import ACTIONS, BASE, expected_costs
    from src.model.stepup import ACTIONS5, STEPUP, costs5
    e4 = expected_costs([p], [amt], [gang], A_MED, BASE)[0]
    e5 = costs5([p], [amt], [gang], A_MED, BASE, STEPUP)[0]
    return (
        {"actions": list(ACTIONS), "costs": [round(float(x), 4) for x in e4],
         "argmin": ACTIONS[int(np.argmin(e4))]},
        {"actions": list(ACTIONS5), "costs": [round(float(x), 4) for x in e5],
         "argmin": ACTIONS5[int(np.argmin(e5))]},
    )


def build():
    from src.agent.disposition import BASE
    from src.model.stepup import SCAN as SU_SCAN, STEPUP

    gt = pd.read_parquet(GT).set_index("TransactionID")
    meta = pd.read_parquet(MERGED, columns=["TransactionID"] + TXN_FIELDS).set_index("TransactionID")

    cases = []
    for pick in PICKS:
        src = Path(pick["src"])
        if not src.exists():
            raise SystemExit(f"案例源文件不存在：{src}")
        r = _load_json(src)
        txn = int(r["txn_id"])
        g = gt.loc[txn]
        p, amt, gang = float(g["p"]), float(g["TransactionAmt"]), float(g["gang_score"])
        four, five = _four_and_five(p, amt, gang)
        row = meta.loc[txn]
        cases.append({
            "key": pick["key"], "label": pick["label"], "teaches": pick["teaches"],
            "decision": pick.get("decision"),
            "caveat": pick.get("caveat"),
            "transaction_id": txn,
            "source_file": str(src.relative_to(PROJECT_ROOT)),
            "mode": r.get("mode"),
            "prompt_version": r.get("prompt_version") or "v1",
            "fields": {k: (None if pd.isna(row[k]) else
                           (float(row[k]) if isinstance(row[k], (int, float, np.floating)) else str(row[k])))
                       for k in TXN_FIELDS},
            "p": round(p, 6),
            "gang_score": round(gang, 4),
            "is_fraud": int(g["isFraud"]),
            "disposition_gt_4": str(g["disposition_gt"]),
            "cost_four": four,
            "cost_five": five,
            # 完整 fact 池：保留 window 与 label_based（时间纪律在页面上要能看见）
            "facts": r.get("facts", []),
            "report": r.get("report"),
            "acceptance": {
                "schema_violations": r.get("schema_violations", []),
                "time_audit_violations": r.get("time_audit_violations", []),
                "pipeline_overrides": r.get("pipeline_overrides", []),
            },
            "cost_usd": r.get("cost_usd", 0.0),
            "tool_calls": r.get("tool_calls", 0),
            "degraded_reason": r.get("degraded_reason"),
            "note": r.get("note"),
        })

    # ---- 全局统计（全部来自已跑出的结果，不重算模型）----
    n = len(gt)
    from src.agent.disposition import argmin_action
    from src.model.stepup import argmin5
    p_all = gt["p"].to_numpy(); a_all = gt["TransactionAmt"].to_numpy()
    g_all = gt["gang_score"].to_numpy()
    four_all = argmin_action(p_all, a_all, g_all, A_MED, BASE)
    five_all = argmin5(p_all, a_all, g_all, A_MED, BASE, STEPUP)
    gate4 = float((four_all == "approve").mean())
    auto5 = float(((five_all == "approve") | (five_all == "stepup")).mean())

    globals_ = {
        "test_window_transactions": int(n),
        "gate_pass_rate_four_tier": round(gate4, 4),
        "auto_rate_five_tier": round(auto5, 4),
        "needs_agent_or_human_four": round(1 - gate4, 4),
        "needs_agent_or_human_five": round(1 - auto5, 4),
        "cost_per_investigation_usd": 0.1131,
        "avg_tool_calls": 5.5,
        "stepup_share": round(float((five_all == "stepup").mean()), 4),
        "stepup_share_grid_range": [0.0, 0.558],
        "four_tier_distribution": {k: int(v) for k, v in
                                   pd.Series(four_all).value_counts().items()},
        "five_tier_distribution": {k: int(v) for k, v in
                                   pd.Series(five_all).value_counts().items()},
        "sources": {
            "cost_per_investigation_usd": "reports/agent_eval_r1.md",
            "avg_tool_calls": "reports/agent_eval_r1.md",
            "stepup_share": "reports/stepup.md",
            "stepup_share_grid_range": "reports/stepup.md（全网格 81 组）",
        },
    }

    params = {
        "four_tier": {k: {"value": v, "status": "[假设]"} for k, v in BASE.items()},
        "stepup": {k: {"value": v, "status": "[假设]", "scan": SU_SCAN.get(k)}
                   for k, v in STEPUP.items()},
        "a_med": {"value": A_MED, "status": "[实测]",
                  "note": "训练窗 [0,125) 内欺诈交易的中位金额"},
        "note": "**四档与 step-up 的成本参数全部为假设值**，本数据无对应结局标签可标定；"
                "所有依赖它们的结论一律报区间、不报点值。",
    }

    doc = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d"),
            "title": "AI Fraud Investigation Copilot —— 离线演示数据包",
            "banner": "离线演示 / 非生产系统",
            "disclaimer": "本数据包全部取自项目既有产物（reports/ 与 data/processed/），"
                          "**未新造任何数据、未调用任何 API**。案例为预置缓存，"
                          "演示不依赖实时服务。",
            "positioning": "离线决策系统 + 已推演工业化路径；**不声称曾在生产环境运行**。",
            "dataset": "Kaggle IEEE-CIS Fraud Detection（Vesta 真实电商交易）",
        },
        "params": params,
        "globals": globals_,
        "cases": cases,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"✅ {len(cases)} 笔案例 → {OUT.relative_to(PROJECT_ROOT)}（{kb:.0f} KB）")
    for c in cases:
        rep = c["report"] or {}
        print(f"  {c['key']:<14} txn {c['transaction_id']} p={c['p']:<9.6f} "
              f"mode={c['mode']:<9} 四档={c['cost_four']['argmin']:<9} "
              f"五档={c['cost_five']['argmin']:<9} facts={len(c['facts']):>2} "
              f"findings={len(rep.get('key_findings', []))}")
    print(f"\n全局：闸门放行 {globals_['gate_pass_rate_four_tier']:.1%} → "
          f"五档自动化 {globals_['auto_rate_five_tier']:.1%}；"
          f"step-up 份额 {globals_['stepup_share']:.1%}")


if __name__ == "__main__":
    build()
