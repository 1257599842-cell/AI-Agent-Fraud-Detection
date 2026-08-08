"""调查报告 schema（AGENT_DESIGN.md 3.1）+ 硬层结构校验（⑦ groundedness 硬层第一块积木）。

为什么是 JSON/结构体而不是散文：eval 硬层要逐字段对账（evidence_ids 存在性、
数字真伪、该弃权未弃权），自由文本让对账代码全废；给人读的话放 summary 字段。

为什么手写校验器而不是 pydantic：不引新依赖，且 validate_report 返回的
violation 清单本身就是硬层 eval 的计数单位（步骤5 的 agent_eval 直接复用）。

三个要害字段（拍板稿 3.1）：
  assertion_strength    —— 抓 overclaiming 的钩子（软层再对照证据支持强度）
  evidence_insufficient —— 合法弃权出口（不给这个，Agent 会被迫硬编）
  confidence            —— 序数（low/medium/high），不是校准概率；真概率归 GBDT

用法（自测，无 LLM）：python -m src.agent.schema
  构造 1 份合法报告 + 7 份注入已知错误的报告，校验器应 0 误放过。
"""

import json

# 四档处置（3.2 期望成本 argmin 的动作空间）
DISPOSITIONS = frozenset({"approve", "decline", "hold", "escalate"})  # 放行/拒绝/挂起/上报
RISK_LEVELS = frozenset({"low", "medium", "high"})
ASSERTION_STRENGTHS = frozenset({"confirmed", "supported", "tentative"})  # 直接证据/间接支持/推测
CONFIDENCES = frozenset({"low", "medium", "high"})

# 顶层字段 → 期望类型（gang_association 允许 None）
REPORT_FIELDS = {
    "txn_id": int,
    "risk_level": str,
    "key_findings": list,
    "gang_association": (dict, type(None)),
    "disposition": str,
    "disposition_rationale": str,
    "confidence": str,
    "evidence_insufficient": bool,
    "summary": str,
}


def _check_evidence_ids(ids, where, known_fact_ids, violations):
    """evidence_ids 必须是非空 str 列表；给了账本就逐个对账（引用不存在的 ID = 编造）。"""
    if not isinstance(ids, list) or not ids:
        violations.append(f"{where}: evidence_ids 缺失或为空（断言必须给证据）")
        return
    for eid in ids:
        if not isinstance(eid, str):
            violations.append(f"{where}: evidence_id 不是字符串: {eid!r}")
        elif known_fact_ids is not None and eid not in known_fact_ids:
            violations.append(f"{where}: 引用了本次调查未返回的证据 ID '{eid}'（编造）")


def validate_report(report, known_fact_ids=None):
    """结构校验：返回违规清单（空列表 = 通过）。

    known_fact_ids: 本次调查 FactRegistry 返回过的全部 fact_id；
    传 None 则跳过引用对账（只查结构）。
    """
    violations = []
    if not isinstance(report, dict):
        return [f"报告不是 JSON 对象: {type(report).__name__}"]

    for field, typ in REPORT_FIELDS.items():
        if field not in report:
            violations.append(f"缺少必填字段 '{field}'")
        elif not isinstance(report[field], typ):
            violations.append(f"字段 '{field}' 类型错误: {type(report[field]).__name__}")
    for field in report:
        if field not in REPORT_FIELDS:
            violations.append(f"未知字段 '{field}'（schema 外字段=幻觉苗头）")
    if violations:
        return violations  # 结构都不对，后面逐项检查没有意义

    if report["risk_level"] not in RISK_LEVELS:
        violations.append(f"risk_level 非法: '{report['risk_level']}'")
    if report["disposition"] not in DISPOSITIONS:
        violations.append(f"disposition 非法: '{report['disposition']}'（只能四档）")
    if report["confidence"] not in CONFIDENCES:
        violations.append(f"confidence 非法: '{report['confidence']}'（序数三档，非概率）")
    if not report["disposition_rationale"].strip():
        violations.append("disposition_rationale 为空（处置必须给理由）")
    if not report["summary"].strip():
        violations.append("summary 为空")

    findings = report["key_findings"]
    if not findings and not report["evidence_insufficient"]:
        violations.append("key_findings 为空却未声明 evidence_insufficient（无发现必须走弃权出口）")
    for i, kf in enumerate(findings):
        where = f"key_findings[{i}]"
        if not isinstance(kf, dict):
            violations.append(f"{where}: 不是对象")
            continue
        if not isinstance(kf.get("finding"), str) or not kf["finding"].strip():
            violations.append(f"{where}: finding 缺失或为空")
        if kf.get("assertion_strength") not in ASSERTION_STRENGTHS:
            violations.append(f"{where}: assertion_strength 非法: {kf.get('assertion_strength')!r}")
        _check_evidence_ids(kf.get("evidence_ids"), where, known_fact_ids, violations)

    gang = report["gang_association"]
    if gang is not None:
        if not isinstance(gang.get("suspected"), bool):
            violations.append("gang_association.suspected 缺失或非布尔")
        elif gang["suspected"]:
            # 团伙断言有成本语义（修订2 的未来暴露项消费它），必须有证据背书
            _check_evidence_ids(gang.get("evidence_ids"), "gang_association", known_fact_ids, violations)
            if not isinstance(gang.get("rationale"), str) or not gang["rationale"].strip():
                violations.append("gang_association: suspected=true 却没有 rationale")

    return violations


def report_from_json(text):
    """解析 LLM 输出为报告 dict；返回 (report | None, 解析错误清单)。

    容忍三种形态（5 笔演习实测：模型会先写分析散文、把 JSON 放文末围栏里）：
    ① 全文即 JSON；② ```json 围栏块（取最后一个）；③ 首 '{' 到末 '}' 的切片。"""
    import re

    s = text.strip()
    candidates = [s]
    candidates += [m.strip() for m in re.findall(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)][::-1]
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j > i:
        candidates.append(s[i:j + 1])
    last_err = "未找到 JSON 对象"
    for c in candidates:
        if not c.startswith("{"):
            continue
        try:
            return json.loads(c), []
        except json.JSONDecodeError as e:
            last_err = str(e)
    return None, [f"JSON 解析失败: {last_err}"]


# ---------------------------------------------------------------- 自测

EXAMPLE_KNOWN_IDS = {"TXN_000", "TXN_001", "GRAPH_000", "GRAPH_001", "STAT_000", "RULE_000", "CASE_000"}

EXAMPLE_REPORT = {
    "txn_id": 3387654,
    "risk_level": "high",
    "key_findings": [
        {
            "finding": "card1=1129 在本笔之前 21 天成熟窗内 prior 欺诈率 0.47（n=59）",
            "evidence_ids": ["GRAPH_001"],
            "assertion_strength": "confirmed",
        },
        {
            "finding": "交易模式与试卡簇案例相似（小额 + credit + 新设备）",
            "evidence_ids": ["CASE_000", "TXN_001"],
            "assertion_strength": "supported",
        },
    ],
    "gang_association": {
        "suspected": True,
        "entities": ["card1=1129"],
        "evidence_ids": ["GRAPH_000", "GRAPH_001"],
        "rationale": "同 card1 fan-out 至 14 个设备，prior 欺诈率高",
    },
    "disposition": "escalate",
    "disposition_rationale": "四档期望成本 argmin：团伙关联使上报的未来暴露项占优",
    "confidence": "high",
    "evidence_insufficient": False,
    "summary": "高风险：实体历史欺诈率高且呈团伙扩散形态，建议上报。",
}


def _self_test():
    import copy

    failures = 0
    ok = validate_report(EXAMPLE_REPORT, EXAMPLE_KNOWN_IDS)
    print(f"[合法报告] 违规 {len(ok)} 条 —— {'PASS' if not ok else 'FAIL: ' + '; '.join(ok)}")
    failures += bool(ok)

    def corrupt(desc, fn):
        nonlocal failures
        r = copy.deepcopy(EXAMPLE_REPORT)
        fn(r)
        v = validate_report(r, EXAMPLE_KNOWN_IDS)
        caught = len(v) > 0
        print(f"[{desc}] {'PASS（抓到: ' + v[0] + '）' if caught else 'FAIL（漏放）'}")
        failures += not caught

    corrupt("引用不存在的 fact_id", lambda r: r["key_findings"][0].update(evidence_ids=["STAT_999"]))
    corrupt("finding 无 evidence_ids", lambda r: r["key_findings"][0].update(evidence_ids=[]))
    corrupt("无发现且不弃权", lambda r: r.update(key_findings=[]))
    corrupt("disposition 超出四档", lambda r: r.update(disposition="block_and_report"))
    corrupt("团伙断言无证据", lambda r: r["gang_association"].update(evidence_ids=[]))
    corrupt("assertion_strength 非法", lambda r: r["key_findings"][1].update(assertion_strength="definitely"))
    corrupt("缺失 summary 字段", lambda r: r.pop("summary"))

    parsed, errs = report_from_json("```json\n" + json.dumps(EXAMPLE_REPORT) + "\n```")
    fence_ok = not errs and parsed == EXAMPLE_REPORT
    print(f"[围栏 JSON 解析] {'PASS' if fence_ok else 'FAIL'}")
    failures += not fence_ok

    print(f"\n自测{'全部通过' if not failures else f'失败 {failures} 项'}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    _self_test()
