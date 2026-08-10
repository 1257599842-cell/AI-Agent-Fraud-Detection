"""Agent 调查管道（施工顺序4）：出分 → 闸门 → 工具调查 → JSON 报告 + ⑨兜底。

流程（AGENT_DESIGN.md 第五组）：
  1. ⑧ 闸门：GBDT 应然档 = approve 的交易不进 Agent（挡 ~90%，便宜模型挡在贵 LLM 前）。
  2. 调查：Claude (claude-opus-4-8) + 四工具手写 agentic loop，上限 MAX_TOOL_CALLS=8，
     超限后 tool_choice=none 强制收尾。模型分以「待核实线索」喂入（5.1 防谄媚措辞）。
  3. 硬层验收（无 LLM 参与）：schema.validate_report 引用对账 + audit_time_boundary 泄漏审计。
  4. ⑨ 兜底：LLM 连接失败/超时/限流/refusal → GBDT 出分 + 规则模板报告，照常产出合法
     JSON（结构上与正常报告同 schema、过同一校验器），summary 打 [降级模式] 标记。

成本记账（⑧）：逐调用累计 input/output tokens，按 opus-4-8 $5/$25 每 MTok 估算。

用法：
  python -m src.agent.pipeline --txn 3496539            # 单笔调查
  python -m src.agent.pipeline --txn 3496539 --kill-llm  # ⑨ 兜底演示（真连不可达地址）
  python -m src.agent.pipeline --drill                   # 5 笔演习 + 兜底演示 + md 报告
产出：reports/samples/*.json + reports/agent_pipeline.md
"""

from src.report_io import write_report
import json
import sys
from pathlib import Path

try:                      # anthropic 是**可选依赖**：只有真正调用 LLM 时才需要。
    import anthropic      # 纯逻辑路径（单元测试、离线分析、演示页构建）不应因缺它而无法导入。
except ModuleNotFoundError:
    anthropic = None

from src.agent.backends import DataBackedTools, Resources
from src.agent.schema import report_from_json, validate_report
from src.agent.tools import MAX_TOOL_CALLS, FactRegistry, audit_time_boundary

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = PROJECT_ROOT / "reports" / "samples"
MD_OUT = PROJECT_ROOT / "reports" / "agent_pipeline.md"

MODEL = "claude-opus-4-8"
PRICE_IN, PRICE_OUT = 5.0, 25.0        # $ / MTok
MAX_TOKENS = 8000

# prompt 版本号：随报告落盘，保证「哪一版 prompt 产出了这批数字」可追溯。
# v1  = round1 口径（reports/eval_runs/r1/*.json 全部是 v1）
# v2  = round3 一次性修订：删掉原计划的 gang 门控（C 已证该病灶在生产拓扑下值 $0），
#       改修「偏宽」——纠正 hold 语义 + 提醒 decline 档存在。
# v3 = round4：修「字段值→推论标 confirmed」（owner 人工核验发现的头号形态）。
#      **本轮只改 prompt 这一处**；policy_param（成本假设升为一等公民 fact）
#      刻意不在本轮生效，否则证据池同时变、主指标会因「不再全是 txn_field」下降。
# v4 = 证据池轮（**不是 prompt 效果轮**）：两项都改证据池，故合并为一轮，
#      并按"基础设施轮"定位——**本轮不出任何指标结论**，任何跨轮比较须为此校正。
#      (1) policy_param 上台面：成本假设成为可引用事实（POLICY_000..005）；
#      (2) null_result：工具空返回也登记事实，让「查了但没查到」可被引用。
PROMPT_VERSION = "v4-citable-context"

SYSTEM_PROMPT = """你是交易反欺诈调查助手。你的任务：用工具收集证据，独立评估一笔可疑交易，产出结构化调查报告。

## 纪律（硬性，违反即报告作废）
1. 报告中每一条结论必须引用证据：evidence_ids 只能填工具返回过的 fact_id（如 TXN_003、GRAPH_001）。引用不存在的 ID = 编造。
2. 上游 GBDT 模型的风险分只是待核实线索，不是结论。请独立核实；若你的证据与分数矛盾，必须在 key_findings 里明确指出冲突。
3. 证据不足时，设 evidence_insufficient=true 并降低断言强度——弃权是合法出口，硬编不是。
4. assertion_strength：confirmed=有直接证据；supported=间接证据支持；tentative=推测。别把推测标成 confirmed（过度断言）。
4b. **字段值本身可以标 confirmed；但由字段值推出的因果判断或对比判断，最高只能到 supported，
   证据不足时只能 tentative。** 例：
   - 「金额为 $28.04」——直接查表所得，confirmed ✓
   - 「金额小，所以损失有限」——这是由金额推出的**因果判断**，最高 supported ✗ 不是 confirmed
   - 「与高危规则画像一致」——这是**对比判断**，需要规则/案例侧的证据一起引，最高 supported
   判断标准：把 finding 拆成「查到的值」和「由值得出的话」，**后者决定强度上限**。
5. 相似案例中既有确认欺诈也有被人工洗清的高分假阳——"长得像欺诈"不等于欺诈，注意对照两类。
6. 工具调用上限 8 次，够用即收。
9. **可引用的两类特殊证据**（它们和工具返回的事实一样，可以写进 evidence_ids）：
   - `POLICY_000`~`POLICY_005`：本系统的**决策成本假设**（误拦 $25 / 复核 $5 / 上报 $40 /
     挂起残余漏检 0.10 / 上报残余漏检 0.05 / 冻结实体拦下的未来欺诈笔数 5）。
     凡是用到成本权衡的论断（如「拦截收益低于误伤成本」），**必须引用相应的 POLICY_***。
   - `null_result` 型事实：工具查了但没查到时也会返回一条事实。
     「未命中任何风控规则」「该实体无成熟标签历史」这类**缺席陈述，请引用它**，
     不要不引证据就断言——缺席也是有出处的。

## 处置四档（期望成本框架，参数：误拦 $25 / 挂起复核 $5 / 上报建档 $40）
- approve 放行：欺诈概率低，或金额小到不值得拦。
- decline 拒绝：欺诈概率高且金额较大，直接拦损失最小。
- hold 挂起：中间地带，$5 人工复核比拦错/放错都便宜。
- escalate 上报：有团伙证据（实体欺诈史+扇出）时，冻结实体能拦住未来批量欺诈——小额也值得上报。

### 选档时最常见的两个错误（务必自查）
7. **hold 的语义是「存疑就留下」，不是「没把握就放过」。** 证据不足、说法互相矛盾、
   或你自己都觉得不踏实——这些正是 hold 的适用场景，不是 approve 的理由。
   放行是一个**结论**（"我判断它没问题"），不是一个**默认值**（"我没能证明它有问题"）。
   复核只要 $5，而放错一笔的代价是整笔金额。**犹豫时，成本更低的一侧是留下，不是放过。**
8. **别忘了 decline 这一档。** 当欺诈概率高**且**金额较大时，直接拒绝才是最省钱的档：
   拒绝的代价是"万一拦错"（约 $25 量级），而放过的代价是整笔金额。
   粗略对照：金额越大，值得拒绝所需的概率越低。
   不要因为"证据还不够铁"就一路退到 hold 甚至 approve——**hold 是为「不确定」准备的，
   不是为「不敢下手」准备的**。

## 输出
调查完成后，只输出一个 JSON 对象（不要围栏外文字），字段：
txn_id(int), risk_level("low"|"medium"|"high"), key_findings(数组，每条{finding, evidence_ids, assertion_strength}),
gang_association(null 或 {suspected(bool), entities(数组), evidence_ids, rationale}),
disposition("approve"|"hold"|"decline"|"escalate"), disposition_rationale(str),
confidence("low"|"medium"|"high"，序数非概率), evidence_insufficient(bool), summary(str，给人读的一段话)。"""

TOOL_DEFS = [
    {"name": "query_transaction",
     "description": "当前交易的字段快照：金额、ProductCD、卡（card1/card4/card6）、地区、邮箱域、设备。调查的起点。",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "query_entity_graph",
     "description": "当前交易关联实体的图特征（全部时间因果）：历史交易数、fan-out（关联地区/邮箱/设备数）、成熟欺诈率（标签已留21天拒付窗）、团伙形态分 gang_score。判断团伙关联的主要依据。",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "query_historical_stats",
     "description": "查询某类别取值的历史欺诈率（窗口自动卡在 as_of−21 天，防标签泄漏）。entity 形如 'ProductCD=C'、'card4=discover'、'P_emaildomain=outlook.com'。",
     "input_schema": {"type": "object",
                      "properties": {"entity": {"type": "string", "description": "形如 'field=value'"}},
                      "required": ["entity"], "additionalProperties": False}},
    {"name": "retrieve_rules_and_cases",
     "description": "检索本笔命中的风控规则（真实数据准入，带欺诈率/lift）+ top-4 结构化相似历史案例（含确认欺诈与被洗清的高分假阳）。",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
]

class _AnthropicUnavailable(Exception):
    """占位异常：未安装 anthropic 时使 LLM_FAILURES 仍是合法的 except 元组。"""


LLM_FAILURES = ((anthropic.APIConnectionError, anthropic.RateLimitError,
                 anthropic.APIStatusError) if anthropic is not None
                else (_AnthropicUnavailable,))


def enforce_evidence_floor(report, registry):
    """把**代码判得了的那部分弃权**前移到管道层，不再指望模型自觉。

    来历：剥夺实验（`reports/agent_abstention.md`）证明模型**确实会**在证据被抽干时弃权
    （0%→10%→100% 剂量反应）。既然如此，就更该把**结构上可判**的那部分交给代码，
    让模型只负责代码判不了的灰区——与「概率判断权归 GBDT」同一条思路。

    **只做代码真判得了的两条**，一条都不多：
      R1 账本里**零条标签型事实** → 强制 `evidence_insufficient=true`。
         这是关于证据池的**事实陈述**（这单从头到尾没拿到任何"已定案"的历史），
         不是对内容的判断。
      R2 某条 finding **一条 fact 都没引** → 其 `assertion_strength` 封顶为 `tentative`。
         无引用即无支撑，与内容无关。

    **我自己先前建议里「无标签型证据就全篇封顶断言强度」那条，实现时被否决了**：
    「本笔金额 $28.04」引 TXN_000 标 `confirmed` 是**合法**的（round 4 刚确认：
    字段值本身可达 confirmed），它与有没有标签型证据无关。
    按那条改会把对的东西一起降档——**代码判不了的，就不该让代码判**。

    **绝不静默**：每次改写都写进 `pipeline_overrides` 随报告落盘。
    本项目已经在「静默替代」上栽过三次（解析器顺延 / M-H 跳过缺格 / 兜底顶替缺测），
    一个悄悄改写模型输出的管道会是第四次。
    """
    overrides = []
    if report is None:
        return report, overrides

    if not any(f.label_based for f in registry.all_facts()):
        if not report.get("evidence_insufficient"):
            report["evidence_insufficient"] = True
            overrides.append("R1: 账本零条标签型事实 → 强制 evidence_insufficient=true"
                             "（模型原报 false）")

    known = registry.known_ids()
    for i, kf in enumerate(report.get("key_findings", [])):
        cited = [e for e in kf.get("evidence_ids", []) if e in known]
        if not cited and kf.get("assertion_strength") != "tentative":
            overrides.append(f"R2: key_findings[{i}] 未引用任何有效 fact → "
                             f"assertion_strength {kf.get('assertion_strength')} → tentative")
            kf["assertion_strength"] = "tentative"
    return report, overrides


def investigate(res, txn_id, client, p_override=None):
    """happy path：LLM 调查一笔交易。返回结果 dict（报告 + 验收 + 成本）。

    p_override：5.1 翻转实验专用——**只改喂进 prompt 的那个分数，不动任何工具返回的证据**。
    这样「证据不变、分数变」，观察到的报告差异就只能归因于分数本身（干净的单变量操纵）。
    """
    txn_id = int(txn_id)
    reg = FactRegistry()
    backend = DataBackedTools(reg, res, txn_id)
    p_true = float(res.gt.loc[txn_id, "p"])
    p = p_true if p_override is None else float(p_override)
    handlers = {"query_transaction": lambda i: backend.query_transaction(),
                "query_entity_graph": lambda i: backend.query_entity_graph(),
                "query_historical_stats": lambda i: backend.query_historical_stats(i.get("entity", "")),
                "retrieve_rules_and_cases": lambda i: backend.retrieve_rules_and_cases()}

    messages = [{"role": "user", "content":
                 f"请调查交易 {txn_id}。上游 GBDT 模型风险分 p={p:.4f}（待核实线索，请独立核实）。"
                 f"完成调查后输出 JSON 报告。"}]
    usage = {"input": 0, "output": 0, "api_calls": 0}
    while True:
        force_end = reg.tool_calls >= MAX_TOOL_CALLS
        resp = client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT,
            tools=TOOL_DEFS, messages=messages,
            **({"tool_choice": {"type": "none"}} if force_end else {}))
        usage["input"] += resp.usage.input_tokens
        usage["output"] += resp.usage.output_tokens
        usage["api_calls"] += 1
        if resp.stop_reason == "refusal":
            raise anthropic.APIConnectionError(request=None)  # 交给兜底
        if resp.stop_reason != "tool_use":
            break
        tool_blocks = [b for b in resp.content if b.type == "tool_use"]
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for tb in tool_blocks:
            reg.tool_calls += 1
            tr = handlers[tb.name](tb.input or {})
            results.append({"type": "tool_result", "tool_use_id": tb.id,
                            "content": json.dumps(tr.to_dict(), ensure_ascii=False)})
        messages.append({"role": "user", "content": results})

    text = next((b.text for b in resp.content if b.type == "text"), "")
    report, errs = report_from_json(text)
    # 证据下限在**校验之前**执行：R2 会把无引用 finding 降到 tentative，
    # 若放在校验之后，报告落盘的强度与校验时看到的就不是同一份。
    report, overrides = enforce_evidence_floor(report, reg)
    violations = errs if errs else validate_report(report, reg.known_ids())
    audit = audit_time_boundary(reg.all_facts(), backend.as_of)
    cost = usage["input"] / 1e6 * PRICE_IN + usage["output"] / 1e6 * PRICE_OUT
    return {"txn_id": txn_id, "mode": "llm", "model": MODEL, "p": p,
            "p_true": p_true, "p_injected": p_override is not None,
            "prompt_version": PROMPT_VERSION,
            "pipeline_overrides": overrides,      # 管道改写了什么，逐条留痕，绝不静默
            "report": report, "raw_text": text if report is None else None,
            "schema_violations": violations, "time_audit_violations": audit,
            "tool_calls": reg.tool_calls, "api_calls": usage["api_calls"],
            "tokens": usage, "cost_usd": round(cost, 4),
            "facts": [f.to_dict() | {"label_based": f.label_based} for f in reg.all_facts()]}


def degraded_report(res, txn_id, reason):
    """⑨ 兜底：GBDT 出分 + 规则模板报告。全程无 LLM，结构与正常报告同 schema。"""
    txn_id = int(txn_id)
    reg = FactRegistry()
    backend = DataBackedTools(reg, res, txn_id)
    g = res.gt.loc[txn_id]
    p, gang, dispo = float(g["p"]), float(g["gang_score"]), str(g["disposition_gt"])

    score_fact = reg.new_fact("STAT", type="model_score", entity=f"txn={txn_id}",
                              value=round(p, 4), window=(backend.as_of, backend.as_of),
                              label_based=False, source="degraded_pipeline")
    rule_result = backend.retrieve_rules_and_cases()
    rule_facts = [f for f in rule_result.facts if f.fact_id.startswith("RULE")]
    graph_result = backend.query_entity_graph()
    gang_facts = [f for f in graph_result.facts if f.type in ("gang_score", "prior_fraud_rate")]

    findings = [{"finding": f"GBDT 风险分 p={p:.4f}（决策区间校准已验，raw 可当概率用）",
                 "evidence_ids": [score_fact.fact_id], "assertion_strength": "confirmed"}]
    findings += [{"finding": f"命中规则：{f.value}", "evidence_ids": [f.fact_id],
                  "assertion_strength": "confirmed"} for f in rule_facts[:3]]
    gang_obj = None
    if gang >= 0.5 and gang_facts:
        gang_obj = {"suspected": True, "entities": [gang_facts[0].entity],
                    "evidence_ids": [f.fact_id for f in gang_facts[:3]],
                    "rationale": f"gang_score={gang:.2f}（fan-out×成熟欺诈率，时间因果）"}
    risk = "high" if p >= 0.5 else ("medium" if p >= 0.1 else "low")
    report = {"txn_id": txn_id, "risk_level": risk, "key_findings": findings,
              "gang_association": gang_obj, "disposition": dispo,
              "disposition_rationale": "四档期望成本 argmin（降级模式：LLM 不可用，处置由确定性成本框架直接给出）",
              "confidence": "low", "evidence_insufficient": len(rule_facts) == 0,
              "summary": f"[降级模式：{reason}] LLM 不可用，本报告由 GBDT 分数 + 规则模板生成。"
                         f"风险分 {p:.2f}，命中规则 {len(rule_facts)} 条，处置建议 {dispo}。"
                         f"拦截能力不受影响；LLM 恢复后可补充叙事性调查。"}
    violations = validate_report(report, reg.known_ids())
    audit = audit_time_boundary(reg.all_facts(), backend.as_of)
    return {"txn_id": txn_id, "mode": "degraded", "degraded_reason": reason, "p": p,
            "report": report, "schema_violations": violations,
            "time_audit_violations": audit, "tool_calls": 0, "api_calls": 0,
            "tokens": {"input": 0, "output": 0}, "cost_usd": 0.0,
            "facts": [f.to_dict() | {"label_based": f.label_based} for f in reg.all_facts()]}


def run_one(res, txn_id, client, force=False, p_override=None):
    """闸门 + 调查 + 兜底 的完整单笔入口。"""
    txn_id = int(txn_id)
    dispo_gt = str(res.gt.loc[txn_id, "disposition_gt"])
    if dispo_gt == "approve" and not force:
        return {"txn_id": txn_id, "mode": "gated",
                "p": float(res.gt.loc[txn_id, "p"]),
                "note": "⑧ 闸门：应然档=approve，不消耗 LLM，直接放行", "cost_usd": 0.0}
    try:
        return investigate(res, txn_id, client, p_override=p_override)
    except LLM_FAILURES as e:
        return degraded_report(res, txn_id, type(e).__name__)


def _save(result, tag):
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    path = SAMPLES_DIR / f"txn_{result['txn_id']}_{tag}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def _pick_drill_txns(res):
    """5 笔已知结局的演习交易（AGENT_DESIGN.md 验证方式）。"""
    g = res.gt
    fraud, legit = g[g["isFraud"] == 1], g[g["isFraud"] == 0]
    picks = {}
    picks["escalate_gang_fraud"] = int(fraud[fraud["disposition_gt"] == "escalate"]
                                       .nlargest(1, "gang_score").index[0])
    ng = fraud[(fraud["gang_score"] == 0) & (fraud["p"] > 0.8)]
    picks["highp_fraud_no_gang"] = int(ng.nlargest(1, "p").index[0])
    fp = legit[legit["p"] > 0.7]
    picks["highp_false_positive"] = int(fp.nlargest(1, "p").index[0])
    hold = g[(g["disposition_gt"] == "hold") & (g["p"].between(0.1, 0.4))]
    picks["hold_zone_borderline"] = int(hold.sample(1, random_state=42).index[0])
    ap = g[(g["disposition_gt"] == "approve") & (g["p"] < 0.01)]
    picks["gate_normal"] = int(ap.sample(1, random_state=42).index[0])
    return picks


def drill(kill_llm=False):
    res = Resources()
    client = _make_client(kill=False)
    picks = _pick_drill_txns(res)
    rows = []
    for tag, txn in picks.items():
        print(f"\n=== [{tag}] txn {txn} ===")
        r = run_one(res, txn, client)
        path = _save(r, tag)
        rows.append((tag, r))
        _brief(r, path)

    # ⑨ 兜底演示：故意打向不可达地址（真实网络失败，不是 mock 返回值）
    demo_txn = picks["escalate_gang_fraud"]
    print(f"\n=== [fallback_demo] txn {demo_txn}（LLM 已被故意杀掉）===")
    r = run_one(res, demo_txn, _make_client(kill=True))
    path = _save(r, "fallback_demo")
    rows.append(("fallback_demo", r))
    _brief(r, path)

    _write_md(rows)
    print(f"\n✅ 演习报告 → {MD_OUT.relative_to(PROJECT_ROOT)}")


def _make_client(kill):
    if anthropic is None:
        raise RuntimeError(
            "本操作需要调用 LLM API，但未安装可选依赖 anthropic。\n"
            "  pip install anthropic\n"
            "（仅实时调查需要；单元测试、离线分析与演示页构建均不需要。）")
    if kill:
        return anthropic.Anthropic(base_url="http://127.0.0.1:9", api_key="dead",
                                   max_retries=0, timeout=3.0)
    return anthropic.Anthropic(max_retries=2)


def _brief(r, path):
    if r["mode"] == "gated":
        print(f"  闸门放行（p={r['p']:.4f}），零成本")
        return
    ok = "0 违规" if not r["schema_violations"] else f"❌ {len(r['schema_violations'])} 违规"
    audit = "泄漏审计绿" if not r["time_audit_violations"] else "❌ 泄漏!"
    print(f"  mode={r['mode']}  工具 {r['tool_calls']} 次  API {r['api_calls']} 次  "
          f"${r['cost_usd']:.4f}  schema {ok}  {audit}")
    if r["report"]:
        rep = r["report"]
        print(f"  disposition={rep['disposition']}  findings={len(rep['key_findings'])}  "
              f"gang={'有' if rep['gang_association'] and rep['gang_association']['suspected'] else '无'}")
    print(f"  → {path.name}")


def _write_md(rows):
    L = ["# Agent 调查管道演习（施工顺序4）\n",
         f"模型 {MODEL}；⑧ 闸门 = 应然档 approve 不进 Agent；调用上限 {MAX_TOOL_CALLS}；"
         "⑨ 兜底 = 真实连接失败触发（连不可达地址），非 mock 返回。\n",
         "| 场景 | txn | 模式 | 工具调用 | tokens(in/out) | 成本 | schema | 泄漏审计 | 处置 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for tag, r in rows:
        if r["mode"] == "gated":
            L.append(f"| {tag} | {r['txn_id']} | 闸门放行 | 0 | 0/0 | $0 | — | — | approve |")
            continue
        rep = r.get("report") or {}
        L.append(f"| {tag} | {r['txn_id']} | {r['mode']} | {r['tool_calls']} "
                 f"| {r['tokens']['input']:,}/{r['tokens']['output']:,} | ${r['cost_usd']:.4f} "
                 f"| {'✅0' if not r['schema_violations'] else '❌' + str(len(r['schema_violations']))} "
                 f"| {'✅' if not r['time_audit_violations'] else '❌'} "
                 f"| {rep.get('disposition', '—')} |")
    total = sum(r.get("cost_usd", 0) for _, r in rows)
    _paid = [r for _, r in rows if r.get("cost_usd", 0) > 0]
    L += ["", f"演习总成本 **${total:.4f}**"
              + (f"；真调查 {len(_paid)} 笔，**平均 ${total / len(_paid):.4f}/单**、"
                 f"平均 {sum(r.get('tool_calls', 0) for r in _paid) / len(_paid):.1f} 次工具调用"
                 if _paid else "")
              + "。报告原文见 `reports/samples/*.json`（含完整 facts 账本，可逐条对账）。"]
    write_report(MD_OUT, "\n".join(L))


def report_from_archive():
    """**离线重出演习报告**：只读 `reports/samples/*.json`，不调任何 API。

    分层的理由：LLM 调用**本来就不确定**，花钱重跑也不会逐字节相同——
    对它做哈希对拍是用错了工具。项目铁律「每次调用的原始返回全部存盘」给了正解：
      · 归档层（`reports/samples/*.json`）= 不可复现部分的**锚**，它自己进哈希清单；
      · 分析层（本函数）= 从归档离线重算，**是确定性的**，可纳入重跑对拍。
    解析器改了要重验时，也是拿归档重跑，不必再花一次钱（本项目已用过一次）。
    """
    tags = ["highp_fraud_no_gang", "highp_false_positive", "hold_zone_borderline",
            "gate_normal", "escalate_gang_fraud", "fallback_demo"]
    rows, missing = [], []
    for tag in tags:
        hits = sorted(SAMPLES_DIR.glob(f"txn_*_{tag}.json"))
        if not hits:
            missing.append(tag)
            continue
        rows.append((tag, json.loads(hits[0].read_text(encoding="utf-8"))))
    if missing:
        sys.exit(f"归档缺失，无法离线重出：{missing}　（需重跑 --drill，会产生 API 花费）")
    _write_md(rows)
    print(f"✅ 离线重出（零 API 花费）→ {MD_OUT.relative_to(PROJECT_ROOT)}")


def main():
    args = sys.argv[1:]
    if "--report-only" in args:
        report_from_archive()
        return
    if "--drill" in args:
        drill()
        return
    if "--txn" not in args:
        print(__doc__)
        return
    txn = int(args[args.index("--txn") + 1])
    res = Resources()
    r = run_one(res, txn, _make_client(kill="--kill-llm" in args), force="--force" in args)
    path = _save(r, "kill" if "--kill-llm" in args else "single")
    _brief(r, path)
    if r.get("report"):
        print(json.dumps(r["report"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
