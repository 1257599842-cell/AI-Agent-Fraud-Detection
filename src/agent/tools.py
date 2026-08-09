"""四个调查工具的接口与返回格式（AGENT_DESIGN.md 2.1/2.2，施工顺序1：先接口后实现）。

工具是"取数管道"不是"能力"：能力来自多轮编排 + 推理。锁死四个，不加假工具；
query_similar_transactions 不单列，作为 retrieve_rules_and_cases 的内部实现（1.4）。

返回格式（焊点级，2.2）：一切事实都是 Fact——唯一 fact_id + 值 + 时间窗 + 样本量
+ 结构型/标签型标记。报告只能引用本次调查返回过的 fact_id（FactRegistry 是账本，
schema.validate_report 拿它对账）。

时间边界契约（每个实现必须遵守；audit_time_boundary 把"泄漏自查"写成代码）：
  - 结构型事实（label_based=False：字段快照/计数/fan-out）：只用 as_of 之前的数据，
    window[1] <= as_of。
  - 标签型事实（label_based=True：prior_fraud_rate/案例结局/欺诈率统计）：只用
    as_of − EMBARGO 之前的标签，window[1] <= as_of − EMBARGO_SECS（拒付延迟 21 天，
    同 src/features/graph_features.py 口径）。
  - 施工提醒1（AGENT_DESIGN.md）：上报档的未来暴露项消费 query_entity_graph 的
    fan-out / prior_fraud_rate——这两个数的时间口径错了，修订4 堵的泄漏会经修订2
    回流。disposition.py 只准吃过审计的 Fact。

用法（自测，无 LLM 无数据）：python -m src.agent.tools
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

EMBARGO_DAYS = 21
EMBARGO_SECS = EMBARGO_DAYS * 86_400
MAX_TOOL_CALLS = 8  # 2.3：单次调查上限，超则强制收尾（管道层执行，兼⑧素材）

# fact_id 前缀 ↔ 工具（ID 形如 GRAPH_003；报告 evidence_ids 只能引用这些）
ID_PREFIXES = {
    "query_transaction": "TXN",
    "query_entity_graph": "GRAPH",
    "query_historical_stats": "STAT",
    "retrieve_rules_and_cases_rule": "RULE",
    "retrieve_rules_and_cases_case": "CASE",
}


@dataclass(frozen=True)
class Fact:
    """原子事实：groundedness 硬层对账的最小单位。"""

    fact_id: str          # 唯一 ID，如 "STAT_003"
    type: str             # 事实类型，如 "prior_fraud_rate" / "txn_field" / "fan_out" / "rule" / "case"
    entity: str           # 所属实体，如 "card1=1129" / "txn=3387654" / "ProductCD=C"
    value: object         # 数值 / 文本 / 结构
    window: tuple         # (start_dt, end_dt) TransactionDT 相对秒；受时间边界契约约束
    label_based: bool     # True=标签型（须留 embargo），False=结构型（只须 < as_of）
    support_n: int = None  # 统计量的样本量（规则文本等无意义处为 None）
    source: str = ""      # 产生它的工具名

    def to_dict(self):
        """喂给 LLM 的 JSON 形态（2.2 示例格式）。"""
        d = {
            "fact_id": self.fact_id,
            "type": self.type,
            "entity": self.entity,
            "value": self.value,
            "window": list(self.window),
        }
        if self.support_n is not None:
            d["support_n"] = self.support_n
        return d


@dataclass
class ToolResult:
    """单次工具调用的返回：facts 进账本，note 给 LLM 的补充说明（如'该实体无历史'）。"""

    tool: str
    facts: list
    note: str = ""

    def to_dict(self):
        return {"tool": self.tool, "facts": [f.to_dict() for f in self.facts], "note": self.note}


# ============ policy_param：把成本假设升为一等公民事实（项目负责人 2026-07-31 拍板）============
#
# **这条改动的来历**：owner 人工核验 17 条时发现的一个架构级不一致——
# Agent 的 system prompt 本身写着成本参数（误拦 $25 / 复核 $5 / 上报 $40），
# 硬层数字对账器的 COST_CONSTS **已把这些常数当合法来源**，
# 而 RUBRIC_V2 要求「只依据给出的证据判断」、证据池里却**没有**这些常数
# → 同一个成本推论，**硬层判它有据、软层判它无据**。
#
# **往严统一**（不是把 COST_CONSTS 从白名单剔除）：理由有二——
#   (a) 反向做会给站得住的推理制造**假的引用缺失**；
#   (b) 一份推荐意见所依赖的假设，本就该出现在审计链上，
#       而它们恰好就是敏感性扫描的那几个参数。
#
# **时间窗**：policy_param 不是观测而是**假设**，没有时间窗。
# 记为 window=(0,0)、label_based=False —— 语义是「先于全部数据即已知」，
# 于是 audit_time_boundary 天然放行，**不需要为它开任何豁免口子**
# （开豁免＝给自己留后门，本项目已多次吃过静默放行的亏）。
POLICY_PARAM_TYPE = "policy_param"

# ============ null_result：让「查了但没查到」也成为可引用的事实 ============
#
# **来历**：证据下限那轮（`reports/agent_evidence_floor.md`）暴露的架构缺陷——
# 工具空返回时不产生任何 Fact，于是「关联实体无任何历史记录，因此无团伙关联证据」
# 这种**正确的缺席陈述**没有 fact_id 可引，只能不引证据地说（触发校验违规、被 R2 降档）
# 或者干脆不说（报告里看不出查过）。**惩罚的是架构缺陷，不是模型的错。**
#
# 修法：空返回也登记一条事实，把**缺席本身**变成可引用的证据。
# 时间窗按被查询的窗口原样记（它确实是"在这个窗口里查过、结果为空"），
# label_based 随被查内容而定——查的是标签型统计（如成熟欺诈率）时为 True，
# 否则为 False；这样 audit_time_boundary 对它的约束与对应的实证事实完全一致，
# **不为它开任何豁免口子**。
NULL_RESULT_TYPE = "null_result"


def null_fact(registry, prefix, entity, what, window, label_based):
    """登记一条「查了但为空」的事实。`what` 说明查的是什么、为什么空。"""
    return registry.new_fact(prefix, type=NULL_RESULT_TYPE, entity=entity,
                             value=f"查询无结果：{what}", window=window,
                             label_based=label_based, source="null_result")


POLICY_DESC = {
    "c_fp": "误拦一笔好交易的成本（假设值；① 已锚 $25，敏感性 $10–100）",
    "c_review": "挂起后人工复核一笔的成本（假设值）",
    "c_report": "上报建档/深查一笔的成本（假设值）",
    "m_h": "挂起后仍漏检的比例（假设值）",
    "m_e": "上报后仍漏检的比例（假设值）",
    "k_future": "冻结团伙实体可拦下的未来欺诈笔数（假设值，**量级未标定**——"
                "gang 效度检验只证了方向 4.6–5.0×，没证这个数）",
}


def policy_facts(registry, params):
    """把成本假设登记为事实，返回 [Fact]。**单一事实来源**：
    四档公式、prompt 文本、硬层数字白名单三处从此指向同一份 params。
    （说明文本放在 POLICY_DESC，不塞进 Fact——Fact 的字段是对账用的，不放散文。）"""
    return [registry.new_fact("POLICY", type=POLICY_PARAM_TYPE,
                              entity=f"policy={k}", value=params[k],
                              window=(0, 0), label_based=False,
                              source="policy_registry")
            for k in POLICY_DESC if k in params]


class FactRegistry:
    """单次调查的事实账本：发 ID、收录所有工具返回、供硬层对账。

    known_ids() → schema.validate_report 的 known_fact_ids；
    get() → eval 硬层反查数字/实体真伪；
    tool_calls 由管道层（步骤4）在每次工具调用时 +1，超 MAX_TOOL_CALLS 强制收尾。
    """

    def __init__(self):
        self._facts = {}
        self._counters = {}
        self.tool_calls = 0

    def new_fact(self, prefix, **kwargs):
        """发号并收录一条 Fact；ID 全局唯一（单次调查内）。"""
        n = self._counters.get(prefix, 0)
        self._counters[prefix] = n + 1
        fact = Fact(fact_id=f"{prefix}_{n:03d}", **kwargs)
        self._facts[fact.fact_id] = fact
        return fact

    def known_ids(self):
        return set(self._facts)

    def get(self, fact_id):
        return self._facts.get(fact_id)

    def all_facts(self):
        return list(self._facts.values())


def audit_time_boundary(facts, as_of, embargo_secs=EMBARGO_SECS):
    """泄漏自查（AGENT_DESIGN.md 验证方式第2条，代码化）：返回违规清单。

    结构型：window[1] <= as_of；标签型：window[1] <= as_of − embargo。
    eval（步骤5）对每单调查的全部 Fact 跑一遍；施工提醒1 的抽查也走这里。
    """
    violations = []
    for f in facts:
        limit = as_of - embargo_secs if f.label_based else as_of
        kind = "标签型" if f.label_based else "结构型"
        if f.window[1] > limit:
            violations.append(
                f"{f.fact_id}({f.type}): {kind}窗口终点 {f.window[1]} > 上限 {limit}（as_of={as_of}）"
            )
    return violations


class InvestigationTools(ABC):
    """四工具接口（2.1 锁死）。实现落地时机：
       query_transaction / query_entity_graph / query_historical_stats —— 步骤4 前，
         数据后端 = data/processed/train_merged.parquet + graph_features.parquet；
       retrieve_rules_and_cases —— 依赖步骤2 知识库（src/agent/knowledge.py）。
    所有实现共享一个 FactRegistry（每单调查新建一个）。"""

    def __init__(self, registry):
        self.registry = registry

    @abstractmethod
    def query_transaction(self, txn_id):
        """本笔交易字段快照：金额 / ProductCD / card1-6 / addr / 邮箱域 / 设备 / as_of。
        全部结构型（label_based=False）；不含任何标签信息。"""

    @abstractmethod
    def query_entity_graph(self, txn_id):
        """关联实体子图（口径 = graph_features.py 两层防泄漏，施工提醒1 的正主）：
          结构型：组合键 prior_cnt、card1 fan-out（只用 as_of 之前的边）；
          标签型：组合键 prior_fraud_rate（标签只取 <= as_of − EMBARGO）。"""

    @abstractmethod
    def query_historical_stats(self, entity, as_of):
        """实体/类别维度历史统计（如 'ProductCD=C' 的欺诈率）。
        欺诈率为标签型：窗口终点 <= as_of − EMBARGO；纯计数为结构型。"""

    @abstractmethod
    def retrieve_rules_and_cases(self, txn_id):
        """规则命中（RULE_*）+ top-4 相似案例（CASE_*）。
        案例：结构化相似主通道 + 案例卡（1.4）；案例窗 [0, as_of − EMBARGO]，
        案例结局是标签型。规则统计量来自训练窗 [0,146) 重算（修订4）。"""


# ---------------------------------------------------------------- 自测

def _self_test():
    from src.agent.schema import validate_report, EXAMPLE_REPORT

    failures = 0
    reg = FactRegistry()
    as_of = 10_000_000  # 假想查询交易的 TransactionDT

    # 合规事实：结构型贴着 as_of，标签型贴着 as_of − embargo
    reg.new_fact("TXN", type="txn_field", entity="txn=3387654", value={"TransactionAmt": 3.5},
                 window=(as_of, as_of), label_based=False, source="query_transaction")
    reg.new_fact("TXN", type="txn_field", entity="txn=3387654", value={"ProductCD": "C"},
                 window=(as_of, as_of), label_based=False, source="query_transaction")
    reg.new_fact("GRAPH", type="fan_out", entity="card1=1129", value=14,
                 window=(0, as_of), label_based=False, support_n=59, source="query_entity_graph")
    reg.new_fact("GRAPH", type="prior_fraud_rate", entity="card1=1129", value=0.47,
                 window=(0, as_of - EMBARGO_SECS), label_based=True, support_n=59,
                 source="query_entity_graph")
    reg.new_fact("STAT", type="fraud_rate", entity="ProductCD=C", value=0.1169,
                 window=(0, as_of - EMBARGO_SECS), label_based=True, support_n=68519,
                 source="query_historical_stats")
    reg.new_fact("RULE", type="rule", entity="rule=card_testing", value="小额试卡簇规则文本",
                 window=(0, as_of - EMBARGO_SECS), label_based=True, source="retrieve_rules_and_cases")
    reg.new_fact("CASE", type="case", entity="case=987", value="案例卡文本",
                 window=(0, as_of - EMBARGO_SECS), label_based=True, source="retrieve_rules_and_cases")

    ids_ok = reg.known_ids() == {"TXN_000", "TXN_001", "GRAPH_000", "GRAPH_001",
                                 "STAT_000", "RULE_000", "CASE_000"}
    print(f"[发号唯一且按前缀递增] {'PASS' if ids_ok else 'FAIL: ' + str(sorted(reg.known_ids()))}")
    failures += not ids_ok

    v = audit_time_boundary(reg.all_facts(), as_of)
    print(f"[合规事实过时间审计] 违规 {len(v)} 条 —— {'PASS' if not v else 'FAIL: ' + v[0]}")
    failures += bool(v)

    # 注入两类已知泄漏，审计必须抓到
    bad = [
        Fact("GRAPH_999", "prior_fraud_rate", "card1=1129", 0.5,
             window=(0, as_of - 3600), label_based=True, source="query_entity_graph"),   # 标签型没留 embargo
        Fact("GRAPH_998", "fan_out", "card1=1129", 20,
             window=(0, as_of + 86_400), label_based=False, source="query_entity_graph"),  # 结构型用了未来边
    ]
    v = audit_time_boundary(bad, as_of)
    caught = len(v) == 2
    print(f"[注入泄漏被审计抓出] {'PASS（' + str(len(v)) + '/2）' if caught else 'FAIL（抓到 ' + str(len(v)) + '/2）'}")
    failures += not caught

    # 账本 ↔ schema 对账闭环：示例报告引用的 ID 恰好全在账本里
    v = validate_report(EXAMPLE_REPORT, reg.known_ids())
    print(f"[账本对账 schema 报告] 违规 {len(v)} 条 —— {'PASS' if not v else 'FAIL: ' + v[0]}")
    failures += bool(v)

    print(f"\n自测{'全部通过' if not failures else f'失败 {failures} 项'}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    _self_test()
