"""四工具的数据后端实现（施工顺序4；接口契约见 tools.py）。

数据源（全部只读）：
  - knowledge.load_meta()：交易字段 + embargo-21 图特征（时间因果，泄漏审计同款）
  - KnowledgeBase：规则库（[0,125) 统计）+ 案例库（检索时按 as_of−21d 过滤）
  - agent_disposition_gt.parquet：test 逐笔 p / gang_score / 应然档（disposition.py 产出）

每个工具返回 ToolResult(facts=[Fact...])，Fact 带 label_based + window，
audit_time_boundary 逐单核查（结构型 <= as_of，标签型 <= as_of − 21d）。
"""

import pandas as pd
from pathlib import Path

from src.agent.knowledge import KnowledgeBase, load_meta
from src.agent.tools import (EMBARGO_SECS, FactRegistry, InvestigationTools,
                             ToolResult, null_fact, policy_facts)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GT_PARQUET = PROJECT_ROOT / "data" / "processed" / "agent_disposition_gt.parquet"

TXN_FIELDS = ["TransactionAmt", "ProductCD", "card1", "card4", "card6",
              "addr1", "P_emaildomain", "DeviceType", "DeviceInfo"]
STAT_FIELDS = frozenset({"ProductCD", "card4", "card6", "DeviceType", "P_emaildomain"})
RATE_COLS = ["card1_prior_fraud_rate", "card1_addr1_prior_fraud_rate",
             "card1_email_prior_fraud_rate", "card1_device_prior_fraud_rate"]


class Resources:
    """一次加载、多单复用的只读数据集。"""

    def __init__(self):
        print("加载数据后端（meta + 知识库 + 应然档）…")
        self.meta = load_meta().set_index("TransactionID", drop=False)
        self.kb = KnowledgeBase()
        self.gt = pd.read_parquet(GT_PARQUET).set_index("TransactionID", drop=False)
        print(f"  meta {len(self.meta):,} 行，规则 {len(self.kb.rules)} 条，"
              f"案例 {len(self.kb.cases):,} 条，test 应然档 {len(self.gt):,} 笔")


class DataBackedTools(InvestigationTools):
    """真实数据后端；每单调查配一个新 FactRegistry。"""

    def __init__(self, registry: FactRegistry, res: Resources, txn_id: int):
        super().__init__(registry)
        self.res = res
        self.txn_id = int(txn_id)
        self.row = res.meta.loc[self.txn_id]
        self.as_of = int(self.row["TransactionDT"])
        # 成本假设升为一等公民事实（总指挥拍板「往严统一」）：让依赖它们的推理
        # 也能落到审计链上，而不是硬层认、软层不认。window=(0,0)=先于全部数据即已知。
        from src.agent.disposition import BASE
        self.policy = policy_facts(registry, BASE)

    # -- 工具一：交易字段快照（结构型） --------------------------------
    def query_transaction(self, txn_id=None):
        facts, missing = [], []
        for f in TXN_FIELDS:
            v = self.row[f]
            if pd.isna(v):
                missing.append(f)
                continue
            v = float(v) if isinstance(v, (int, float)) and f == "TransactionAmt" else \
                (int(v) if f in ("card1",) else (float(v) if f == "addr1" else str(v)))
            facts.append(self.registry.new_fact(
                "TXN", type=f"txn_field:{f}", entity=f"txn={self.txn_id}", value=v,
                window=(self.as_of, self.as_of), label_based=False,
                source="query_transaction"))
        note = f"as_of(TransactionDT)={self.as_of}"
        if missing:
            note += f"；缺失字段：{','.join(missing)}"
        return ToolResult("query_transaction", facts, note)

    # -- 工具二：实体图特征（结构型 + 标签型，graph_features 口径） ----
    def query_entity_graph(self, txn_id=None):
        r, reg, facts = self.row, self.registry, []
        card1 = f"card1={int(r['card1'])}"
        for col, ent in [("card1_prior_cnt", card1),
                         ("card1_addr1_prior_cnt", f"{card1}|addr1")]:
            facts.append(reg.new_fact("GRAPH", type=col.replace("card1_", "").replace("addr1_", ""),
                                      entity=ent, value=int(r[col]),
                                      window=(0, self.as_of), label_based=False,
                                      source="query_entity_graph"))
        for other in ("addr1", "email", "device"):
            facts.append(reg.new_fact("GRAPH", type=f"fanout_{other}", entity=card1,
                                      value=int(r[f"card1_fanout_{other}"]),
                                      window=(0, self.as_of), label_based=False,
                                      source="query_entity_graph"))
        lab_end = self.as_of - EMBARGO_SECS
        n_rate = 0
        for col in RATE_COLS:
            v = r[col]
            if pd.isna(v):
                continue
            n_rate += 1
            ent = col.replace("_prior_fraud_rate", "").replace("card1_", "card1|")
            facts.append(reg.new_fact("GRAPH", type="prior_fraud_rate",
                                      entity=ent if ent != "card1" else card1,
                                      value=round(float(v), 4),
                                      window=(0, lab_end), label_based=True,
                                      source="query_entity_graph"))
        if self.txn_id in self.res.gt.index:
            g = float(self.res.gt.loc[self.txn_id, "gang_score"])
            facts.append(reg.new_fact("GRAPH", type="gang_score", entity=card1,
                                      value=round(g, 3), window=(0, lab_end),
                                      label_based=True, source="query_entity_graph"))
        if not n_rate:
            facts.append(null_fact(reg, "GRAPH", card1,
                                   "该实体各组合键均无成熟标签历史（prior_fraud_rate 全缺失）",
                                   (0, self.as_of - EMBARGO_SECS), True))
        note = "" if n_rate else "该实体组合键均无成熟标签历史（prior_fraud_rate 全缺失）"
        return ToolResult("query_entity_graph", facts, note)

    # -- 工具三：类别历史欺诈率（标签型，窗口卡 as_of−21d） ------------
    def query_historical_stats(self, entity, as_of=None):
        try:
            field, value = str(entity).split("=", 1)
        except ValueError:
            return ToolResult("query_historical_stats", [], f"entity 格式错误：{entity!r}，应为 'ProductCD=C'")
        if field not in STAT_FIELDS:
            return ToolResult("query_historical_stats", [],
                              f"不支持的字段 {field!r}，可用：{sorted(STAT_FIELDS)}")
        lab_end = self.as_of - EMBARGO_SECS
        m = (self.res.meta[field].astype(str) == value) & (self.res.meta["TransactionDT"] <= lab_end)
        n = int(m.sum())
        if n == 0:
            f = null_fact(self.registry, "STAT", str(entity),
                          f"{entity} 在成熟窗 [0, as_of−21d] 内无任何样本", (0, lab_end), True)
            return ToolResult("query_historical_stats", [f], f"{entity} 在成熟窗内无样本")
        rate = float(self.res.meta.loc[m, "isFraud"].mean())
        fact = self.registry.new_fact("STAT", type="fraud_rate", entity=str(entity),
                                     value=round(rate, 4), window=(0, lab_end),
                                     label_based=True, support_n=n,
                                     source="query_historical_stats")
        return ToolResult("query_historical_stats", [fact],
                          f"窗口已卡 as_of−21d；全体基率≈3.5% 供对照")

    # -- 工具四：规则命中 + 相似案例（知识库后端） ---------------------
    def retrieve_rules_and_cases(self, txn_id=None):
        reg, facts = self.registry, []
        hits = self.res.kb.match_rules(self.row)
        for ru in hits:
            facts.append(reg.new_fact("RULE", type="rule", entity=f"rule={ru['rule_id']}",
                                      value=ru["text"], window=tuple(ru["window"]),
                                      label_based=True, support_n=ru["support_n"],
                                      source="retrieve_rules_and_cases"))
        cases = self.res.kb.retrieve_cases(self.row, self.as_of)
        for c in cases:
            facts.append(reg.new_fact("CASE", type="case",
                                      entity=f"case={int(c['TransactionID'])}",
                                      value=c["card_text"],
                                      window=(int(c["TransactionDT"]), int(c["TransactionDT"])),
                                      label_based=True,
                                      source="retrieve_rules_and_cases"))
        lab_end = self.as_of - EMBARGO_SECS
        if not hits:
            facts.append(null_fact(reg, "RULE", f"txn={self.txn_id}",
                                   "本笔未命中任何风控规则", (0, lab_end), True))
        if not cases:
            facts.append(null_fact(reg, "CASE", f"txn={self.txn_id}",
                                   "相似案例检索无返回", (0, lab_end), True))
        return ToolResult("retrieve_rules_and_cases", facts,
                          f"命中规则 {len(hits)} 条，相似案例 {len(cases)} 条"
                          "（案例含确认欺诈与被洗清的高分假阳两类，注意对照）")
