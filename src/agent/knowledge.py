"""Agent 知识库（AGENT_DESIGN.md 第一组，施工顺序2）：规则库 + 案例库 + 结构化检索。

规则库（1.1 + 修订4）：候选规则清单 → 在 [0,125) 天窗上算真实欺诈率 → 数据准入
  （lift>=1.5 且样本够）才收录，拒收的也报告。窗口终点取 125 = 146−21：任何
  test 交易（day>=146）都满足「标签窗终点 <= t−21d」，audit_time_boundary 全绿。

案例库（1.2）：~3k 条，池 ⊆ 训练窗 [0,146)，检索时再按 dt <= as_of − EMBARGO 过滤
  （池静态、成熟性逐查询保证）。两类来源天然不同：
    正例 = isFraud=1（拒付确认，不依赖模型分；按月分层抽样）
    负例 = 模型高分但 isFraud=0 的假阳（"被调查后洗清"）——专训一个只见 day<104
      的模型给 [104,146) 打分挖取，不用 baseline 对自己训练集的记忆分。

检索（1.4）：结构化相似主通道——同实体(card1/组合键) > 同模式(ProductCD+金额档+
  卡类型) > 同品类；案例入库时渲染成自然语言案例卡（card_text）。向量粗排层
  （BGE/Chroma）暂缓：等管道跑通看检索质量再定（待决项，见 reports/agent_knowledge.md）。

用法（项目根、已激活 .venv）：python -m src.agent.knowledge
产出：data/processed/agent_rules.json + agent_cases.parquet + reports/agent_knowledge.md
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.agent.tools import EMBARGO_SECS
from src.model.train_baseline import LGB_PARAMS, prepare

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MERGED = PROJECT_ROOT / "data" / "processed" / "train_merged.parquet"
GRAPH = PROJECT_ROOT / "data" / "processed" / "graph_features.parquet"
RULES_OUT = PROJECT_ROOT / "data" / "processed" / "agent_rules.json"
CASES_OUT = PROJECT_ROOT / "data" / "processed" / "agent_cases.parquet"
REPORT = PROJECT_ROOT / "reports" / "agent_knowledge.md"

SECS_PER_DAY = 86_400
RULE_END_DAY = 125     # 规则标签窗 [0,125)：= 146−21，对所有 test 交易过时间审计
CASE_END_DAY = 146     # 案例池 ⊆ 训练窗
FP_FIT_END = 90        # 挖假阳的模型：fit day<90，val [90,104)，给 [104,146) 打分
FP_SCORE_START = 104
N_POS = N_NEG = 1_500
TOP_K = 4
SEED = 42

# 准入门槛：模式欺诈率 >= 1.5×基率，且触发样本/欺诈数足够（避免小样本噪声规则）
MIN_LIFT, MIN_SUPPORT, MIN_FRAUD_N = 1.5, 500, 30

META_COLS = ["TransactionID", "TransactionDT", "isFraud", "TransactionAmt", "ProductCD",
             "card1", "card4", "card6", "addr1", "P_emaildomain", "DeviceInfo",
             "DeviceType", "id_01"]
GRAPH_COLS_USED = ["card1_prior_cnt", "card1_prior_fraud_rate",
                   "card1_addr1_prior_cnt", "card1_addr1_prior_fraud_rate",
                   "card1_email_prior_fraud_rate", "card1_device_prior_fraud_rate",
                   "card1_fanout_addr1", "card1_fanout_email", "card1_fanout_device"]
AMT_BINS = [0, 1, 10, 50, 100, 300, 1000, np.inf]
AMT_LABELS = ["<$1", "$1-10", "$10-50", "$50-100", "$100-300", "$300-1k", ">$1k"]


# ---------------------------------------------------------------- 触发器（建库与运行时共用同一套语义）

def _cond_mask(df, field, op, value):
    """向量化条件求值；NaN 一律不触发。"""
    s = df[field]
    if op == "eq":
        m = s == value
    elif op == "lt":
        m = s < value
    elif op == "ge":
        m = s >= value
    elif op == "isnull":
        m = s.isna()
    elif op == "notnull":
        m = s.notna()
    else:
        raise ValueError(f"未知 op: {op}")
    return np.asarray(m == True)  # noqa: E712  —— 把可能的 NA 压成 False


def eval_trigger(conditions, df):
    """AND 组合的多条件触发器；df 为 DataFrame（建库）或单行 dict/Series（运行时）。"""
    frame = pd.DataFrame([df]) if not isinstance(df, pd.DataFrame) else df
    m = np.ones(len(frame), dtype=bool)
    for field, op, value in conditions:
        m &= _cond_mask(frame, field, op, value)
    return m if isinstance(df, pd.DataFrame) else bool(m[0])


def _cond_desc(conditions):
    ops = {"eq": "=", "lt": "<", "ge": ">=", "isnull": " 缺失", "notnull": " 存在"}
    return " 且 ".join(f"{f}{ops[op]}{'' if value is None else value}" for f, op, value in conditions)


# ---------------------------------------------------------------- 规则库

def build_rules(meta):
    """候选规则 → [0,125) 窗真实欺诈率 → 数据准入。返回 (admitted, rejected)。"""
    W = meta[meta["day"] < RULE_END_DAY]
    base = float(W["isFraud"].mean())
    w_lo, w_hi = int(W["TransactionDT"].min()), int(W["TransactionDT"].max())
    print(f"  规则窗 [0,{RULE_END_DAY})天：{len(W):,} 行，基率 {base:.3%}")

    candidates = []
    # ① 类别值扫描：高危取值自动成为候选（每值一条，数据说了算）
    for f in ["ProductCD", "card4", "card6", "DeviceType", "P_emaildomain"]:
        vc = W[f].value_counts()
        for v in vc.index[vc >= MIN_SUPPORT]:
            candidates.append((f"R_{f}_{v}".replace(".", "_"), f"高危取值 {f}={v}",
                               [(f, "eq", v)]))
    # ② 领域模式（EDA/图特征结论的规则化）
    p99_amt = float(W["TransactionAmt"].quantile(0.99))
    p99_cnt = float(W["card1_prior_cnt"].quantile(0.99))
    candidates += [
        ("R_AMT_MICRO", "微额试卡（card-testing）", [("TransactionAmt", "lt", 1.0)]),
        ("R_C_LOW_AMT", "ProductCD C 且小额", [("ProductCD", "eq", "C"), ("TransactionAmt", "lt", 10.0)]),
        ("R_AMT_P99", f"大额（>= p99=${p99_amt:.0f}）", [("TransactionAmt", "ge", p99_amt)]),
        ("R_PRIOR_FRAUD_CARD1", "实体欺诈史：card1", [("card1_prior_fraud_rate", "ge", 0.10)]),
        ("R_PRIOR_FRAUD_CARD1_ADDR1", "实体欺诈史：card1+addr1", [("card1_addr1_prior_fraud_rate", "ge", 0.10)]),
        ("R_PRIOR_FRAUD_CARD1_EMAIL", "实体欺诈史：card1+邮箱", [("card1_email_prior_fraud_rate", "ge", 0.10)]),
        ("R_PRIOR_FRAUD_CARD1_DEVICE", "实体欺诈史：card1+设备", [("card1_device_prior_fraud_rate", "ge", 0.10)]),
        ("R_FANOUT_DEVICE", "团伙扩散：card1 关联 >=10 设备", [("card1_fanout_device", "ge", 10)]),
        ("R_FANOUT_EMAIL", "团伙扩散：card1 关联 >=10 邮箱", [("card1_fanout_email", "ge", 10)]),
        ("R_FANOUT_ADDR", "团伙扩散：card1 关联 >=10 地区", [("card1_fanout_addr1", "ge", 10)]),
        ("R_NEW_ENTITY", "新实体首现（card1+addr1 无历史）", [("card1_addr1_prior_cnt", "eq", 0)]),
        ("R_VELOCITY", f"高频复用（card1 历史 >= p99={p99_cnt:.0f} 笔）", [("card1_prior_cnt", "ge", p99_cnt)]),
        ("R_NO_IDENTITY", "无设备/身份采集记录", [("id_01", "isnull", None)]),
        ("R_HAS_IDENTITY", "有设备/身份采集记录", [("id_01", "notnull", None)]),
    ]

    admitted, rejected = [], []
    y = W["isFraud"].to_numpy()
    for rid, name, conds in candidates:
        m = eval_trigger(conds, W)
        n, fn = int(m.sum()), int(y[m].sum())
        rate = fn / n if n else 0.0
        lift = rate / base if base else 0.0
        rec = {"rule_id": rid, "name": name, "conditions": [list(c) for c in conds],
               "rate": round(rate, 4), "lift": round(lift, 2), "support_n": n, "fraud_n": fn,
               "base_rate": round(base, 4), "window": [w_lo, w_hi], "label_based": True}
        if n >= MIN_SUPPORT and fn >= MIN_FRAUD_N and lift >= MIN_LIFT:
            rec["text"] = (f"{name}（{_cond_desc(conds)}）：训练窗 [0,{RULE_END_DAY})天 内该模式"
                           f"欺诈率 {rate:.2%}，为基率 {base:.2%} 的 {lift:.1f} 倍"
                           f"（触发 {n:,} 笔，其中欺诈 {fn:,}）。")
            admitted.append(rec)
        else:
            why = ("lift 不足" if lift < MIN_LIFT else
                   "触发样本不足" if n < MIN_SUPPORT else "欺诈数不足")
            rec["rejected_reason"] = f"{why}（lift={lift:.2f}, n={n}, fraud={fn}）"
            rejected.append(rec)
    admitted.sort(key=lambda r: -r["lift"])
    print(f"  候选 {len(candidates)} 条 → 准入 {len(admitted)} 条 / 拒收 {len(rejected)} 条")
    return admitted, rejected


# ---------------------------------------------------------------- 案例库

def _mine_fp_scores(meta):
    """训 fit<90 + val[90,104) 的模型，给 [104,146) 打分（挖'高分假阳'负例用）。"""
    X, y, day = prepare()
    fit_m, val_m = day < FP_FIT_END, (day >= FP_FIT_END) & (day < FP_SCORE_START)
    score_m = (day >= FP_SCORE_START) & (day < CASE_END_DAY)
    booster = lgb.train(LGB_PARAMS, lgb.Dataset(X[fit_m], label=y[fit_m]),
                        num_boost_round=2000,
                        valid_sets=[lgb.Dataset(X[val_m], label=y[val_m])],
                        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    scores = pd.Series(np.nan, index=meta.index)
    scores.loc[score_m] = booster.predict(X[score_m], num_iteration=booster.best_iteration)
    print(f"  FP 挖掘模型 best_iter={booster.best_iteration}，已给 {int(score_m.sum()):,} 行打分")
    return scores


def _render_card(r):
    """案例卡：喂 LLM 的自然语言形态（结构化字段仍单独存，检索/对账用结构化）。"""
    outcome = ("确认欺诈（拒付举报）" if r["isFraud"] == 1
               else f"高分假阳（人工洗清，当时模型分 {r['score']:.2f}）")
    ent = (f"card1={r['card1']}" + (f", addr1={r['addr1']:.0f}" if pd.notna(r["addr1"]) else ""))
    hist = (f"实体历史：prior {int(r['card1_addr1_prior_cnt'])} 笔"
            + (f"、成熟欺诈率 {r['card1_addr1_prior_fraud_rate']:.0%}"
               if pd.notna(r["card1_addr1_prior_fraud_rate"]) else "、无成熟标签")
            + f"；设备 fan-out {int(r['card1_fanout_device'])}")
    return (f"案例 #{int(r['TransactionID'])}（day {int(r['day'])}）：{outcome}。"
            f"金额 ${r['TransactionAmt']:.2f}（{r['amt_bucket']}），ProductCD={r['ProductCD']}，"
            f"{r['card4']}/{r['card6']}，邮箱域={r['P_emaildomain'] if pd.notna(r['P_emaildomain']) else '缺失'}，"
            f"设备={r['DeviceInfo'] if pd.notna(r['DeviceInfo']) else '缺失'}，{ent}。{hist}。")


def build_cases(meta):
    """正例（拒付确认，按月分层）+ 负例（模型高分假阳），各 ~1500。"""
    meta = meta.copy()
    meta["score"] = _mine_fp_scores(meta)
    pool = meta[meta["day"] < CASE_END_DAY]

    pos_all = pool[pool["isFraud"] == 1]
    frac = min(1.0, N_POS / len(pos_all))
    pos = (pos_all.groupby(pos_all["day"] // 30, group_keys=False)
           .sample(frac=frac, random_state=SEED))
    neg = (pool[(pool["isFraud"] == 0) & pool["score"].notna()]
           .nlargest(N_NEG, "score"))
    cases = pd.concat([pos, neg], ignore_index=True)
    cases["amt_bucket"] = pd.cut(cases["TransactionAmt"], AMT_BINS,
                                 labels=AMT_LABELS, right=False).astype(str)
    cases["card_text"] = cases.apply(_render_card, axis=1)
    print(f"  案例池：正例 {len(pos):,}（day {pos['day'].min()}~{pos['day'].max()}）"
          f" + 负例 {len(neg):,}（假阳分 {neg['score'].min():.2f}~{neg['score'].max():.2f}）")
    return cases


# ---------------------------------------------------------------- 检索（结构化相似主通道）

class KnowledgeBase:
    """加载已建产物；match_rules / retrieve_cases 是步骤4 retrieve_rules_and_cases 的后端。"""

    def __init__(self, rules_path=RULES_OUT, cases_path=CASES_OUT):
        self.rules = json.loads(Path(rules_path).read_text(encoding="utf-8"))["admitted"]
        self.cases = pd.read_parquet(cases_path)

    def match_rules(self, row):
        """row: 单笔交易（含图特征）的 dict/Series → 命中的规则列表。"""
        return [r for r in self.rules
                if eval_trigger([tuple(c) for c in r["conditions"]], row)]

    def retrieve_cases(self, row, as_of, k=TOP_K):
        """结构化相似 top-k；只看 dt <= as_of − EMBARGO 的成熟案例（标签型时间契约）。"""
        pool = self.cases[self.cases["TransactionDT"] <= as_of - EMBARGO_SECS]
        if pool.empty:
            return []
        g = lambda f: row[f] if pd.notna(row.get(f, np.nan)) else None  # noqa: E731
        amt_b = int(np.digitize(row["TransactionAmt"], AMT_BINS) - 1)
        pool_b = np.digitize(pool["TransactionAmt"].to_numpy(), AMT_BINS) - 1

        sim = np.zeros(len(pool))
        same_card1 = np.asarray((pool["card1"] == g("card1")) if g("card1") is not None else False)
        sim += 8.0 * same_card1                                     # 同实体最强
        if g("addr1") is not None:
            sim += 4.0 * (same_card1 & np.asarray(pool["addr1"] == g("addr1")))
        for f, w in [("ProductCD", 2.0), ("card6", 1.0), ("card4", 1.0),
                     ("P_emaildomain", 1.5), ("DeviceInfo", 1.0)]:
            if g(f) is not None:
                sim += w * np.asarray(pool[f] == g(f))
        sim += 1.5 * (pool_b == amt_b) + 0.75 * (np.abs(pool_b - amt_b) == 1)
        if pd.notna(row.get("card1_addr1_prior_fraud_rate", np.nan)) and row["card1_addr1_prior_fraud_rate"] >= 0.1:
            sim += 1.5 * np.asarray(pool["card1_addr1_prior_fraud_rate"] >= 0.1)
        if pd.notna(row.get("card1_fanout_device", np.nan)) and row["card1_fanout_device"] >= 10:
            sim += 1.0 * np.asarray(pool["card1_fanout_device"] >= 10)

        top = (pool.assign(sim=sim)
               .sort_values(["sim", "TransactionDT"], ascending=False).head(k))
        return top.to_dict("records")


# ---------------------------------------------------------------- 建库主流程

def load_meta():
    meta = pd.read_parquet(MERGED, columns=META_COLS)
    graph = pd.read_parquet(GRAPH, columns=["TransactionID"] + GRAPH_COLS_USED)
    meta = meta.merge(graph, on="TransactionID", how="left")
    day = meta["TransactionDT"] // SECS_PER_DAY
    meta["day"] = (day - day.min()).astype(int)
    return meta


def _sanity(kb, meta):
    """3 笔 test 交易过一遍检索 + 时间纪律断言；返回报告片段。"""
    test = meta[meta["day"] >= CASE_END_DAY]
    fraud = test[test["isFraud"] == 1]
    with_hist = fraud[fraud["card1_addr1_prior_fraud_rate"] >= 0.1]
    if with_hist.empty:
        with_hist = fraud.nlargest(1, "card1_addr1_prior_fraud_rate")
    micro = fraud[fraud["TransactionAmt"] < 1]
    if micro.empty:  # test 窗无 <$1 欺诈（试卡簇是训练窗现象）→ 退取最小额欺诈
        micro = fraud.nsmallest(1, "TransactionAmt")
    picks = [
        ("有实体欺诈史的 test 欺诈", with_hist.iloc[0]),
        ("小额端 test 欺诈", micro.iloc[0]),
        ("普通 test 正常交易", test[test["isFraud"] == 0].iloc[100]),
    ]
    lines, n_bad = [], 0
    for desc, row in picks:
        as_of = int(row["TransactionDT"])
        hits = kb.match_rules(row)
        cases = kb.retrieve_cases(row, as_of)
        # 时间纪律断言（audit_time_boundary 的知识库版）
        bad = [r["rule_id"] for r in hits if r["window"][1] > as_of - EMBARGO_SECS]
        bad += [int(c["TransactionID"]) for c in cases
                if c["TransactionDT"] > as_of - EMBARGO_SECS]
        n_bad += len(bad)
        lines += [f"### {desc}（txn {int(row['TransactionID'])}, day {int(row['day'])}, "
                  f"${row['TransactionAmt']:.2f}, isFraud={int(row['isFraud'])}）",
                  f"- 命中规则 {len(hits)} 条：{', '.join(r['rule_id'] for r in hits) or '（无）'}",
                  f"- top-{TOP_K} 案例（sim | 结局）：" + "；".join(
                      f"#{int(c['TransactionID'])}（{c['sim']:.1f} | "
                      f"{'欺诈' if c['isFraud'] else '假阳'}）" for c in cases),
                  f"- 时间纪律：{'✅ 全部 <= as_of − 21d' if not bad else '❌ 违规 ' + str(bad)}", ""]
        print(f"  [{desc}] 规则 {len(hits)} 条，案例 {len(cases)} 条，"
              f"时间纪律 {'OK' if not bad else 'VIOLATION!'}")
    return lines, n_bad


def _write_report(admitted, rejected, cases, sanity_lines):
    pos, neg = cases[cases["isFraud"] == 1], cases[cases["isFraud"] == 0]
    L = ["# Agent 知识库（施工顺序2）—— 规则库 + 案例库 + 结构化检索\n",
         f"时间纪律：规则统计窗 **[0,{RULE_END_DAY}) 天**（=146−21，对所有 test 交易满足"
         f" <= t−21d）；案例池 ⊆ [0,{CASE_END_DAY}) 天，检索时再按 dt <= as_of − 21d 过滤。\n",
         f"## 规则库：候选 {len(admitted) + len(rejected)} → 准入 {len(admitted)}"
         f"（lift>={MIN_LIFT} 且 n>={MIN_SUPPORT} 且欺诈数>={MIN_FRAUD_N}）\n",
         "| rule_id | 模式 | 欺诈率 | lift | 触发 n | 欺诈 n |",
         "|---|---|---|---|---|---|"]
    L += [f"| {r['rule_id']} | {_cond_desc([tuple(c) for c in r['conditions']])} "
          f"| {r['rate']:.2%} | {r['lift']:.1f}× | {r['support_n']:,} | {r['fraud_n']:,} |"
          for r in admitted]
    L += ["", "<details><summary>被数据拒收的候选（诚实记录：规则是准入的，不是编的）</summary>", ""]
    L += [f"- {r['rule_id']}：{r['rejected_reason']}" for r in rejected]
    L += ["</details>", "",
          "## 案例库构成",
          f"- 正例 {len(pos):,}（拒付确认；day {int(pos['day'].min())}~{int(pos['day'].max())}，按月分层）；"
          f"负例 {len(neg):,}（模型高分假阳，分 {neg['score'].min():.2f}~{neg['score'].max():.2f}，"
          f"day {int(neg['day'].min())}~{int(neg['day'].max())}——负例只在有模型分的 [104,146) 挖，"
          "与「假阳来自模型上线后」的现实语义一致）。",
          "- 诚实边界（1.2）：正例只含**被举报出来的**欺诈 = 选择性偏差样本（⑤ 在 RAG 层的翻版）；"
          "标签有传播性，同实体多案例非独立作案。",
          "",
          "## 检索 sanity（3 笔 test 交易）", ""]
    L += sanity_lines
    L += ["## 待决项",
          "- 向量粗排层（BGE/Chroma，拍板稿 1.3）本步未装：结构化相似是 1.4 拍板的主通道，"
          "20 条量级规则命中用触发器即可。等步骤4 管道跑通、看检索质量再决定是否为案例卡加向量粗排。"]
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print(f"✅ 报告 → {REPORT.relative_to(PROJECT_ROOT)}")


def main(sanity_only=False):
    print("读取 meta（交易字段 + 图特征）…")
    meta = load_meta()

    if sanity_only:  # 产物已在盘上，只重跑检索 sanity + 报告（免重训 FP 模型）
        obj = json.loads(RULES_OUT.read_text(encoding="utf-8"))
        admitted, rejected = obj["admitted"], obj["rejected"]
        cases = pd.read_parquet(CASES_OUT)
    else:
        print("建规则库 …")
        admitted, rejected = build_rules(meta)
        RULES_OUT.write_text(json.dumps({"admitted": admitted, "rejected": rejected},
                                        ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"✅ 规则 → {RULES_OUT.relative_to(PROJECT_ROOT)}")

        print("建案例库（含 FP 挖掘模型训练，几分钟）…")
        cases = build_cases(meta)
        cases.to_parquet(CASES_OUT, index=False)
        print(f"✅ 案例 → {CASES_OUT.relative_to(PROJECT_ROOT)}")

    print("检索 sanity …")
    kb = KnowledgeBase()
    sanity_lines, n_bad = _sanity(kb, meta)
    _write_report(admitted, rejected, cases, sanity_lines)
    if n_bad:
        raise SystemExit(f"时间纪律违规 {n_bad} 处，见报告")
    print("\n知识库建成，时间纪律全绿。")


if __name__ == "__main__":
    import sys

    main(sanity_only="--sanity-only" in sys.argv)
