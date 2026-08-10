"""工程层（四层架构的第四层）：两个端点 + 兜底。

  POST /score        —— 便宜通道：**收原始交易字段** → 在线算 27 列历史特征 → 真调 GBDT
                        → 四档期望成本 argmin + ⑧ 闸门判定。**不调 LLM。**
  GET  /demo/score   —— 薄壳：按 TransactionID 取出原始行，喂给上面**同一条路径**。
  POST /investigate  —— 贵通道：跑 Agent 调查，返回结构化报告 + 硬层验收 + 成本。
  GET  /healthz      —— 存活与资源加载状态。

设计取舍（工程层「够用就行」，不在工具链上抠）：
  * 资源（59 万行元数据 + 知识库 + 应然档）与**持久化模型**只在启动时加载一次。
  * **只有一条打分路径。** `/score` 之前接受 `transaction_id`、去离线表里查一个算好的 `p`——
    那不是打分，是查表。现在它收原始字段、在线算特征、真的调用模型。
    demo 需要方便，就用 `/demo/score` 这层薄壳取出原始行再喂进同一条路径——
    **不留第二个打分实现**：两条实现 = 「只有一条真理」当场破产。
  * `/score` 与 `/investigate` 的分工就是 ⑧ 闸门本身：便宜模型挡在贵 LLM 前面。
    `/score` 会直接告诉调用方「这笔要不要进 Agent」（`should_investigate`）。
  * ⑨ 兜底原样复用：LLM 不可用时 `/investigate` 仍返回合法报告（`mode="degraded"`），
    **HTTP 仍是 200**——降级是产品行为，不是服务故障。真出错才 5xx。

本地跑：uvicorn src.serving.app:app --port 8000
"""

from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载重资源；失败不静默——让容器起不来好过起来了给错数。"""
    from src.agent.backends import Resources
    STATE["res"] = Resources()
    STATE["ready"] = True
    yield
    STATE.clear()


app = FastAPI(title="AI Fraud Investigation Copilot",
              description="交易反欺诈：GBDT 打分 + Agent 调查（离线决策系统）",
              version="1.0.0", lifespan=lifespan)


class TxnRequest(BaseModel):
    transaction_id: int = Field(..., description="test 窗内的 TransactionID")


class RawTxnRequest(BaseModel):
    """一笔**没见过的**交易。历史特征由服务在线算，不由调用方提供。

    `fields` 放原始表字段（TransactionAmt / ProductCD / card* / C* / V* …）；
    缺的列按缺失处理——生产里上游本来就可能缺字段，**缺失不该让服务 500**。
    """
    transaction_dt: int = Field(..., description="交易时刻（TransactionDT，相对秒）")
    fields: dict = Field(default_factory=dict, description="原始表字段")
    transaction_id: int | None = Field(
        None, description="仅用于与离线逐行对账时打破 dt 并列；生产无此物，见 feature_store 文档")


class ScoreResponse(BaseModel):
    transaction_id: int | None = None
    p: float = Field(..., description="GBDT 欺诈概率（raw，未做 blanket 校准——见 ③ 的结论）")
    gang_score: float
    expected_costs: dict = Field(..., description="四档期望成本，单位 $")
    disposition: str = Field(..., description="期望成本 argmin")
    should_investigate: bool = Field(..., description="⑧ 闸门：应然档≠approve 才值得进 Agent")
    online_features: dict = Field(default_factory=dict,
                                  description="在线算出的 27 列历史特征（可审计）")
    latency_ms: dict = Field(default_factory=dict, description="特征查询 / 模型推理耗时")


def _model():
    """按需加载持久化模型与 FeatureStore（首次调用时建，之后复用）。

    历史事件只灌 **< T0（第 146 天）** 的部分——服务不该"看过"测试期，
    否则在线打分就带上了未来信息。
    """
    if "booster" not in STATE:
        import json

        import lightgbm as lgb

        from src.serving.feature_store import FeatureStore
        mdir = ROOT / "models"
        if not (mdir / "scoring_model.txt").exists():
            raise HTTPException(503, "打分模型未落盘，先跑 python -m src.agent.disposition")
        STATE["booster"] = lgb.Booster(model_file=str(mdir / "scoring_model.txt"))
        STATE["feat_cols"] = json.loads((mdir / "feature_columns.json").read_text())
        STATE["cat_levels"] = json.loads((mdir / "categorical_levels.json").read_text())
        res = _res()
        st = FeatureStore()
        hist = res.meta[res.meta["TransactionDT"] // 86400
                        - res.meta["TransactionDT"].min() // 86400 < 146]
        st.append_frame(hist)
        STATE["store"] = st
        print(f"  在线特征库：预灌 {len(hist):,} 笔历史（仅 day < 146）")
    return STATE["store"], STATE["booster"], STATE["feat_cols"]


def _res():
    if not STATE.get("ready"):
        raise HTTPException(503, "资源尚未加载完成")
    return STATE["res"]


@app.get("/healthz")
def healthz():
    return {"status": "ok" if STATE.get("ready") else "loading",
            "resources_loaded": bool(STATE.get("ready"))}


@app.post("/score", response_model=ScoreResponse)
def score(req: RawTxnRequest):
    """便宜通道：**真的打分**。

    收原始字段 → FeatureStore 在线算 27 列历史特征 → 调用持久化 GBDT
    → 四档期望成本 argmin。全程纯 CPU、不调 LLM。

    历史特征**必须由服务算**，不能让调用方传——否则时间纪律就交给了上游，
    而上游没有理由知道「标签型事实要留 21 天」。
    """
    import time

    import numpy as np
    import pandas as pd

    from src.agent.disposition import ACTIONS, BASE, expected_costs, gang_score
    from src.serving.feature_store import FEATURE_COLUMNS

    store, booster, feat_cols = _model()
    txn = dict(req.fields)
    txn["TransactionDT"] = req.transaction_dt

    t0 = time.perf_counter()
    hist = store.get_features(txn, tiebreak_id=req.transaction_id)
    t_feat = (time.perf_counter() - t0) * 1000

    row = {**txn, **hist}
    # **类别列必须用训练时的层级重建**，不能就地 astype("category")：
    # 单行数据只会得到 1 个层级，LightGBM 记着训练时的集合 → 直接报错。
    # 这是训练/线上不一致的第二种经典形态（第一种是 NaN 语义）。
    cat = STATE["cat_levels"]
    X = pd.DataFrame([{c: row.get(c, np.nan) for c in feat_cols}])
    for c, levels in cat.items():
        v = X[c].iloc[0]
        X[c] = pd.Categorical([None if pd.isna(v) else str(v)], categories=levels)
    # 数值列必须显式转 float：FeatureStore 的 NULL 用 None 表示，
    # 单行 DataFrame 里 None 会让整列变成 object dtype，LightGBM 直接拒收。
    # **这是 NaN 语义的第三种形态**（另两种：比较时 nan!=nan、类别层级不匹配）。
    num = [c for c in feat_cols if c not in cat]
    X[num] = X[num].apply(pd.to_numeric, errors="coerce").astype("float64")
    t1 = time.perf_counter()
    p = float(booster.predict(X)[0])
    t_model = (time.perf_counter() - t1) * 1000

    amt = float(txn.get("TransactionAmt") or 0.0)
    gang = float(gang_score(hist["card1_fanout_device"],
                            hist["card1_prior_fraud_rate"] or 0.0,
                            hist["card1_prior_fraud_cnt"]))
    E = expected_costs([p], [amt], [gang], 76.02, BASE)[0]
    dispo = ACTIONS[int(np.argmin(E))]
    return ScoreResponse(
        transaction_id=req.transaction_id, p=p, gang_score=gang,
        expected_costs={a: round(float(v), 4) for a, v in zip(ACTIONS, E)},
        disposition=dispo, should_investigate=(dispo != "approve"),
        online_features={k: hist[k] for k in FEATURE_COLUMNS},
        latency_ms={"features": round(t_feat, 2), "model": round(t_model, 2)})


@app.get("/demo/score", response_model=ScoreResponse)
def demo_score(transaction_id: int):
    """演示薄壳：按 ID 取出**原始行**，喂给上面同一条打分路径。

    它只做「取原始字段」这一件事——**不复制任何打分逻辑**。
    """
    res = _res()
    if transaction_id not in res.meta.index:
        raise HTTPException(404, f"transaction_id {transaction_id} 不在数据集内")
    row = res.meta.loc[transaction_id].to_dict()
    dt = int(row.pop("TransactionDT"))
    row.pop("isFraud", None)
    return score(RawTxnRequest(transaction_dt=dt, fields=row,
                               transaction_id=transaction_id))


@app.post("/investigate")
def investigate(req: TxnRequest, force: bool = False):
    """贵通道：Agent 调查。

    force=False 时遵守 ⑧ 闸门（应然档=approve 直接放行、零 LLM 成本）。
    LLM 不可用 → ⑨ 兜底降级报告，仍返回 200（降级是产品行为，不是服务故障）。
    """
    from src.agent.pipeline import _make_client, run_one
    res = _res()
    if req.transaction_id not in res.gt.index:
        raise HTTPException(404, f"transaction_id {req.transaction_id} 不在 test 窗应然档表内")
    try:
        client = _make_client(kill=False)
        out = run_one(res, req.transaction_id, client, force=force)
    except Exception as e:                       # 非 LLM 类故障才算服务错误
        raise HTTPException(500, f"{type(e).__name__}: {e}") from e
    return {
        "transaction_id": out["txn_id"],
        "mode": out.get("mode"),                 # llm / degraded / gated
        "prompt_version": out.get("prompt_version"),
        "p": out.get("p"),
        "report": out.get("report"),
        "acceptance": {                          # 硬层验收随响应一起返回，便于上游审计
            "schema_violations": out.get("schema_violations", []),
            "time_audit_violations": out.get("time_audit_violations", []),
        },
        "cost_usd": out.get("cost_usd", 0.0),
        "tool_calls": out.get("tool_calls", 0),
        "note": out.get("note") or out.get("degraded_reason"),
    }
