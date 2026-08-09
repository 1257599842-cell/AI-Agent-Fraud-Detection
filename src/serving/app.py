"""工程层（四层架构的第四层）：两个端点 + 兜底。

  POST /score        —— 便宜通道：GBDT 出分 + 四档期望成本 argmin + ⑧ 闸门判定。**不调 LLM。**
  POST /investigate  —— 贵通道：跑 Agent 调查，返回结构化报告 + 硬层验收 + 成本。
  GET  /healthz      —— 存活与资源加载状态。

设计取舍（工程层「够用就行」，不在工具链上抠）：
  * 资源（59 万行元数据 + 知识库 + 应然档）**只在启动时加载一次**，请求期零 IO。
  * `/score` 与 `/investigate` 的分工就是 ⑧ 闸门本身：便宜模型挡在贵 LLM 前面。
    `/score` 会直接告诉调用方「这笔要不要进 Agent」（`should_investigate`）。
  * ⑨ 兜底原样复用：LLM 不可用时 `/investigate` 仍返回合法报告（`mode="degraded"`），
    **HTTP 仍是 200**——降级是产品行为，不是服务故障。真出错才 5xx。

本地跑：uvicorn src.serving.app:app --port 8000
"""

from contextlib import asynccontextmanager

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


class ScoreResponse(BaseModel):
    transaction_id: int
    p: float = Field(..., description="GBDT 欺诈概率（raw，未做 blanket 校准——见 ③ 的结论）")
    gang_score: float
    expected_costs: dict = Field(..., description="四档期望成本，单位 $")
    disposition: str = Field(..., description="期望成本 argmin")
    should_investigate: bool = Field(..., description="⑧ 闸门：应然档≠approve 才值得进 Agent")


def _res():
    if not STATE.get("ready"):
        raise HTTPException(503, "资源尚未加载完成")
    return STATE["res"]


@app.get("/healthz")
def healthz():
    return {"status": "ok" if STATE.get("ready") else "loading",
            "resources_loaded": bool(STATE.get("ready"))}


@app.post("/score", response_model=ScoreResponse)
def score(req: TxnRequest):
    """便宜通道：纯 CPU，无 LLM。四档成本与应然档口径与离线完全一致（同一份 BASE）。"""
    import numpy as np

    from src.agent.disposition import ACTIONS, BASE, expected_costs
    res = _res()
    if req.transaction_id not in res.gt.index:
        raise HTTPException(404, f"transaction_id {req.transaction_id} 不在 test 窗应然档表内")
    g = res.gt.loc[req.transaction_id]
    p, amt, gang = float(g["p"]), float(g["TransactionAmt"]), float(g["gang_score"])
    a_med = 76.02
    E = expected_costs([p], [amt], [gang], a_med, BASE)[0]
    dispo = ACTIONS[int(np.argmin(E))]
    return ScoreResponse(
        transaction_id=req.transaction_id, p=p, gang_score=gang,
        expected_costs={a: round(float(v), 4) for a, v in zip(ACTIONS, E)},
        disposition=dispo, should_investigate=(dispo != "approve"))


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
