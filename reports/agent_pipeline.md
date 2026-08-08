# Agent 调查管道演习（施工顺序4）

模型 claude-opus-4-8；⑧ 闸门=应然档 approve 不进 Agent；调用上限 8；⑨ 兜底=真实连接失败触发（连不可达地址）。
首轮暴露解析 bug（模型先散文后围栏 JSON）→ 修 report_from_json 三形态解析 → 离线重验全绿（未重花 API）。

| 场景 | txn | 模式 | 工具 | tokens(in/out) | 成本 | schema | 泄漏审计 | Agent 处置 |
|---|---|---|---|---|---|---|---|---|
| escalate_gang_fraud | 3474965 | llm | 6 | 15,447/2,707 | $0.1449 | ✅0 | ✅ | escalate |
| highp_fraud_no_gang | 3556658 | llm | 6 | 10,810/2,309 | $0.1118 | ✅0 | ✅ | hold |
| highp_false_positive | 3569699 | llm | 5 | 11,791/2,362 | $0.1180 | ✅0 | ✅ | escalate |
| hold_zone_borderline | 3539590 | llm | 7 | 9,755/2,102 | $0.1013 | ✅0 | ✅ | approve |
| gate_normal | 3549323 | 闸门放行 | 0 | 0/0 | $0 | — | — | approve |
| fallback_demo | 3474965 | degraded | 0 | 0/0 | $0 | ✅0 | ✅ | escalate |

4 笔真调查总成本 **$0.4760**（平均 $0.1190/单，平均 6.0 次工具调用）——⑧ 的第一批真实数字。
报告原文 + 完整 facts 账本见 `reports/samples/*.json`，逐条可对账。

## 待第 5 步 eval 的观察（不在本步修）
- 模型无视「只输出 JSON」指令，先写分析散文再给围栏 JSON（4/4）——解析已兜住，指令遵从进 eval。
- Agent 处置 vs 应然档存在分歧（如 hold_zone 笔 Agent 给 approve）——正是层1 处置一致性要量化的东西。