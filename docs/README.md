# 项目文档

[返回项目首页](../README.md) · [交互演示](https://1257599842-cell.github.io/AI-Agent-Fraud-Detection/)

## 当前文档

| 文档 | 用途 |
|---|---|
| [结果与证据导航](EVIDENCE.md) | 按验证问题查找实验报告、实现与限制 |
| [复现指南](REPRODUCE.md) | 从无数据检查到本地模型、API 和浏览器验收 |
| [实验协议](../EXPERIMENT_PROTOCOL.md) | 时间窗口、标签可得性、各版本实现边界 |
| [模型卡](../MODEL_CARD.md) | 数据、模型、评估、假设与已知限制 |

## 审查与历史

以下材料保留其记录日期与原有结论，不代表所有计划已经落地。判断当前实现时，优先阅读上方实验协议和源码。

| 材料 | 性质 |
|---|---|
| [独立审查与修复记录](audits/AUDIT_REPORT.md) | 2026-09-13 的专项检查、修正及未解决证据缺口 |
| [机器检查摘要](audits/AUDIT_CHECKS.json) | 与该次审查配套的检查记录，不是实时 CI 状态 |
| [银行指标与评估专项复核](audits/BANK_EVIDENCE_REVIEW.md) | KS、评分卡及 judge 独立参照的复核 |
| [模型卡引用快照](audits/MODEL_CARD_SOURCES.json) | 内容摘要，用于发现模型卡及其引用源变化 |
| [Agent 历史设计登记](design/AGENT_DESIGN.md) | 开工前设计与后续订正；不是当前系统规格 |

## 实验产物

- [报告目录](../reports)：由生成器写出的实验结果；查找结论建议先用[证据导航](EVIDENCE.md)。
- [生成器与归档清单](../reports/_manifest.json)：记录生成命令、内容摘要和冻结件。校验摘要只证明内容未变，不证明结论成立。
- [历史 Agent 轮次](../reports/eval_runs)：原始返回与评估记录；不可通过重写归档美化历史效果。

## 源码目录

~~~text
src/
  features/    数据加载、时间感知图特征、SQL 对账
  model/       时序训练、校准、消融、成本与评分卡实验
  agent/       决策公式、工具、事实账本、检索与报告验收
  eval/        报告评估、受控实验、归档完整性检查
  serving/     API、历史特征查询、回放与网页构建
tests/         合成数据、mock 和归档上的回归测试
~~~
