# Evidence guide · 结果与证据导航

这个项目研究三个连续问题：**交易风险怎样预测、风险怎样变成处置、调查结论怎样被核验。** 本页按问题组织已有代码和报告，不新增实验结论。

## 1. 关联历史有没有预测增量？

| 想核对的结论 | 证据 | 应一起看的限制 |
|---|---|---|
| 同一时间切分和配置，加入 15 列图特征，PR-AUC 0.5645 → 0.6032 | [主对照](../reports/graph_vs_tabular.md) · [实现](../src/model/graph_vs_tabular.py) | 主实验的训练、早停标签未统一施加 21 天成熟隔离 |
| top 2% 容量下 recall 42.3% → 45.2%；top 0.5% 则 14.2% → 13.9% | [完整工作点](../reports/graph_vs_tabular.md) | 不是所有容量均改善；不能只挑一个工作点外推 |
| 增益主要来自成熟标签历史，裸 fan-out 增量有限 | [特征消融](../reports/graph_feature_ablation.md) · [特征实现](../src/features/graph_features.py) | 未跑 GNN 对照；匿名字段含义未知 |
| 拟合和选择标签均在 day 146 前按合成延迟成熟时，仍有图增量 | [独立标签可得性审计](../reports/label_availability_audit.md) | PR-AUC 0.5086 → 0.5531；流程改变，不能与主实验绝对分数直接归因比较；仍复用了最终时间窗 |
| 图标签延迟、历史年龄改变时，结果会变 | [60 天延迟](../reports/graph_vs_tabular_e60.md) · [等宽历史窗](../reports/embargo_age_control.md) | 训练期邻居数未完全对齐，不是纯年龄因果效应或通用最优窗口 |

补充：[KS 与分层表](../reports/bank_metrics.md)、[WOE/IV 评分卡对照](../reports/scorecard.md)、[标签泄漏审计](../reports/graph_leak_audit.md)。

## 2. 为什么不直接设 0.5 阈值，或者让 LLM 拍板？

| 想核对的问题 | 证据 | 应一起看的限制 |
|---|---|---|
| 同一个分数在不同金额、误拦代价下为什么对应不同动作？ | [四动作成本](../reports/disposition.md) · [公式实现](../src/agent/disposition.py) | 干预成本和未来收益系假设，输入 raw score 未校准，输出不等于真实最优业务策略 |
| 第五动作 step-up 带来什么变化？ | [五动作扩展](../reports/stepup.md) · [小额边界推导](../reports/small_amount_floor.md) | 仅离线分析和沙盘，API 没有接通 OTP/3DS |
| 校准和重采样是否总有帮助？ | [校准](../reports/calibration.md) · [类别不平衡消融](../reports/imbalance_ablation.md) | 独立实验、特定窗口和配置；不同容量点表现不同 |
| “成本降低 26.1%”可以当上线效果吗？ | [历史阈值扫描](../reports/cost_sensitive.md) | 不可以：在同一评估窗标签上选阈值，是事后模型化结果 |
| 选择性标签与漂移怎么办？ | [选择性偏差演示](../reports/selective_bias.md) · [漂移监控](../reports/drift_monitor.md) | 前者是人为遮蔽；后者未接自动重训，构造噪声实验不等于真实线上退化 |

## 3. 调查报告能核验到哪一层？

- [事实账本与两级数字对账](../reports/agent_grounding.md)：r1 的 565 条数值 finding 中，522 条引用充分、43 条引用不完整；2,672 个数字经机械检查未发现事实池外数字。**这不是语义正确率，也不是“零幻觉”。**
- [证据解释 vs 决策](../reports/agent_evidence_vs_decision.md)：历史对照支持将证据组织与成本决策拆开；一致率的参照不是独立真值。
- [自主性活动面](../reports/agent_autonomy_surface.md)：四轮 318 份归档的事实集合高度集中；770/770 次统计查询都查询本笔字段值。**归档事实不等于完整调用轨迹；尚未完成固定流程/单次 LLM/多轮工具循环的三臂对照。**
- [缺陷分类](../reports/agent_defect_taxonomy.md)：少量人工核查用于识别错误形态，不能外推总体错误率；LLM-as-judge 未作为可靠软层验收依据。
- [受控翻转](../reports/agent_flip_experiment.md)、[证据不足实验](../reports/agent_abstention.md)：小规模受控干预，不是一般化的鲁棒性保证。
- [当前管道](../src/agent/pipeline.py)、[工具后端](../src/agent/backends.py)：v5 限制最多 8 次工具请求尝试，校验不通过的草稿隔离并模板降级。**当前代码保护与历史付费评估分属不同版本，不能混算。**

案例检索是**结构化匹配排序**，不是 embedding 向量检索。案例库经过选择，不能把返回案例的正负比例当作欺诈概率。历史报告中的“人工洗清”等措辞没有真实处理记录支持；保留归档供核查，不将这些措辞升级为事实。

## 4. 实验特征与服务特征对得上吗？

| 检查 | 结果 | 边界 |
|---|---|---|
| SQL / pandas 逐值对账 | 590,540 笔 × 15 列一致 | 同数据、同时间/空值口径；包含错误窗函数的负对照 |
| 先查后写在线回放 | 3,000 笔 × 27 列一致 | 特定同秒密集窗口、单并发；不是生产流量或最终留出集 |
| 单笔查询延迟 | 回放 p95 19.4 ms | 本机历史测量，不是 HTTP 吞吐压测 |

证据：[SQL 对账](../reports/sql_vs_pandas_reconciliation.md)、[在线回放](../reports/online_replay.md)、[FastAPI 实现](../src/serving/app.py)。

`/score` 对原始字段计算特征并评分，但预灌历史不会随请求追加；`/investigate` 仍使用已有交易 ID 的离线证据。**这两个入口尚未组成任意新交易的实时端到端调查链路。**

## 5. 从哪里继续？

- [实验协议](../EXPERIMENT_PROTOCOL.md)：版本、时间窗与实现边界的短索引。
- [模型卡](../MODEL_CARD.md)：模型、数据、假设和已知限制。
- [独立审查](audits/AUDIT_REPORT.md)：修正记录，不代替原始实验。
- [全部文档](README.md)：当前指南、设计历史与专项审查的分层入口。
- [报告清单](../reports/_manifest.json)：报告、生成器与冻结件的登记；哈希检查只证明字节未变，不证明结论成立。
- [分层复现指南](REPRODUCE.md)：无需数据浏览 → 无需付费调用检查 → 本地数据复现。

原始数据与模型不随仓库发布；数据使用遵循 IEEE-CIS 比赛条款。公开展示只使用仓库中已有的案例归档。
