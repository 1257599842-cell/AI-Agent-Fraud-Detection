# 纯表 vs 表+图 干净对照（硬点⑥ 量化依据）

同切分（fit<132/test≥146）、同 LGB 配置，只差图特征。**delta 可能≈0 = 诚实结论（图信号已被匿名特征吸收）。**

## 结果（test）

| 臂 | PR-AUC | ROC-AUC | recall@0.5% | recall@1% | recall@2% |
|----|--------|---------|-------------|-----------|-----------|
| 纯表 | 0.5645 | 0.9138 | 0.142 | 0.262 | 0.423 |
| 表+图 | 0.6032 | 0.9306 | 0.139 | 0.268 | 0.452 |
| **delta** | **+0.0387** | **+0.0168** | -0.003 | +0.005 | +0.029 |

## 图特征重要性（共 446 特征）

| 图特征 | gain 排名 | gain |
|--------|-----------|------|
| card1_addr1_prior_fraud_rate | #3/446 | 32422 |
| card1_email_prior_fraud_rate | #8/446 | 25760 |
| card1_prior_fraud_rate | #11/446 | 22420 |
| card1_fanout_device | #22/446 | 10659 |
| card1_addr1_prior_cnt | #23/446 | 10292 |
| card1_prior_cnt | #25/446 | 9624 |
| card1_fanout_addr1 | #28/446 | 9179 |
| card1_email_prior_cnt | #31/446 | 7692 |
| card1_fanout_email | #38/446 | 6317 |
| card1_prior_fraud_cnt | #43/446 | 5661 |
| card1_addr1_prior_fraud_cnt | #61/446 | 3549 |
| card1_device_prior_fraud_rate | #72/446 | 2773 |
| card1_email_prior_fraud_cnt | #75/446 | 2690 |
| card1_device_prior_cnt | #112/446 | 1328 |
| card1_device_prior_fraud_cnt | #155/446 | 600 |

## 结论（按实际数字，接 ⑥）
- 表+图相对纯表：PR-AUC +0.0387、ROC-AUC +0.0168。最有用的图特征是 `card1_addr1_prior_fraud_rate`（排名 #3）。
- 若 delta 微弱：印证「图信号已被 C1-C14/V 匿名计数特征吸收」→ **用轻量图特征（度/prior 欺诈率/fan-out）而非 GNN**：轻量特征已吃掉大部分图价值，GNN 的边际增量不值其训练/推理/服务复杂度。
- 时间因果：结构型（prior_cnt/fan-out）只用 t 之前的边；标签型（prior_fraud_rate）邻居标签留 21 天 embargo——图特征版硬点②，标签+结构两层泄漏都防住。
- 团伙叙事（喂②风控/④业务安全两张皮）：组合键（card1+邮箱/设备）的 prior_fraud_rate 与 fan-out 是「共享稀有实体」的团伙信号。

<!-- HUMAN:BEGIN -->
## 判读（人写 · 接 ⑥）
> 泄漏审计的两行对比表已升级为机器生成：见 `reports/graph_leak_audit.md`
> （`python -m src.model.graph_leak_audit`）。此处只留判读。


- 预期 delta≈0 被推翻：**轻量时间因果图特征在 CPU 上挖出真实 +0.039 PR-AUC（top-3 特征），且过泄漏审计。**
- **⑥ 叙事（反转、更强）**：不是"图信号已被匿名特征吸收"，而是"我用便宜的 GBDT + 时间因果实体聚合就拿到了图/团伙的主要价值；GNN 只能追剩下的残差——那点残差值不值它的训练/推理/服务复杂度，正是 GTAN 对照臂要量化的"。
- **团伙叙事（喂②风控/④业务安全两张皮）**：`card1_addr1_prior_fraud_rate`、`card1_fanout_device`（一卡扩散到多设备）就是"共享稀有实体"的团伙信号，现在有 top-3 重要性背书。
- **GTAN 决策**：图特征既然有真增益，GNN 或有残差可挖——但主价值已被 CPU 特征吃掉。是否租卡跑 GTAN，取决于愿不愿为残差付复杂度，可留待 Agent 层之后再定。
<!-- HUMAN:END -->
