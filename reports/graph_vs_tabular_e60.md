# 纯表 vs 表+图 干净对照（硬点⑥ 量化依据）

同切分（fit<132/test≥146）、同 LGB 配置，只差图特征。**delta 可能≈0 = 诚实结论（图信号已被匿名特征吸收）。**

## 结果（test）

| 臂 | PR-AUC | ROC-AUC | recall@0.5% | recall@1% | recall@2% |
|----|--------|---------|-------------|-----------|-----------|
| 纯表 | 0.5645 | 0.9138 | 0.142 | 0.262 | 0.423 |
| 表+图 | 0.5878 | 0.9223 | 0.137 | 0.264 | 0.443 |
| **delta** | **+0.0234** | **+0.0086** | -0.006 | +0.002 | +0.020 |

## 图特征重要性（共 446 特征）

| 图特征 | gain 排名 | gain |
|--------|-----------|------|
| card1_prior_fraud_rate | #15/446 | 16535 |
| card1_addr1_prior_fraud_rate | #17/446 | 14593 |
| card1_fanout_device | #18/446 | 14367 |
| card1_addr1_prior_cnt | #19/446 | 14287 |
| card1_prior_cnt | #20/446 | 13341 |
| card1_email_prior_fraud_rate | #21/446 | 13183 |
| card1_fanout_addr1 | #24/446 | 12300 |
| card1_email_prior_cnt | #26/446 | 11823 |
| card1_fanout_email | #33/446 | 9038 |
| card1_prior_fraud_cnt | #57/446 | 4560 |
| card1_device_prior_cnt | #84/446 | 2411 |
| card1_email_prior_fraud_cnt | #87/446 | 2253 |
| card1_addr1_prior_fraud_cnt | #102/446 | 1893 |
| card1_device_prior_fraud_rate | #118/446 | 1373 |
| card1_device_prior_fraud_cnt | #185/446 | 480 |

## 结论（按实际数字，接 ⑥）
- 表+图相对纯表：PR-AUC +0.0234、ROC-AUC +0.0086。最有用的图特征是 `card1_prior_fraud_rate`（排名 #15）。
- 若 delta 微弱：印证「图信号已被 C1-C14/V 匿名计数特征吸收」→ **用轻量图特征（度/prior 欺诈率/fan-out）而非 GNN**：轻量特征已吃掉大部分图价值，GNN 的边际增量不值其训练/推理/服务复杂度。
- 时间因果：结构型（prior_cnt/fan-out）只用 t 之前的边；标签型（prior_fraud_rate）邻居标签留 21 天 embargo——图特征版硬点②，标签+结构两层泄漏都防住。
- 团伙叙事（喂②风控/④业务安全两张皮）：组合键（card1+邮箱/设备）的 prior_fraud_rate 与 fan-out 是「共享稀有实体」的团伙信号。
