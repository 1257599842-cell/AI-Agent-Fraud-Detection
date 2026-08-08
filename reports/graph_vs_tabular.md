# 纯表 vs 表+图 干净对照（防守点⑥ 量化依据）

同切分（fit<132/test≥146）、同 LGB 配置，只差图特征。**预期是 delta≈0，实测反转：图特征显著有用，且通过泄漏审计。**

## 主结果（21 天 embargo，与 baseline 假设一致）

| 臂 | PR-AUC | ROC-AUC | recall@0.5% | recall@1% | recall@2% |
|----|--------|---------|-------------|-----------|-----------|
| 纯表（431 特征）| 0.5645 | 0.9138 | 0.142 | 0.262 | 0.425 |
| 表+图（+15 图特征）| 0.6032 | 0.9306 | 0.147 | 0.268 | 0.454 |
| **delta** | **+0.0387** | **+0.0168** | +0.005 | +0.005 | +0.029 |

## 图特征重要性（共 446 特征，gain 排名）

| 图特征 | 排名 | gain |
|--------|------|------|
| card1_addr1_prior_fraud_rate | **#3** | 32,422 |
| card1_email_prior_fraud_rate | **#8** | 25,760 |
| card1_prior_fraud_rate | **#11** | 22,420 |
| card1_fanout_device | #22 | 10,659 |
| card1_addr1_prior_cnt | #23 | 10,292 |
| card1_prior_cnt | #25 | 9,624 |
| card1_fanout_addr1 | #28 | 9,179 |
| card1_email_prior_cnt | #31 | 7,692 |
| card1_fanout_email | #38 | 6,317 |
| card1_prior_fraud_cnt | #43 | 5,661 |
| card1_addr1_prior_fraud_cnt | #61 | 3,549 |
| card1_device_prior_fraud_rate | #72 | 2,773 |
| card1_email_prior_fraud_cnt | #75 | 2,690 |
| card1_device_prior_cnt | #112 | 1,328 |
| card1_device_prior_fraud_cnt | #155 | 600 |

3 个 `prior_fraud_rate`（组合键的"这卡+地址/邮箱以前欺诈过"）冲进 top-11 —— repeat-offender / 团伙信号。

## 泄漏审计（surprising 结果必做）

意外地强 + 标签衍生特征 = 泄漏高发区。验证：把 embargo 从 21 天拉到 **60 天**（3× 更保守的标签延迟假设），看增益是崩塌（泄漏）还是存活（真信号）。

| embargo | PR-AUC delta |
|---------|-------------|
| 21 天 | +0.0387 |
| 60 天 | **+0.0234** |

**结论：增益缩水但不崩塌（+0.039→+0.023）→ 不是泄漏，是真 repeat-offender 信号。** 缩的那部分正是"新鲜度"——21 天给的欺诈历史更新鲜、更有预测力（与前面 embargo 实验同源）。+0.023 是保守下界，+0.039 是与 baseline 一致的假设。（两层时间因果：结构型 prior_cnt/fan-out 只用 t 之前的边；标签型 prior_fraud_rate 邻居标签留 embargo。）

## 结论（接 ⑥）

- 预期 delta≈0 被推翻：**轻量时间因果图特征在 CPU 上挖出真实 +0.039 PR-AUC（top-3 特征），且过泄漏审计。**
- **⑥ 叙事（反转、更强）**：不是"图信号已被匿名特征吸收"，而是"我用便宜的 GBDT + 时间因果实体聚合就拿到了图/团伙的主要价值；GNN 只能追剩下的残差——那点残差值不值它的训练/推理/服务复杂度，正是 GTAN 对照臂要量化的"。
- **团伙叙事（喂②风控/④业务安全两张皮）**：`card1_addr1_prior_fraud_rate`、`card1_fanout_device`（一卡扩散到多设备）就是"共享稀有实体"的团伙信号，现在有 top-3 重要性背书。
- **GTAN 决策**：图特征既然有真增益，GNN 或有残差可挖——但主价值已被 CPU 特征吃掉。是否租卡跑 GTAN，取决于愿不愿为残差付复杂度，可留待 Agent 层之后再定。
