# SQL/DuckDB 与 pandas 的逐列对账

> 成功判据只有一条：**同一套特征、两种实现、逐列一致**。
> 不新增特征、不做查询优化、不碰 ML 侧。

- pandas 产出：`graph_features.parquet` 590,540 行 × 15 特征
- SQL 产出　：`graph_features_sql.parquet` 590,540 行 × 15 特征
- 行数一致：✅；TransactionID 集合一致：✅

## 逐列对账

| 特征列 | 类型 | pandas 缺失 | SQL 缺失 | 不一致行数 | 最大绝对差 | 结论 |
|---|---|---|---|---|---|---|
| `card1_prior_cnt` | int64 | 0 | 0 | 0 | 0 | ✅ 一致 |
| `card1_prior_fraud_cnt` | int64 | 0 | 0 | 0 | 0 | ✅ 一致 |
| `card1_prior_fraud_rate` | float64 | 108,828 | 108,828 | 0 | 0 | ✅ 一致 |
| `card1_addr1_prior_cnt` | int64 | 0 | 0 | 0 | 0 | ✅ 一致 |
| `card1_addr1_prior_fraud_cnt` | int64 | 0 | 0 | 0 | 0 | ✅ 一致 |
| `card1_addr1_prior_fraud_rate` | float64 | 190,536 | 190,536 | 0 | 0 | ✅ 一致 |
| `card1_email_prior_cnt` | int64 | 0 | 0 | 0 | 0 | ✅ 一致 |
| `card1_email_prior_fraud_cnt` | int64 | 0 | 0 | 0 | 0 | ✅ 一致 |
| `card1_email_prior_fraud_rate` | float64 | 215,385 | 215,385 | 0 | 0 | ✅ 一致 |
| `card1_device_prior_cnt` | int64 | 0 | 0 | 0 | 0 | ✅ 一致 |
| `card1_device_prior_fraud_cnt` | int64 | 0 | 0 | 0 | 0 | ✅ 一致 |
| `card1_device_prior_fraud_rate` | float64 | 533,604 | 533,604 | 0 | 0 | ✅ 一致 |
| `card1_fanout_addr1` | int64 | 0 | 0 | 0 | 0 | ✅ 一致 |
| `card1_fanout_email` | int64 | 0 | 0 | 0 | 0 | ✅ 一致 |
| `card1_fanout_device` | int64 | 0 | 0 | 0 | 0 | ✅ 一致 |

## 关键特征分布对照

| 特征 | 实现 | 非空数 | 均值 | 中位 | p99 | 最大 |
|---|---|---|---|---|---|---|
| `card1_prior_fraud_rate` | pandas | 481,712 | 0.029643 | 0.012270 | 0.281579 | 1.000000 |
| `card1_prior_fraud_rate` | SQL | 481,712 | 0.029643 | 0.012270 | 0.281579 | 1.000000 |
| `card1_addr1_prior_fraud_rate` | pandas | 400,004 | 0.018692 | 0.000000 | 0.267857 | 1.000000 |
| `card1_addr1_prior_fraud_rate` | SQL | 400,004 | 0.018692 | 0.000000 | 0.267857 | 1.000000 |
| `card1_email_prior_fraud_rate` | pandas | 375,155 | 0.029792 | 0.002053 | 0.400000 | 1.000000 |
| `card1_email_prior_fraud_rate` | SQL | 375,155 | 0.029792 | 0.002053 | 0.400000 | 1.000000 |
| `card1_device_prior_fraud_rate` | pandas | 56,936 | 0.048619 | 0.000000 | 0.833333 | 1.000000 |
| `card1_device_prior_fraud_rate` | SQL | 56,936 | 0.048619 | 0.000000 | 0.833333 | 1.000000 |
| `card1_fanout_addr1` | pandas | 590,540 | 21.630572 | 17.000000 | 60.000000 | 64.000000 |
| `card1_fanout_addr1` | SQL | 590,540 | 21.630572 | 17.000000 | 60.000000 | 64.000000 |
| `card1_fanout_email` | pandas | 590,540 | 14.550420 | 13.000000 | 40.000000 | 43.000000 |
| `card1_fanout_email` | SQL | 590,540 | 14.550420 | 13.000000 | 40.000000 | 43.000000 |
| `card1_fanout_device` | pandas | 590,540 | 27.889889 | 7.000000 | 385.000000 | 614.000000 |
| `card1_fanout_device` | SQL | 590,540 | 27.889889 | 7.000000 | 385.000000 | 614.000000 |

## 结论

✅ **全部特征列逐行一致**（浮点按 rtol=1e-9 比较），缺失模式亦一致。

两处**本来最容易对不上**的地方，是这次实现里唯一需要动脑的部分：

1. **`ROWS` 与 `RANGE` 不能混用**。
   pandas 的 `prior_cnt` 取组内**位置**（`prior_cnt[s+i]=i`），dt 并列时靠前那行**算**在内；
   而 `obs_cnt` 取的是 `dt <= t − embargo` 的**取值**条件。
   → 前者必须是 `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING`（且 ORDER BY 带 `transaction_id` 复刻 lexsort 的稳定次序），后者必须是 `RANGE BETWEEN UNBOUNDED PRECEDING AND 1814400 PRECEDING`。
   **负对照证明这不是风格问题**：把 `prior_cnt` 换成 RANGE 帧后，
   pandas vs ROWS 版不一致 **0** 行、pandas vs RANGE 版不一致 **166** 行（card1 键；其余键 72/77/4）。
   全局有 17,191 行 dt 并列，但只有**同组内**的并列才会造成差异——先前我按全局并列数说事是夸大了，这里按实测值订正。

2. **NULL 键必须各自成组**。
   pandas 侧 `codes_group` 让 NA 行各自独立成组（早先塌成巨型组是个已修的 bug）；
   SQL 的 `GROUP BY`/`PARTITION BY` 默认把 NULL 视为相等、会重新塌成一组。
   → 用 `COALESCE(key, '\x00NA#' || transaction_id)` 复刻。
   复合键在两侧都天然传播 NULL（pandas 字符串相加、SQL `||`），这一点无需额外处理。

> 顺带一提：**「窗口帧类型」正好对应本项目的两层防泄漏**——
> 结构型只问「在不在之前」（ROWS），标签型还要问「标签熟没熟」（RANGE + embargo 偏移）。
> 同一条纪律，在 pandas 里是两段代码，在 SQL 里是两种帧。
