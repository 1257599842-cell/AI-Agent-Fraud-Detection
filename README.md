# AI Fraud Investigation Copilot

> 交易反欺诈**风险评分**（GBDT）+ **AI Agent 自动调查**（RAG / 工具调用 / 调查报告 + 处置建议）+ 人工复核。
> 2026 秋招旗舰项目。完整设计与工作守则见 [`CLAUDE.md`](./CLAUDE.md)，实时进度见 [`PROGRESS.md`](./PROGRESS.md)，**面试答案库见 [`INTERVIEW.md`](./INTERVIEW.md)**。
> **交付文档见 [`MODEL_CARD.md`](./MODEL_CARD.md)** —— 模型卡 / 模型验证报告，含用途边界、口径、假设、**已知局限与偏差**、监控与退化触发、降级方案、以及"已推演但未实现"的工业化路径。

这份 README 是**滚动复盘** —— 边做边写，最后直接变成面试叙事与简历素材。
**当前状态**：四层全部落地（ML 核心 / 图特征 / Agent + eval / FastAPI·Docker），十个防守点全部有真实数字；
另有一份[《作废清单》](./INTERVIEW.md)记录测过之后决定**不用**的十一件事。

---

## 演示（离线，零依赖）

> **双击即开，不联网、不调用任何 API、不产生任何费用**——页面内嵌 7 笔预置案例的完整证据链。

```bash
open reports/demo/index.html      # 或直接双击该文件；不需要装任何东西
```

### ① 决策成本沙盘 —— 阈值不是拍 0.5，是解出来的

![决策成本沙盘](reports/demo/shots/light_sandbox_1280x720.png)

拖动「欺诈概率」与「交易金额」两个滑块，**五档动作的期望成本实时重算，最低者即处置建议**。
图示工作点（p=0.30 / $20）最优动作是**加验证**（$2.41），而不是挂起（$5.95）——
因为这笔金额太小，**掏 $5 让人复核在经济上不划算**，直接放行又要吃 0.30 的欺诈概率。
**这不是拍出来的规则，是五个期望成本比出来的。** 成本参数全部标注为 `[假设]` 并可现场调整。

### ② 调查案卷 —— 每条结论都必须挂证据，且证据带时间边界

![调查案卷与证据链](reports/demo/shots/dark_case_1280x720.png)

展开任意一条结论，可见它引用的原始 fact、取值、以及**该 fact 的时间窗与类型**：
**标签型**事实（欺诈率、gang_score）的窗口右端**至少比本笔交易早 21 天**——这就是**标签成熟期（embargo）**：
拒付要等这么久才回来，训练时不能假装当时就知道。**结构型**事实（fan-out）则只要求发生在本笔之前。
> 图中这笔的看点：GBDT 只给了 **p=0.007**，Agent 独立核出该卡扇出 375 台设备、成熟窗欺诈率 5.59%，
> 写下「模型分与证据严重冲突」并建议上报——**但成本明细显示公式仍判放行**（网络项在这个分数上翻不了档）。
> 这正是本项目把**处置权交回公式、只让 Agent 负责取证**的现场证据：它的证据发现是对的，算术不是。

页面另含 **⑧ 闸门漏斗**（**90.3%** 的交易根本不进 Agent）与 **⑨ 兜底开关**
（模拟 LLM 不可用 → 降级报告仍通过同一套校验器、服务返回 200 而非 5xx）。

---

## 一句话定位

模型负责**预测**，Agent 负责**解释 / 检索 / 调工具 / 给建议**，人负责**拍板**。
深度 > 广度：少数几个必被追问的硬点，做到能扛住二十分钟追问的深度。

**对外诚实定位**：这是**离线决策系统 + 已推演工业化路径**，**不声称"生产级"**（叙事红线见 CLAUDE.md 第九节）。

## 数据

- **Kaggle IEEE-CIS Fraud Detection**（Vesta 真实电商交易，约 59 万笔，欺诈率约 3.5%）。
- 两张表 `train_transaction` + `train_identity`，按 `TransactionID` 关联（仅约 24% 交易有 identity）。
- **线下评估集从 train 按时间切分**得到（+21 天 embargo 防标签泄漏）；Kaggle 官方 test 集无标签，**不参与任何线下评估**。

## 架构（四层）

1. **ML 核心**（CPU）：LightGBM 风险评分 + 特征工程（表特征 + 图特征）。
2. ~~可选 GNN 对照臂（GTAN）~~ —— **已放弃且未跑**：分组消融证明结构信号已被匿名 C/V 特征吸收（fan-out 组 −0.0008），GNN 要追的正是被吸收的部分 → 用量化依据替代了这次实验（见下表）。
3. **Agent 层**（调 LLM API）：出分 → RAG 检索规则/案例 → 工具调用 → 调查报告 + 处置建议 → 人工复核。
4. **工程层**：FastAPI、Docker、ML/Agent eval、token 成本监控、兜底降级、漂移监控。

## 十个面试防守点（= 项目灵魂，逐条攻克）

1. 代价敏感阈值（FP/FN 不对称 + 复核容量）
2. 时间切分防泄漏 + 标签延迟
3. 概率校准
4. 不平衡的正确处理（为何慎用 SMOTE）——**反例消融**：重采样/加权 recall 不升、ECE 从 0.008 爆到 0.09（图 10）
5. 选择性偏差（放行的欺诈看不到）——**已做成可演示实验**：naive「放行=legit」ROC 0.872→0.848（−0.024），2% 随机抽检救回 +0.018（图 09）
6. GBDT vs GNN 的工程取舍——**对照 +0.039 PR-AUC**（过泄漏审计）；消融证增益全来自实体标签历史、结构特征≈0（已被匿名特征吸收）→ GNN 追的正是被吸收的结构；团伙 fan-out 展品（图 11）
7. Agent 的 eval（groundedness / 幻觉率 / 处置一致性）——**能硬则硬**：硬层（编造数字 0、引用完整 **92.4%**、泄漏审计 100%、处置一致率 51–66%）；**软层 LLM-as-judge 试过并作废**（要验 judge 需要一个非 LLM 的参照，而我没有）
8. Agent 的成本与延迟（GBDT 当闸门 / 缓存）——每单 $0.113、5.5 次工具调用；**闸门的第二重身份 = 防 Agent 过度上报的行为护栏**
9. 兜底降级（LLM 挂了照常出分）——**真触发过**，降级报告过同一套校验器
10. 人在环 + 反馈回流 + 偏差缓解；**概念漂移已做成可演示实验**：分窗监控 + 触发器（诚实结论：ROC 半年稳/0 告警、PR 更敏感，图 08）

> 每条防守点的 30 秒 / 2 分钟答案见 [`INTERVIEW.md`](./INTERVIEW.md)。

## 复现

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# 数据：配置 Kaggle CLI 后 kaggle competitions download -c ieee-fraud-detection
```

## 进度

见 [`PROGRESS.md`](./PROGRESS.md)。

## 关键数字（记 delta，不记绝对值；跑出来再填，禁止编造）

> 面试官不关心 AUC 0.92，关心你带来多少**改善**。所有指标记成 X→Y / 差多少。
>
> ⚠️ **版本基线**：下表所有 Agent 侧指标取自 **v3 证据池**（运行 r1 / r3 / r4）。
> 其后有一次基础设施轮 **v4-citable-context**（缺席事实 `null_result` + 成本假设 `policy_param` 进证据池并可被引用），
> **证据池已变，v4 之后的运行不可与本表数字直接比较**——跨版本比较必须重取基线。本表数字如实描述其所属版本。

| 指标（delta 形式） | 数值 |
|------|------|
| 朴素 0.5 → 代价敏感阈值 t*=0.078（c_FP=$25）：有效拦截率 recall | 0.338→0.623，总成本省 26% |
| ↳ 乐观 gap 归因（等量旧窗对照）：数据量 vs 新鲜度 | 数据量≈0(+0.005)·新鲜度−0.037 |
| 概率校准：决策区间验证 | raw top2% gap 0.002（本已较准）；后验校准反伤尾（gap→0.07，42天仍0.056）→ 不 blanket 套 |
| 选择性偏差演示：naive 衰减 → 随机抽检救回 | ROC 0.872→0.848（−0.024）→ 救回 +0.018 |
| 不平衡反例消融：重采样毁校准 | recall 不升、ECE 0.008→0.09（11×）、决策区间预测 0.70→0.93 虚高 |
| 时间切分：线下→线上（21天 embargo）乐观 gap | PR-AUC 0.5645→0.5323（Δ−0.032）·ROC 0.914→0.903（Δ−0.011）|
| 概念漂移监控：冻结模型 6 个月滚动窗 | ROC 0.881~0.918、**0 次触发**（δ=0.02）；**PR-AUC 波动 ~0.09 更敏感 → 正类监控该盯 PR** |
| 图特征增益（纯表→表+图）：PR-AUC | 0.5645→0.6032（+0.039，60天embargo仍+0.023）|
| 图增益的分组消融：增益来自哪 | **~100% 来自实体标签历史**；度/fan-out 结构 ≈0（已被匿名 C/V 吸收）→ GNN 对照臂据此**放弃**（未跑，不作数字声称）|
| Agent eval 硬层：结构合规 / 编造数字 / 引用完整 / 泄漏审计 | 99% / **0**（2,672 个数字全量对账，`true_ungrounded` 0/565）/ **92.4%**(finding 级 522/565) / 100% |
| Agent 处置一致性（层1，vs 代价最优档，dev 100 笔） | **51–66%**（432 组成本参数网格；**单点不可报**——100 笔里 40 笔的应然档随参数改档）。**金额口径经 jackknife 判定不可用**（n_eff 29.4、单笔占 68%、剔 2 笔排名翻转）|
| Agent 处置的成本口径（HT 加权，每万笔） | 生产拓扑 **$22,914** < 全放行 $28,417；纯公式 argmin $13,811 → **Agent 价值在证据叙事，处置服从成本框架**（CI 宽，报排序不报点值）|
| Agent 每单调查成本 / 工具调用 | **$0.113** / 平均 **5.5 次**（上限 8）|
| 网络项效度（四档框架里唯一未验证的假设） | 高 gang 实体**后续 30 天**欺诈率 lift **4.6–5.0×**（两个不重叠时间窗独立复制：t0=146→4.95×、t0=115→4.56×；在"同样有欺诈史"的实体内部比，避免同义反复；留一实体后 4.66×）；拆解：密度 3.60–5.53× / **fan-out 1.72–2.64×**（prior_cnt 分层内池化仍 3.15×）→ **方向已验证、量级未标定，保守记账保留** |
| ⚡ 同一特征、两个相反答案（⑥ 与网络项的焊接） | fan-out 对"**这一笔**是否欺诈"增量 ≈0（−0.0008，已被匿名 C/V 吸收）；对"**该实体未来**是否再作案"有 1.72–2.64× → **"有没有用"要先问"对哪个目标"** |
| 证据层 vs 决策层（① 的依据，已做多数类基线校正） | 证据层 89%（基线 57%，**+32pp**）vs 决策层 57%（基线 45%，**+11pp**）→ **净差 +21pp** → 取证归 Agent、算术归公式。效度层面（对 isFraud 的分离度）CI 跨 0，**未测出，正确的锚未跑** |
| 谄媚/独立性（5.1 翻转，含假阳对照 + 第四臂） | 分数拉高 150×，证据层翻转 **0/15**；冲突检出剂量梯度 0% → 27% → 47% → 100%，关键对照 **+73pp**（CI [+51,+96]）|
| eval 驱动的迭代（round 3，预注册一次性修订） | **干净的零**：偏宽分歧率 28%→27%（CI [−13,+11]）、decline 1/100→1/100。配对迁移显示 20 笔改档、方向全对，但改对 7/改坏 6/换个错法 7 → **听见了、照做了，照做不能让它更准** → 处置权交回闭式解 |
| Agent eval 软层（LLM-as-judge） | **已作废、不报数**——判定标准与参照方同源、少数类仅 3 条 → 判别力不可测（详见 PROGRESS ⚠️ 修正节）|
| 容量工作点 top1%（无 embargo）：precision / recall(=有效拦截率) | 89.7% / 26.2% |
| （参考）baseline AUC（无 embargo）：ROC-AUC / PR-AUC | 0.9138 / 0.5645 |

---

## 同一套特征的两种实现（pandas / SQL·DuckDB）

风控与数仓岗位普遍把 SQL 当硬门槛，而本项目的特征流水线原本是纯 pandas。
因此把**同一套特征**用 DuckDB + 小星型模型重写了一遍。

**范围声明**：不新增任何特征、不做查询优化、不碰 ML 侧。
**唯一成功判据**：与 pandas 产出**逐列对账**——15 个特征列全部逐行一致
（浮点 rtol=1e-9），缺失模式与分布同样一致。详见
[`reports/sql_vs_pandas_reconciliation.md`](reports/sql_vs_pandas_reconciliation.md)。

### 小星型模型

```mermaid
erDiagram
    fact_transaction  }o--|| dim_card    : card_sk
    fact_transaction  }o--|| dim_addr    : addr_sk
    fact_transaction  }o--|| dim_email   : email_sk
    fact_transaction  }o--|| dim_device  : device_sk
    fact_transaction  }o--|| dim_product : product_sk
    fact_transaction  }o--|| dim_date    : date_sk
    fact_transaction  ||--|| fact_graph_feature : transaction_id

    fact_transaction {
        bigint transaction_id PK
        bigint dt
        bigint date_sk FK
        bigint card_sk FK
        bigint addr_sk FK
        bigint email_sk FK
        bigint device_sk FK
        bigint product_sk FK
        double amt
        tinyint is_fraud
    }
    dim_card    { bigint card_sk PK
                  bigint card1
                  varchar card4
                  varchar card6 }
    dim_addr    { bigint addr_sk PK
                  double addr1
                  double addr2 }
    dim_email   { bigint email_sk PK
                  varchar p_emaildomain
                  varchar r_emaildomain }
    dim_device  { bigint device_sk PK
                  varchar device_info
                  varchar device_type }
    dim_product { bigint product_sk PK
                  varchar product_cd }
    dim_date    { bigint date_sk PK
                  bigint day
                  bigint week_idx
                  bigint dow_idx }
    fact_graph_feature {
        bigint transaction_id PK
        bigint card1_prior_cnt
        bigint card1_prior_fraud_cnt
        double card1_prior_fraud_rate
        bigint card1_fanout_device
    }
```

| 表 | 行数 | 说明 |
|---|---|---|
| `fact_transaction` | 590,540 | 事实表，粒度=一笔交易 |
| `fact_graph_feature` | 590,540 | 特征事实表，15 列图特征 |
| `dim_card` | 14,318 | card1/card4/card6 |
| `dim_device` | 1,943 | DeviceInfo/DeviceType |
| `dim_email` | 743 | 收/发件邮箱域 |
| `dim_addr` | 438 | addr1/addr2 |
| `dim_date` | 183 | 相对天（TransactionDT 非日历时间） |
| `dim_product` | 5 | ProductCD |

建表 DDL：[`src/features/sql/01_star_schema.sql`](src/features/sql/01_star_schema.sql)　
特征 SQL：[`src/features/sql/02_graph_features.sql`](src/features/sql/02_graph_features.sql)　
构建+对账：`python -m src.features.build_duckdb`

### 这次重写里唯一需要动脑的两处

**1. `ROWS` 与 `RANGE` 必须分开用——它正好就是本项目的两层防泄漏。**

| | 语义 | 窗口帧 |
|---|---|---|
| 结构型（`prior_cnt` / fan-out） | 只问「在不在本行之前」 | `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING` |
| 标签型（`prior_fraud_cnt`） | 还要问「标签熟没熟」 | `RANGE BETWEEN UNBOUNDED PRECEDING AND 1814400 PRECEDING` |

pandas 侧 `prior_cnt` 取的是组内**位置**（`prior_cnt[s+i]=i`，dt 并列时靠前那行算在内），
而 `obs_cnt` 取的是 `dt ≤ t−embargo` 的**取值**条件。用错帧型就对不上。

**负对照证明这不是风格问题**：把 `prior_cnt` 改成 RANGE 帧后，
pandas vs ROWS 版不一致 **0** 行、vs RANGE 版不一致 **166** 行。
（一个通不过负对照的对账，通过了也说明不了什么。）

**2. NULL 键必须各自成组。**
pandas 侧 `codes_group` 让缺失值各自独立成组（早先塌成一个巨型组是已修的 bug）；
而 SQL 的 `PARTITION BY` 默认把 NULL 视作相等、会重新塌回去。
用 `COALESCE(key, '\x00NA#' || transaction_id)` 复刻。

> 一句话：**同一条时间因果纪律，在 pandas 里是两段手写循环，在 SQL 里是两种窗口帧。**
