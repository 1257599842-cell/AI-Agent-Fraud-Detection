# Agent 层 [W5] 拍板终稿 —— 设计决策登记表（审阅修订版）

## Context

ML 统计核心已定稿（防守点①②③④⑤⑥⑩，全部真实 delta）。Agent 层是最后三个防守点（⑦eval / ⑧成本 / ⑨兜底）的载体。owner 提交了五组拍板初稿，本次会话以专家审阅通过，**修订 5 处**后定稿。本文件 = 开工前的最终决策登记表；批准即拍板。**本次不动工**，动工按文末施工顺序另起。

## 审阅修订（5 处，已并入下方登记表）

1. **③ 一致性修复（措辞级，最高优先）**：四档 argmin 喂 **raw 概率**，不喂 Platt/Iso 输出（③ 实测 blanket 校准伤决策区间尾部：top2% gap 0.002→0.067）。链条改述为「③决策区间可靠性**验证** → ①代价敏感 → 四档」——③ 是概率的前置资质检查，非加工步骤；漂移打坏校准时由 ③⑩ 控制回路触发重校。
2. **上报档加网络项（防四档塌三档）**：E_上报 = c_report + 残余错误 − P(团伙关联)×预期未来暴露；未来暴露 ≈ 关联实体 fan-out × prior_fraud_rate × 中位金额（`src/features/graph_features.py` 现成量）。①⑥ 由此焊成一条链，gang_association 字段获得成本语义。
3. **5.1 翻转实验指标下沉到证据层**：处置档是喂入概率的 argmin 函数，处置翻转率混淆"合规算术跟随"与"谄媚证据扭曲"，不可用。改测：key_findings/gang_association/evidence_insufficient/assertion_strength 的稳定率 + **冲突检出率**（喂错分时 Agent 是否报告"分与证据矛盾"）。
4. **规则库数字防泄漏**：入库统计量（ProductCD C 欺诈率、邮箱域欺诈率等）全部在训练窗 **[0,146)** 重算（现 EDA 数字含测试窗，不可直接入库）。案例库的时间纪律（[0, t−21d]）同样适用于规则库。
5. **分层 eval 集的成本数字要加权**：「每万笔期望损失 X→Y」在欺诈富集的分层集上须按逆抽样概率加权（Horvitz–Thompson），或另留自然分布切片算钱；分层集只按层报一致率/幻觉率。

## 已定决策登记表（骨架，勿在施工中擅改）

### 第一组 · 知识层
- **1.1 规则库**：自写 15–25 条，从真实 EDA/图特征结论提炼；**统计量在 [0,146) 重算**（修订4）。
- **1.2 案例库**：[0, t−21d] 滚动窗；正例（成熟窗 isFraud=1 分层抽样）+ 负例（同窗 GBDT 高分假阳）各 ~1,500，共 ~3,000；检索 top-4。诚实边界：案例库本身是选择性偏差样本（⑤ 在 RAG 层的翻版），主动讲。标签有传播性，讲团伙时别说成 N 次独立作案。
- **1.3 embedding/向量库**：BGE 或 Qwen 自托管 + Chroma/FAISS，CPU。
- **1.4 检索粒度**：一案一条、一规则一 chunk。**案例检索主通道 = 结构化相似**（同组合键/金额档/ProductCD），作为 `retrieve_rules_and_cases` 内部实现；向量检索用于规则（文本主场）+ 案例卡粗排。案例入库前渲染成模板化自然语言案例卡，embedding 吃卡不吃原始行。

### 第二组 · 工具层
- **2.1 工具集锁死四个**：query_transaction / query_entity_graph / query_historical_stats（卡时间边界）/ retrieve_rules_and_cases。不加假工具；缺口话术 =「接入范式同构 + 每种新数据源的坑我在 IEEE-CIS 上踩过一遍」（不说"本质变化不大"）。query_similar_transactions 不单列，并入 1.4 检索后端。
- **2.2 返回格式（焊点级）**：结构化 JSON + fact_id 唯一 + 报告强制 evidence_ids 引用 + 每个事实带 window/as-of（时间边界可审计）。
- **2.3 调用上限**：8 次/单，超则强制收尾（兼 ⑧ 素材）。

### 第三组 · 输出层
- **3.1 报告 schema**：txn_id / risk_level / key_findings[]（finding + evidence_ids + assertion_strength）/ gang_association / disposition（四档）/ disposition_rationale / confidence（序数，非校准概率）/ evidence_insufficient / summary。JSON 不散文；人读的放 summary。
- **3.2 四档 ground truth = 期望成本 argmin**（含修订1、2）：
  - 放行 E = p·金额；拒绝 E = (1−p)·c_FP
  - 挂起 E = c_review + p·漏检率_人·金额 + (1−p)·误拦率_人·c_FP
  - 上报 E = c_report + 更低残余 − P(团伙)×未来暴露（图特征估计）
  - p = **raw** GBDT 概率（③ 决策区间验证背书）。成本参数全部做敏感性扫描 + (p,金额,团伙证据) 空间四档分区图（新展品）。
  - Agent 可凭 evidence_insufficient/明示 rationale 偏离公式，偏离由一致性第一层捕获。

### 第四组 · Eval（⑦，叙事领衔）
- **总原则**：能硬则硬；软层只留推理合理性。
- **Groundedness**：硬层（代码对账 evidence_ids 存在性/数字实体真伪）→ 事实层通过率；软层（judge+人工锚）→ 推理层通过率。分开报。
- **处置一致性（最硬）**：层1 实际档 vs 应然档（代码比）；层2 代入真标签算实际期望成本 vs naive 全放行 vs 完美后见。**成本总额按修订5 加权**。
- **幻觉率**：编造率（硬）/ 该弃权未弃权率（半硬，真标签反查）/ 过度断言率（软，assertion_strength vs 证据强度）。
- **eval 集**：200 条 = 4 层×50（高分真欺诈/高分假阳/中分模糊/低分正常），取自测试窗 [146,182]。
- **judge**：DeepSeek/Qwen（与被评 Claude 分家）；人工锚 60 条，报 judge-人工一致率。

### 第五组 · 实验与工程
- **5.1 喂分 + 翻转实验**（含修订3）：喂真分 vs 翻转分，测证据层稳定率 + 冲突检出率；prompt 把分当"待核实线索"。概率判断权归 GBDT。
- **5.2 成本（⑧）**：GBDT 闸门挡 ~~96.5%~~ + 上限 8 + DeepSeek 起草/Claude 把关；记每单 token 成本、平均调用次数。
  > ⚠️ **2026-08-08 订正**：`96.5%` 是本文件成稿时（2026-07-25，早于 `disposition.py`）的**估计值**，非实测。
  > 实测闸门放行率（四档 approve 直接放行、不进 Agent）= **90.3%**（`reports/stepup.md`，
  > 与 `demo_data.json` 的 `gate_pass_rate_four_tier=0.9029` 一致）。
  > **注意数值撞车**：`96.5%` 另有一处**完全不同**的合法用法——bootstrap 概率 `P(生产拓扑 < 全放行)=96.5%`
  > （见 INTERVIEW ⑧）。两个量不相干却同值，极易串用，引用前必须回源。
- **5.3 兜底（⑨）**：LLM 挂/超时/限流 → GBDT 出分 + 规则模板报告；主链路不依赖外部 LLM。

### 参数初值（全部标注为假设 + 敏感性扫描）
| 参数 | 初值 | 扫描 |
|---|---|---|
| c_FP | $25（已锚） | $10–100 |
| c_review | $5 | $2–20 |
| c_report | $40 | $20–100 |
| 人工复核漏检率 | 10% | 5–20% |
| 上报残余漏检率 | 5% | 2–10% |

### 终裁
- **GTAN 对照臂：放弃**（⑥ 已有分组消融量化依据，一句"边际收益 vs 复杂度"带过）。
- **模型分工**：Agent 主模型 = Claude API；judge + 起草 = DeepSeek/Qwen。
- 待办提醒：Kaggle token 仍未 Expire 重新生成（2026-06-30 遗留）。

## 施工顺序（拍板后另起会话/下一步执行，小步走）
1. 报告 schema + 四工具接口与返回格式（`src/agent/tools.py`、`src/agent/schema.py`）——先接口后实现
2. 规则库（训练窗重算统计量）+ 案例库构建（`src/agent/knowledge.py`）
3. 四档成本框架 + 分区图（`src/agent/disposition.py`，复用 `src/model/cost_sensitive.py` 口径）
4. Agent 调查管道（出分→检索→工具→报告）+ ⑨ 兜底
5. eval 三指标 + eval 集构建（`src/eval/agent_eval.py`）
6. 5.1 翻转实验 + 3.2 敏感性扫描 + ⑧ 成本统计
7. 每步更新 PROGRESS.md

## 验证方式
- 每个硬层 eval 指标可用代码在无 LLM 情况下自测（mock 报告注入已知错误 → 应被抓出）。
- 泄漏自查：规则库/案例库/工具返回的任何统计量时间戳 ≤ 查询交易时间 − 21d。
- 端到端：抽 5 笔已知结局交易跑全管道，人工核报告与 evidence_ids 对账。

## 施工提醒（owner 补充，2026-07-25，不改骨架）
1. **修订 2 的未来暴露项自身要卡时间边界**：E_上报 里的 fan-out × prior_fraud_rate 必须只用 t 之前的边（结构型）/ t−21d 之前的标签（标签型）——即完全复用 `graph_features.py` 的两层防泄漏口径，`disposition.py` 施工时逐项确认。否则修订 4 堵上的泄漏从修订 2 漏回来。
2. **冲突检出率要配对照组**（实验期落实）：喂错分测「真冲突检出率」的同时，喂对分测「假冲突误报率」——两个一起报才是完整独立性画像，否则"逢分必疑"的 Agent 也能刷满检出率。
