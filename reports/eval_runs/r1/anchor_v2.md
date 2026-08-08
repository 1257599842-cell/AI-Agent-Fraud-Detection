# round3 盲标表（缩表版，17 条 · 估时 1.5–2h）

> **标注前请勿打开 `anchor_v2_manifest.json`**：它记着每条属于哪个臂（旧少数类 / DISPUTE / 漂移抽查 / 模板筛出），
> 知道了会让你对某些条目手更紧、对另一些手更松，把要测的东西污染掉。
> 表内条目按交易号排序、臂别不可见；旧条不显示历史标注，请重新独立判断。

> **本表与 judge（arm3）逐字同卷**：判据同下、证据同为整单证据池。

> **这份表的定位**：不是「人工锚定研究」，是 **owner 亲手核过的缺陷分类学 + 金标漂移方向**。n 太小，产出**一律不报比率**，只报存在性、方向与原始计数。

## 判据（RUBRIC_V2，与 judge system prompt 程序化同源）

```
【reasoning_valid】默认 true。只有下列三种之一成立才判 false；未列入的一律 true：
(a) 证据反证结论：引用的 fact 直接指向与结论相反的方向；
(b) 证据与结论无关：引用的 fact 与主张之间无逻辑连接（挂错证据）；
(c) 非因果跳跃：断言了一个行为类别/机制，而所引 fact 类型根本不足以确立它（判 c 必须能指出缺失的那类证据，指不出就不是 c）。
以下一律判 true（round1 在此系统性过严）：对冲/条件化（"分数高但有同卡假阳先例、不宜单独作拒绝依据"是正确权衡证据）、不完整、恰当的不确定（明示"证据不足以单独定性"）、某前提未挂 fact 但结论由已引证据独立成立、程度形容偏松偏紧（基率±50%内的形容词是判断不是误读）。
若"值/属性未挂对应 fact 但在整单证据池里能找到出处"——这是引用瑕疵，不因此判 false。
判定困难 → 判 true。不确定时倒向 true。

【overclaim】先假定 reasoning_valid=true，只问 assertion_strength 是否与证据匹配；不得因推理/接地问题顺带判 true。
按"主张范围 vs fact 范围"定该主张可达最高档：
- confirmed：主张语义范围=所引 fact 范围且直接检索所得（字段值/实体级图特征 gang_score·fanout·prior_fraud_rate/类别统计量，按本义引用）；
- supported：主张超出任一单条 fact，但由≥2 条 fact 或"一条统计事实+明示推理链"导出；
- tentative：仅凭单一弱/宽信号（人群级规则 lift<2、覆盖 6万-11万笔）、或关键对照缺失、或明示推测。
overclaim=true 仅当 assertion_strength 超过该最高档。典型 true：引 prior_fraud_rate=0.47（实体历史）却断言"本笔是欺诈"（主张范围越过 fact 范围）；纯字段值→"卡测试/欺诈农场画像"标 supported。
从宽（一律 false）：标低于可达档（保守偏置）；标 supported 且有真实统计事实或≥2-fact 链，即使措辞对冲。
注意：gang_score/fanout/prior_fraud_rate 是实体个体特征，可达 confirmed；ProductCD=C 11.27% 这类人群先验单用时最高只到 tentative。

【few-shot】
① finding 称 3 条确认欺诈案例均在 addr1=387 与本笔 addr1=325 不同，但所引案例明写 addr1=325 → {"reasoning_valid":false,"overclaim":false}（证据反证，情形a）
② finding 称"未命中任何风控规则"，所引唯一证据是一条确认欺诈案例 → {"reasoning_valid":false,"overclaim":false}（证据无关，情形b）
③ 仅凭 6 条字段值断言"典型卡测试/欺诈农场画像"，无 rule/stat/实体证据 → {"reasoning_valid":false,"overclaim":true}（跳跃c+强度虚高）
④ "GBDT p=0.64 与证据方向一致，但有同卡假阳先例、不宜直接作拒绝依据；金额极小拦截收益有限"，引实体欺诈率+2条同卡假阳+金额 → {"reasoning_valid":true,"overclaim":false}（对冲是权衡，不判N）
⑤ finding 称"4 条中 3 条假阳"但只挂 3 条案例，核心区分论证证据完整 → {"reasoning_valid":true,"overclaim":false}（案例计数失真是引用瑕疵，不进 reasoning）
⑥ "该 card1 成熟欺诈率 16.67%、card1+设备 31.58%、card1+邮箱 10.53%"三条直接查表 → {"reasoning_valid":true,"overclaim":false}（confirmed 恰当）
⑦ "5124 笔约 95% 非欺诈、gang_score 0.516 中等、未返回假阳对照，无法单独定性欺诈" → {"reasoning_valid":true,"overclaim":false}（tentative 恰当，恰当的不确定）
```

共 17 条。每条在末尾两行的冒号后填 Y 或 N —— 第一行 Y=推理成立、第二行 Y=断言强度超过证据。不确定 → 填 Y 并在行尾写 uncertain。

---

# 交易 3481803（GBDT p=0.0048）
<details><summary>本单【全部证据池】——核实被断言的值是否在整单里有据；「有据但该 finding 没引」属引用瑕疵，按 v2 不判 N</summary>

- `TXN_000`: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3481803", "value": 34.0, "window": [12935315, 12935315], "label_based": false}
- `TXN_001`: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3481803", "value": "W", "window": [12935315, 12935315], "label_based": false}
- `TXN_002`: {"fact_id": "TXN_002", "type": "txn_field:card1", "entity": "txn=3481803", "value": 15224, "window": [12935315, 12935315], "label_based": false}
- `TXN_003`: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3481803", "value": "mastercard", "window": [12935315, 12935315], "label_based": false}
- `TXN_004`: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3481803", "value": "debit", "window": [12935315, 12935315], "label_based": false}
- `TXN_005`: {"fact_id": "TXN_005", "type": "txn_field:addr1", "entity": "txn=3481803", "value": 441.0, "window": [12935315, 12935315], "label_based": false}
- `TXN_006`: {"fact_id": "TXN_006", "type": "txn_field:P_emaildomain", "entity": "txn=3481803", "value": "gmail.com", "window": [12935315, 12935315], "label_based": false}
- `GRAPH_000`: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=15224", "value": 84, "window": [0, 12935315], "label_based": false}
- `GRAPH_001`: {"fact_id": "GRAPH_001", "type": "prior_cnt", "entity": "card1=15224|addr1", "value": 4, "window": [0, 12935315], "label_based": false}
- `GRAPH_002`: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=15224", "value": 19, "window": [0, 12935315], "label_based": false}
- `GRAPH_003`: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=15224", "value": 5, "window": [0, 12935315], "label_based": false}
- `GRAPH_004`: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=15224", "value": 1, "window": [0, 12935315], "label_based": false}
- `GRAPH_005`: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=15224", "value": 0.0, "window": [0, 11120915], "label_based": true}
- `GRAPH_006`: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|addr1", "value": 0.0, "window": [0, 11120915], "label_based": true}
- `GRAPH_007`: {"fact_id": "GRAPH_007", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.0, "window": [0, 11120915], "label_based": true}
- `GRAPH_008`: {"fact_id": "GRAPH_008", "type": "gang_score", "entity": "card1=15224", "value": 0.0, "window": [0, 11120915], "label_based": true}
- `CASE_000`: {"fact_id": "CASE_000", "type": "case", "entity": "case=3414443", "value": "案例 #3414443（day 123）：确认欺诈（拒付举报）。金额 $35.00（$10-50），ProductCD=W，mastercard/debit，邮箱域=gmail.com，设备=缺失，card1=17131, addr1=264。实体历史：prior 1110 笔、成熟欺诈率 1%；设备 fan-out 12。", "window": [10796864, 10796864], "label_based": true}
- `CASE_001`: {"fact_id": "CASE_001", "type": "case", "entity": "case=3413040", "value": "案例 #3413040（day 123）：高分假阳（人工洗清，当时模型分 0.17）。金额 $49.00（$10-50），ProductCD=W，mastercard/debit，邮箱域=gmail.com，设备=缺失，card1=14089, addr1=264。实体历史：prior 13 笔、成熟欺诈率 0%；设备 fan-out 1。", "window": [10777337, 10777337], "label_based": true}
- `CASE_002`: {"fact_id": "CASE_002", "type": "case", "entity": "case=3392785", "value": "案例 #3392785（day 117）：高分假阳（人工洗清，当时模型分 0.28）。金额 $49.00（$10-50），ProductCD=W，mastercard/debit，邮箱域=gmail.com，设备=缺失，card1=12257, addr1=110。实体历史：prior 12 笔、成熟欺诈率 0%；设备 fan-out 0。", "window": [10256268, 10256268], "label_based": true}
- `CASE_003`: {"fact_id": "CASE_003", "type": "case", "entity": "case=3387416", "value": "案例 #3387416（day 115）：高分假阳（人工洗清，当时模型分 0.25）。金额 $49.00（$10-50），ProductCD=W，mastercard/debit，邮箱域=gmail.com，设备=缺失，card1=9992, addr1=204。实体历史：prior 70 笔、成熟欺诈率 0%；设备 fan-out 5。", "window": [10098291, 10098291], "label_based": true}
- `STAT_000`: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=W", "value": 0.0205, "window": [0, 11120915], "support_n": 318733, "label_based": true}
- `STAT_001`: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "P_emaildomain=gmail.com", "value": 0.0439, "window": [0, 11120915], "support_n": 169477, "label_based": true}
- `STAT_002`: {"fact_id": "STAT_002", "type": "fraud_rate", "entity": "card4=mastercard", "value": 0.0352, "window": [0, 11120915], "support_n": 139893, "label_based": true}

</details>

## 3481803#4（断言强度=supported）
finding：No risk rules were triggered, and 3 of the 4 similar historical cases were high-scoring false positives (manually cleared). The single confirmed-fraud case (CASE_000) came from an entity with 1110 priors, 1% fraud rate and device fanout 12 — a profile that does NOT match this transaction's clean, low-fanout entity.
该 finding 引用的证据：
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3414443", "value": "案例 #3414443（day 123）：确认欺诈（拒付举报）。金额 $35.00（$10-50），ProductCD=W，mastercard/debit，邮箱域=gmail.com，设备=缺失，card1=17131, addr1=264。实体历史：prior 1110 笔、成熟欺诈率 1%；设备 fan-out 12。", "window": [10796864, 10796864], "label_based": true}
  - CASE_001: {"fact_id": "CASE_001", "type": "case", "entity": "case=3413040", "value": "案例 #3413040（day 123）：高分假阳（人工洗清，当时模型分 0.17）。金额 $49.00（$10-50），ProductCD=W，mastercard/debit，邮箱域=gmail.com，设备=缺失，card1=14089, addr1=264。实体历史：prior 13 笔、成熟欺诈率 0%；设备 fan-out 1。", "window": [10777337, 10777337], "label_based": true}
  - CASE_002: {"fact_id": "CASE_002", "type": "case", "entity": "case=3392785", "value": "案例 #3392785（day 117）：高分假阳（人工洗清，当时模型分 0.28）。金额 $49.00（$10-50），ProductCD=W，mastercard/debit，邮箱域=gmail.com，设备=缺失，card1=12257, addr1=110。实体历史：prior 12 笔、成熟欺诈率 0%；设备 fan-out 0。", "window": [10256268, 10256268], "label_based": true}
  - CASE_003: {"fact_id": "CASE_003", "type": "case", "entity": "case=3387416", "value": "案例 #3387416（day 115）：高分假阳（人工洗清，当时模型分 0.25）。金额 $49.00（$10-50），ProductCD=W，mastercard/debit，邮箱域=gmail.com，设备=缺失，card1=9992, addr1=204。实体历史：prior 70 笔、成熟欺诈率 0%；设备 fan-out 5。", "window": [10098291, 10098291], "label_based": true}
reasoning_valid: true
注：主张为条件化对比，缺引的本笔画像值（prior=84/fanout=1）在池中有据（GRAPH_004/005），属引用瑕疵，未命中反证/无关/非因果跳跃三类 N。
overclaim: false
“不匹配”是关系判断，由本笔低 fanout(1)/低 prior(84) 与 CASE_000 高 fanout(12)/高 prior(1110) 多事实对比推导，预设 supported 未超过可达范围，属保守偏置。

# 交易 3481840（GBDT p=0.0282）
<details><summary>本单【全部证据池】——核实被断言的值是否在整单里有据；「有据但该 finding 没引」属引用瑕疵，按 v2 不判 N</summary>

- `TXN_000`: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3481840", "value": 445.0, "window": [12935968, 12935968], "label_based": false}
- `TXN_001`: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3481840", "value": "W", "window": [12935968, 12935968], "label_based": false}
- `TXN_002`: {"fact_id": "TXN_002", "type": "txn_field:card1", "entity": "txn=3481840", "value": 16326, "window": [12935968, 12935968], "label_based": false}
- `TXN_003`: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3481840", "value": "visa", "window": [12935968, 12935968], "label_based": false}
- `TXN_004`: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3481840", "value": "credit", "window": [12935968, 12935968], "label_based": false}
- `TXN_005`: {"fact_id": "TXN_005", "type": "txn_field:addr1", "entity": "txn=3481840", "value": 204.0, "window": [12935968, 12935968], "label_based": false}
- `TXN_006`: {"fact_id": "TXN_006", "type": "txn_field:P_emaildomain", "entity": "txn=3481840", "value": "yahoo.com", "window": [12935968, 12935968], "label_based": false}
- `GRAPH_000`: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=16326", "value": 570, "window": [0, 12935968], "label_based": false}
- `GRAPH_001`: {"fact_id": "GRAPH_001", "type": "prior_cnt", "entity": "card1=16326|addr1", "value": 90, "window": [0, 12935968], "label_based": false}
- `GRAPH_002`: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=16326", "value": 41, "window": [0, 12935968], "label_based": false}
- `GRAPH_003`: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=16326", "value": 25, "window": [0, 12935968], "label_based": false}
- `GRAPH_004`: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=16326", "value": 20, "window": [0, 12935968], "label_based": false}
- `GRAPH_005`: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=16326", "value": 0.0236, "window": [0, 11121568], "label_based": true}
- `GRAPH_006`: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|addr1", "value": 0.0, "window": [0, 11121568], "label_based": true}
- `GRAPH_007`: {"fact_id": "GRAPH_007", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.0217, "window": [0, 11121568], "label_based": true}
- `GRAPH_008`: {"fact_id": "GRAPH_008", "type": "gang_score", "entity": "card1=16326", "value": 0.236, "window": [0, 11121568], "label_based": true}
- `RULE_000`: {"fact_id": "RULE_000", "type": "rule", "entity": "rule=R_card6_credit", "value": "高危取值 card6=credit（card6=credit）：训练窗 [0,125)天 内该模式欺诈率 6.65%，为基率 3.51% 的 1.9 倍（触发 112,717 笔，其中欺诈 7,501）。", "window": [86400, 10886394], "support_n": 112717, "label_based": true}
- `CASE_000`: {"fact_id": "CASE_000", "type": "case", "entity": "case=3168182", "value": "案例 #3168182（day 45）：确认欺诈（拒付举报）。金额 $150.00（$100-300），ProductCD=R，visa/credit，邮箱域=gmail.com，设备=MacOS，card1=16326, addr1=299。实体历史：prior 62 笔、成熟欺诈率 2%；设备 fan-out 17。", "window": [3982912, 3982912], "label_based": true}
- `CASE_001`: {"fact_id": "CASE_001", "type": "case", "entity": "case=3154125", "value": "案例 #3154125（day 40）：确认欺诈（拒付举报）。金额 $75.00（$50-100），ProductCD=H，visa/credit，邮箱域=hotmail.com，设备=Windows，card1=16326, addr1=299。实体历史：prior 61 笔、成熟欺诈率 3%；设备 fan-out 16。", "window": [3574643, 3574643], "label_based": true}
- `CASE_002`: {"fact_id": "CASE_002", "type": "case", "entity": "case=3413887", "value": "案例 #3413887（day 123）：高分假阳（人工洗清，当时模型分 0.15）。金额 $315.00（$300-1k），ProductCD=W，visa/credit，邮箱域=yahoo.com，设备=缺失，card1=7585, addr1=264。实体历史：prior 207 笔、成熟欺诈率 5%；设备 fan-out 66。", "window": [10788805, 10788805], "label_based": true}
- `CASE_003`: {"fact_id": "CASE_003", "type": "case", "entity": "case=3413885", "value": "案例 #3413885（day 123）：高分假阳（人工洗清，当时模型分 0.15）。金额 $315.00（$300-1k），ProductCD=W，visa/credit，邮箱域=yahoo.com，设备=缺失，card1=7585, addr1=264。实体历史：prior 206 笔、成熟欺诈率 5%；设备 fan-out 66。", "window": [10788777, 10788777], "label_based": true}
- `STAT_000`: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "card6=credit", "value": 0.0665, "window": [0, 11121568], "support_n": 114336, "label_based": true}
- `STAT_001`: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "P_emaildomain=yahoo.com", "value": 0.0232, "window": [0, 11121568], "support_n": 74983, "label_based": true}
- `STAT_002`: {"fact_id": "STAT_002", "type": "fraud_rate", "entity": "ProductCD=W", "value": 0.0205, "window": [0, 11121568], "support_n": 318769, "label_based": true}

</details>

## 3481840#4（断言强度=tentative）
finding：同卡的确认欺诈案例(#3168182/#3154125)与本笔行为簇不同：它们在 addr1=299、ProductCD=R/H、金额 $75-150，而本笔 addr1=204、ProductCD=W、$445，不落入同一欺诈模式。
该 finding 引用的证据：
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3168182", "value": "案例 #3168182（day 45）：确认欺诈（拒付举报）。金额 $150.00（$100-300），ProductCD=R，visa/credit，邮箱域=gmail.com，设备=MacOS，card1=16326, addr1=299。实体历史：prior 62 笔、成熟欺诈率 2%；设备 fan-out 17。", "window": [3982912, 3982912], "label_based": true}
  - CASE_001: {"fact_id": "CASE_001", "type": "case", "entity": "case=3154125", "value": "案例 #3154125（day 40）：确认欺诈（拒付举报）。金额 $75.00（$50-100），ProductCD=H，visa/credit，邮箱域=hotmail.com，设备=Windows，card1=16326, addr1=299。实体历史：prior 61 笔、成熟欺诈率 3%；设备 fan-out 16。", "window": [3574643, 3574643], "label_based": true}
  - TXN_000: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3481840", "value": 445.0, "window": [12935968, 12935968], "label_based": false}
  - TXN_001: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3481840", "value": "W", "window": [12935968, 12935968], "label_based": false}
  - TXN_005: {"fact_id": "TXN_005", "type": "txn_field:addr1", "entity": "txn=3481840", "value": 204.0, "window": [12935968, 12935968], "label_based": false}
reasoning_valid: true
引用证据直接提供了本笔与案例的对比字段值（addr1/ProductCD/Amt），方向一致且相关，未命中反证/无关/跳跃三类 N
overclaim: false
预设 tentative，而“模式不同”由多字段对比推导，最高可达 supported，标注 tentative 属保守偏置，未越界。

# 交易 3484095（GBDT p=0.7664）
<details><summary>本单【全部证据池】——核实被断言的值是否在整单里有据；「有据但该 finding 没引」属引用瑕疵，按 v2 不判 N</summary>

- `TXN_000`: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3484095", "value": 226.0, "window": [13024906, 13024906], "label_based": false}
- `TXN_001`: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3484095", "value": "W", "window": [13024906, 13024906], "label_based": false}
- `TXN_002`: {"fact_id": "TXN_002", "type": "txn_field:card1", "entity": "txn=3484095", "value": 13526, "window": [13024906, 13024906], "label_based": false}
- `TXN_003`: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3484095", "value": "visa", "window": [13024906, 13024906], "label_based": false}
- `TXN_004`: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3484095", "value": "credit", "window": [13024906, 13024906], "label_based": false}
- `TXN_005`: {"fact_id": "TXN_005", "type": "txn_field:addr1", "entity": "txn=3484095", "value": 420.0, "window": [13024906, 13024906], "label_based": false}
- `TXN_006`: {"fact_id": "TXN_006", "type": "txn_field:P_emaildomain", "entity": "txn=3484095", "value": "gmail.com", "window": [13024906, 13024906], "label_based": false}
- `GRAPH_000`: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=13526", "value": 292, "window": [0, 13024906], "label_based": false}
- `GRAPH_001`: {"fact_id": "GRAPH_001", "type": "prior_cnt", "entity": "card1=13526|addr1", "value": 13, "window": [0, 13024906], "label_based": false}
- `GRAPH_002`: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=13526", "value": 33, "window": [0, 13024906], "label_based": false}
- `GRAPH_003`: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=13526", "value": 17, "window": [0, 13024906], "label_based": false}
- `GRAPH_004`: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=13526", "value": 7, "window": [0, 13024906], "label_based": false}
- `GRAPH_005`: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=13526", "value": 0.2191, "window": [0, 11210506], "label_based": true}
- `GRAPH_006`: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|addr1", "value": 0.1, "window": [0, 11210506], "label_based": true}
- `GRAPH_007`: {"fact_id": "GRAPH_007", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.3976, "window": [0, 11210506], "label_based": true}
- `GRAPH_008`: {"fact_id": "GRAPH_008", "type": "gang_score", "entity": "card1=13526", "value": 0.7, "window": [0, 11210506], "label_based": true}
- `RULE_000`: {"fact_id": "RULE_000", "type": "rule", "entity": "rule=R_PRIOR_FRAUD_CARD1_EMAIL", "value": "实体欺诈史：card1+邮箱（card1_email_prior_fraud_rate>=0.1）：训练窗 [0,125)天 内该模式欺诈率 19.11%，为基率 3.51% 的 5.4 倍（触发 16,801 笔，其中欺诈 3,210）。", "window": [86400, 10886394], "support_n": 16801, "label_based": true}
- `RULE_001`: {"fact_id": "RULE_001", "type": "rule", "entity": "rule=R_PRIOR_FRAUD_CARD1", "value": "实体欺诈史：card1（card1_prior_fraud_rate>=0.1）：训练窗 [0,125)天 内该模式欺诈率 17.32%，为基率 3.51% 的 4.9 倍（触发 21,568 笔，其中欺诈 3,735）。", "window": [86400, 10886394], "support_n": 21568, "label_based": true}
- `RULE_002`: {"fact_id": "RULE_002", "type": "rule", "entity": "rule=R_PRIOR_FRAUD_CARD1_ADDR1", "value": "实体欺诈史：card1+addr1（card1_addr1_prior_fraud_rate>=0.1）：训练窗 [0,125)天 内该模式欺诈率 13.49%，为基率 3.51% 的 3.8 倍（触发 9,065 笔，其中欺诈 1,223）。", "window": [86400, 10886394], "support_n": 9065, "label_based": true}
- `RULE_003`: {"fact_id": "RULE_003", "type": "rule", "entity": "rule=R_card6_credit", "value": "高危取值 card6=credit（card6=credit）：训练窗 [0,125)天 内该模式欺诈率 6.65%，为基率 3.51% 的 1.9 倍（触发 112,717 笔，其中欺诈 7,501）。", "window": [86400, 10886394], "support_n": 112717, "label_based": true}
- `CASE_000`: {"fact_id": "CASE_000", "type": "case", "entity": "case=3371213", "value": "案例 #3371213（day 110）：确认欺诈（拒付举报）。金额 $490.00（$300-1k），ProductCD=W，visa/credit，邮箱域=yahoo.com，设备=缺失，card1=13526, addr1=226。实体历史：prior 12 笔、成熟欺诈率 0%；设备 fan-out 7。", "window": [9642194, 9642194], "label_based": true}
- `CASE_001`: {"fact_id": "CASE_001", "type": "case", "entity": "case=3345699", "value": "案例 #3345699（day 101）：确认欺诈（拒付举报）。金额 $300.00（$300-1k），ProductCD=R，visa/credit，邮箱域=gmail.com，设备=MacOS，card1=13526, addr1=181。实体历史：prior 20 笔、成熟欺诈率 0%；设备 fan-out 7。", "window": [8888922, 8888922], "label_based": true}
- `CASE_002`: {"fact_id": "CASE_002", "type": "case", "entity": "case=3399390", "value": "案例 #3399390（day 119）：高分假阳（人工洗清，当时模型分 0.14）。金额 $280.00（$100-300），ProductCD=W，visa/credit，邮箱域=gmail.com，设备=缺失，card1=17947, addr1=310。实体历史：prior 9 笔、成熟欺诈率 33%；设备 fan-out 7。", "window": [10425838, 10425838], "label_based": true}
- `CASE_003`: {"fact_id": "CASE_003", "type": "case", "entity": "case=3394367", "value": "案例 #3394367（day 118）：高分假阳（人工洗清，当时模型分 0.28）。金额 $280.00（$100-300），ProductCD=W，visa/credit，邮箱域=gmail.com，设备=缺失，card1=17947, addr1=310。实体历史：prior 8 笔、成熟欺诈率 33%；设备 fan-out 7。", "window": [10282728, 10282728], "label_based": true}
- `STAT_000`: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "P_emaildomain=gmail.com", "value": 0.0439, "window": [0, 11210506], "support_n": 170490, "label_based": true}
- `STAT_001`: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "ProductCD=W", "value": 0.0205, "window": [0, 11210506], "support_n": 320783, "label_based": true}
- `STAT_002`: {"fact_id": "STAT_002", "type": "fraud_rate", "entity": "card4=visa", "value": 0.0347, "window": [0, 11210506], "support_n": 287805, "label_based": true}

</details>

## 3484095#5（断言强度=supported）
finding：交易本身通用属性接近基率、不构成风险源：gmail 4.39%、ProductCD=W 2.05%、visa 3.47%，均在基率3.5%附近。GBDT 分0.7664高分主要由实体欺诈史解释，我的独立证据与该分数方向一致（无冲突）。
该 finding 引用的证据：
  - STAT_000: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "P_emaildomain=gmail.com", "value": 0.0439, "window": [0, 11210506], "support_n": 170490, "label_based": true}
  - STAT_001: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "ProductCD=W", "value": 0.0205, "window": [0, 11210506], "support_n": 320783, "label_based": true}
  - STAT_002: {"fact_id": "STAT_002", "type": "fraud_rate", "entity": "card4=visa", "value": 0.0347, "window": [0, 11210506], "support_n": 287805, "label_based": true}
reasoning_valid: true
主张分两层：通用属性低值有据（STAT_000~002）；“实体史解释高分”缺引的实体史值在池中有据（GRAPH_005=0.2191），属引用瑕疵，未命中三类 N。
overclaim: false
预设 supported，“实体史解释高分”为归因判断，需多条事实推理，可达 supported 档，未超过；标注保守，不判越界

# 交易 3490572（GBDT p=0.0561）
<details><summary>本单【全部证据池】——核实被断言的值是否在整单里有据；「有据但该 finding 没引」属引用瑕疵，按 v2 不判 N</summary>

- `TXN_000`: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3490572", "value": 300.0, "window": [13195950, 13195950], "label_based": false}
- `TXN_001`: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3490572", "value": "H", "window": [13195950, 13195950], "label_based": false}
- `TXN_002`: {"fact_id": "TXN_002", "type": "txn_field:card1", "entity": "txn=3490572", "value": 15066, "window": [13195950, 13195950], "label_based": false}
- `TXN_003`: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3490572", "value": "mastercard", "window": [13195950, 13195950], "label_based": false}
- `TXN_004`: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3490572", "value": "credit", "window": [13195950, 13195950], "label_based": false}
- `TXN_005`: {"fact_id": "TXN_005", "type": "txn_field:addr1", "entity": "txn=3490572", "value": 226.0, "window": [13195950, 13195950], "label_based": false}
- `TXN_006`: {"fact_id": "TXN_006", "type": "txn_field:P_emaildomain", "entity": "txn=3490572", "value": "yahoo.com", "window": [13195950, 13195950], "label_based": false}
- `TXN_007`: {"fact_id": "TXN_007", "type": "txn_field:DeviceType", "entity": "txn=3490572", "value": "desktop", "window": [13195950, 13195950], "label_based": false}
- `TXN_008`: {"fact_id": "TXN_008", "type": "txn_field:DeviceInfo", "entity": "txn=3490572", "value": "MacOS", "window": [13195950, 13195950], "label_based": false}
- `GRAPH_000`: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=15066", "value": 6699, "window": [0, 13195950], "label_based": false}
- `GRAPH_001`: {"fact_id": "GRAPH_001", "type": "prior_cnt", "entity": "card1=15066|addr1", "value": 51, "window": [0, 13195950], "label_based": false}
- `GRAPH_002`: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=15066", "value": 61, "window": [0, 13195950], "label_based": false}
- `GRAPH_003`: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=15066", "value": 39, "window": [0, 13195950], "label_based": false}
- `GRAPH_004`: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=15066", "value": 80, "window": [0, 13195950], "label_based": false}
- `GRAPH_005`: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=15066", "value": 0.0404, "window": [0, 11381550], "label_based": true}
- `GRAPH_006`: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|addr1", "value": 0.0, "window": [0, 11381550], "label_based": true}
- `GRAPH_007`: {"fact_id": "GRAPH_007", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.0108, "window": [0, 11381550], "label_based": true}
- `GRAPH_008`: {"fact_id": "GRAPH_008", "type": "prior_fraud_rate", "entity": "card1|device", "value": 0.019, "window": [0, 11381550], "label_based": true}
- `GRAPH_009`: {"fact_id": "GRAPH_009", "type": "gang_score", "entity": "card1=15066", "value": 0.404, "window": [0, 11381550], "label_based": true}
- `RULE_000`: {"fact_id": "RULE_000", "type": "rule", "entity": "rule=R_HAS_IDENTITY", "value": "有设备/身份采集记录（id_01 存在）：训练窗 [0,125)天 内该模式欺诈率 7.44%，为基率 3.51% 的 2.1 倍（触发 112,859 笔，其中欺诈 8,397）。", "window": [86400, 10886394], "support_n": 112859, "label_based": true}
- `RULE_001`: {"fact_id": "RULE_001", "type": "rule", "entity": "rule=R_card6_credit", "value": "高危取值 card6=credit（card6=credit）：训练窗 [0,125)天 内该模式欺诈率 6.65%，为基率 3.51% 的 1.9 倍（触发 112,717 笔，其中欺诈 7,501）。", "window": [86400, 10886394], "support_n": 112717, "label_based": true}
- `RULE_002`: {"fact_id": "RULE_002", "type": "rule", "entity": "rule=R_DeviceType_desktop", "value": "高危取值 DeviceType=desktop（DeviceType=desktop）：训练窗 [0,125)天 内该模式欺诈率 6.09%，为基率 3.51% 的 1.7 倍（触发 67,488 笔，其中欺诈 4,113）。", "window": [86400, 10886394], "support_n": 67488, "label_based": true}
- `CASE_000`: {"fact_id": "CASE_000", "type": "case", "entity": "case=3377102", "value": "案例 #3377102（day 112）：高分假阳（人工洗清，当时模型分 0.18）。金额 $100.00（$100-300），ProductCD=H，mastercard/credit，邮箱域=gmail.com，设备=MacOS，card1=15066, addr1=325。实体历史：prior 576 笔、成熟欺诈率 4%；设备 fan-out 72。", "window": [9778932, 9778932], "label_based": true}
- `CASE_001`: {"fact_id": "CASE_001", "type": "case", "entity": "case=3244210", "value": "案例 #3244210（day 70）：确认欺诈（拒付举报）。金额 $450.00（$300-1k），ProductCD=H，mastercard/credit，邮箱域=gmail.com，设备=iOS Device，card1=15066, addr1=330。实体历史：prior 197 笔、成熟欺诈率 12%；设备 fan-out 63。", "window": [6170475, 6170475], "label_based": true}
- `CASE_002`: {"fact_id": "CASE_002", "type": "case", "entity": "case=3384178", "value": "案例 #3384178（day 114）：高分假阳（人工洗清，当时模型分 0.17）。金额 $317.50（$300-1k），ProductCD=W，mastercard/credit，邮箱域=yahoo.com，设备=缺失，card1=15066, addr1=264。实体历史：prior 375 笔、成熟欺诈率 2%；设备 fan-out 74。", "window": [10006166, 10006166], "label_based": true}
- `CASE_003`: {"fact_id": "CASE_003", "type": "case", "entity": "case=3380213", "value": "案例 #3380213（day 113）：高分假阳（人工洗清，当时模型分 0.20）。金额 $450.00（$300-1k），ProductCD=W，mastercard/credit，邮箱域=yahoo.com，设备=缺失，card1=15066, addr1=264。实体历史：prior 372 笔、成熟欺诈率 2%；设备 fan-out 73。", "window": [9859342, 9859342], "label_based": true}
- `STAT_000`: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=H", "value": 0.0462, "window": [0, 11381550], "support_n": 29001, "label_based": true}
- `STAT_001`: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "P_emaildomain=yahoo.com", "value": 0.023, "window": [0, 11381550], "support_n": 76344, "label_based": true}

</details>

## 3490572#1（断言强度=supported）
finding：card1=15066 历史成熟欺诈率 4.04%，仅略高于全体基率 3.5%，未见异常抬升；实体交易量大（6699 笔）、扇出相对交易量属中等（addr1 61 / email 39 / device 80）。
该 finding 引用的证据：
  - GRAPH_000: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=15066", "value": 6699, "window": [0, 13195950], "label_based": false}
  - GRAPH_002: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=15066", "value": 61, "window": [0, 13195950], "label_based": false}
  - GRAPH_003: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=15066", "value": 39, "window": [0, 13195950], "label_based": false}
  - GRAPH_004: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=15066", "value": 80, "window": [0, 13195950], "label_based": false}
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=15066", "value": 0.0404, "window": [0, 11381550], "label_based": true}
reasoning_valid: true
主张为实体特征描述与程度判断，核心数值（欺诈率/交易量/扇出）均有据，程度形容词（略高于/中等）属合理评价，未命中反证/无关/跳跃三类 N
overclaim: false
预设 supported，实体特征值可达 confirmed，但“未见异常”与“中等”为多事实对比/程度判断，标 supported 未超过可达最高档，属保守偏置

# 交易 3508739（GBDT p=0.5262）
<details><summary>本单【全部证据池】——核实被断言的值是否在整单里有据；「有据但该 finding 没引」属引用瑕疵，按 v2 不判 N</summary>

- `TXN_000`: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3508739", "value": 28.035, "window": [13711788, 13711788], "label_based": false}
- `TXN_001`: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3508739", "value": "C", "window": [13711788, 13711788], "label_based": false}
- `TXN_002`: {"fact_id": "TXN_002", "type": "txn_field:card1", "entity": "txn=3508739", "value": 9500, "window": [13711788, 13711788], "label_based": false}
- `TXN_003`: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3508739", "value": "visa", "window": [13711788, 13711788], "label_based": false}
- `TXN_004`: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3508739", "value": "debit", "window": [13711788, 13711788], "label_based": false}
- `TXN_005`: {"fact_id": "TXN_005", "type": "txn_field:P_emaildomain", "entity": "txn=3508739", "value": "gmail.com", "window": [13711788, 13711788], "label_based": false}
- `TXN_006`: {"fact_id": "TXN_006", "type": "txn_field:DeviceType", "entity": "txn=3508739", "value": "desktop", "window": [13711788, 13711788], "label_based": false}
- `TXN_007`: {"fact_id": "TXN_007", "type": "txn_field:DeviceInfo", "entity": "txn=3508739", "value": "Windows", "window": [13711788, 13711788], "label_based": false}
- `GRAPH_000`: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=9500", "value": 12392, "window": [0, 13711788], "label_based": false}
- `GRAPH_001`: {"fact_id": "GRAPH_001", "type": "prior_cnt", "entity": "card1=9500|addr1", "value": 0, "window": [0, 13711788], "label_based": false}
- `GRAPH_002`: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=9500", "value": 53, "window": [0, 13711788], "label_based": false}
- `GRAPH_003`: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=9500", "value": 40, "window": [0, 13711788], "label_based": false}
- `GRAPH_004`: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=9500", "value": 65, "window": [0, 13711788], "label_based": false}
- `GRAPH_005`: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=9500", "value": 0.0341, "window": [0, 11897388], "label_based": true}
- `GRAPH_006`: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.0311, "window": [0, 11897388], "label_based": true}
- `GRAPH_007`: {"fact_id": "GRAPH_007", "type": "prior_fraud_rate", "entity": "card1|device", "value": 0.0419, "window": [0, 11897388], "label_based": true}
- `GRAPH_008`: {"fact_id": "GRAPH_008", "type": "gang_score", "entity": "card1=9500", "value": 0.341, "window": [0, 11897388], "label_based": true}
- `STAT_000`: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=C", "value": 0.1132, "window": [0, 11897388], "support_n": 55353, "label_based": true}
- `STAT_001`: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "P_emaildomain=gmail.com", "value": 0.0441, "window": [0, 11897388], "support_n": 178681, "label_based": true}
- `RULE_000`: {"fact_id": "RULE_000", "type": "rule", "entity": "rule=R_ProductCD_C", "value": "高危取值 ProductCD=C（ProductCD=C）：训练窗 [0,125)天 内该模式欺诈率 11.27%，为基率 3.51% 的 3.2 倍（触发 51,857 笔，其中欺诈 5,844）。", "window": [86400, 10886394], "support_n": 51857, "label_based": true}
- `RULE_001`: {"fact_id": "RULE_001", "type": "rule", "entity": "rule=R_NEW_ENTITY", "value": "新实体首现（card1+addr1 无历史）（card1_addr1_prior_cnt=0）：训练窗 [0,125)天 内该模式欺诈率 7.78%，为基率 3.51% 的 2.2 倍（触发 82,512 笔，其中欺诈 6,417）。", "window": [86400, 10886394], "support_n": 82512, "label_based": true}
- `RULE_002`: {"fact_id": "RULE_002", "type": "rule", "entity": "rule=R_HAS_IDENTITY", "value": "有设备/身份采集记录（id_01 存在）：训练窗 [0,125)天 内该模式欺诈率 7.44%，为基率 3.51% 的 2.1 倍（触发 112,859 笔，其中欺诈 8,397）。", "window": [86400, 10886394], "support_n": 112859, "label_based": true}
- `RULE_003`: {"fact_id": "RULE_003", "type": "rule", "entity": "rule=R_DeviceType_desktop", "value": "高危取值 DeviceType=desktop（DeviceType=desktop）：训练窗 [0,125)天 内该模式欺诈率 6.09%，为基率 3.51% 的 1.7 倍（触发 67,488 笔，其中欺诈 4,113）。", "window": [86400, 10886394], "support_n": 67488, "label_based": true}
- `CASE_000`: {"fact_id": "CASE_000", "type": "case", "entity": "case=3429815", "value": "案例 #3429815（day 129）：确认欺诈（拒付举报）。金额 $58.41（$50-100），ProductCD=C，visa/debit，邮箱域=gmail.com，设备=GT-I9060M Build/KTU84P，card1=9500。实体历史：prior 0 笔、无成熟标签；设备 fan-out 62。", "window": [11242395, 11242395], "label_based": true}
- `CASE_001`: {"fact_id": "CASE_001", "type": "case", "entity": "case=3379531", "value": "案例 #3379531（day 112）：高分假阳（人工洗清，当时模型分 0.16）。金额 $11.00（$10-50），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=9500, addr1=204。实体历史：prior 2970 笔、成熟欺诈率 3%；设备 fan-out 56。", "window": [9846709, 9846709], "label_based": true}
- `CASE_002`: {"fact_id": "CASE_002", "type": "case", "entity": "case=3407175", "value": "案例 #3407175（day 122）：高分假阳（人工洗清，当时模型分 0.20）。金额 $20.00（$10-50），ProductCD=S，visa/debit，邮箱域=缺失，设备=Windows，card1=9500, addr1=204。实体历史：prior 3173 笔、成熟欺诈率 2%；设备 fan-out 60。", "window": [10629442, 10629442], "label_based": true}
- `CASE_003`: {"fact_id": "CASE_003", "type": "case", "entity": "case=3386048", "value": "案例 #3386048（day 115）：高分假阳（人工洗清，当时模型分 0.37）。金额 $25.00（$10-50），ProductCD=H，visa/debit，邮箱域=hotmail.com，设备=Windows，card1=9500, addr1=204。实体历史：prior 3024 笔、成熟欺诈率 2%；设备 fan-out 57。", "window": [10074504, 10074504], "label_based": true}

</details>

## 3508739#5（断言强度=confirmed）
finding：Transaction amount is very small ($28.04), limiting worst-case loss if it is fraud.
该 finding 引用的证据：
  - TXN_000: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3508739", "value": 28.035, "window": [13711788, 13711788], "label_based": false}
reasoning_valid: true
金额值有据（TXN_000），金额小则损失有限为常识推论，未命中反证/无关/行为类别跳跃三类 N
overclaim: true
预设 confirmed，但主张包含“损失有限”的因果推论，非纯字段本义引用，最高仅达 supported，超档越界

# 交易 3510638（GBDT p=0.1144）
<details><summary>本单【全部证据池】——核实被断言的值是否在整单里有据；「有据但该 finding 没引」属引用瑕疵，按 v2 不判 N</summary>

- `TXN_000`: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3510638", "value": 82.0, "window": [13745692, 13745692], "label_based": false}
- `TXN_001`: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3510638", "value": "W", "window": [13745692, 13745692], "label_based": false}
- `TXN_002`: {"fact_id": "TXN_002", "type": "txn_field:card1", "entity": "txn=3510638", "value": 14426, "window": [13745692, 13745692], "label_based": false}
- `TXN_003`: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3510638", "value": "mastercard", "window": [13745692, 13745692], "label_based": false}
- `TXN_004`: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3510638", "value": "debit", "window": [13745692, 13745692], "label_based": false}
- `TXN_005`: {"fact_id": "TXN_005", "type": "txn_field:addr1", "entity": "txn=3510638", "value": 272.0, "window": [13745692, 13745692], "label_based": false}
- `TXN_006`: {"fact_id": "TXN_006", "type": "txn_field:P_emaildomain", "entity": "txn=3510638", "value": "gmail.com", "window": [13745692, 13745692], "label_based": false}
- `GRAPH_000`: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=14426", "value": 729, "window": [0, 13745692], "label_based": false}
- `GRAPH_001`: {"fact_id": "GRAPH_001", "type": "prior_cnt", "entity": "card1=14426|addr1", "value": 666, "window": [0, 13745692], "label_based": false}
- `GRAPH_002`: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=14426", "value": 17, "window": [0, 13745692], "label_based": false}
- `GRAPH_003`: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=14426", "value": 12, "window": [0, 13745692], "label_based": false}
- `GRAPH_004`: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=14426", "value": 7, "window": [0, 13745692], "label_based": false}
- `GRAPH_005`: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=14426", "value": 0.0129, "window": [0, 11931292], "label_based": true}
- `GRAPH_006`: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|addr1", "value": 0.0141, "window": [0, 11931292], "label_based": true}
- `GRAPH_007`: {"fact_id": "GRAPH_007", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.0246, "window": [0, 11931292], "label_based": true}
- `GRAPH_008`: {"fact_id": "GRAPH_008", "type": "gang_score", "entity": "card1=14426", "value": 0.09, "window": [0, 11931292], "label_based": true}
- `CASE_000`: {"fact_id": "CASE_000", "type": "case", "entity": "case=3450124", "value": "案例 #3450124（day 136）：确认欺诈（拒付举报）。金额 $77.00（$50-100），ProductCD=W，mastercard/debit，邮箱域=gmail.com，设备=缺失，card1=14098, addr1=325。实体历史：prior 34 笔、成熟欺诈率 4%；设备 fan-out 3。", "window": [11912322, 11912322], "label_based": true}
- `CASE_001`: {"fact_id": "CASE_001", "type": "case", "entity": "case=3448336", "value": "案例 #3448336（day 136）：确认欺诈（拒付举报）。金额 $53.00（$50-100），ProductCD=W，mastercard/debit，邮箱域=gmail.com，设备=缺失，card1=12469, addr1=269。实体历史：prior 2 笔、成熟欺诈率 0%；设备 fan-out 1。", "window": [11847541, 11847541], "label_based": true}
- `CASE_002`: {"fact_id": "CASE_002", "type": "case", "entity": "case=3442090", "value": "案例 #3442090（day 133）：高分假阳（人工洗清，当时模型分 0.32）。金额 $68.50（$50-100），ProductCD=W，mastercard/debit，邮箱域=gmail.com，设备=缺失，card1=5376, addr1=441。实体历史：prior 8 笔、成熟欺诈率 0%；设备 fan-out 4。", "window": [11642390, 11642390], "label_based": true}
- `CASE_003`: {"fact_id": "CASE_003", "type": "case", "entity": "case=3441599", "value": "案例 #3441599（day 133）：高分假阳（人工洗清，当时模型分 0.59）。金额 $77.00（$50-100），ProductCD=W，mastercard/debit，邮箱域=gmail.com，设备=缺失，card1=5376, addr1=441。实体历史：prior 7 笔、成熟欺诈率 0%；设备 fan-out 4。", "window": [11633446, 11633446], "label_based": true}
- `STAT_000`: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=W", "value": 0.0207, "window": [0, 11931292], "support_n": 339398, "label_based": true}
- `STAT_001`: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "P_emaildomain=gmail.com", "value": 0.044, "window": [0, 11931292], "support_n": 179392, "label_based": true}
- `STAT_002`: {"fact_id": "STAT_002", "type": "fraud_rate", "entity": "card4=mastercard", "value": 0.035, "window": [0, 11931292], "support_n": 148187, "label_based": true}

</details>

## 3510638#0（断言强度=supported）
finding：小额交易 $82，ProductCD=W（该品类历史欺诈率2.07%，低于全体基率3.5%），mastercard/debit（3.5%，基线水平），拦截误伤成本高于潜在损失。
该 finding 引用的证据：
  - TXN_000: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3510638", "value": 82.0, "window": [13745692, 13745692], "label_based": false}
  - TXN_001: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3510638", "value": "W", "window": [13745692, 13745692], "label_based": false}
  - TXN_003: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3510638", "value": "mastercard", "window": [13745692, 13745692], "label_based": false}
  - TXN_004: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3510638", "value": "debit", "window": [13745692, 13745692], "label_based": false}
  - STAT_000: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=W", "value": 0.0207, "window": [0, 11931292], "support_n": 339398, "label_based": true}
  - STAT_002: {"fact_id": "STAT_002", "type": "fraud_rate", "entity": "card4=mastercard", "value": 0.035, "window": [0, 11931292], "support_n": 148187, "label_based": true}
reasoning_valid: false
引用 STAT_002（card4=mastercard）不足以推出 mastercard/debit 组合基线；“误伤成本>损失”需成本数据支撑，证据池缺失该类证据，属非因果跳跃。但之前在ml层确实有过类似的结论，因此我怀疑是不是需要结合更大背景比如整个项目的。
overclaim: true
预设 supported，“成本收益权衡”断言需成本证据，证据池无，实际最高仅 tentative，超档越界

# 交易 3511656（GBDT p=0.9768）
<details><summary>本单【全部证据池】——核实被断言的值是否在整单里有据；「有据但该 finding 没引」属引用瑕疵，按 v2 不判 N</summary>

- `TXN_000`: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3511656", "value": 53.856, "window": [13796620, 13796620], "label_based": false}
- `TXN_001`: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3511656", "value": "C", "window": [13796620, 13796620], "label_based": false}
- `TXN_002`: {"fact_id": "TXN_002", "type": "txn_field:card1", "entity": "txn=3511656", "value": 5458, "window": [13796620, 13796620], "label_based": false}
- `TXN_003`: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3511656", "value": "mastercard", "window": [13796620, 13796620], "label_based": false}
- `TXN_004`: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3511656", "value": "credit", "window": [13796620, 13796620], "label_based": false}
- `TXN_005`: {"fact_id": "TXN_005", "type": "txn_field:P_emaildomain", "entity": "txn=3511656", "value": "gmail.com", "window": [13796620, 13796620], "label_based": false}
- `TXN_006`: {"fact_id": "TXN_006", "type": "txn_field:DeviceType", "entity": "txn=3511656", "value": "mobile", "window": [13796620, 13796620], "label_based": false}
- `GRAPH_000`: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=5458", "value": 149, "window": [0, 13796620], "label_based": false}
- `GRAPH_001`: {"fact_id": "GRAPH_001", "type": "prior_cnt", "entity": "card1=5458|addr1", "value": 0, "window": [0, 13796620], "label_based": false}
- `GRAPH_002`: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=5458", "value": 6, "window": [0, 13796620], "label_based": false}
- `GRAPH_003`: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=5458", "value": 10, "window": [0, 13796620], "label_based": false}
- `GRAPH_004`: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=5458", "value": 12, "window": [0, 13796620], "label_based": false}
- `GRAPH_005`: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=5458", "value": 0.0803, "window": [0, 11982220], "label_based": true}
- `GRAPH_006`: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.16, "window": [0, 11982220], "label_based": true}
- `GRAPH_007`: {"fact_id": "GRAPH_007", "type": "gang_score", "entity": "card1=5458", "value": 0.803, "window": [0, 11982220], "label_based": true}
- `RULE_000`: {"fact_id": "RULE_000", "type": "rule", "entity": "rule=R_PRIOR_FRAUD_CARD1_EMAIL", "value": "实体欺诈史：card1+邮箱（card1_email_prior_fraud_rate>=0.1）：训练窗 [0,125)天 内该模式欺诈率 19.11%，为基率 3.51% 的 5.4 倍（触发 16,801 笔，其中欺诈 3,210）。", "window": [86400, 10886394], "support_n": 16801, "label_based": true}
- `RULE_001`: {"fact_id": "RULE_001", "type": "rule", "entity": "rule=R_ProductCD_C", "value": "高危取值 ProductCD=C（ProductCD=C）：训练窗 [0,125)天 内该模式欺诈率 11.27%，为基率 3.51% 的 3.2 倍（触发 51,857 笔，其中欺诈 5,844）。", "window": [86400, 10886394], "support_n": 51857, "label_based": true}
- `RULE_002`: {"fact_id": "RULE_002", "type": "rule", "entity": "rule=R_DeviceType_mobile", "value": "高危取值 DeviceType=mobile（DeviceType=mobile）：训练窗 [0,125)天 内该模式欺诈率 9.88%，为基率 3.51% 的 2.8 倍（触发 42,771 笔，其中欺诈 4,226）。", "window": [86400, 10886394], "support_n": 42771, "label_based": true}
- `RULE_003`: {"fact_id": "RULE_003", "type": "rule", "entity": "rule=R_NEW_ENTITY", "value": "新实体首现（card1+addr1 无历史）（card1_addr1_prior_cnt=0）：训练窗 [0,125)天 内该模式欺诈率 7.78%，为基率 3.51% 的 2.2 倍（触发 82,512 笔，其中欺诈 6,417）。", "window": [86400, 10886394], "support_n": 82512, "label_based": true}
- `RULE_004`: {"fact_id": "RULE_004", "type": "rule", "entity": "rule=R_HAS_IDENTITY", "value": "有设备/身份采集记录（id_01 存在）：训练窗 [0,125)天 内该模式欺诈率 7.44%，为基率 3.51% 的 2.1 倍（触发 112,859 笔，其中欺诈 8,397）。", "window": [86400, 10886394], "support_n": 112859, "label_based": true}
- `RULE_005`: {"fact_id": "RULE_005", "type": "rule", "entity": "rule=R_card6_credit", "value": "高危取值 card6=credit（card6=credit）：训练窗 [0,125)天 内该模式欺诈率 6.65%，为基率 3.51% 的 1.9 倍（触发 112,717 笔，其中欺诈 7,501）。", "window": [86400, 10886394], "support_n": 112717, "label_based": true}
- `CASE_000`: {"fact_id": "CASE_000", "type": "case", "entity": "case=3390348", "value": "案例 #3390348（day 116）：确认欺诈（拒付举报）。金额 $67.07（$50-100），ProductCD=C，mastercard/credit，邮箱域=gmail.com，设备=缺失，card1=5458。实体历史：prior 0 笔、无成熟标签；设备 fan-out 8。", "window": [10181414, 10181414], "label_based": true}
- `CASE_001`: {"fact_id": "CASE_001", "type": "case", "entity": "case=3415043", "value": "案例 #3415043（day 124）：高分假阳（人工洗清，当时模型分 0.43）。金额 $18.38（$10-50），ProductCD=C，mastercard/credit，邮箱域=gmail.com，设备=Moto G (5) Plus Build/NPN25.137-82，card1=5458。实体历史：prior 0 笔、无成熟标签；设备 fan-out 8。", "window": [10807118, 10807118], "label_based": true}
- `CASE_002`: {"fact_id": "CASE_002", "type": "case", "entity": "case=3393012", "value": "案例 #3393012（day 117）：高分假阳（人工洗清，当时模型分 0.52）。金额 $26.59（$10-50），ProductCD=C，mastercard/credit，邮箱域=gmail.com，设备=Windows，card1=5458。实体历史：prior 0 笔、无成熟标签；设备 fan-out 8。", "window": [10259255, 10259255], "label_based": true}
- `CASE_003`: {"fact_id": "CASE_003", "type": "case", "entity": "case=3387104", "value": "案例 #3387104（day 115）：高分假阳（人工洗清，当时模型分 0.36）。金额 $11.78（$10-50），ProductCD=C，mastercard/credit，邮箱域=gmail.com，设备=Windows，card1=5458。实体历史：prior 0 笔、无成熟标签；设备 fan-out 8。", "window": [10093298, 10093298], "label_based": true}
- `STAT_000`: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=C", "value": 0.113, "window": [0, 11982220], "support_n": 55630, "label_based": true}
- `STAT_001`: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "P_emaildomain=gmail.com", "value": 0.044, "window": [0, 11982220], "support_n": 179832, "label_based": true}
- `STAT_002`: {"fact_id": "STAT_002", "type": "fraud_rate", "entity": "card4=mastercard", "value": 0.035, "window": [0, 11982220], "support_n": 148527, "label_based": true}

</details>

## 3511656#2（断言强度=confirmed）
finding：本笔命中高危规则 R_PRIOR_FRAUD_CARD1_EMAIL（card1+邮箱历史欺诈率 19.11%，5.4x lift，支持样本 16801），并叠加 ProductCD=C（11.3%，STAT 佐证）、DeviceType=mobile、card6=credit 等多条正 lift 规则。
该 finding 引用的证据：
  - RULE_000: {"fact_id": "RULE_000", "type": "rule", "entity": "rule=R_PRIOR_FRAUD_CARD1_EMAIL", "value": "实体欺诈史：card1+邮箱（card1_email_prior_fraud_rate>=0.1）：训练窗 [0,125)天 内该模式欺诈率 19.11%，为基率 3.51% 的 5.4 倍（触发 16,801 笔，其中欺诈 3,210）。", "window": [86400, 10886394], "support_n": 16801, "label_based": true}
  - RULE_001: {"fact_id": "RULE_001", "type": "rule", "entity": "rule=R_ProductCD_C", "value": "高危取值 ProductCD=C（ProductCD=C）：训练窗 [0,125)天 内该模式欺诈率 11.27%，为基率 3.51% 的 3.2 倍（触发 51,857 笔，其中欺诈 5,844）。", "window": [86400, 10886394], "support_n": 51857, "label_based": true}
  - STAT_000: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=C", "value": 0.113, "window": [0, 11982220], "support_n": 55630, "label_based": true}
  - RULE_002: {"fact_id": "RULE_002", "type": "rule", "entity": "rule=R_DeviceType_mobile", "value": "高危取值 DeviceType=mobile（DeviceType=mobile）：训练窗 [0,125)天 内该模式欺诈率 9.88%，为基率 3.51% 的 2.8 倍（触发 42,771 笔，其中欺诈 4,226）。", "window": [86400, 10886394], "support_n": 42771, "label_based": true}
  - RULE_005: {"fact_id": "RULE_005", "type": "rule", "entity": "rule=R_card6_credit", "value": "高危取值 card6=credit（card6=credit）：训练窗 [0,125)天 内该模式欺诈率 6.65%，为基率 3.51% 的 1.9 倍（触发 112,717 笔，其中欺诈 7,501）。", "window": [86400, 10886394], "support_n": 112717, "label_based": true}
reasoning_valid: true
每条引用证据（RULE_000/001/002/005、STAT_000）均直接对应主张中命中的规则及其统计值，无反证/无关/跳跃。
overclaim: false
预设 confirmed，主张仅为规则命中事实及统计值直接引用，未超出所引事实范围，恰达 confirmed 档

# 交易 3516609（GBDT p=0.6041）
<details><summary>本单【全部证据池】——核实被断言的值是否在整单里有据；「有据但该 finding 没引」属引用瑕疵，按 v2 不判 N</summary>

- `TXN_000`: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3516609", "value": 15.247, "window": [13946708, 13946708], "label_based": false}
- `TXN_001`: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3516609", "value": "C", "window": [13946708, 13946708], "label_based": false}
- `TXN_002`: {"fact_id": "TXN_002", "type": "txn_field:card1", "entity": "txn=3516609", "value": 10568, "window": [13946708, 13946708], "label_based": false}
- `TXN_003`: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3516609", "value": "visa", "window": [13946708, 13946708], "label_based": false}
- `TXN_004`: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3516609", "value": "credit", "window": [13946708, 13946708], "label_based": false}
- `TXN_005`: {"fact_id": "TXN_005", "type": "txn_field:P_emaildomain", "entity": "txn=3516609", "value": "gmail.com", "window": [13946708, 13946708], "label_based": false}
- `TXN_006`: {"fact_id": "TXN_006", "type": "txn_field:DeviceType", "entity": "txn=3516609", "value": "mobile", "window": [13946708, 13946708], "label_based": false}
- `TXN_007`: {"fact_id": "TXN_007", "type": "txn_field:DeviceInfo", "entity": "txn=3516609", "value": "LG-M700 Build/NMF26X", "window": [13946708, 13946708], "label_based": false}
- `GRAPH_000`: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=10568", "value": 830, "window": [0, 13946708], "label_based": false}
- `GRAPH_001`: {"fact_id": "GRAPH_001", "type": "prior_cnt", "entity": "card1=10568|addr1", "value": 0, "window": [0, 13946708], "label_based": false}
- `GRAPH_002`: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=10568", "value": 13, "window": [0, 13946708], "label_based": false}
- `GRAPH_003`: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=10568", "value": 12, "window": [0, 13946708], "label_based": false}
- `GRAPH_004`: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=10568", "value": 152, "window": [0, 13946708], "label_based": false}
- `GRAPH_005`: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=10568", "value": 0.2019, "window": [0, 12132308], "label_based": true}
- `GRAPH_006`: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.2593, "window": [0, 12132308], "label_based": true}
- `GRAPH_007`: {"fact_id": "GRAPH_007", "type": "gang_score", "entity": "card1=10568", "value": 1.0, "window": [0, 12132308], "label_based": true}
- `RULE_000`: {"fact_id": "RULE_000", "type": "rule", "entity": "rule=R_PRIOR_FRAUD_CARD1_EMAIL", "value": "实体欺诈史：card1+邮箱（card1_email_prior_fraud_rate>=0.1）：训练窗 [0,125)天 内该模式欺诈率 19.11%，为基率 3.51% 的 5.4 倍（触发 16,801 笔，其中欺诈 3,210）。", "window": [86400, 10886394], "support_n": 16801, "label_based": true}
- `RULE_001`: {"fact_id": "RULE_001", "type": "rule", "entity": "rule=R_PRIOR_FRAUD_CARD1", "value": "实体欺诈史：card1（card1_prior_fraud_rate>=0.1）：训练窗 [0,125)天 内该模式欺诈率 17.32%，为基率 3.51% 的 4.9 倍（触发 21,568 笔，其中欺诈 3,735）。", "window": [86400, 10886394], "support_n": 21568, "label_based": true}
- `RULE_002`: {"fact_id": "RULE_002", "type": "rule", "entity": "rule=R_ProductCD_C", "value": "高危取值 ProductCD=C（ProductCD=C）：训练窗 [0,125)天 内该模式欺诈率 11.27%，为基率 3.51% 的 3.2 倍（触发 51,857 笔，其中欺诈 5,844）。", "window": [86400, 10886394], "support_n": 51857, "label_based": true}
- `RULE_003`: {"fact_id": "RULE_003", "type": "rule", "entity": "rule=R_DeviceType_mobile", "value": "高危取值 DeviceType=mobile（DeviceType=mobile）：训练窗 [0,125)天 内该模式欺诈率 9.88%，为基率 3.51% 的 2.8 倍（触发 42,771 笔，其中欺诈 4,226）。", "window": [86400, 10886394], "support_n": 42771, "label_based": true}
- `RULE_004`: {"fact_id": "RULE_004", "type": "rule", "entity": "rule=R_NEW_ENTITY", "value": "新实体首现（card1+addr1 无历史）（card1_addr1_prior_cnt=0）：训练窗 [0,125)天 内该模式欺诈率 7.78%，为基率 3.51% 的 2.2 倍（触发 82,512 笔，其中欺诈 6,417）。", "window": [86400, 10886394], "support_n": 82512, "label_based": true}
- `RULE_005`: {"fact_id": "RULE_005", "type": "rule", "entity": "rule=R_HAS_IDENTITY", "value": "有设备/身份采集记录（id_01 存在）：训练窗 [0,125)天 内该模式欺诈率 7.44%，为基率 3.51% 的 2.1 倍（触发 112,859 笔，其中欺诈 8,397）。", "window": [86400, 10886394], "support_n": 112859, "label_based": true}
- `RULE_006`: {"fact_id": "RULE_006", "type": "rule", "entity": "rule=R_card6_credit", "value": "高危取值 card6=credit（card6=credit）：训练窗 [0,125)天 内该模式欺诈率 6.65%，为基率 3.51% 的 1.9 倍（触发 112,717 笔，其中欺诈 7,501）。", "window": [86400, 10886394], "support_n": 112717, "label_based": true}
- `CASE_000`: {"fact_id": "CASE_000", "type": "case", "entity": "case=3454224", "value": "案例 #3454224（day 138）：确认欺诈（拒付举报）。金额 $39.59（$10-50），ProductCD=C，visa/credit，邮箱域=gmail.com，设备=SM-J320M Build/LMY47V，card1=10568。实体历史：prior 0 笔、无成熟标签；设备 fan-out 136。", "window": [12029531, 12029531], "label_based": true}
- `CASE_001`: {"fact_id": "CASE_001", "type": "case", "entity": "case=3435401", "value": "案例 #3435401（day 131）：高分假阳（人工洗清，当时模型分 0.16）。金额 $47.28（$10-50），ProductCD=C，visa/credit，邮箱域=gmail.com，设备=HUAWEI RIO-L03 Build/HUAWEIRIO-L03，card1=10568。实体历史：prior 0 笔、无成熟标签；设备 fan-out 129。", "window": [11413925, 11413925], "label_based": true}
- `CASE_002`: {"fact_id": "CASE_002", "type": "case", "entity": "case=3433291", "value": "案例 #3433291（day 130）：高分假阳（人工洗清，当时模型分 0.44）。金额 $10.64（$10-50），ProductCD=C，visa/credit，邮箱域=gmail.com，设备=缺失，card1=10568。实体历史：prior 0 笔、无成熟标签；设备 fan-out 128。", "window": [11371973, 11371973], "label_based": true}
- `CASE_003`: {"fact_id": "CASE_003", "type": "case", "entity": "case=3417508", "value": "案例 #3417508（day 124）：高分假阳（人工洗清，当时模型分 0.30）。金额 $10.45（$10-50），ProductCD=C，visa/credit，邮箱域=gmail.com，设备=缺失，card1=10568, addr1=284。实体历史：prior 5 笔、成熟欺诈率 0%；设备 fan-out 122。", "window": [10876924, 10876924], "label_based": true}
- `STAT_000`: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=C", "value": 0.1131, "window": [0, 12132308], "support_n": 56152, "label_based": true}
- `STAT_001`: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "P_emaildomain=gmail.com", "value": 0.044, "window": [0, 12132308], "support_n": 181726, "label_based": true}

</details>

## 3516609#1（断言强度=supported）
finding：Strong gang/entity-abuse signature: gang_score=1.0 with device fan-out of 152 (plus addr1 fan-out 13, email fan-out 12) on a single card1 — consistent with one card cycling many devices/identities.
该 finding 引用的证据：
  - GRAPH_007: {"fact_id": "GRAPH_007", "type": "gang_score", "entity": "card1=10568", "value": 1.0, "window": [0, 12132308], "label_based": true}
  - GRAPH_004: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=10568", "value": 152, "window": [0, 13946708], "label_based": false}
  - GRAPH_002: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=10568", "value": 13, "window": [0, 13946708], "label_based": false}
  - GRAPH_003: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=10568", "value": 12, "window": [0, 13946708], "label_based": false}
reasoning_valid: true
fanout_device=152 直接表明多设备共用，“consistent with”为条件化表述，未新增需要时序证据的机制断言
overclaim: false
预设 supported，由 gang_score 及三个 fanout 值综合归因，需多事实综合，最高可达 supported，未越界


# 交易 3529881（GBDT p=0.0015）
<details><summary>本单【全部证据池】——核实被断言的值是否在整单里有据；「有据但该 finding 没引」属引用瑕疵，按 v2 不判 N</summary>

- `TXN_000`: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3529881", "value": 226.0, "window": [14323198, 14323198], "label_based": false}
- `TXN_001`: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3529881", "value": "W", "window": [14323198, 14323198], "label_based": false}
- `TXN_002`: {"fact_id": "TXN_002", "type": "txn_field:card1", "entity": "txn=3529881", "value": 2157, "window": [14323198, 14323198], "label_based": false}
- `TXN_003`: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3529881", "value": "visa", "window": [14323198, 14323198], "label_based": false}
- `TXN_004`: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3529881", "value": "debit", "window": [14323198, 14323198], "label_based": false}
- `TXN_005`: {"fact_id": "TXN_005", "type": "txn_field:addr1", "entity": "txn=3529881", "value": 272.0, "window": [14323198, 14323198], "label_based": false}
- `TXN_006`: {"fact_id": "TXN_006", "type": "txn_field:P_emaildomain", "entity": "txn=3529881", "value": "aol.com", "window": [14323198, 14323198], "label_based": false}
- `GRAPH_000`: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=2157", "value": 807, "window": [0, 14323198], "label_based": false}
- `GRAPH_001`: {"fact_id": "GRAPH_001", "type": "prior_cnt", "entity": "card1=2157|addr1", "value": 661, "window": [0, 14323198], "label_based": false}
- `GRAPH_002`: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=2157", "value": 31, "window": [0, 14323198], "label_based": false}
- `GRAPH_003`: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=2157", "value": 17, "window": [0, 14323198], "label_based": false}
- `GRAPH_004`: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=2157", "value": 11, "window": [0, 14323198], "label_based": false}
- `GRAPH_005`: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=2157", "value": 0.0115, "window": [0, 12508798], "label_based": true}
- `GRAPH_006`: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|addr1", "value": 0.0105, "window": [0, 12508798], "label_based": true}
- `GRAPH_007`: {"fact_id": "GRAPH_007", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.0, "window": [0, 12508798], "label_based": true}
- `GRAPH_008`: {"fact_id": "GRAPH_008", "type": "gang_score", "entity": "card1=2157", "value": 0.115, "window": [0, 12508798], "label_based": true}
- `STAT_000`: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=W", "value": 0.0209, "window": [0, 12508798], "support_n": 353155, "label_based": true}
- `STAT_001`: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "card4=visa", "value": 0.0348, "window": [0, 12508798], "support_n": 313909, "label_based": true}
- `STAT_002`: {"fact_id": "STAT_002", "type": "fraud_rate", "entity": "P_emaildomain=aol.com", "value": 0.0223, "window": [0, 12508798], "support_n": 23044, "label_based": true}
- `CASE_000`: {"fact_id": "CASE_000", "type": "case", "entity": "case=3464518", "value": "案例 #3464518（day 142）：确认欺诈（拒付举报）。金额 $107.95（$100-300），ProductCD=W，visa/debit，邮箱域=缺失，设备=缺失，card1=2157, addr1=220。实体历史：prior 2 笔、成熟欺诈率 0%；设备 fan-out 11。", "window": [12358419, 12358419], "label_based": true}
- `CASE_001`: {"fact_id": "CASE_001", "type": "case", "entity": "case=3378408", "value": "案例 #3378408（day 112）：高分假阳（人工洗清，当时模型分 0.24）。金额 $280.00（$100-300），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=2157, addr1=512。实体历史：prior 5 笔、无成熟标签；设备 fan-out 10。", "window": [9828377, 9828377], "label_based": true}
- `CASE_002`: {"fact_id": "CASE_002", "type": "case", "entity": "case=3378405", "value": "案例 #3378405（day 112）：高分假阳（人工洗清，当时模型分 0.15）。金额 $265.00（$100-300），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=2157, addr1=512。实体历史：prior 4 笔、无成熟标签；设备 fan-out 10。", "window": [9828298, 9828298], "label_based": true}
- `CASE_003`: {"fact_id": "CASE_003", "type": "case", "entity": "case=3378325", "value": "案例 #3378325（day 112）：高分假阳（人工洗清，当时模型分 0.18）。金额 $280.00（$100-300），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=2157, addr1=512。实体历史：prior 3 笔、无成熟标签；设备 fan-out 10。", "window": [9827163, 9827163], "label_based": true}
- `STAT_003`: {"fact_id": "STAT_003", "type": "fraud_rate", "entity": "card6=debit", "value": 0.024, "window": [0, 12508798], "support_n": 356671, "label_based": true}

</details>

## 3529881#3（断言强度=supported）
finding：无风控规则命中；相似案例中三笔同 card1 家族的高分案例均被人工洗清为假阳，唯一确认欺诈案例(#3464518)画像不同（prior 仅 2 笔、成熟欺诈率 0%），与本笔长历史低欺诈率账户不匹配。
该 finding 引用的证据：
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3464518", "value": "案例 #3464518（day 142）：确认欺诈（拒付举报）。金额 $107.95（$100-300），ProductCD=W，visa/debit，邮箱域=缺失，设备=缺失，card1=2157, addr1=220。实体历史：prior 2 笔、成熟欺诈率 0%；设备 fan-out 11。", "window": [12358419, 12358419], "label_based": true}
  - CASE_001: {"fact_id": "CASE_001", "type": "case", "entity": "case=3378408", "value": "案例 #3378408（day 112）：高分假阳（人工洗清，当时模型分 0.24）。金额 $280.00（$100-300），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=2157, addr1=512。实体历史：prior 5 笔、无成熟标签；设备 fan-out 10。", "window": [9828377, 9828377], "label_based": true}
  - CASE_002: {"fact_id": "CASE_002", "type": "case", "entity": "case=3378405", "value": "案例 #3378405（day 112）：高分假阳（人工洗清，当时模型分 0.15）。金额 $265.00（$100-300），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=2157, addr1=512。实体历史：prior 4 笔、无成熟标签；设备 fan-out 10。", "window": [9828298, 9828298], "label_based": true}
  - CASE_003: {"fact_id": "CASE_003", "type": "case", "entity": "case=3378325", "value": "案例 #3378325（day 112）：高分假阳（人工洗清，当时模型分 0.18）。金额 $280.00（$100-300），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=2157, addr1=512。实体历史：prior 3 笔、无成熟标签；设备 fan-out 10。", "window": [9827163, 9827163], "label_based": true}
reasoning_valid: true
主张为画像对比，缺引的本笔 prior=807/fraud_rate=1.15% 在池中有据（GRAPH_000/005），属引用瑕疵，未命中反证/无关/跳跃三类 N
overclaim: false
预设 supported，“不匹配”由本笔与 CASE_000 的 prior/fraud_rate 等多字段对比推导，最高可达 supported，未越界

# 交易 3531514（GBDT p=0.0010）
<details><summary>本单【全部证据池】——核实被断言的值是否在整单里有据；「有据但该 finding 没引」属引用瑕疵，按 v2 不判 N</summary>

- `TXN_000`: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3531514", "value": 24.95, "window": [14356742, 14356742], "label_based": false}
- `TXN_001`: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3531514", "value": "W", "window": [14356742, 14356742], "label_based": false}
- `TXN_002`: {"fact_id": "TXN_002", "type": "txn_field:card1", "entity": "txn=3531514", "value": 2518, "window": [14356742, 14356742], "label_based": false}
- `TXN_003`: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3531514", "value": "visa", "window": [14356742, 14356742], "label_based": false}
- `TXN_004`: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3531514", "value": "debit", "window": [14356742, 14356742], "label_based": false}
- `TXN_005`: {"fact_id": "TXN_005", "type": "txn_field:addr1", "entity": "txn=3531514", "value": 512.0, "window": [14356742, 14356742], "label_based": false}
- `TXN_006`: {"fact_id": "TXN_006", "type": "txn_field:P_emaildomain", "entity": "txn=3531514", "value": "gmail.com", "window": [14356742, 14356742], "label_based": false}
- `GRAPH_000`: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=2518", "value": 25, "window": [0, 14356742], "label_based": false}
- `GRAPH_001`: {"fact_id": "GRAPH_001", "type": "prior_cnt", "entity": "card1=2518|addr1", "value": 25, "window": [0, 14356742], "label_based": false}
- `GRAPH_002`: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=2518", "value": 1, "window": [0, 14356742], "label_based": false}
- `GRAPH_003`: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=2518", "value": 3, "window": [0, 14356742], "label_based": false}
- `GRAPH_004`: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=2518", "value": 2, "window": [0, 14356742], "label_based": false}
- `GRAPH_005`: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=2518", "value": 0.0, "window": [0, 12542342], "label_based": true}
- `GRAPH_006`: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|addr1", "value": 0.0, "window": [0, 12542342], "label_based": true}
- `GRAPH_007`: {"fact_id": "GRAPH_007", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.0, "window": [0, 12542342], "label_based": true}
- `GRAPH_008`: {"fact_id": "GRAPH_008", "type": "gang_score", "entity": "card1=2518", "value": 0.0, "window": [0, 12542342], "label_based": true}
- `CASE_000`: {"fact_id": "CASE_000", "type": "case", "entity": "case=3464403", "value": "案例 #3464403（day 142）：确认欺诈（拒付举报）。金额 $39.00（$10-50），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=7664, addr1=204。实体历史：prior 27 笔、成熟欺诈率 78%；设备 fan-out 28。", "window": [12355498, 12355498], "label_based": true}
- `CASE_001`: {"fact_id": "CASE_001", "type": "case", "entity": "case=3462257", "value": "案例 #3462257（day 141）：确认欺诈（拒付举报）。金额 $24.00（$10-50），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=13780, addr1=441。实体历史：prior 749 笔、成熟欺诈率 1%；设备 fan-out 10。", "window": [12273437, 12273437], "label_based": true}
- `CASE_002`: {"fact_id": "CASE_002", "type": "case", "entity": "case=3461179", "value": "案例 #3461179（day 140）：确认欺诈（拒付举报）。金额 $15.00（$10-50），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=13780, addr1=441。实体历史：prior 745 笔、成熟欺诈率 1%；设备 fan-out 10。", "window": [12254662, 12254662], "label_based": true}
- `CASE_003`: {"fact_id": "CASE_003", "type": "case", "entity": "case=3450689", "value": "案例 #3450689（day 136）：高分假阳（人工洗清，当时模型分 0.18）。金额 $49.00（$10-50），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=1078, addr1=123。实体历史：prior 277 笔、成熟欺诈率 4%；设备 fan-out 6。", "window": [11921951, 11921951], "label_based": true}
- `STAT_000`: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=W", "value": 0.0209, "window": [0, 12542342], "support_n": 354532, "label_based": true}
- `STAT_001`: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "P_emaildomain=gmail.com", "value": 0.0441, "window": [0, 12542342], "support_n": 186743, "label_based": true}
- `STAT_002`: {"fact_id": "STAT_002", "type": "fraud_rate", "entity": "card4=visa", "value": 0.0348, "window": [0, 12542342], "support_n": 314975, "label_based": true}

</details>

## 3531514#3（断言强度=supported）
finding：相似确认欺诈案例的实体普遍具备高成熟欺诈率（如78%）或高设备扇出（10-28），与本笔低扇出、0%欺诈率的实体画像不符；本笔更接近被人工洗清的高分假阳模式。
该 finding 引用的证据：
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3464403", "value": "案例 #3464403（day 142）：确认欺诈（拒付举报）。金额 $39.00（$10-50），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=7664, addr1=204。实体历史：prior 27 笔、成熟欺诈率 78%；设备 fan-out 28。", "window": [12355498, 12355498], "label_based": true}
  - CASE_001: {"fact_id": "CASE_001", "type": "case", "entity": "case=3462257", "value": "案例 #3462257（day 141）：确认欺诈（拒付举报）。金额 $24.00（$10-50），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=13780, addr1=441。实体历史：prior 749 笔、成熟欺诈率 1%；设备 fan-out 10。", "window": [12273437, 12273437], "label_based": true}
  - CASE_003: {"fact_id": "CASE_003", "type": "case", "entity": "case=3450689", "value": "案例 #3450689（day 136）：高分假阳（人工洗清，当时模型分 0.18）。金额 $49.00（$10-50），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=1078, addr1=123。实体历史：prior 277 笔、成熟欺诈率 4%；设备 fan-out 6。", "window": [11921951, 11921951], "label_based": true}
  - GRAPH_004: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=2518", "value": 2, "window": [0, 14356742], "label_based": false}
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=2518", "value": 0.0, "window": [0, 12542342], "label_based": true}
reasoning_valid: true
引用证据完整支撑确认欺诈与假阳案例的特征对比，未命中反证/无关/跳跃三类 N
overclaim: false
预设 supported，“不匹配”及“更接近假阳”由多案例多字段对比推理，最高可达 supported，未越界

# 交易 3534425（GBDT p=0.9914）
<details><summary>本单【全部证据池】——核实被断言的值是否在整单里有据；「有据但该 finding 没引」属引用瑕疵，按 v2 不判 N</summary>

- `TXN_000`: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3534425", "value": 150.0, "window": [14442360, 14442360], "label_based": false}
- `TXN_001`: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3534425", "value": "H", "window": [14442360, 14442360], "label_based": false}
- `TXN_002`: {"fact_id": "TXN_002", "type": "txn_field:card1", "entity": "txn=3534425", "value": 7170, "window": [14442360, 14442360], "label_based": false}
- `TXN_003`: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3534425", "value": "mastercard", "window": [14442360, 14442360], "label_based": false}
- `TXN_004`: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3534425", "value": "debit", "window": [14442360, 14442360], "label_based": false}
- `TXN_005`: {"fact_id": "TXN_005", "type": "txn_field:addr1", "entity": "txn=3534425", "value": 325.0, "window": [14442360, 14442360], "label_based": false}
- `TXN_006`: {"fact_id": "TXN_006", "type": "txn_field:P_emaildomain", "entity": "txn=3534425", "value": "gmail.com", "window": [14442360, 14442360], "label_based": false}
- `TXN_007`: {"fact_id": "TXN_007", "type": "txn_field:DeviceType", "entity": "txn=3534425", "value": "desktop", "window": [14442360, 14442360], "label_based": false}
- `TXN_008`: {"fact_id": "TXN_008", "type": "txn_field:DeviceInfo", "entity": "txn=3534425", "value": "Windows", "window": [14442360, 14442360], "label_based": false}
- `GRAPH_000`: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=7170", "value": 70, "window": [0, 14442360], "label_based": false}
- `GRAPH_001`: {"fact_id": "GRAPH_001", "type": "prior_cnt", "entity": "card1=7170|addr1", "value": 59, "window": [0, 14442360], "label_based": false}
- `GRAPH_002`: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=7170", "value": 5, "window": [0, 14442360], "label_based": false}
- `GRAPH_003`: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=7170", "value": 7, "window": [0, 14442360], "label_based": false}
- `GRAPH_004`: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=7170", "value": 5, "window": [0, 14442360], "label_based": false}
- `GRAPH_005`: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=7170", "value": 0.0147, "window": [0, 12627960], "label_based": true}
- `GRAPH_006`: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|addr1", "value": 0.0, "window": [0, 12627960], "label_based": true}
- `GRAPH_007`: {"fact_id": "GRAPH_007", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.0, "window": [0, 12627960], "label_based": true}
- `GRAPH_008`: {"fact_id": "GRAPH_008", "type": "prior_fraud_rate", "entity": "card1|device", "value": 0.0, "window": [0, 12627960], "label_based": true}
- `GRAPH_009`: {"fact_id": "GRAPH_009", "type": "gang_score", "entity": "card1=7170", "value": 0.0, "window": [0, 12627960], "label_based": true}
- `RULE_000`: {"fact_id": "RULE_000", "type": "rule", "entity": "rule=R_HAS_IDENTITY", "value": "有设备/身份采集记录（id_01 存在）：训练窗 [0,125)天 内该模式欺诈率 7.44%，为基率 3.51% 的 2.1 倍（触发 112,859 笔，其中欺诈 8,397）。", "window": [86400, 10886394], "support_n": 112859, "label_based": true}
- `RULE_001`: {"fact_id": "RULE_001", "type": "rule", "entity": "rule=R_DeviceType_desktop", "value": "高危取值 DeviceType=desktop（DeviceType=desktop）：训练窗 [0,125)天 内该模式欺诈率 6.09%，为基率 3.51% 的 1.7 倍（触发 67,488 笔，其中欺诈 4,113）。", "window": [86400, 10886394], "support_n": 67488, "label_based": true}
- `CASE_000`: {"fact_id": "CASE_000", "type": "case", "entity": "case=3235916", "value": "案例 #3235916（day 67）：确认欺诈（拒付举报）。金额 $150.00（$100-300），ProductCD=H，mastercard/debit，邮箱域=gmail.com，设备=Windows，card1=7919, addr1=225。实体历史：prior 46 笔、成熟欺诈率 0%；设备 fan-out 6。", "window": [5936959, 5936959], "label_based": true}
- `CASE_001`: {"fact_id": "CASE_001", "type": "case", "entity": "case=3460024", "value": "案例 #3460024（day 140）：高分假阳（人工洗清，当时模型分 0.16）。金额 $100.00（$100-300），ProductCD=H，visa/debit，邮箱域=gmail.com，设备=Windows，card1=12501, addr1=272。实体历史：prior 318 笔、成熟欺诈率 1%；设备 fan-out 32。", "window": [12233739, 12233739], "label_based": true}
- `CASE_002`: {"fact_id": "CASE_002", "type": "case", "entity": "case=3434397", "value": "案例 #3434397（day 130）：高分假阳（人工洗清，当时模型分 0.24）。金额 $150.00（$100-300），ProductCD=H，visa/debit，邮箱域=gmail.com，设备=Windows，card1=7508, addr1=220。实体历史：prior 583 笔、成熟欺诈率 6%；设备 fan-out 23。", "window": [11393475, 11393475], "label_based": true}
- `CASE_003`: {"fact_id": "CASE_003", "type": "case", "entity": "case=3420881", "value": "案例 #3420881（day 125）：高分假阳（人工洗清，当时模型分 0.20）。金额 $150.00（$100-300），ProductCD=H，mastercard/debit，邮箱域=gmail.com，设备=HTCD100LVWPP，card1=16053, addr1=325。实体历史：prior 43 笔、成熟欺诈率 3%；设备 fan-out 3。", "window": [10967405, 10967405], "label_based": true}
- `STAT_000`: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=H", "value": 0.0463, "window": [0, 12627960], "support_n": 30020, "label_based": true}
- `STAT_001`: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "P_emaildomain=gmail.com", "value": 0.0441, "window": [0, 12627960], "support_n": 187740, "label_based": true}
- `STAT_002`: {"fact_id": "STAT_002", "type": "fraud_rate", "entity": "card4=mastercard", "value": 0.035, "window": [0, 12627960], "support_n": 154977, "label_based": true}

</details>

## 3534425#5（断言强度=tentative）
finding：相似案例中 3/4（#3460024、#3434397、#3420881）为人工洗清的高分假阳，画像（$150、ProductCD=H、gmail、Windows、debit）与本笔高度一致；仅 1 例（#3235916）确认欺诈但属不同 card1/addr1。对照倾向假阳。
该 finding 引用的证据：
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3235916", "value": "案例 #3235916（day 67）：确认欺诈（拒付举报）。金额 $150.00（$100-300），ProductCD=H，mastercard/debit，邮箱域=gmail.com，设备=Windows，card1=7919, addr1=225。实体历史：prior 46 笔、成熟欺诈率 0%；设备 fan-out 6。", "window": [5936959, 5936959], "label_based": true}
  - CASE_001: {"fact_id": "CASE_001", "type": "case", "entity": "case=3460024", "value": "案例 #3460024（day 140）：高分假阳（人工洗清，当时模型分 0.16）。金额 $100.00（$100-300），ProductCD=H，visa/debit，邮箱域=gmail.com，设备=Windows，card1=12501, addr1=272。实体历史：prior 318 笔、成熟欺诈率 1%；设备 fan-out 32。", "window": [12233739, 12233739], "label_based": true}
  - CASE_002: {"fact_id": "CASE_002", "type": "case", "entity": "case=3434397", "value": "案例 #3434397（day 130）：高分假阳（人工洗清，当时模型分 0.24）。金额 $150.00（$100-300），ProductCD=H，visa/debit，邮箱域=gmail.com，设备=Windows，card1=7508, addr1=220。实体历史：prior 583 笔、成熟欺诈率 6%；设备 fan-out 23。", "window": [11393475, 11393475], "label_based": true}
  - CASE_003: {"fact_id": "CASE_003", "type": "case", "entity": "case=3420881", "value": "案例 #3420881（day 125）：高分假阳（人工洗清，当时模型分 0.20）。金额 $150.00（$100-300），ProductCD=H，mastercard/debit，邮箱域=gmail.com，设备=HTCD100LVWPP，card1=16053, addr1=325。实体历史：prior 43 笔、成熟欺诈率 3%；设备 fan-out 3。", "window": [10967405, 10967405], "label_based": true}
reasoning_valid: true
主张为案例对比与倾向判断，缺引的本笔画像值在池中有据（TXN_000/001/006/007/004/002/005），属引用瑕疵，未命中反证/无关/跳跃三类 N
overclaim: false
预设 tentative，而“倾向假阳”由多案例多维度对比综合推导，最高可达 supported，标注 tentativ 属保守偏置，未越界


# 交易 3537070（GBDT p=0.0009）
<details><summary>本单【全部证据池】——核实被断言的值是否在整单里有据；「有据但该 finding 没引」属引用瑕疵，按 v2 不判 N</summary>

- `TXN_000`: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3537070", "value": 26.95, "window": [14518560, 14518560], "label_based": false}
- `TXN_001`: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3537070", "value": "W", "window": [14518560, 14518560], "label_based": false}
- `TXN_002`: {"fact_id": "TXN_002", "type": "txn_field:card1", "entity": "txn=3537070", "value": 17399, "window": [14518560, 14518560], "label_based": false}
- `TXN_003`: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3537070", "value": "mastercard", "window": [14518560, 14518560], "label_based": false}
- `TXN_004`: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3537070", "value": "debit", "window": [14518560, 14518560], "label_based": false}
- `TXN_005`: {"fact_id": "TXN_005", "type": "txn_field:addr1", "entity": "txn=3537070", "value": 204.0, "window": [14518560, 14518560], "label_based": false}
- `TXN_006`: {"fact_id": "TXN_006", "type": "txn_field:P_emaildomain", "entity": "txn=3537070", "value": "gmail.com", "window": [14518560, 14518560], "label_based": false}
- `GRAPH_000`: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=17399", "value": 1768, "window": [0, 14518560], "label_based": false}
- `GRAPH_001`: {"fact_id": "GRAPH_001", "type": "prior_cnt", "entity": "card1=17399|addr1", "value": 1664, "window": [0, 14518560], "label_based": false}
- `GRAPH_002`: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=17399", "value": 30, "window": [0, 14518560], "label_based": false}
- `GRAPH_003`: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=17399", "value": 21, "window": [0, 14518560], "label_based": false}
- `GRAPH_004`: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=17399", "value": 11, "window": [0, 14518560], "label_based": false}
- `GRAPH_005`: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=17399", "value": 0.0077, "window": [0, 12704160], "label_based": true}
- `GRAPH_006`: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|addr1", "value": 0.0081, "window": [0, 12704160], "label_based": true}
- `GRAPH_007`: {"fact_id": "GRAPH_007", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.0014, "window": [0, 12704160], "label_based": true}
- `GRAPH_008`: {"fact_id": "GRAPH_008", "type": "gang_score", "entity": "card1=17399", "value": 0.077, "window": [0, 12704160], "label_based": true}
- `STAT_000`: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=W", "value": 0.0208, "window": [0, 12704160], "support_n": 358853, "label_based": true}
- `STAT_001`: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "P_emaildomain=gmail.com", "value": 0.044, "window": [0, 12704160], "support_n": 188785, "label_based": true}
- `CASE_000`: {"fact_id": "CASE_000", "type": "case", "entity": "case=3092931", "value": "案例 #3092931（day 23）：确认欺诈（拒付举报）。金额 $300.00（$300-1k），ProductCD=H，mastercard/debit，邮箱域=gmail.com，设备=iOS Device，card1=17399, addr1=204。实体历史：prior 244 笔、成熟欺诈率 0%；设备 fan-out 8。", "window": [2094826, 2094826], "label_based": true}
- `CASE_001`: {"fact_id": "CASE_001", "type": "case", "entity": "case=3414443", "value": "案例 #3414443（day 123）：确认欺诈（拒付举报）。金额 $35.00（$10-50），ProductCD=W，mastercard/debit，邮箱域=gmail.com，设备=缺失，card1=17131, addr1=264。实体历史：prior 1110 笔、成熟欺诈率 1%；设备 fan-out 12。", "window": [10796864, 10796864], "label_based": true}
- `CASE_002`: {"fact_id": "CASE_002", "type": "case", "entity": "case=3226470", "value": "案例 #3226470（day 64）：确认欺诈（拒付举报）。金额 $39.00（$10-50），ProductCD=W，mastercard/debit，邮箱域=gmail.com，设备=缺失，card1=10057, addr1=269。实体历史：prior 84 笔、成熟欺诈率 4%；设备 fan-out 19。", "window": [5669289, 5669289], "label_based": true}
- `CASE_003`: {"fact_id": "CASE_003", "type": "case", "entity": "case=3428587", "value": "案例 #3428587（day 128）：高分假阳（人工洗清，当时模型分 0.16）。金额 $50.00（$50-100），ProductCD=W，mastercard/debit，邮箱域=gmail.com，设备=缺失，card1=17055, addr1=325。实体历史：prior 232 笔、成熟欺诈率 1%；设备 fan-out 13。", "window": [11216344, 11216344], "label_based": true}

</details>

## 3537070#4（断言强度=supported）
finding：同一 card1=17399 存在一笔确认欺诈(CASE_000)，但其画像不同（ProductCD=H、$300、iOS 设备、早于本笔约 143 天），且该卡成熟欺诈率仍低至 0.77%，不足以推翻本笔低风险判断；与之相对，几乎同画像(W/mastercard/gmail)的高分案例 CASE_003 曾被人工洗清为假阳。
该 finding 引用的证据：
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3092931", "value": "案例 #3092931（day 23）：确认欺诈（拒付举报）。金额 $300.00（$300-1k），ProductCD=H，mastercard/debit，邮箱域=gmail.com，设备=iOS Device，card1=17399, addr1=204。实体历史：prior 244 笔、成熟欺诈率 0%；设备 fan-out 8。", "window": [2094826, 2094826], "label_based": true}
  - CASE_003: {"fact_id": "CASE_003", "type": "case", "entity": "case=3428587", "value": "案例 #3428587（day 128）：高分假阳（人工洗清，当时模型分 0.16）。金额 $50.00（$50-100），ProductCD=W，mastercard/debit，邮箱域=gmail.com，设备=缺失，card1=17055, addr1=325。实体历史：prior 232 笔、成熟欺诈率 1%；设备 fan-out 13。", "window": [11216344, 11216344], "label_based": true}
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=17399", "value": 0.0077, "window": [0, 12704160], "label_based": true}
reasoning_valid: true
缺引的本笔画像/时间值在池中有据（TXN_000/001/006/004/003/005），属引用瑕疵；“143天”为程度估算不判误读，未命中三类 N。
overclaim: false
预设 supported，低风险判断由画像对比+时间差异+低欺诈率+假阳参照多事实综合权衡，最高可达 supported，未越界

# 交易 3547972（GBDT p=0.8144）
<details><summary>本单【全部证据池】——核实被断言的值是否在整单里有据；「有据但该 finding 没引」属引用瑕疵，按 v2 不判 N</summary>

- `TXN_000`: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3547972", "value": 35.658, "window": [14857168, 14857168], "label_based": false}
- `TXN_001`: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3547972", "value": "C", "window": [14857168, 14857168], "label_based": false}
- `TXN_002`: {"fact_id": "TXN_002", "type": "txn_field:card1", "entity": "txn=3547972", "value": 10086, "window": [14857168, 14857168], "label_based": false}
- `TXN_003`: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3547972", "value": "mastercard", "window": [14857168, 14857168], "label_based": false}
- `TXN_004`: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3547972", "value": "credit", "window": [14857168, 14857168], "label_based": false}
- `TXN_005`: {"fact_id": "TXN_005", "type": "txn_field:P_emaildomain", "entity": "txn=3547972", "value": "gmail.com", "window": [14857168, 14857168], "label_based": false}
- `TXN_006`: {"fact_id": "TXN_006", "type": "txn_field:DeviceType", "entity": "txn=3547972", "value": "desktop", "window": [14857168, 14857168], "label_based": false}
- `TXN_007`: {"fact_id": "TXN_007", "type": "txn_field:DeviceInfo", "entity": "txn=3547972", "value": "Windows", "window": [14857168, 14857168], "label_based": false}
- `GRAPH_000`: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=10086", "value": 429, "window": [0, 14857168], "label_based": false}
- `GRAPH_001`: {"fact_id": "GRAPH_001", "type": "prior_cnt", "entity": "card1=10086|addr1", "value": 0, "window": [0, 14857168], "label_based": false}
- `GRAPH_002`: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=10086", "value": 11, "window": [0, 14857168], "label_based": false}
- `GRAPH_003`: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=10086", "value": 15, "window": [0, 14857168], "label_based": false}
- `GRAPH_004`: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=10086", "value": 63, "window": [0, 14857168], "label_based": false}
- `GRAPH_005`: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=10086", "value": 0.2251, "window": [0, 13042768], "label_based": true}
- `GRAPH_006`: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.4214, "window": [0, 13042768], "label_based": true}
- `GRAPH_007`: {"fact_id": "GRAPH_007", "type": "prior_fraud_rate", "entity": "card1|device", "value": 0.1538, "window": [0, 13042768], "label_based": true}
- `GRAPH_008`: {"fact_id": "GRAPH_008", "type": "gang_score", "entity": "card1=10086", "value": 1.0, "window": [0, 13042768], "label_based": true}
- `RULE_000`: {"fact_id": "RULE_000", "type": "rule", "entity": "rule=R_PRIOR_FRAUD_CARD1_DEVICE", "value": "实体欺诈史：card1+设备（card1_device_prior_fraud_rate>=0.1）：训练窗 [0,125)天 内该模式欺诈率 21.04%，为基率 3.51% 的 6.0 倍（触发 4,920 笔，其中欺诈 1,035）。", "window": [86400, 10886394], "support_n": 4920, "label_based": true}
- `RULE_001`: {"fact_id": "RULE_001", "type": "rule", "entity": "rule=R_PRIOR_FRAUD_CARD1_EMAIL", "value": "实体欺诈史：card1+邮箱（card1_email_prior_fraud_rate>=0.1）：训练窗 [0,125)天 内该模式欺诈率 19.11%，为基率 3.51% 的 5.4 倍（触发 16,801 笔，其中欺诈 3,210）。", "window": [86400, 10886394], "support_n": 16801, "label_based": true}
- `RULE_002`: {"fact_id": "RULE_002", "type": "rule", "entity": "rule=R_PRIOR_FRAUD_CARD1", "value": "实体欺诈史：card1（card1_prior_fraud_rate>=0.1）：训练窗 [0,125)天 内该模式欺诈率 17.32%，为基率 3.51% 的 4.9 倍（触发 21,568 笔，其中欺诈 3,735）。", "window": [86400, 10886394], "support_n": 21568, "label_based": true}
- `RULE_003`: {"fact_id": "RULE_003", "type": "rule", "entity": "rule=R_ProductCD_C", "value": "高危取值 ProductCD=C（ProductCD=C）：训练窗 [0,125)天 内该模式欺诈率 11.27%，为基率 3.51% 的 3.2 倍（触发 51,857 笔，其中欺诈 5,844）。", "window": [86400, 10886394], "support_n": 51857, "label_based": true}
- `RULE_004`: {"fact_id": "RULE_004", "type": "rule", "entity": "rule=R_NEW_ENTITY", "value": "新实体首现（card1+addr1 无历史）（card1_addr1_prior_cnt=0）：训练窗 [0,125)天 内该模式欺诈率 7.78%，为基率 3.51% 的 2.2 倍（触发 82,512 笔，其中欺诈 6,417）。", "window": [86400, 10886394], "support_n": 82512, "label_based": true}
- `RULE_005`: {"fact_id": "RULE_005", "type": "rule", "entity": "rule=R_HAS_IDENTITY", "value": "有设备/身份采集记录（id_01 存在）：训练窗 [0,125)天 内该模式欺诈率 7.44%，为基率 3.51% 的 2.1 倍（触发 112,859 笔，其中欺诈 8,397）。", "window": [86400, 10886394], "support_n": 112859, "label_based": true}
- `RULE_006`: {"fact_id": "RULE_006", "type": "rule", "entity": "rule=R_card6_credit", "value": "高危取值 card6=credit（card6=credit）：训练窗 [0,125)天 内该模式欺诈率 6.65%，为基率 3.51% 的 1.9 倍（触发 112,717 笔，其中欺诈 7,501）。", "window": [86400, 10886394], "support_n": 112717, "label_based": true}
- `RULE_007`: {"fact_id": "RULE_007", "type": "rule", "entity": "rule=R_DeviceType_desktop", "value": "高危取值 DeviceType=desktop（DeviceType=desktop）：训练窗 [0,125)天 内该模式欺诈率 6.09%，为基率 3.51% 的 1.7 倍（触发 67,488 笔，其中欺诈 4,113）。", "window": [86400, 10886394], "support_n": 67488, "label_based": true}
- `CASE_000`: {"fact_id": "CASE_000", "type": "case", "entity": "case=3429950", "value": "案例 #3429950（day 129）：确认欺诈（拒付举报）。金额 $35.41（$10-50），ProductCD=C，mastercard/credit，邮箱域=gmail.com，设备=SM-G610M Build/MMB29K，card1=10086。实体历史：prior 0 笔、无成熟标签；设备 fan-out 51。", "window": [11250420, 11250420], "label_based": true}
- `CASE_001`: {"fact_id": "CASE_001", "type": "case", "entity": "case=3228898", "value": "案例 #3228898（day 65）：确认欺诈（拒付举报）。金额 $16.61（$10-50），ProductCD=C，mastercard/credit，邮箱域=gmail.com，设备=SM-G955U Build/NRD90M，card1=10086。实体历史：prior 0 笔、无成熟标签；设备 fan-out 33。", "window": [5713262, 5713262], "label_based": true}
- `CASE_002`: {"fact_id": "CASE_002", "type": "case", "entity": "case=3228862", "value": "案例 #3228862（day 65）：确认欺诈（拒付举报）。金额 $15.47（$10-50），ProductCD=C，mastercard/credit，邮箱域=gmail.com，设备=rv:58.0，card1=10086。实体历史：prior 0 笔、无成熟标签；设备 fan-out 33。", "window": [5712272, 5712272], "label_based": true}
- `CASE_003`: {"fact_id": "CASE_003", "type": "case", "entity": "case=3395115", "value": "案例 #3395115（day 118）：高分假阳（人工洗清，当时模型分 0.17）。金额 $41.45（$10-50），ProductCD=C，mastercard/credit，邮箱域=hotmail.com，设备=Windows，card1=10086。实体历史：prior 0 笔、无成熟标签；设备 fan-out 45。", "window": [10303506, 10303506], "label_based": true}

</details>

## 3547972#6（断言强度=confirmed）
finding：Transaction amount is small ($35.66), which lowers per-txn loss but does not offset entity-level gang risk.
该 finding 引用的证据：
  - TXN_000: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3547972", "value": 35.658, "window": [14857168, 14857168], "label_based": false}
reasoning_valid: true
前半句金额→损失有据；后半句 gang risk 相关值在池中有据（GRAPH_008=1.0/RULE_000）但未引，属引用瑕疵，不判 N。
overclaim: true
预设 confirmed，但“不能抵消 gang risk”需金额+gang风险双事实对比，单条金额事实最高仅 tentative，超档越界

# 交易 3551044（GBDT p=0.9192）
<details><summary>本单【全部证据池】——核实被断言的值是否在整单里有据；「有据但该 finding 没引」属引用瑕疵，按 v2 不判 N</summary>

- `TXN_000`: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3551044", "value": 74.101, "window": [14940839, 14940839], "label_based": false}
- `TXN_001`: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3551044", "value": "C", "window": [14940839, 14940839], "label_based": false}
- `TXN_002`: {"fact_id": "TXN_002", "type": "txn_field:card1", "entity": "txn=3551044", "value": 4461, "window": [14940839, 14940839], "label_based": false}
- `TXN_003`: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3551044", "value": "mastercard", "window": [14940839, 14940839], "label_based": false}
- `TXN_004`: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3551044", "value": "debit", "window": [14940839, 14940839], "label_based": false}
- `TXN_005`: {"fact_id": "TXN_005", "type": "txn_field:P_emaildomain", "entity": "txn=3551044", "value": "gmail.com", "window": [14940839, 14940839], "label_based": false}
- `TXN_006`: {"fact_id": "TXN_006", "type": "txn_field:DeviceType", "entity": "txn=3551044", "value": "desktop", "window": [14940839, 14940839], "label_based": false}
- `TXN_007`: {"fact_id": "TXN_007", "type": "txn_field:DeviceInfo", "entity": "txn=3551044", "value": "Windows", "window": [14940839, 14940839], "label_based": false}
- `GRAPH_000`: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=4461", "value": 2587, "window": [0, 14940839], "label_based": false}
- `GRAPH_001`: {"fact_id": "GRAPH_001", "type": "prior_cnt", "entity": "card1=4461|addr1", "value": 0, "window": [0, 14940839], "label_based": false}
- `GRAPH_002`: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=4461", "value": 20, "window": [0, 14940839], "label_based": false}
- `GRAPH_003`: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=4461", "value": 18, "window": [0, 14940839], "label_based": false}
- `GRAPH_004`: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=4461", "value": 275, "window": [0, 14940839], "label_based": false}
- `GRAPH_005`: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=4461", "value": 0.069, "window": [0, 13126439], "label_based": true}
- `GRAPH_006`: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.1143, "window": [0, 13126439], "label_based": true}
- `GRAPH_007`: {"fact_id": "GRAPH_007", "type": "prior_fraud_rate", "entity": "card1|device", "value": 0.043, "window": [0, 13126439], "label_based": true}
- `GRAPH_008`: {"fact_id": "GRAPH_008", "type": "gang_score", "entity": "card1=4461", "value": 0.69, "window": [0, 13126439], "label_based": true}
- `RULE_000`: {"fact_id": "RULE_000", "type": "rule", "entity": "rule=R_PRIOR_FRAUD_CARD1_EMAIL", "value": "实体欺诈史：card1+邮箱（card1_email_prior_fraud_rate>=0.1）：训练窗 [0,125)天 内该模式欺诈率 19.11%，为基率 3.51% 的 5.4 倍（触发 16,801 笔，其中欺诈 3,210）。", "window": [86400, 10886394], "support_n": 16801, "label_based": true}
- `RULE_001`: {"fact_id": "RULE_001", "type": "rule", "entity": "rule=R_ProductCD_C", "value": "高危取值 ProductCD=C（ProductCD=C）：训练窗 [0,125)天 内该模式欺诈率 11.27%，为基率 3.51% 的 3.2 倍（触发 51,857 笔，其中欺诈 5,844）。", "window": [86400, 10886394], "support_n": 51857, "label_based": true}
- `RULE_002`: {"fact_id": "RULE_002", "type": "rule", "entity": "rule=R_NEW_ENTITY", "value": "新实体首现（card1+addr1 无历史）（card1_addr1_prior_cnt=0）：训练窗 [0,125)天 内该模式欺诈率 7.78%，为基率 3.51% 的 2.2 倍（触发 82,512 笔，其中欺诈 6,417）。", "window": [86400, 10886394], "support_n": 82512, "label_based": true}
- `RULE_003`: {"fact_id": "RULE_003", "type": "rule", "entity": "rule=R_HAS_IDENTITY", "value": "有设备/身份采集记录（id_01 存在）：训练窗 [0,125)天 内该模式欺诈率 7.44%，为基率 3.51% 的 2.1 倍（触发 112,859 笔，其中欺诈 8,397）。", "window": [86400, 10886394], "support_n": 112859, "label_based": true}
- `RULE_004`: {"fact_id": "RULE_004", "type": "rule", "entity": "rule=R_DeviceType_desktop", "value": "高危取值 DeviceType=desktop（DeviceType=desktop）：训练窗 [0,125)天 内该模式欺诈率 6.09%，为基率 3.51% 的 1.7 倍（触发 67,488 笔，其中欺诈 4,113）。", "window": [86400, 10886394], "support_n": 67488, "label_based": true}
- `CASE_000`: {"fact_id": "CASE_000", "type": "case", "entity": "case=3255322", "value": "案例 #3255322（day 74）：确认欺诈（拒付举报）。金额 $62.11（$50-100），ProductCD=C，mastercard/debit，邮箱域=gmail.com，设备=Windows，card1=4461。实体历史：prior 0 笔、无成熟标签；设备 fan-out 165。", "window": [6488220, 6488220], "label_based": true}
- `CASE_001`: {"fact_id": "CASE_001", "type": "case", "entity": "case=3133514", "value": "案例 #3133514（day 33）：确认欺诈（拒付举报）。金额 $64.71（$50-100），ProductCD=C，mastercard/debit，邮箱域=gmail.com，设备=Windows，card1=4461。实体历史：prior 0 笔、无成熟标签；设备 fan-out 113。", "window": [3013160, 3013160], "label_based": true}
- `CASE_002`: {"fact_id": "CASE_002", "type": "case", "entity": "case=3121135", "value": "案例 #3121135（day 29）：确认欺诈（拒付举报）。金额 $19.24（$10-50），ProductCD=C，mastercard/debit，邮箱域=gmail.com，设备=Windows，card1=4461。实体历史：prior 0 笔、无成熟标签；设备 fan-out 104。", "window": [2675032, 2675032], "label_based": true}
- `CASE_003`: {"fact_id": "CASE_003", "type": "case", "entity": "case=3454902", "value": "案例 #3454902（day 138）：高分假阳（人工洗清，当时模型分 0.17）。金额 $73.65（$50-100），ProductCD=C，mastercard/debit，邮箱域=gmail.com，设备=SM-G925I Build/NRD90M，card1=4461。实体历史：prior 0 笔、无成熟标签；设备 fan-out 243。", "window": [12066211, 12066211], "label_based": true}
- `STAT_000`: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=C", "value": 0.1138, "window": [0, 13126439], "support_n": 59008, "label_based": true}

</details>

## 3551044#2（断言强度=tentative）
finding：对照假阳案例：唯一被人工洗清的高分案例 #3454902 在设备上与本笔不同（移动端 SM-G925I，而非 Windows 桌面），本笔匹配的是欺诈簇而非被洗清簇，降低了假阳可能。
该 finding 引用的证据：
  - CASE_003: {"fact_id": "CASE_003", "type": "case", "entity": "case=3454902", "value": "案例 #3454902（day 138）：高分假阳（人工洗清，当时模型分 0.17）。金额 $73.65（$50-100），ProductCD=C，mastercard/debit，邮箱域=gmail.com，设备=SM-G925I Build/NRD90M，card1=4461。实体历史：prior 0 笔、无成熟标签；设备 fan-out 243。", "window": [12066211, 12066211], "label_based": true}
  - TXN_006: {"fact_id": "TXN_006", "type": "txn_field:DeviceType", "entity": "txn=3551044", "value": "desktop", "window": [14940839, 14940839], "label_based": false}
  - TXN_007: {"fact_id": "TXN_007", "type": "txn_field:DeviceInfo", "entity": "txn=3551044", "value": "Windows", "window": [14940839, 14940839], "label_based": false}
reasoning_valid: true
“匹配欺诈簇”所需的欺诈案例设备值在池中有据（CASE_000/001/002 均为 Windows），属引用瑕疵，未命中三类 N。
overclaim: false
预设 tentative，“匹配欺诈簇+降低假阳可能”需多案例设备对比综合判断，最高可达 supported，标 tentativ 属保守偏置，未越界。

# 交易 3552038（GBDT p=0.7380）
<details><summary>本单【全部证据池】——核实被断言的值是否在整单里有据；「有据但该 finding 没引」属引用瑕疵，按 v2 不判 N</summary>

- `TXN_000`: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3552038", "value": 81.092, "window": [14962186, 14962186], "label_based": false}
- `TXN_001`: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3552038", "value": "C", "window": [14962186, 14962186], "label_based": false}
- `TXN_002`: {"fact_id": "TXN_002", "type": "txn_field:card1", "entity": "txn=3552038", "value": 13832, "window": [14962186, 14962186], "label_based": false}
- `TXN_003`: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3552038", "value": "mastercard", "window": [14962186, 14962186], "label_based": false}
- `TXN_004`: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3552038", "value": "debit", "window": [14962186, 14962186], "label_based": false}
- `TXN_005`: {"fact_id": "TXN_005", "type": "txn_field:P_emaildomain", "entity": "txn=3552038", "value": "hotmail.com", "window": [14962186, 14962186], "label_based": false}
- `TXN_006`: {"fact_id": "TXN_006", "type": "txn_field:DeviceType", "entity": "txn=3552038", "value": "mobile", "window": [14962186, 14962186], "label_based": false}
- `GRAPH_000`: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=13832", "value": 1975, "window": [0, 14962186], "label_based": false}
- `GRAPH_001`: {"fact_id": "GRAPH_001", "type": "prior_cnt", "entity": "card1=13832|addr1", "value": 0, "window": [0, 14962186], "label_based": false}
- `GRAPH_002`: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=13832", "value": 17, "window": [0, 14962186], "label_based": false}
- `GRAPH_003`: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=13832", "value": 16, "window": [0, 14962186], "label_based": false}
- `GRAPH_004`: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=13832", "value": 249, "window": [0, 14962186], "label_based": false}
- `GRAPH_005`: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=13832", "value": 0.0939, "window": [0, 13147786], "label_based": true}
- `GRAPH_006`: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.077, "window": [0, 13147786], "label_based": true}
- `GRAPH_007`: {"fact_id": "GRAPH_007", "type": "gang_score", "entity": "card1=13832", "value": 0.939, "window": [0, 13147786], "label_based": true}
- `RULE_000`: {"fact_id": "RULE_000", "type": "rule", "entity": "rule=R_ProductCD_C", "value": "高危取值 ProductCD=C（ProductCD=C）：训练窗 [0,125)天 内该模式欺诈率 11.27%，为基率 3.51% 的 3.2 倍（触发 51,857 笔，其中欺诈 5,844）。", "window": [86400, 10886394], "support_n": 51857, "label_based": true}
- `RULE_001`: {"fact_id": "RULE_001", "type": "rule", "entity": "rule=R_DeviceType_mobile", "value": "高危取值 DeviceType=mobile（DeviceType=mobile）：训练窗 [0,125)天 内该模式欺诈率 9.88%，为基率 3.51% 的 2.8 倍（触发 42,771 笔，其中欺诈 4,226）。", "window": [86400, 10886394], "support_n": 42771, "label_based": true}
- `RULE_002`: {"fact_id": "RULE_002", "type": "rule", "entity": "rule=R_NEW_ENTITY", "value": "新实体首现（card1+addr1 无历史）（card1_addr1_prior_cnt=0）：训练窗 [0,125)天 内该模式欺诈率 7.78%，为基率 3.51% 的 2.2 倍（触发 82,512 笔，其中欺诈 6,417）。", "window": [86400, 10886394], "support_n": 82512, "label_based": true}
- `RULE_003`: {"fact_id": "RULE_003", "type": "rule", "entity": "rule=R_HAS_IDENTITY", "value": "有设备/身份采集记录（id_01 存在）：训练窗 [0,125)天 内该模式欺诈率 7.44%，为基率 3.51% 的 2.1 倍（触发 112,859 笔，其中欺诈 8,397）。", "window": [86400, 10886394], "support_n": 112859, "label_based": true}
- `RULE_004`: {"fact_id": "RULE_004", "type": "rule", "entity": "rule=R_P_emaildomain_hotmail_com", "value": "高危取值 P_emaildomain=hotmail.com（P_emaildomain=hotmail.com）：训练窗 [0,125)天 内该模式欺诈率 5.32%，为基率 3.51% 的 1.5 倍（触发 34,076 笔，其中欺诈 1,813）。", "window": [86400, 10886394], "support_n": 34076, "label_based": true}
- `CASE_000`: {"fact_id": "CASE_000", "type": "case", "entity": "case=3403835", "value": "案例 #3403835（day 120）：高分假阳（人工洗清，当时模型分 0.23）。金额 $38.79（$10-50），ProductCD=C，mastercard/debit，邮箱域=hotmail.com，设备=缺失，card1=13832, addr1=284。实体历史：prior 9 笔、成熟欺诈率 0%；设备 fan-out 197。", "window": [10527695, 10527695], "label_based": true}
- `CASE_001`: {"fact_id": "CASE_001", "type": "case", "entity": "case=3398080", "value": "案例 #3398080（day 119）：高分假阳（人工洗清，当时模型分 0.97）。金额 $12.45（$10-50），ProductCD=C，mastercard/debit，邮箱域=hotmail.com，设备=缺失，card1=13832。实体历史：prior 0 笔、无成熟标签；设备 fan-out 193。", "window": [10379248, 10379248], "label_based": true}
- `CASE_002`: {"fact_id": "CASE_002", "type": "case", "entity": "case=3397474", "value": "案例 #3397474（day 118）：高分假阳（人工洗清，当时模型分 0.97）。金额 $12.45（$10-50），ProductCD=C，mastercard/debit，邮箱域=hotmail.com，设备=缺失，card1=13832。实体历史：prior 0 笔、无成熟标签；设备 fan-out 193。", "window": [10365974, 10365974], "label_based": true}
- `CASE_003`: {"fact_id": "CASE_003", "type": "case", "entity": "case=3372368", "value": "案例 #3372368（day 110）：确认欺诈（拒付举报）。金额 $16.55（$10-50），ProductCD=C，mastercard/debit，邮箱域=hotmail.com，设备=缺失，card1=13832。实体历史：prior 0 笔、无成熟标签；设备 fan-out 175。", "window": [9661252, 9661252], "label_based": true}
- `STAT_000`: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=C", "value": 0.1139, "window": [0, 13147786], "support_n": 59113, "label_based": true}
- `STAT_001`: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "P_emaildomain=hotmail.com", "value": 0.0516, "window": [0, 13147786], "support_n": 39129, "label_based": true}

</details>

## 3552038#0（断言强度=confirmed）
finding：交易本身为小额($81.09)、ProductCD=C、mastercard/debit、hotmail.com、移动设备——与命中的高危规则和相似案例的画像高度一致。
该 finding 引用的证据：
  - TXN_000: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3552038", "value": 81.092, "window": [14962186, 14962186], "label_based": false}
  - TXN_001: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3552038", "value": "C", "window": [14962186, 14962186], "label_based": false}
  - TXN_003: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3552038", "value": "mastercard", "window": [14962186, 14962186], "label_based": false}
  - TXN_004: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3552038", "value": "debit", "window": [14962186, 14962186], "label_based": false}
  - TXN_005: {"fact_id": "TXN_005", "type": "txn_field:P_emaildomain", "entity": "txn=3552038", "value": "hotmail.com", "window": [14962186, 14962186], "label_based": false}
  - TXN_006: {"fact_id": "TXN_006", "type": "txn_field:DeviceType", "entity": "txn=3552038", "value": "mobile", "window": [14962186, 14962186], "label_based": false}
reasoning_valid: true
事实层（交易属性）有据；“与规则/案例一致”所需的规则/案例值在池中有据但未引，属引用瑕疵，不判 N。
overclaim: true
预设 confirmed，但“与画像一致”为对比关系判断，需规则/案例参照，单引本笔属性最高仅 supported，超档越界

# 交易 3563867（GBDT p=0.7718）
<details><summary>本单【全部证据池】——核实被断言的值是否在整单里有据；「有据但该 finding 没引」属引用瑕疵，按 v2 不判 N</summary>

- `TXN_000`: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3563867", "value": 75.0, "window": [15349847, 15349847], "label_based": false}
- `TXN_001`: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3563867", "value": "H", "window": [15349847, 15349847], "label_based": false}
- `TXN_002`: {"fact_id": "TXN_002", "type": "txn_field:card1", "entity": "txn=3563867", "value": 10165, "window": [15349847, 15349847], "label_based": false}
- `TXN_003`: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3563867", "value": "mastercard", "window": [15349847, 15349847], "label_based": false}
- `TXN_004`: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3563867", "value": "credit", "window": [15349847, 15349847], "label_based": false}
- `TXN_005`: {"fact_id": "TXN_005", "type": "txn_field:addr1", "entity": "txn=3563867", "value": 204.0, "window": [15349847, 15349847], "label_based": false}
- `TXN_006`: {"fact_id": "TXN_006", "type": "txn_field:P_emaildomain", "entity": "txn=3563867", "value": "anonymous.com", "window": [15349847, 15349847], "label_based": false}
- `TXN_007`: {"fact_id": "TXN_007", "type": "txn_field:DeviceType", "entity": "txn=3563867", "value": "mobile", "window": [15349847, 15349847], "label_based": false}
- `TXN_008`: {"fact_id": "TXN_008", "type": "txn_field:DeviceInfo", "entity": "txn=3563867", "value": "FRD-L09 Build/HUAWEIFRD-L09", "window": [15349847, 15349847], "label_based": false}
- `GRAPH_000`: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=10165", "value": 67, "window": [0, 15349847], "label_based": false}
- `GRAPH_001`: {"fact_id": "GRAPH_001", "type": "prior_cnt", "entity": "card1=10165|addr1", "value": 1, "window": [0, 15349847], "label_based": false}
- `GRAPH_002`: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=10165", "value": 22, "window": [0, 15349847], "label_based": false}
- `GRAPH_003`: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=10165", "value": 11, "window": [0, 15349847], "label_based": false}
- `GRAPH_004`: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=10165", "value": 7, "window": [0, 15349847], "label_based": false}
- `GRAPH_005`: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=10165", "value": 0.0, "window": [0, 13535447], "label_based": true}
- `GRAPH_006`: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|addr1", "value": 0.0, "window": [0, 13535447], "label_based": true}
- `GRAPH_007`: {"fact_id": "GRAPH_007", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.0, "window": [0, 13535447], "label_based": true}
- `GRAPH_008`: {"fact_id": "GRAPH_008", "type": "gang_score", "entity": "card1=10165", "value": 0.0, "window": [0, 13535447], "label_based": true}
- `RULE_000`: {"fact_id": "RULE_000", "type": "rule", "entity": "rule=R_DeviceType_mobile", "value": "高危取值 DeviceType=mobile（DeviceType=mobile）：训练窗 [0,125)天 内该模式欺诈率 9.88%，为基率 3.51% 的 2.8 倍（触发 42,771 笔，其中欺诈 4,226）。", "window": [86400, 10886394], "support_n": 42771, "label_based": true}
- `RULE_001`: {"fact_id": "RULE_001", "type": "rule", "entity": "rule=R_HAS_IDENTITY", "value": "有设备/身份采集记录（id_01 存在）：训练窗 [0,125)天 内该模式欺诈率 7.44%，为基率 3.51% 的 2.1 倍（触发 112,859 笔，其中欺诈 8,397）。", "window": [86400, 10886394], "support_n": 112859, "label_based": true}
- `RULE_002`: {"fact_id": "RULE_002", "type": "rule", "entity": "rule=R_card6_credit", "value": "高危取值 card6=credit（card6=credit）：训练窗 [0,125)天 内该模式欺诈率 6.65%，为基率 3.51% 的 1.9 倍（触发 112,717 笔，其中欺诈 7,501）。", "window": [86400, 10886394], "support_n": 112717, "label_based": true}
- `CASE_000`: {"fact_id": "CASE_000", "type": "case", "entity": "case=3146773", "value": "案例 #3146773（day 37）：确认欺诈（拒付举报）。金额 $200.00（$100-300），ProductCD=H，mastercard/credit，邮箱域=anonymous.com，设备=Windows，card1=8223, addr1=315。实体历史：prior 0 笔、无成熟标签；设备 fan-out 4。", "window": [3359391, 3359391], "label_based": true}
- `CASE_001`: {"fact_id": "CASE_001", "type": "case", "entity": "case=3387190", "value": "案例 #3387190（day 115）：确认欺诈（拒付举报）。金额 $50.00（$50-100），ProductCD=H，visa/credit，邮箱域=anonymous.com，设备=Windows，card1=7157, addr1=296。实体历史：prior 10 笔、成熟欺诈率 0%；设备 fan-out 4。", "window": [10094654, 10094654], "label_based": true}
- `CASE_002`: {"fact_id": "CASE_002", "type": "case", "entity": "case=3283268", "value": "案例 #3283268（day 83）：确认欺诈（拒付举报）。金额 $75.00（$50-100），ProductCD=H，visa/credit，邮箱域=anonymous.com，设备=rv:58.0，card1=15063, addr1=310。实体历史：prior 38 笔、成熟欺诈率 8%；设备 fan-out 47。", "window": [7324354, 7324354], "label_based": true}
- `CASE_003`: {"fact_id": "CASE_003", "type": "case", "entity": "case=3206270", "value": "案例 #3206270（day 58）：确认欺诈（拒付举报）。金额 $50.00（$50-100），ProductCD=H，visa/credit，邮箱域=anonymous.com，设备=Windows，card1=10616, addr1=310。实体历史：prior 57 笔、成熟欺诈率 0%；设备 fan-out 47。", "window": [5154178, 5154178], "label_based": true}
- `STAT_000`: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "P_emaildomain=anonymous.com", "value": 0.0229, "window": [0, 13535447], "support_n": 32557, "label_based": true}
- `STAT_001`: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "ProductCD=H", "value": 0.047, "window": [0, 13535447], "support_n": 30786, "label_based": true}

</details>

## 3563867#1（断言强度=supported）
finding：该卡扇出为 22 地区/11 邮箱/7 设备，对 67 笔活跃卡属正常范围，且卡-地址与卡-邮箱历史欺诈率均为 0%，不构成团伙扇出特征
该 finding 引用的证据：
  - GRAPH_002: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=10165", "value": 22, "window": [0, 15349847], "label_based": false}
  - GRAPH_003: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=10165", "value": 11, "window": [0, 15349847], "label_based": false}
  - GRAPH_004: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=10165", "value": 7, "window": [0, 15349847], "label_based": false}
  - GRAPH_006: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|addr1", "value": 0.0, "window": [0, 13535447], "label_based": true}
  - GRAPH_007: {"fact_id": "GRAPH_007", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.0, "window": [0, 13535447], "label_based": true}
reasoning_valid: true
缺引的 prior_cnt=67 在池中有据（GRAPH_000），属引用瑕疵；“正常范围/不构成团伙”为多事实程度判断，未命中三类 N。
overclaim: false
预设 supported，扇出/欺诈率值可达 confirmed，但“正常范围”与“不构成团伙”需多事实综合判断，最高 supported，未越界

# 交易 3568967（GBDT p=0.0017）
<details><summary>本单【全部证据池】——核实被断言的值是否在整单里有据；「有据但该 finding 没引」属引用瑕疵，按 v2 不判 N</summary>

- `TXN_000`: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3568967", "value": 117.0, "window": [15541283, 15541283], "label_based": false}
- `TXN_001`: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3568967", "value": "W", "window": [15541283, 15541283], "label_based": false}
- `TXN_002`: {"fact_id": "TXN_002", "type": "txn_field:card1", "entity": "txn=3568967", "value": 12695, "window": [15541283, 15541283], "label_based": false}
- `TXN_003`: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3568967", "value": "visa", "window": [15541283, 15541283], "label_based": false}
- `TXN_004`: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3568967", "value": "debit", "window": [15541283, 15541283], "label_based": false}
- `TXN_005`: {"fact_id": "TXN_005", "type": "txn_field:addr1", "entity": "txn=3568967", "value": 126.0, "window": [15541283, 15541283], "label_based": false}
- `TXN_006`: {"fact_id": "TXN_006", "type": "txn_field:P_emaildomain", "entity": "txn=3568967", "value": "verizon.net", "window": [15541283, 15541283], "label_based": false}
- `GRAPH_000`: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=12695", "value": 6982, "window": [0, 15541283], "label_based": false}
- `GRAPH_001`: {"fact_id": "GRAPH_001", "type": "prior_cnt", "entity": "card1=12695|addr1", "value": 673, "window": [0, 15541283], "label_based": false}
- `GRAPH_002`: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=12695", "value": 38, "window": [0, 15541283], "label_based": false}
- `GRAPH_003`: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=12695", "value": 28, "window": [0, 15541283], "label_based": false}
- `GRAPH_004`: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=12695", "value": 41, "window": [0, 15541283], "label_based": false}
- `GRAPH_005`: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=12695", "value": 0.0254, "window": [0, 13726883], "label_based": true}
- `GRAPH_006`: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|addr1", "value": 0.0255, "window": [0, 13726883], "label_based": true}
- `GRAPH_007`: {"fact_id": "GRAPH_007", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.0, "window": [0, 13726883], "label_based": true}
- `GRAPH_008`: {"fact_id": "GRAPH_008", "type": "gang_score", "entity": "card1=12695", "value": 0.254, "window": [0, 13726883], "label_based": true}
- `CASE_000`: {"fact_id": "CASE_000", "type": "case", "entity": "case=3419828", "value": "案例 #3419828（day 125）：确认欺诈（拒付举报）。金额 $117.00（$100-300），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=12695, addr1=325。实体历史：prior 4074 笔、成熟欺诈率 2%；设备 fan-out 33。", "window": [10948383, 10948383], "label_based": true}
- `CASE_001`: {"fact_id": "CASE_001", "type": "case", "entity": "case=3389691", "value": "案例 #3389691（day 116）：高分假阳（人工洗清，当时模型分 0.15）。金额 $107.00（$100-300），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=12695, addr1=325。实体历史：prior 3777 笔、成熟欺诈率 2%；设备 fan-out 31。", "window": [10169091, 10169091], "label_based": true}
- `CASE_002`: {"fact_id": "CASE_002", "type": "case", "entity": "case=3377263", "value": "案例 #3377263（day 112）：确认欺诈（拒付举报）。金额 $206.00（$100-300），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=12695, addr1=325。实体历史：prior 3657 笔、成熟欺诈率 2%；设备 fan-out 31。", "window": [9795221, 9795221], "label_based": true}
- `CASE_003`: {"fact_id": "CASE_003", "type": "case", "entity": "case=3311366", "value": "案例 #3311366（day 91）：确认欺诈（拒付举报）。金额 $213.00（$100-300），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=12695, addr1=325。实体历史：prior 2988 笔、成熟欺诈率 2%；设备 fan-out 28。", "window": [8020179, 8020179], "label_based": true}
- `STAT_000`: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "P_emaildomain=verizon.net", "value": 0.0085, "window": [0, 13726883], "support_n": 2459, "label_based": true}
- `STAT_001`: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "ProductCD=W", "value": 0.0206, "window": [0, 13726883], "support_n": 387313, "label_based": true}
- `STAT_002`: {"fact_id": "STAT_002", "type": "fraud_rate", "entity": "card4=visa", "value": 0.0345, "window": [0, 13726883], "support_n": 340851, "label_based": true}

</details>

## 3568967#0（断言强度=supported）
finding：交易本身为低风险画像：$117 小额，ProductCD=W，visa/debit，邮箱 verizon.net。各类别历史欺诈率均在全体基率(~3.5%)及以下——verizon.net 0.85%、ProductCD=W 2.06%、visa 3.45%，无单一属性拉高风险。
该 finding 引用的证据：
  - TXN_000: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3568967", "value": 117.0, "window": [15541283, 15541283], "label_based": false}
  - TXN_001: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3568967", "value": "W", "window": [15541283, 15541283], "label_based": false}
  - TXN_003: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3568967", "value": "visa", "window": [15541283, 15541283], "label_based": false}
  - TXN_004: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3568967", "value": "debit", "window": [15541283, 15541283], "label_based": false}
  - TXN_006: {"fact_id": "TXN_006", "type": "txn_field:P_emaildomain", "entity": "txn=3568967", "value": "verizon.net", "window": [15541283, 15541283], "label_based": false}
  - STAT_000: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "P_emaildomain=verizon.net", "value": 0.0085, "window": [0, 13726883], "support_n": 2459, "label_based": true}
  - STAT_001: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "ProductCD=W", "value": 0.0206, "window": [0, 13726883], "support_n": 387313, "label_based": true}
  - STAT_002: {"fact_id": "STAT_002", "type": "fraud_rate", "entity": "card4=visa", "value": 0.0345, "window": [0, 13726883], "support_n": 340851, "label_based": true}
reasoning_valid: true
引用证据完整支撑本笔属性及其对应欺诈率，均 ≤ 基率 3.5%，未命中反证/无关/跳跃三类 N。
overclaim: false
预设 supported，属性值及 STAT 可达 confirmed，但“低风险画像”为多属性综合评估，最高 supported，未越界。