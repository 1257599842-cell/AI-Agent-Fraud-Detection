# 人工锚定盲标表（先独立标完再对 judge，勿先看 judge 结果）

rubric（与 judge 逐字同卷）：
- reasoning_valid：该 finding 引用的证据能否支撑其结论？（证据→结论的推理成立=Y；跳跃、误读证据、结论与证据无关=N）
- overclaim：该 finding 的断言强度是否超过证据支撑？（confirmed 需直接证据；supported 需间接证据；把推测写成确定=Y 过度断言；恰当或偏保守=N）

每条在两行末尾填 Y 或 N。

## 3480386#1（断言强度=supported）
finding：card1=2616 + addr1 组合的成熟欺诈率高达 18.37%，规则 R_PRIOR_FRAUD_CARD1_ADDR1 也确认该模式约 13.5%（基率 3.8 倍），是本笔最主要的风险线索。
证据：
  - GRAPH_006: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|addr1", "value": 0.1837, "window": [0, 11054537], "label_based": true}
  - RULE_000: {"fact_id": "RULE_000", "type": "rule", "entity": "rule=R_PRIOR_FRAUD_CARD1_ADDR1", "value": "实体欺诈史：card1+addr1（card1_addr1_prior_fraud_rate>=0.1）：训练窗 [0,125)天 内该模式欺诈率 13.49%，为基率 3.51% 的 3.8 倍（触发 9,065 笔，其中欺诈 1,223）。", "window": [86400, 10886394], "support_n": 9065, "label_based": true}
reasoning_valid: Y
overclaim: N

## 3480871#6（断言强度=supported）
finding：GBDT p=0.6407 与证据方向一致（实体欺诈史+团伙信号），但存在同卡假阳先例，分数不宜直接作为拒绝依据；金额极小($27.97)使得单笔拦截收益有限。
证据：
  - TXN_000: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3480871", "value": 27.973, "window": [12879564, 12879564], "label_based": false}
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=4329", "value": 0.2125, "window": [0, 11065164], "label_based": true}
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3366322", "value": "案例 #3366322（day 108）：高分假阳（人工洗清，当时模型分 0.26）。金额 $32.02（$10-50），ProductCD=C，visa/credit，邮箱域=gmail.com，设备=缺失，card1=4329。实体历史：prior 0 笔、无成熟标签；设备 fan-out 12。", "window": [9492217, 9492217], "label_based": true}
  - CASE_001: {"fact_id": "CASE_001", "type": "case", "entity": "case=3363953", "value": "案例 #3363953（day 108）：高分假阳（人工洗清，当时模型分 0.18）。金额 $22.96（$10-50），ProductCD=C，visa/credit，邮箱域=gmail.com，设备=缺失，card1=4329。实体历史：prior 0 笔、无成熟标签；设备 fan-out 12。", "window": [9417717, 9417717], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3481073#2（断言强度=confirmed）
finding：命中的风控规则均为弱-中等类别信号：ProductCD=C 欺诈率 11.26%（3.2x），card6=credit 6.65%（1.9x），DeviceType=desktop 6.09%（1.7x）；这些是群体先验而非个体行为证据。
证据：
  - STAT_000: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=C", "value": 0.1126, "window": [0, 11072926], "support_n": 52651, "label_based": true}
  - RULE_003: {"fact_id": "RULE_003", "type": "rule", "entity": "rule=R_card6_credit", "value": "高危取值 card6=credit（card6=credit）：训练窗 [0,125)天 内该模式欺诈率 6.65%，为基率 3.51% 的 1.9 倍（触发 112,717 笔，其中欺诈 7,501）。", "window": [86400, 10886394], "support_n": 112717, "label_based": true}
  - RULE_004: {"fact_id": "RULE_004", "type": "rule", "entity": "rule=R_DeviceType_desktop", "value": "高危取值 DeviceType=desktop（DeviceType=desktop）：训练窗 [0,125)天 内该模式欺诈率 6.09%，为基率 3.51% 的 1.7 倍（触发 67,488 笔，其中欺诈 4,113）。", "window": [86400, 10886394], "support_n": 67488, "label_based": true}
  - TXN_001: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3481073", "value": "C", "window": [12887326, 12887326], "label_based": false}
  - TXN_004: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3481073", "value": "credit", "window": [12887326, 12887326], "label_based": false}
  - TXN_006: {"fact_id": "TXN_006", "type": "txn_field:DeviceType", "entity": "txn=3481073", "value": "desktop", "window": [12887326, 12887326], "label_based": false}
reasoning_valid: Y
overclaim: N

## 3481803#4（断言强度=supported）
finding：No risk rules were triggered, and 3 of the 4 similar historical cases were high-scoring false positives (manually cleared). The single confirmed-fraud case (CASE_000) came from an entity with 1110 priors, 1% fraud rate and device fanout 12 — a profile that does NOT match this transaction's clean, low-fanout entity.
证据：
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3414443", "value": "案例 #3414443（day 123）：确认欺诈（拒付举报）。金额 $35.00（$10-50），ProductCD=W，mastercard/debit，邮箱域=gmail.com，设备=缺失，card1=17131, addr1=264。实体历史：prior 1110 笔、成熟欺诈率 1%；设备 fan-out 12。", "window": [10796864, 10796864], "label_based": true}
  - CASE_001: {"fact_id": "CASE_001", "type": "case", "entity": "case=3413040", "value": "案例 #3413040（day 123）：高分假阳（人工洗清，当时模型分 0.17）。金额 $49.00（$10-50），ProductCD=W，mastercard/debit，邮箱域=gmail.com，设备=缺失，card1=14089, addr1=264。实体历史：prior 13 笔、成熟欺诈率 0%；设备 fan-out 1。", "window": [10777337, 10777337], "label_based": true}
  - CASE_002: {"fact_id": "CASE_002", "type": "case", "entity": "case=3392785", "value": "案例 #3392785（day 117）：高分假阳（人工洗清，当时模型分 0.28）。金额 $49.00（$10-50），ProductCD=W，mastercard/debit，邮箱域=gmail.com，设备=缺失，card1=12257, addr1=110。实体历史：prior 12 笔、成熟欺诈率 0%；设备 fan-out 0。", "window": [10256268, 10256268], "label_based": true}
  - CASE_003: {"fact_id": "CASE_003", "type": "case", "entity": "case=3387416", "value": "案例 #3387416（day 115）：高分假阳（人工洗清，当时模型分 0.25）。金额 $49.00（$10-50），ProductCD=W，mastercard/debit，邮箱域=gmail.com，设备=缺失，card1=9992, addr1=204。实体历史：prior 70 笔、成熟欺诈率 0%；设备 fan-out 5。", "window": [10098291, 10098291], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3481803#5（断言强度=supported）
finding：The independent evidence corroborates the upstream GBDT low score (p=0.0048); no conflict found.
证据：
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=15224", "value": 0.0, "window": [0, 11120915], "label_based": true}
  - STAT_000: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=W", "value": 0.0205, "window": [0, 11120915], "support_n": 318733, "label_based": true}
  - CASE_001: {"fact_id": "CASE_001", "type": "case", "entity": "case=3413040", "value": "案例 #3413040（day 123）：高分假阳（人工洗清，当时模型分 0.17）。金额 $49.00（$10-50），ProductCD=W，mastercard/debit，邮箱域=gmail.com，设备=缺失，card1=14089, addr1=264。实体历史：prior 13 笔、成熟欺诈率 0%；设备 fan-out 1。", "window": [10777337, 10777337], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3481840#3（断言强度=supported）
finding：本笔的 card1+addr1(204) 组合有 90 笔历史且欺诈率为 0%，该卡在此地区的行为记录干净。
证据：
  - TXN_005: {"fact_id": "TXN_005", "type": "txn_field:addr1", "entity": "txn=3481840", "value": 204.0, "window": [12935968, 12935968], "label_based": false}
  - GRAPH_001: {"fact_id": "GRAPH_001", "type": "prior_cnt", "entity": "card1=16326|addr1", "value": 90, "window": [0, 12935968], "label_based": false}
  - GRAPH_006: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|addr1", "value": 0.0, "window": [0, 11121568], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3481840#4（断言强度=tentative）
finding：同卡的确认欺诈案例(#3168182/#3154125)与本笔行为簇不同：它们在 addr1=299、ProductCD=R/H、金额 $75-150，而本笔 addr1=204、ProductCD=W、$445，不落入同一欺诈模式。
证据：
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3168182", "value": "案例 #3168182（day 45）：确认欺诈（拒付举报）。金额 $150.00（$100-300），ProductCD=R，visa/credit，邮箱域=gmail.com，设备=MacOS，card1=16326, addr1=299。实体历史：prior 62 笔、成熟欺诈率 2%；设备 fan-out 17。", "window": [3982912, 3982912], "label_based": true}
  - CASE_001: {"fact_id": "CASE_001", "type": "case", "entity": "case=3154125", "value": "案例 #3154125（day 40）：确认欺诈（拒付举报）。金额 $75.00（$50-100），ProductCD=H，visa/credit，邮箱域=hotmail.com，设备=Windows，card1=16326, addr1=299。实体历史：prior 61 笔、成熟欺诈率 3%；设备 fan-out 16。", "window": [3574643, 3574643], "label_based": true}
  - TXN_000: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3481840", "value": 445.0, "window": [12935968, 12935968], "label_based": false}
  - TXN_001: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3481840", "value": "W", "window": [12935968, 12935968], "label_based": false}
  - TXN_005: {"fact_id": "TXN_005", "type": "txn_field:addr1", "entity": "txn=3481840", "value": 204.0, "window": [12935968, 12935968], "label_based": false}
reasoning_valid: Y
overclaim: N

## 3484095#5（断言强度=supported）
finding：交易本身通用属性接近基率、不构成风险源：gmail 4.39%、ProductCD=W 2.05%、visa 3.47%，均在基率3.5%附近。GBDT 分0.7664高分主要由实体欺诈史解释，我的独立证据与该分数方向一致（无冲突）。
证据：
  - STAT_000: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "P_emaildomain=gmail.com", "value": 0.0439, "window": [0, 11210506], "support_n": 170490, "label_based": true}
  - STAT_001: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "ProductCD=W", "value": 0.0205, "window": [0, 11210506], "support_n": 320783, "label_based": true}
  - STAT_002: {"fact_id": "STAT_002", "type": "fraud_rate", "entity": "card4=visa", "value": 0.0347, "window": [0, 11210506], "support_n": 287805, "label_based": true}
reasoning_valid: N
overclaim: Y

## 3490572#1（断言强度=supported）
finding：card1=15066 历史成熟欺诈率 4.04%，仅略高于全体基率 3.5%，未见异常抬升；实体交易量大（6699 笔）、扇出相对交易量属中等（addr1 61 / email 39 / device 80）。
证据：
  - GRAPH_000: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=15066", "value": 6699, "window": [0, 13195950], "label_based": false}
  - GRAPH_002: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=15066", "value": 61, "window": [0, 13195950], "label_based": false}
  - GRAPH_003: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=15066", "value": 39, "window": [0, 13195950], "label_based": false}
  - GRAPH_004: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=15066", "value": 80, "window": [0, 13195950], "label_based": false}
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=15066", "value": 0.0404, "window": [0, 11381550], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3490572#5（断言强度=supported）
finding：ProductCD=H 类目欺诈率 4.62%、yahoo.com 域 2.3%，均无显著抬升；与上游 GBDT 低分 p=0.0561 一致，无证据冲突。
证据：
  - STAT_000: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=H", "value": 0.0462, "window": [0, 11381550], "support_n": 29001, "label_based": true}
  - STAT_001: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "P_emaildomain=yahoo.com", "value": 0.023, "window": [0, 11381550], "support_n": 76344, "label_based": true}
reasoning_valid: Y
overclaim: N

## 3492272#5（断言强度=supported）
finding：证据与 GBDT 低分 p=0.0527 一致，无冲突；实体级证据进一步支撑低风险判断。
证据：
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "gang_score", "entity": "card1=7702", "value": 0.0, "window": [0, 11407922], "label_based": true}
  - STAT_001: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "ProductCD=W", "value": 0.0206, "window": [0, 11407922], "support_n": 326417, "label_based": true}
reasoning_valid: Y
overclaim: N

## 3493969#0（断言强度=confirmed）
finding：关联卡 card1=9633 拥有极高历史欺诈率 16.84%（成熟标签窗口），约为全体基率3.5%的近5倍，且历史交易量大（3,673笔），非偶发。
证据：
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=9633", "value": 0.1684, "window": [0, 11468824], "label_based": true}
  - GRAPH_000: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=9633", "value": 3673, "window": [0, 13283224], "label_based": false}
  - TXN_002: {"fact_id": "TXN_002", "type": "txn_field:card1", "entity": "txn=3493969", "value": 9633, "window": [13283224, 13283224], "label_based": false}
reasoning_valid: Y
overclaim: N

## 3493969#5（断言强度=supported）
finding：同一 card1=9633 存在3笔已确认欺诈相似案例（拒付举报，profile完全一致）；虽亦有1笔被人工洗清的高分假阳，但该假阳当时模型分仅0.14且实体prior=0/无成熟标签，与本笔（实体已积累3673笔且16.84%成熟欺诈率）情形显著不同，不足以抵消风险。
证据：
  - CASE_001: {"fact_id": "CASE_001", "type": "case", "entity": "case=3214250", "value": "案例 #3214250（day 61）：确认欺诈（拒付举报）。金额 $77.34（$50-100），ProductCD=C，visa/debit，邮箱域=gmail.com，设备=Windows，card1=9633。实体历史：prior 0 笔、无成熟标签；设备 fan-out 226。", "window": [5364245, 5364245], "label_based": true}
  - CASE_002: {"fact_id": "CASE_002", "type": "case", "entity": "case=3214246", "value": "案例 #3214246（day 61）：确认欺诈（拒付举报）。金额 $77.34（$50-100），ProductCD=C，visa/debit，邮箱域=gmail.com，设备=Windows，card1=9633。实体历史：prior 0 笔、无成熟标签；设备 fan-out 226。", "window": [5364145, 5364145], "label_based": true}
  - CASE_003: {"fact_id": "CASE_003", "type": "case", "entity": "case=3214142", "value": "案例 #3214142（day 61）：确认欺诈（拒付举报）。金额 $77.34（$50-100），ProductCD=C，visa/debit，邮箱域=gmail.com，设备=Windows，card1=9633。实体历史：prior 0 笔、无成熟标签；设备 fan-out 226。", "window": [5362178, 5362178], "label_based": true}
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3387322", "value": "案例 #3387322（day 115）：高分假阳（人工洗清，当时模型分 0.14）。金额 $63.86（$50-100），ProductCD=C，visa/debit，邮箱域=gmail.com，设备=Windows，card1=9633。实体历史：prior 0 笔、无成熟标签；设备 fan-out 304。", "window": [10096729, 10096729], "label_based": true}
  - GRAPH_000: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=9633", "value": 3673, "window": [0, 13283224], "label_based": false}
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=9633", "value": 0.1684, "window": [0, 11468824], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3494922#0（断言强度=confirmed）
finding：交易金额极小（$14.33），单笔直接损失有限；此金额下误拦成本($25)已超过交易本身价值。
证据：
  - TXN_000: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3494922", "value": 14.325, "window": [13300719, 13300719], "label_based": false}
reasoning_valid: Y
overclaim: N

## 3501730#1（断言强度=supported）
finding：card1=11162 为成熟高频实体：754 笔历史交易，成熟欺诈率2.38%低于基率，card1|email 组合欺诈率为0%，无欺诈聚集迹象。
证据：
  - GRAPH_000: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=11162", "value": 754, "window": [0, 13468740], "label_based": false}
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=11162", "value": 0.0238, "window": [0, 11654340], "label_based": true}
  - GRAPH_007: {"fact_id": "GRAPH_007", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.0, "window": [0, 11654340], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3510638#0（断言强度=supported）
finding：小额交易 $82，ProductCD=W（该品类历史欺诈率2.07%，低于全体基率3.5%），mastercard/debit（3.5%，基线水平），拦截误伤成本高于潜在损失。
证据：
  - TXN_000: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3510638", "value": 82.0, "window": [13745692, 13745692], "label_based": false}
  - TXN_001: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3510638", "value": "W", "window": [13745692, 13745692], "label_based": false}
  - TXN_003: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3510638", "value": "mastercard", "window": [13745692, 13745692], "label_based": false}
  - TXN_004: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3510638", "value": "debit", "window": [13745692, 13745692], "label_based": false}
  - STAT_000: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=W", "value": 0.0207, "window": [0, 11931292], "support_n": 339398, "label_based": true}
  - STAT_002: {"fact_id": "STAT_002", "type": "fraud_rate", "entity": "card4=mastercard", "value": 0.035, "window": [0, 11931292], "support_n": 148187, "label_based": true}
reasoning_valid: Y
overclaim: N

## 3510638#1（断言强度=supported）
finding：关联卡 card1=14426 为成熟高频卡：729 笔历史交易，成熟欺诈率仅 1.29%，明显低于全体基率 3.5%，属良性主体特征。
证据：
  - GRAPH_000: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=14426", "value": 729, "window": [0, 13745692], "label_based": false}
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=14426", "value": 0.0129, "window": [0, 11931292], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3511656#4（断言强度=supported）
finding：部分单笔特征风险有限：邮箱域 gmail.com 历史欺诈率 4.4%、card4=mastercard 3.5%，均接近或等于基率，非独立风险来源；真正驱动风险的是 card1 实体本身。
证据：
  - STAT_001: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "P_emaildomain=gmail.com", "value": 0.044, "window": [0, 11982220], "support_n": 179832, "label_based": true}
  - STAT_002: {"fact_id": "STAT_002", "type": "fraud_rate", "entity": "card4=mastercard", "value": 0.035, "window": [0, 11982220], "support_n": 148527, "label_based": true}
reasoning_valid: Y
overclaim: N

## 3512319#5（断言强度=supported）
finding：上游 GBDT 分 p=0.7753 与独立证据方向一致（同为高风险），无冲突。
证据：
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=12730", "value": 0.1402, "window": [0, 11995266], "label_based": true}
  - RULE_000: {"fact_id": "RULE_000", "type": "rule", "entity": "rule=R_PRIOR_FRAUD_CARD1_EMAIL", "value": "实体欺诈史：card1+邮箱（card1_email_prior_fraud_rate>=0.1）：训练窗 [0,125)天 内该模式欺诈率 19.11%，为基率 3.51% 的 5.4 倍（触发 16,801 笔，其中欺诈 3,210）。", "window": [86400, 10886394], "support_n": 16801, "label_based": true}
reasoning_valid: Y
overclaim: N

## 3512903#1（断言强度=confirmed）
finding：card1=4504 gang_score=1.0（满分），且扇出异常：关联 89 台设备、16 个地区、15 个邮箱，呈典型团伙/批量欺诈形态。
证据：
  - GRAPH_007: {"fact_id": "GRAPH_007", "type": "gang_score", "entity": "card1=4504", "value": 1.0, "window": [0, 12006482], "label_based": true}
  - GRAPH_004: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=4504", "value": 89, "window": [0, 13820882], "label_based": false}
  - GRAPH_002: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=4504", "value": 16, "window": [0, 13820882], "label_based": false}
  - GRAPH_003: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=4504", "value": 15, "window": [0, 13820882], "label_based": false}
reasoning_valid: Y
overclaim: N

## 3512903#4（断言强度=supported）
finding：card1+addr1 无历史（新实体首现），命中新实体规则（7.78%，lift 2.2），与团伙不断更换地区/设备的模式吻合。
证据：
  - GRAPH_001: {"fact_id": "GRAPH_001", "type": "prior_cnt", "entity": "card1=4504|addr1", "value": 0, "window": [0, 13820882], "label_based": false}
  - RULE_003: {"fact_id": "RULE_003", "type": "rule", "entity": "rule=R_NEW_ENTITY", "value": "新实体首现（card1+addr1 无历史）（card1_addr1_prior_cnt=0）：训练窗 [0,125)天 内该模式欺诈率 7.78%，为基率 3.51% 的 2.2 倍（触发 82,512 笔，其中欺诈 6,417）。", "window": [86400, 10886394], "support_n": 82512, "label_based": true}
reasoning_valid: Y
overclaim: N

## 3515681#4（断言强度=supported）
finding：无风控规则命中；4条相似案例中3条为人工洗清的高分假阳（模型分0.14-0.19），仅1条确认欺诈且金额小($54)。本笔画像更贴近假阳簇。
证据：
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3407448", "value": "案例 #3407448（day 122）：高分假阳（人工洗清，当时模型分 0.14）。金额 $141.00（$100-300），ProductCD=W，visa/debit，邮箱域=yahoo.com，设备=缺失，card1=7508, addr1=420。实体历史：prior 209 笔、成熟欺诈率 1%；设备 fan-out 22。", "window": [10634612, 10634612], "label_based": true}
  - CASE_002: {"fact_id": "CASE_002", "type": "case", "entity": "case=3452535", "value": "案例 #3452535（day 137）：高分假阳（人工洗清，当时模型分 0.16）。金额 $226.00（$100-300），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=7508, addr1=220。实体历史：prior 611 笔、成熟欺诈率 5%；设备 fan-out 23。", "window": [11987725, 11987725], "label_based": true}
  - CASE_003: {"fact_id": "CASE_003", "type": "case", "entity": "case=3434064", "value": "案例 #3434064（day 130）：高分假阳（人工洗清，当时模型分 0.19）。金额 $160.50（$100-300），ProductCD=W，visa/debit，邮箱域=缺失，设备=缺失，card1=7508, addr1=220。实体历史：prior 577 笔、成熟欺诈率 6%；设备 fan-out 22。", "window": [11386852, 11386852], "label_based": true}
  - CASE_001: {"fact_id": "CASE_001", "type": "case", "entity": "case=3393822", "value": "案例 #3393822（day 117）：确认欺诈（拒付举报）。金额 $54.00（$50-100），ProductCD=W，visa/debit，邮箱域=yahoo.com，设备=缺失，card1=7508, addr1=220。实体历史：prior 517 笔、成熟欺诈率 4%；设备 fan-out 22。", "window": [10272821, 10272821], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3515768#3（断言强度=confirmed）
finding：各类别历史欺诈率均在或低于全体基率(≈3.5%)：yahoo.com 2.29%、ProductCD=W 2.07%、visa 3.47%，无任何高危类别信号。
证据：
  - STAT_000: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "P_emaildomain=yahoo.com", "value": 0.0229, "window": [0, 12092567], "support_n": 80311, "label_based": true}
  - STAT_001: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "ProductCD=W", "value": 0.0207, "window": [0, 12092567], "support_n": 343449, "label_based": true}
  - STAT_002: {"fact_id": "STAT_002", "type": "fraud_rate", "entity": "card4=visa", "value": 0.0347, "window": [0, 12092567], "support_n": 306024, "label_based": true}
reasoning_valid: Y
overclaim: N

## 3515816#1（断言强度=supported）
finding：card1=8755 呈团伙/养卡形态：gang_score=1.0，在 1022 笔历史交易上扇出到 137 台设备、17 个地区、16 个邮箱，设备扇出异常高。
证据：
  - GRAPH_007: {"fact_id": "GRAPH_007", "type": "gang_score", "entity": "card1=8755", "value": 1.0, "window": [0, 12093423], "label_based": true}
  - GRAPH_000: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=8755", "value": 1022, "window": [0, 13907823], "label_based": false}
  - GRAPH_004: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=8755", "value": 137, "window": [0, 13907823], "label_based": false}
  - GRAPH_002: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=8755", "value": 17, "window": [0, 13907823], "label_based": false}
  - GRAPH_003: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=8755", "value": 16, "window": [0, 13907823], "label_based": false}
reasoning_valid: Y
overclaim: N

## 3516609#1（断言强度=supported）
finding：Strong gang/entity-abuse signature: gang_score=1.0 with device fan-out of 152 (plus addr1 fan-out 13, email fan-out 12) on a single card1 — consistent with one card cycling many devices/identities.
证据：
  - GRAPH_007: {"fact_id": "GRAPH_007", "type": "gang_score", "entity": "card1=10568", "value": 1.0, "window": [0, 12132308], "label_based": true}
  - GRAPH_004: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=10568", "value": 152, "window": [0, 13946708], "label_based": false}
  - GRAPH_002: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=10568", "value": 13, "window": [0, 13946708], "label_based": false}
  - GRAPH_003: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=10568", "value": 12, "window": [0, 13946708], "label_based": false}
reasoning_valid: Y
overclaim: N

## 3516609#4（断言强度=supported）
finding：Transaction-attribute risk is only moderate on its own: ProductCD=C historical fraud rate 11.31% and rule lift 3.2x, DeviceType=mobile 2.8x, card6=credit 1.9x, while P_emaildomain=gmail.com is near-benign at 4.4% — so surface attributes do not by themselves justify a decline.
证据：
  - STAT_000: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=C", "value": 0.1131, "window": [0, 12132308], "support_n": 56152, "label_based": true}
  - RULE_002: {"fact_id": "RULE_002", "type": "rule", "entity": "rule=R_ProductCD_C", "value": "高危取值 ProductCD=C（ProductCD=C）：训练窗 [0,125)天 内该模式欺诈率 11.27%，为基率 3.51% 的 3.2 倍（触发 51,857 笔，其中欺诈 5,844）。", "window": [86400, 10886394], "support_n": 51857, "label_based": true}
  - RULE_003: {"fact_id": "RULE_003", "type": "rule", "entity": "rule=R_DeviceType_mobile", "value": "高危取值 DeviceType=mobile（DeviceType=mobile）：训练窗 [0,125)天 内该模式欺诈率 9.88%，为基率 3.51% 的 2.8 倍（触发 42,771 笔，其中欺诈 4,226）。", "window": [86400, 10886394], "support_n": 42771, "label_based": true}
  - RULE_006: {"fact_id": "RULE_006", "type": "rule", "entity": "rule=R_card6_credit", "value": "高危取值 card6=credit（card6=credit）：训练窗 [0,125)天 内该模式欺诈率 6.65%，为基率 3.51% 的 1.9 倍（触发 112,717 笔，其中欺诈 7,501）。", "window": [86400, 10886394], "support_n": 112717, "label_based": true}
  - STAT_001: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "P_emaildomain=gmail.com", "value": 0.044, "window": [0, 12132308], "support_n": 181726, "label_based": true}
reasoning_valid: Y
overclaim: N

## 3519769#2（断言强度=supported）
finding：团伙形态弱：gang_score=0.309，扇出适中（37 地区/16 邮箱/11 设备）相对 285 笔历史属正常，非批量欺诈形态。
证据：
  - GRAPH_009: {"fact_id": "GRAPH_009", "type": "gang_score", "entity": "card1=11690", "value": 0.309, "window": [0, 12232212], "label_based": true}
  - GRAPH_002: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=11690", "value": 37, "window": [0, 14046612], "label_based": false}
  - GRAPH_003: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=11690", "value": 16, "window": [0, 14046612], "label_based": false}
  - GRAPH_004: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=11690", "value": 11, "window": [0, 14046612], "label_based": false}
  - GRAPH_000: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=11690", "value": 285, "window": [0, 14046612], "label_based": false}
reasoning_valid: Y
overclaim: N

## 3519778#1（断言强度=supported）
finding：各类别历史欺诈率均接近基率(≈3.5%)：ProductCD=R 3.63%、gmail 4.41%、visa 3.48%，均无法解释上游 0.9986 的极高分。
证据：
  - STAT_000: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=R", "value": 0.0363, "window": [0, 12232442], "support_n": 32229, "label_based": true}
  - STAT_001: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "P_emaildomain=gmail.com", "value": 0.0441, "window": [0, 12232442], "support_n": 182910, "label_based": true}
  - STAT_002: {"fact_id": "STAT_002", "type": "fraud_rate", "entity": "card4=visa", "value": 0.0348, "window": [0, 12232442], "support_n": 308499, "label_based": true}
reasoning_valid: Y
overclaim: N

## 3521213#2（断言强度=supported）
finding：本笔card1+addr1为首次配对（prior_cnt=0），叠加超高扇出，符合该卡在多地区/设备间批量铺开的模式。
证据：
  - GRAPH_001: {"fact_id": "GRAPH_001", "type": "prior_cnt", "entity": "card1=9633|addr1", "value": 0, "window": [0, 14070619], "label_based": false}
  - RULE_002: {"fact_id": "RULE_002", "type": "rule", "entity": "rule=R_NEW_ENTITY", "value": "新实体首现（card1+addr1 无历史）（card1_addr1_prior_cnt=0）：训练窗 [0,125)天 内该模式欺诈率 7.78%，为基率 3.51% 的 2.2 倍（触发 82,512 笔，其中欺诈 6,417）。", "window": [86400, 10886394], "support_n": 82512, "label_based": true}
reasoning_valid: N
overclaim: Y

## 3523036#4（断言强度=tentative）
finding：同实体亦存在被人工洗清的高分假阳（案例 #3396790，当时模型分仅 0.28），说明该实体并非 100% 欺诈；但该假阳当时模型分低，与本笔 p=0.9463 情形不同，不足以推翻高风险判断。
证据：
  - CASE_002: {"fact_id": "CASE_002", "type": "case", "entity": "case=3396790", "value": "案例 #3396790（day 118）：高分假阳（人工洗清，当时模型分 0.28）。金额 $50.21（$50-100），ProductCD=C，visa/credit，邮箱域=gmail.com，设备=缺失，card1=3643。实体历史：prior 0 笔、无成熟标签；设备 fan-out 9。", "window": [10354739, 10354739], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3523094#5（断言强度=supported）
finding：上游GBDT分p=0.0091（极低）与独立证据一致，无冲突。
证据：
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=8256", "value": 0.0, "window": [0, 12320310], "label_based": true}
  - STAT_000: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=R", "value": 0.0362, "window": [0, 12320310], "support_n": 32364, "label_based": true}
  - TXN_000: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3523094", "value": 100.0, "window": [14134710, 14134710], "label_based": false}
reasoning_valid: Y
overclaim: N

## 3523865#2（断言强度=supported）
finding：上游 GBDT 风险分 p=0.9435 与独立实体证据一致，不存在冲突——高分由实体历史欺诈率而非单纯特征长相支撑。
证据：
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=10568", "value": 0.2016, "window": [0, 12335787], "label_based": true}
  - GRAPH_006: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.2569, "window": [0, 12335787], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3528434#1（断言强度=supported）
finding：card1+email combination shows an even higher 24.29% mature fraud rate, reinforcing that the current gmail.com pairing is high-risk despite gmail's own benign base rate.
证据：
  - GRAPH_006: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.2429, "window": [0, 12453811], "label_based": true}
  - STAT_001: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "P_emaildomain=gmail.com", "value": 0.0441, "window": [0, 12453811], "support_n": 185685, "label_based": true}
reasoning_valid: Y
overclaim: N

## 3529881#4（断言强度=supported）
finding：上游 GBDT 风险分 p=0.0015（极低）与独立证据一致，无冲突，可采信。
证据：
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=2157", "value": 0.0115, "window": [0, 12508798], "label_based": true}
  - STAT_000: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=W", "value": 0.0209, "window": [0, 12508798], "support_n": 353155, "label_based": true}
  - STAT_001: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "card4=visa", "value": 0.0348, "window": [0, 12508798], "support_n": 313909, "label_based": true}
reasoning_valid: Y
overclaim: N

## 3529936#2（断言强度=supported）
finding：GBDT风险分p=0.5136与证据冲突：属性基率与实体历史均指向低欺诈概率，模型分明显偏高。
证据：
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=11839", "value": 0.0306, "window": [0, 12509567], "label_based": true}
  - GRAPH_006: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|addr1", "value": 0.0116, "window": [0, 12509567], "label_based": true}
  - STAT_000: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=W", "value": 0.0209, "window": [0, 12509567], "support_n": 353192, "label_based": true}
  - STAT_002: {"fact_id": "STAT_002", "type": "fraud_rate", "entity": "card6=debit", "value": 0.024, "window": [0, 12509567], "support_n": 356703, "label_based": true}
reasoning_valid: Y
overclaim: N

## 3531514#3（断言强度=supported）
finding：相似确认欺诈案例的实体普遍具备高成熟欺诈率（如78%）或高设备扇出（10-28），与本笔低扇出、0%欺诈率的实体画像不符；本笔更接近被人工洗清的高分假阳模式。
证据：
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3464403", "value": "案例 #3464403（day 142）：确认欺诈（拒付举报）。金额 $39.00（$10-50），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=7664, addr1=204。实体历史：prior 27 笔、成熟欺诈率 78%；设备 fan-out 28。", "window": [12355498, 12355498], "label_based": true}
  - CASE_001: {"fact_id": "CASE_001", "type": "case", "entity": "case=3462257", "value": "案例 #3462257（day 141）：确认欺诈（拒付举报）。金额 $24.00（$10-50），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=13780, addr1=441。实体历史：prior 749 笔、成熟欺诈率 1%；设备 fan-out 10。", "window": [12273437, 12273437], "label_based": true}
  - CASE_003: {"fact_id": "CASE_003", "type": "case", "entity": "case=3450689", "value": "案例 #3450689（day 136）：高分假阳（人工洗清，当时模型分 0.18）。金额 $49.00（$10-50），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=1078, addr1=123。实体历史：prior 277 笔、成熟欺诈率 4%；设备 fan-out 6。", "window": [11921951, 11921951], "label_based": true}
  - GRAPH_004: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=2518", "value": 2, "window": [0, 14356742], "label_based": false}
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=2518", "value": 0.0, "window": [0, 12542342], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3537070#2（断言强度=supported）
finding：实体扇出与团伙形态均低：1768 笔对应 addr1 扇出 30 / email 21 / device 11，gang_score 仅 0.077，无团伙聚集特征。
证据：
  - GRAPH_002: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=17399", "value": 30, "window": [0, 14518560], "label_based": false}
  - GRAPH_003: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=17399", "value": 21, "window": [0, 14518560], "label_based": false}
  - GRAPH_004: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=17399", "value": 11, "window": [0, 14518560], "label_based": false}
  - GRAPH_008: {"fact_id": "GRAPH_008", "type": "gang_score", "entity": "card1=17399", "value": 0.077, "window": [0, 12704160], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3537070#4（断言强度=supported）
finding：同一 card1=17399 存在一笔确认欺诈(CASE_000)，但其画像不同（ProductCD=H、$300、iOS 设备、早于本笔约 143 天），且该卡成熟欺诈率仍低至 0.77%，不足以推翻本笔低风险判断；与之相对，几乎同画像(W/mastercard/gmail)的高分案例 CASE_003 曾被人工洗清为假阳。
证据：
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3092931", "value": "案例 #3092931（day 23）：确认欺诈（拒付举报）。金额 $300.00（$300-1k），ProductCD=H，mastercard/debit，邮箱域=gmail.com，设备=iOS Device，card1=17399, addr1=204。实体历史：prior 244 笔、成熟欺诈率 0%；设备 fan-out 8。", "window": [2094826, 2094826], "label_based": true}
  - CASE_003: {"fact_id": "CASE_003", "type": "case", "entity": "case=3428587", "value": "案例 #3428587（day 128）：高分假阳（人工洗清，当时模型分 0.16）。金额 $50.00（$50-100），ProductCD=W，mastercard/debit，邮箱域=gmail.com，设备=缺失，card1=17055, addr1=325。实体历史：prior 232 笔、成熟欺诈率 1%；设备 fan-out 13。", "window": [11216344, 11216344], "label_based": true}
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=17399", "value": 0.0077, "window": [0, 12704160], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3537178#4（断言强度=supported）
finding：gmail.com and visa carry no lift (4.4% and 3.5%, at/near base), so email/card-brand signals do not support fraud.
证据：
  - STAT_001: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "P_emaildomain=gmail.com", "value": 0.044, "window": [0, 12706794], "support_n": 188830, "label_based": true}
  - STAT_002: {"fact_id": "STAT_002", "type": "fraud_rate", "entity": "card4=visa", "value": 0.0347, "window": [0, 12706794], "support_n": 318526, "label_based": true}
reasoning_valid: Y
overclaim: N

## 3537316#1（断言强度=confirmed）
finding：The card1+email combination shows an even higher matured fraud rate of 34.05%, and this exact rule (R_PRIOR_FRAUD_CARD1_EMAIL) fired with 19.11% historical fraud, 5.4x lift over 16,801 supporting cases.
证据：
  - GRAPH_006: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.3405, "window": [0, 12710281], "label_based": true}
  - RULE_000: {"fact_id": "RULE_000", "type": "rule", "entity": "rule=R_PRIOR_FRAUD_CARD1_EMAIL", "value": "实体欺诈史：card1+邮箱（card1_email_prior_fraud_rate>=0.1）：训练窗 [0,125)天 内该模式欺诈率 19.11%，为基率 3.51% 的 5.4 倍（触发 16,801 笔，其中欺诈 3,210）。", "window": [86400, 10886394], "support_n": 16801, "label_based": true}
reasoning_valid: Y
overclaim: N

## 3537316#3（断言强度=supported）
finding：Transaction-level attributes (ProductCD=C at 11.4% fraud, card6=credit, DeviceType=desktop) are moderately elevated risk categories, consistent with the GBDT score direction.
证据：
  - STAT_000: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=C", "value": 0.114, "window": [0, 12710281], "support_n": 57815, "label_based": true}
  - RULE_002: {"fact_id": "RULE_002", "type": "rule", "entity": "rule=R_ProductCD_C", "value": "高危取值 ProductCD=C（ProductCD=C）：训练窗 [0,125)天 内该模式欺诈率 11.27%，为基率 3.51% 的 3.2 倍（触发 51,857 笔，其中欺诈 5,844）。", "window": [86400, 10886394], "support_n": 51857, "label_based": true}
  - RULE_005: {"fact_id": "RULE_005", "type": "rule", "entity": "rule=R_card6_credit", "value": "高危取值 card6=credit（card6=credit）：训练窗 [0,125)天 内该模式欺诈率 6.65%，为基率 3.51% 的 1.9 倍（触发 112,717 笔，其中欺诈 7,501）。", "window": [86400, 10886394], "support_n": 112717, "label_based": true}
  - RULE_006: {"fact_id": "RULE_006", "type": "rule", "entity": "rule=R_DeviceType_desktop", "value": "高危取值 DeviceType=desktop（DeviceType=desktop）：训练窗 [0,125)天 内该模式欺诈率 6.09%，为基率 3.51% 的 1.7 倍（触发 67,488 笔，其中欺诈 4,113）。", "window": [86400, 10886394], "support_n": 67488, "label_based": true}
reasoning_valid: Y
overclaim: N

## 3537557#3（断言强度=supported）
finding：该实体自身成熟欺诈率仅3.96%、card1+email 组合5.57%，仅略高于基率，实体层面并非明确高危；其9744笔历史交易表明是活跃真实卡号。
证据：
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=15885", "value": 0.0396, "window": [0, 12726834], "label_based": true}
  - GRAPH_006: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.0557, "window": [0, 12726834], "label_based": true}
  - GRAPH_000: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=15885", "value": 9744, "window": [0, 14541234], "label_based": false}
reasoning_valid: Y
overclaim: N

## 3537557#5（断言强度=tentative）
finding：上游 GBDT 分 p=0.1017 与本次核实的类别先验（约9-11%）方向一致，未发现明显矛盾，但分数无法区分该实体内部的欺诈/假阳分化。
证据：
  - STAT_000: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=C", "value": 0.1139, "window": [0, 12726834], "support_n": 57858, "label_based": true}
  - STAT_001: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "P_emaildomain=outlook.com", "value": 0.0904, "window": [0, 12726834], "support_n": 4204, "label_based": true}
  - CASE_002: {"fact_id": "CASE_002", "type": "case", "entity": "case=3442484", "value": "案例 #3442484（day 133）：高分假阳（人工洗清，当时模型分 0.36）。金额 $3.46（$1-10），ProductCD=C，visa/debit，邮箱域=outlook.com，设备=LG-D680 Build/KOT49I，card1=15885。实体历史：prior 0 笔、无成熟标签；设备 fan-out 527。", "window": [11648717, 11648717], "label_based": true}
  - CASE_003: {"fact_id": "CASE_003", "type": "case", "entity": "case=3435597", "value": "案例 #3435597（day 131）：高分假阳（人工洗清，当时模型分 0.15）。金额 $4.14（$1-10），ProductCD=C，visa/debit，邮箱域=outlook.com，设备=LG-D680 Build/KOT49I，card1=15885。实体历史：prior 0 笔、无成熟标签；设备 fan-out 524。", "window": [11422424, 11422424], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3547972#2（断言强度=supported）
finding：Abusive fan-out on a single card1: 63 devices, 15 email domains, 11 addr1 values — a bot/gang pattern rather than a single legitimate consumer.
证据：
  - GRAPH_002: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=10086", "value": 11, "window": [0, 14857168], "label_based": false}
  - GRAPH_003: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=10086", "value": 15, "window": [0, 14857168], "label_based": false}
  - GRAPH_004: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=10086", "value": 63, "window": [0, 14857168], "label_based": false}
reasoning_valid: Y
overclaim: N

## 3547972#6（断言强度=confirmed）
finding：Transaction amount is small ($35.66), which lowers per-txn loss but does not offset entity-level gang risk.
证据：
  - TXN_000: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3547972", "value": 35.658, "window": [14857168, 14857168], "label_based": false}
reasoning_valid: Y
overclaim: N

## 3548749#0（断言强度=confirmed）
finding：交易本体为小额 $48.95，ProductCD=W、visa/debit、gmail.com，字段本身平平；各类别历史欺诈率接近或略高于全体基率(gmail 4.38%、W 2.08%、visa 3.45%)，单看类别不构成强信号。
证据：
  - TXN_000: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3548749", "value": 48.95, "window": [14867471, 14867471], "label_based": false}
  - TXN_001: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3548749", "value": "W", "window": [14867471, 14867471], "label_based": false}
  - TXN_003: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3548749", "value": "visa", "window": [14867471, 14867471], "label_based": false}
  - TXN_004: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3548749", "value": "debit", "window": [14867471, 14867471], "label_based": false}
  - TXN_006: {"fact_id": "TXN_006", "type": "txn_field:P_emaildomain", "entity": "txn=3548749", "value": "gmail.com", "window": [14867471, 14867471], "label_based": false}
  - STAT_000: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "P_emaildomain=gmail.com", "value": 0.0438, "window": [0, 13053071], "support_n": 192849, "label_based": true}
  - STAT_001: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "ProductCD=W", "value": 0.0208, "window": [0, 13053071], "support_n": 367287, "label_based": true}
  - STAT_002: {"fact_id": "STAT_002", "type": "fraud_rate", "entity": "card4=visa", "value": 0.0345, "window": [0, 13053071], "support_n": 325411, "label_based": true}
reasoning_valid: Y
overclaim: N

## 3550069#0（断言强度=supported）
finding：上游 GBDT 风险分 p=0.0046（极低）与实体级证据严重冲突：该卡实体历史成熟欺诈率 5.16%、card1|email 组合达 7.93%，均高于全体基率 ~3.5%，且团伙形态分 gang_score=0.516（中高）。模型分不可采信为放行依据。
证据：
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=7585", "value": 0.0516, "window": [0, 13108895], "label_based": true}
  - GRAPH_007: {"fact_id": "GRAPH_007", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.0793, "window": [0, 13108895], "label_based": true}
  - GRAPH_008: {"fact_id": "GRAPH_008", "type": "gang_score", "entity": "card1=7585", "value": 0.516, "window": [0, 13108895], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3550069#4（断言强度=tentative）
finding：反向对照：该卡 5124 笔中约 95% 为非欺诈，gang_score 仅 0.516（中等非极端），且本次未返回被洗清的高分假阳案例作为对照，故无法将本笔单独 confirmed 为欺诈，存在合法交易可能。
证据：
  - GRAPH_000: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=7585", "value": 5124, "window": [0, 14923295], "label_based": false}
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=7585", "value": 0.0516, "window": [0, 13108895], "label_based": true}
  - GRAPH_008: {"fact_id": "GRAPH_008", "type": "gang_score", "entity": "card1=7585", "value": 0.516, "window": [0, 13108895], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3551044#2（断言强度=tentative）
finding：对照假阳案例：唯一被人工洗清的高分案例 #3454902 在设备上与本笔不同（移动端 SM-G925I，而非 Windows 桌面），本笔匹配的是欺诈簇而非被洗清簇，降低了假阳可能。
证据：
  - CASE_003: {"fact_id": "CASE_003", "type": "case", "entity": "case=3454902", "value": "案例 #3454902（day 138）：高分假阳（人工洗清，当时模型分 0.17）。金额 $73.65（$50-100），ProductCD=C，mastercard/debit，邮箱域=gmail.com，设备=SM-G925I Build/NRD90M，card1=4461。实体历史：prior 0 笔、无成熟标签；设备 fan-out 243。", "window": [12066211, 12066211], "label_based": true}
  - TXN_006: {"fact_id": "TXN_006", "type": "txn_field:DeviceType", "entity": "txn=3551044", "value": "desktop", "window": [14940839, 14940839], "label_based": false}
  - TXN_007: {"fact_id": "TXN_007", "type": "txn_field:DeviceInfo", "entity": "txn=3551044", "value": "Windows", "window": [14940839, 14940839], "label_based": false}
reasoning_valid: N
overclaim: N

## 3552038#3（断言强度=supported）
finding：对照冲突：同一 card1 也有 3 笔被人工洗清的高分假阳(#3403835/#3398080/#3397474)，其中两笔当时模型分 0.97 仍被判定为非欺诈；说明该卡上单笔交易的欺诈标签混杂，'长得像'不等于欺诈。
证据：
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3403835", "value": "案例 #3403835（day 120）：高分假阳（人工洗清，当时模型分 0.23）。金额 $38.79（$10-50），ProductCD=C，mastercard/debit，邮箱域=hotmail.com，设备=缺失，card1=13832, addr1=284。实体历史：prior 9 笔、成熟欺诈率 0%；设备 fan-out 197。", "window": [10527695, 10527695], "label_based": true}
  - CASE_001: {"fact_id": "CASE_001", "type": "case", "entity": "case=3398080", "value": "案例 #3398080（day 119）：高分假阳（人工洗清，当时模型分 0.97）。金额 $12.45（$10-50），ProductCD=C，mastercard/debit，邮箱域=hotmail.com，设备=缺失，card1=13832。实体历史：prior 0 笔、无成熟标签；设备 fan-out 193。", "window": [10379248, 10379248], "label_based": true}
  - CASE_002: {"fact_id": "CASE_002", "type": "case", "entity": "case=3397474", "value": "案例 #3397474（day 118）：高分假阳（人工洗清，当时模型分 0.97）。金额 $12.45（$10-50），ProductCD=C，mastercard/debit，邮箱域=hotmail.com，设备=缺失，card1=13832。实体历史：prior 0 笔、无成熟标签；设备 fan-out 193。", "window": [10365974, 10365974], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3552038#5（断言强度=tentative）
finding：GBDT 风险分 0.738 方向上与实体高风险一致，但该卡历史同时含确认欺诈与被洗清假阳，故不能仅凭该分数或单笔画像直接判欺诈——证据支持'实体级风险'强于'本笔已欺诈'。
证据：
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=13832", "value": 0.0939, "window": [0, 13147786], "label_based": true}
  - CASE_003: {"fact_id": "CASE_003", "type": "case", "entity": "case=3372368", "value": "案例 #3372368（day 110）：确认欺诈（拒付举报）。金额 $16.55（$10-50），ProductCD=C，mastercard/debit，邮箱域=hotmail.com，设备=缺失，card1=13832。实体历史：prior 0 笔、无成熟标签；设备 fan-out 175。", "window": [9661252, 9661252], "label_based": true}
  - CASE_001: {"fact_id": "CASE_001", "type": "case", "entity": "case=3398080", "value": "案例 #3398080（day 119）：高分假阳（人工洗清，当时模型分 0.97）。金额 $12.45（$10-50），ProductCD=C，mastercard/debit，邮箱域=hotmail.com，设备=缺失，card1=13832。实体历史：prior 0 笔、无成熟标签；设备 fan-out 193。", "window": [10379248, 10379248], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3552965#1（断言强度=confirmed）
finding：该 card1 gang_score=1.0（满分），且扇出异常：19 台设备、11 个邮箱、4 个 addr1 共用一张卡，符合团伙批量作案形态。
证据：
  - GRAPH_007: {"fact_id": "GRAPH_007", "type": "gang_score", "entity": "card1=3643", "value": 1.0, "window": [0, 13193832], "label_based": true}
  - GRAPH_004: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=3643", "value": 19, "window": [0, 15008232], "label_based": false}
  - GRAPH_003: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=3643", "value": 11, "window": [0, 15008232], "label_based": false}
  - GRAPH_002: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=3643", "value": 4, "window": [0, 15008232], "label_based": false}
reasoning_valid: Y
overclaim: N

## 3554924#1（断言强度=supported）
finding：ProductCD=C 属高危类别，独立核实的历史欺诈率为 11.38%（约为基率3.5%的3.2倍），这与上游GBDT给出的低分 p=0.0186 存在明显冲突——单看类别该笔并不低风险。
证据：
  - TXN_001: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3554924", "value": "C", "window": [15058351, 15058351], "label_based": false}
  - STAT_000: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=C", "value": 0.1138, "window": [0, 13243951], "support_n": 59422, "label_based": true}
  - RULE_000: {"fact_id": "RULE_000", "type": "rule", "entity": "rule=R_ProductCD_C", "value": "高危取值 ProductCD=C（ProductCD=C）：训练窗 [0,125)天 内该模式欺诈率 11.27%，为基率 3.51% 的 3.2 倍（触发 51,857 笔，其中欺诈 5,844）。", "window": [86400, 10886394], "support_n": 51857, "label_based": true}
reasoning_valid: Y
overclaim: N

## 3556671#4（断言强度=tentative）
finding：card1+邮箱 历史欺诈率 14.29%（边界触发 R_PRIOR_FRAUD_CARD1_EMAIL），高于基率但同样受小样本影响，为次要辅证。
证据：
  - GRAPH_007: {"fact_id": "GRAPH_007", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.1429, "window": [0, 13296516], "label_based": true}
  - RULE_000: {"fact_id": "RULE_000", "type": "rule", "entity": "rule=R_PRIOR_FRAUD_CARD1_EMAIL", "value": "实体欺诈史：card1+邮箱（card1_email_prior_fraud_rate>=0.1）：训练窗 [0,125)天 内该模式欺诈率 19.11%，为基率 3.51% 的 5.4 倍（触发 16,801 笔，其中欺诈 3,210）。", "window": [86400, 10886394], "support_n": 16801, "label_based": true}
reasoning_valid: Y
overclaim: N

## 3558979#2（断言强度=supported）
finding：上游 GBDT 分 p=0.1203 高于基率，但金额仅 $49；相似案例显示本实体的小额交易（$77/$92）均被人工洗清为假阳，与本笔 $49 画像高度吻合，说明分数偏高更可能是实体高活跃度导致的假阳倾向。
证据：
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3441552", "value": "案例 #3441552（day 133）：高分假阳（人工洗清，当时模型分 0.20）。金额 $92.00（$50-100），ProductCD=W，mastercard/credit，邮箱域=gmail.com，设备=缺失，card1=15066, addr1=204。实体历史：prior 549 笔、成熟欺诈率 2%；设备 fan-out 78。", "window": [11632497, 11632497], "label_based": true}
  - CASE_003: {"fact_id": "CASE_003", "type": "case", "entity": "case=3442492", "value": "案例 #3442492（day 133）：高分假阳（人工洗清，当时模型分 0.16）。金额 $77.00（$50-100），ProductCD=W，mastercard/credit，邮箱域=hotmail.com，设备=缺失，card1=15066, addr1=204。实体历史：prior 553 笔、成熟欺诈率 2%；设备 fan-out 78。", "window": [11648849, 11648849], "label_based": true}
  - TXN_000: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3558979", "value": 49.0, "window": [15191515, 15191515], "label_based": false}
reasoning_valid: Y
overclaim: N

## 3560198#1（断言强度=supported）
finding：card1=5009 呈现明显扇出/团伙形态：关联 5 个地区、9 个邮箱域、29 台设备，与单一持卡人正常行为不符，指向共享卡资源的团伙操作。
证据：
  - GRAPH_002: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=5009", "value": 5, "window": [0, 15216444], "label_based": false}
  - GRAPH_003: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=5009", "value": 9, "window": [0, 15216444], "label_based": false}
  - GRAPH_004: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=5009", "value": 29, "window": [0, 15216444], "label_based": false}
reasoning_valid: Y
overclaim: N

## 3562477#1（断言强度=supported）
finding：The card1|addr1 combination (186 priors) has a 0% mature fraud rate, and this addr1=264 is consistent with the card's established footprint.
证据：
  - GRAPH_001: {"fact_id": "GRAPH_001", "type": "prior_cnt", "entity": "card1=6550|addr1", "value": 186, "window": [0, 15287066], "label_based": false}
  - GRAPH_006: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|addr1", "value": 0.0, "window": [0, 13472666], "label_based": true}
  - TXN_005: {"fact_id": "TXN_005", "type": "txn_field:addr1", "entity": "txn=3562477", "value": 264.0, "window": [15287066, 15287066], "label_based": false}
reasoning_valid: Y
overclaim: N

## 3568042#4（断言强度=tentative）
finding：相似案例中该交易画像（W/visa/debit/gmail/设备缺失/$300-1k）同时匹配确认欺诈与被洗清假阳两类，'长得像欺诈'不成立独立结论；本笔实体历史干净是关键区分点。
证据：
  - CASE_001: {"fact_id": "CASE_001", "type": "case", "entity": "case=3470839", "value": "案例 #3470839（day 144）：确认欺诈（拒付举报）。金额 $554.00（$300-1k），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=11207, addr1=126。实体历史：prior 755 笔、成熟欺诈率 1%；设备 fan-out 30。", "window": [12592495, 12592495], "label_based": true}
  - CASE_003: {"fact_id": "CASE_003", "type": "case", "entity": "case=3452151", "value": "案例 #3452151（day 137）：确认欺诈（拒付举报）。金额 $445.00（$300-1k），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=3326, addr1=485。实体历史：prior 95 笔、成熟欺诈率 0%；设备 fan-out 2。", "window": [11979298, 11979298], "label_based": true}
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3471687", "value": "案例 #3471687（day 144）：高分假阳（人工洗清，当时模型分 0.15）。金额 $412.00（$300-1k），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=12544, addr1=494。实体历史：prior 378 笔、成熟欺诈率 0%；设备 fan-out 39。", "window": [12610608, 12610608], "label_based": true}
  - CASE_002: {"fact_id": "CASE_002", "type": "case", "entity": "case=3469544", "value": "案例 #3469544（day 144）：高分假阳（人工洗清，当时模型分 0.23）。金额 $445.00（$300-1k），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=17868, addr1=472。实体历史：prior 420 笔、成熟欺诈率 0%；设备 fan-out 9。", "window": [12535896, 12535896], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3568967#0（断言强度=supported）
finding：交易本身为低风险画像：$117 小额，ProductCD=W，visa/debit，邮箱 verizon.net。各类别历史欺诈率均在全体基率(~3.5%)及以下——verizon.net 0.85%、ProductCD=W 2.06%、visa 3.45%，无单一属性拉高风险。
证据：
  - TXN_000: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3568967", "value": 117.0, "window": [15541283, 15541283], "label_based": false}
  - TXN_001: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3568967", "value": "W", "window": [15541283, 15541283], "label_based": false}
  - TXN_003: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3568967", "value": "visa", "window": [15541283, 15541283], "label_based": false}
  - TXN_004: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3568967", "value": "debit", "window": [15541283, 15541283], "label_based": false}
  - TXN_006: {"fact_id": "TXN_006", "type": "txn_field:P_emaildomain", "entity": "txn=3568967", "value": "verizon.net", "window": [15541283, 15541283], "label_based": false}
  - STAT_000: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "P_emaildomain=verizon.net", "value": 0.0085, "window": [0, 13726883], "support_n": 2459, "label_based": true}
  - STAT_001: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "ProductCD=W", "value": 0.0206, "window": [0, 13726883], "support_n": 387313, "label_based": true}
  - STAT_002: {"fact_id": "STAT_002", "type": "fraud_rate", "entity": "card4=visa", "value": 0.0345, "window": [0, 13726883], "support_n": 340851, "label_based": true}
reasoning_valid: Y
overclaim: N

## 3572418#0（断言强度=supported）
finding：本笔卡实体 card1=3698 历史成熟欺诈率 26.76%（21天拒付窗已成熟），约为基率 3.5% 的 7.6 倍，且有 77 笔历史交易样本充足，命中规则 R_PRIOR_FRAUD_CARD1（4.9x lift）。这是本笔最强风险证据。
证据：
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=3698", "value": 0.2676, "window": [0, 13832593], "label_based": true}
  - GRAPH_000: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=3698", "value": 77, "window": [0, 15646993], "label_based": false}
  - RULE_000: {"fact_id": "RULE_000", "type": "rule", "entity": "rule=R_PRIOR_FRAUD_CARD1", "value": "实体欺诈史：card1（card1_prior_fraud_rate>=0.1）：训练窗 [0,125)天 内该模式欺诈率 17.32%，为基率 3.51% 的 4.9 倍（触发 21,568 笔，其中欺诈 3,735）。", "window": [86400, 10886394], "support_n": 21568, "label_based": true}
reasoning_valid: Y
overclaim: N
