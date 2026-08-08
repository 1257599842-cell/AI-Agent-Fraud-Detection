# 富集盲标表 round2（judge-flag 筛出的少数类候选；仍是盲标，勿看 judge 结果）

背景：round1 judge κ≈0.05 且系统性偏严；本表用 judge 的 flag 当筛子放大少数类，
你独立盲标后即得 **judge-flag 精确率**（你认可几条是真问题）。同卷 rubric：
- reasoning_valid：该 finding 引用的证据能否支撑其结论？（证据→结论的推理成立=Y；跳跃、误读证据、结论与证据无关=N）
- overclaim：该 finding 的断言强度是否超过证据支撑？（confirmed 需直接证据；supported 需间接证据；把推测写成确定=Y 过度断言；恰当或偏保守=N）

共 40 条，每条两行末尾填 Y 或 N。

## 3477135#3（断言强度=tentative）
finding：相似案例两类并存需对照：同指纹（W/mastercard/debit/yahoo/同card1）既有1笔确认欺诈(#3313952)，也有3笔被人工洗清的高分假阳(#3392275/#3413955/#3377912)。'长得像欺诈'在此实体上有大量假阳先例，不足以判定本笔欺诈。
证据：
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3313952", "value": "案例 #3313952（day 92）：确认欺诈（拒付举报）。金额 $151.00（$100-300），ProductCD=W，mastercard/debit，邮箱域=yahoo.com，设备=缺失，card1=10057, addr1=269。实体历史：prior 113 笔、成熟欺诈率 20%；设备 fan-out 20。", "window": [8041434, 8041434], "label_based": true}
  - CASE_001: {"fact_id": "CASE_001", "type": "case", "entity": "case=3392275", "value": "案例 #3392275（day 117）：高分假阳（人工洗清，当时模型分 0.27）。金额 $59.00（$50-100），ProductCD=W，mastercard/debit，邮箱域=yahoo.com，设备=缺失，card1=10057, addr1=315。实体历史：prior 374 笔、成熟欺诈率 0%；设备 fan-out 22。", "window": [10246588, 10246588], "label_based": true}
  - CASE_002: {"fact_id": "CASE_002", "type": "case", "entity": "case=3413955", "value": "案例 #3413955（day 123）：高分假阳（人工洗清，当时模型分 0.16）。金额 $117.00（$100-300），ProductCD=W，mastercard/debit，邮箱域=gmail.com，设备=缺失，card1=10057, addr1=441。实体历史：prior 159 笔、成熟欺诈率 0%；设备 fan-out 22。", "window": [10789735, 10789735], "label_based": true}
  - CASE_003: {"fact_id": "CASE_003", "type": "case", "entity": "case=3377912", "value": "案例 #3377912（day 112）：高分假阳（人工洗清，当时模型分 0.22）。金额 $213.00（$100-300），ProductCD=W，mastercard/debit，邮箱域=gmail.com，设备=缺失，card1=10057, addr1=269。实体历史：prior 160 笔、成熟欺诈率 20%；设备 fan-out 22。", "window": [9819852, 9819852], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3480039#4（断言强度=supported）
finding：交易画像（ProductCD=C, visa/credit, gmail.com, Windows desktop, 小额 $31.60）与多条命中规则一致；ProductCD=C 历史欺诈率 11.27%（lift 3.2x）。
证据：
  - TXN_001: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3480039", "value": "C", "window": [12862856, 12862856], "label_based": false}
  - TXN_000: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3480039", "value": 31.601, "window": [12862856, 12862856], "label_based": false}
  - STAT_000: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=C", "value": 0.1127, "window": [0, 11048456], "support_n": 52510, "label_based": true}
  - RULE_002: {"fact_id": "RULE_002", "type": "rule", "entity": "rule=R_ProductCD_C", "value": "高危取值 ProductCD=C（ProductCD=C）：训练窗 [0,125)天 内该模式欺诈率 11.27%，为基率 3.51% 的 3.2 倍（触发 51,857 笔，其中欺诈 5,844）。", "window": [86400, 10886394], "support_n": 51857, "label_based": true}
  - RULE_005: {"fact_id": "RULE_005", "type": "rule", "entity": "rule=R_card6_credit", "value": "高危取值 card6=credit（card6=credit）：训练窗 [0,125)天 内该模式欺诈率 6.65%，为基率 3.51% 的 1.9 倍（触发 112,717 笔，其中欺诈 7,501）。", "window": [86400, 10886394], "support_n": 112717, "label_based": true}
  - RULE_006: {"fact_id": "RULE_006", "type": "rule", "entity": "rule=R_DeviceType_desktop", "value": "高危取值 DeviceType=desktop（DeviceType=desktop）：训练窗 [0,125)天 内该模式欺诈率 6.09%，为基率 3.51% 的 1.7 倍（触发 67,488 笔，其中欺诈 4,113）。", "window": [86400, 10886394], "support_n": 67488, "label_based": true}
reasoning_valid: Y
overclaim: N

## 3480336#0（断言强度=confirmed）
finding：交易金额极小（$24.90），单笔误拦损失有限，但金额小不能单独作为放行理由，因实体层面存在团伙风险。
证据：
  - TXN_000: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3480336", "value": 24.899, "window": [12868166, 12868166], "label_based": false}
reasoning_valid: Y
overclaim: N

## 3482079#4（断言强度=supported）
finding：GBDT分p=0.0757低于同型被洗清假阳案例的模型分(0.57/0.65)，与'倾向假阳'方向一致，无明显冲突；分数与我的证据判断相符。
证据：
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3385391", "value": "案例 #3385391（day 115）：高分假阳（人工洗清，当时模型分 0.57）。金额 $380.00（$300-1k），ProductCD=W，discover/credit，邮箱域=gmail.com，设备=缺失，card1=2616, addr1=325。实体历史：prior 243 笔、成熟欺诈率 6%；设备 fan-out 58。", "window": [10032153, 10032153], "label_based": true}
  - CASE_002: {"fact_id": "CASE_002", "type": "case", "entity": "case=3387641", "value": "案例 #3387641（day 115）：高分假阳（人工洗清，当时模型分 0.65）。金额 $884.00（$300-1k），ProductCD=W，discover/credit，邮箱域=yahoo.com，设备=缺失，card1=2616, addr1=325。实体历史：prior 252 笔、成熟欺诈率 6%；设备 fan-out 58。", "window": [10101609, 10101609], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3488161#2（断言强度=supported）
finding：未命中任何风控规则。
证据：
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3374123", "value": "案例 #3374123（day 111）：确认欺诈（拒付举报）。金额 $1089.00（>$1k），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=12544, addr1=476。实体历史：prior 848 笔、成熟欺诈率 4%；设备 fan-out 34。", "window": [9706569, 9706569], "label_based": true}
reasoning_valid: N
overclaim: N

## 3494664#4（断言强度=supported）
finding：上游 GBDT 分 p=0.6997 与证据冲突：各类别基率均接近或低于全体基率、实体成熟且欺诈率低、近乎相同的历史案例被人工洗清——独立证据不支持高欺诈概率结论。
证据：
  - STAT_000: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=W", "value": 0.0206, "window": [0, 11481839], "support_n": 328164, "label_based": true}
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=15066", "value": 0.0409, "window": [0, 11481839], "label_based": true}
  - CASE_002: {"fact_id": "CASE_002", "type": "case", "entity": "case=3371517", "value": "案例 #3371517（day 110）：高分假阳（人工洗清，当时模型分 0.17）。金额 $117.00（$100-300），ProductCD=W，mastercard/credit，邮箱域=gmail.com，设备=缺失，card1=15066, addr1=330。实体历史：prior 293 笔、成熟欺诈率 8%；设备 fan-out 71。", "window": [9647698, 9647698], "label_based": true}
  - STAT_001: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "card4=mastercard", "value": 0.0352, "window": [0, 11481839], "support_n": 143665, "label_based": true}
reasoning_valid: Y
overclaim: N

## 3497324#4（断言强度=tentative）
finding：上游 GBDT 分 p=0.0879 属低-中区间，与我方独立证据（干净配对、低成熟欺诈率、低gang_score）一致，无冲突。
证据：
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=11919", "value": 0.0167, "window": [0, 11553190], "label_based": true}
  - GRAPH_006: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|addr1", "value": 0.0, "window": [0, 11553190], "label_based": true}
  - GRAPH_007: {"fact_id": "GRAPH_007", "type": "gang_score", "entity": "card1=11919", "value": 0.167, "window": [0, 11553190], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3498960#5（断言强度=supported）
finding：上游 GBDT 分 p=0.0080 极低，与独立证据（低实体欺诈率、同形案例被洗清）方向一致，无冲突
证据：
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=12839", "value": 0.0161, "window": [0, 11577029], "label_based": true}
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3387860", "value": "案例 #3387860（day 115）：高分假阳（人工洗清，当时模型分 0.21）。金额 $59.00（$50-100），ProductCD=W，visa/debit，邮箱域=缺失，设备=缺失，card1=12839, addr1=264。实体历史：prior 2340 笔、成熟欺诈率 1%；设备 fan-out 29。", "window": [10105191, 10105191], "label_based": true}
  - CASE_001: {"fact_id": "CASE_001", "type": "case", "entity": "case=3383025", "value": "案例 #3383025（day 114）：高分假阳（人工洗清，当时模型分 0.15）。金额 $59.00（$50-100），ProductCD=W，visa/debit，邮箱域=缺失，设备=缺失，card1=12839, addr1=264。实体历史：prior 2304 笔、成熟欺诈率 1%；设备 fan-out 29。", "window": [9954846, 9954846], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3502069#0（断言强度=confirmed）
finding：交易金额极小（$16.72），远低于误拦成本 $25，拦错的期望损失高于放错。
证据：
  - TXN_000: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3502069", "value": 16.723, "window": [13473579, 13473579], "label_based": false}
reasoning_valid: Y
overclaim: N

## 3502069#2（断言强度=confirmed）
finding：本实体图特征干净，与团伙欺诈画像相悖：gang_score=0.0，邮箱扇出仅 2、设备扇出仅 2，无成熟欺诈标签历史。这与风险分构成显著冲突。
证据：
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "gang_score", "entity": "card1=15974", "value": 0.0, "window": [0, 11659179], "label_based": true}
  - GRAPH_003: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=15974", "value": 2, "window": [0, 13473579], "label_based": false}
  - GRAPH_004: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=15974", "value": 2, "window": [0, 13473579], "label_based": false}
  - GRAPH_000: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=15974", "value": 8, "window": [0, 13473579], "label_based": false}
reasoning_valid: Y
overclaim: N

## 3504164#2（断言强度=confirmed）
finding：同一 card1=12778 上存在确认欺诈案例 #3091774（拒付举报），金额、ProductCD=C、mastercard/credit、gmail 域与本笔高度一致，构成直接同卡欺诈先例。
证据：
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3091774", "value": "案例 #3091774（day 23）：确认欺诈（拒付举报）。金额 $16.56（$10-50），ProductCD=C，mastercard/credit，邮箱域=gmail.com，设备=缺失，card1=12778。实体历史：prior 0 笔、无成熟标签；设备 fan-out 4。", "window": [2076488, 2076488], "label_based": true}
  - TXN_001: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3504164", "value": "C", "window": [13549395, 13549395], "label_based": false}
  - TXN_003: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3504164", "value": "mastercard", "window": [13549395, 13549395], "label_based": false}
  - TXN_004: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3504164", "value": "credit", "window": [13549395, 13549395], "label_based": false}
  - TXN_005: {"fact_id": "TXN_005", "type": "txn_field:P_emaildomain", "entity": "txn=3504164", "value": "gmail.com", "window": [13549395, 13549395], "label_based": false}
reasoning_valid: Y
overclaim: N

## 3504164#5（断言强度=supported）
finding：证据冲突：上游 GBDT 风险分 p=0.0947 偏低，与实体级成熟欺诈史（15.24%/33.33%）、gang_score=1.0 及同卡确认欺诈案例严重矛盾；应以实体证据为准，模型分不可信。
证据：
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=12778", "value": 0.1524, "window": [0, 11734995], "label_based": true}
  - GRAPH_006: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.3333, "window": [0, 11734995], "label_based": true}
  - GRAPH_007: {"fact_id": "GRAPH_007", "type": "gang_score", "entity": "card1=12778", "value": 1.0, "window": [0, 11734995], "label_based": true}
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3091774", "value": "案例 #3091774（day 23）：确认欺诈（拒付举报）。金额 $16.56（$10-50），ProductCD=C，mastercard/credit，邮箱域=gmail.com，设备=缺失，card1=12778。实体历史：prior 0 笔、无成熟标签；设备 fan-out 4。", "window": [2076488, 2076488], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3505282#0（断言强度=supported）
finding：交易本体呈典型卡测试/欺诈农场画像：ProductCD=C、金额极小($15.72)、mastercard/debit、hotmail 邮箱、移动设备。
证据：
  - TXN_000: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3505282", "value": 15.722, "window": [13575020, 13575020], "label_based": false}
  - TXN_001: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3505282", "value": "C", "window": [13575020, 13575020], "label_based": false}
  - TXN_003: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3505282", "value": "mastercard", "window": [13575020, 13575020], "label_based": false}
  - TXN_004: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3505282", "value": "debit", "window": [13575020, 13575020], "label_based": false}
  - TXN_005: {"fact_id": "TXN_005", "type": "txn_field:P_emaildomain", "entity": "txn=3505282", "value": "hotmail.com", "window": [13575020, 13575020], "label_based": false}
  - TXN_006: {"fact_id": "TXN_006", "type": "txn_field:DeviceType", "entity": "txn=3505282", "value": "mobile", "window": [13575020, 13575020], "label_based": false}
reasoning_valid: N
overclaim: Y

## 3510143#0（断言强度=confirmed）
finding：交易本身金额极小（$27.05），单笔误拦成本低，产品类型为高危 ProductCD=C。
证据：
  - TXN_000: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3510143", "value": 27.051, "window": [13736132, 13736132], "label_based": false}
  - TXN_001: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3510143", "value": "C", "window": [13736132, 13736132], "label_based": false}
reasoning_valid: Y
overclaim: N

## 3510143#4（断言强度=supported）
finding：与上游 GBDT 分 0.9944 存在部分冲突：4 条几乎完全同构（同 card1=9633、C、visa/debit、gmail）的相似历史案例全部为被人工洗清的高分假阳，且其中一条实体 prior 23 笔的成熟欺诈率为 0%——说明该卡下的部分小额交易在个案层面确会被判定为合法，模型极高分对单笔的可靠性需谨慎。
证据：
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3447349", "value": "案例 #3447349（day 135）：高分假阳（人工洗清，当时模型分 0.16）。金额 $16.23（$10-50），ProductCD=C，visa/debit，邮箱域=gmail.com，设备=SM-J701M Build/NRD90M，card1=9633。实体历史：prior 0 笔、无成熟标签；设备 fan-out 336。", "window": [11826004, 11826004], "label_based": true}
  - CASE_001: {"fact_id": "CASE_001", "type": "case", "entity": "case=3447341", "value": "案例 #3447341（day 135）：高分假阳（人工洗清，当时模型分 0.30）。金额 $16.23（$10-50），ProductCD=C，visa/debit，邮箱域=gmail.com，设备=SM-J701M Build/NRD90M，card1=9633。实体历史：prior 0 笔、无成熟标签；设备 fan-out 336。", "window": [11825769, 11825769], "label_based": true}
  - CASE_002: {"fact_id": "CASE_002", "type": "case", "entity": "case=3447337", "value": "案例 #3447337（day 135）：高分假阳（人工洗清，当时模型分 0.26）。金额 $16.23（$10-50），ProductCD=C，visa/debit，邮箱域=gmail.com，设备=SM-J701M Build/NRD90M，card1=9633。实体历史：prior 0 笔、无成熟标签；设备 fan-out 336。", "window": [11825653, 11825653], "label_based": true}
  - CASE_003: {"fact_id": "CASE_003", "type": "case", "entity": "case=3439640", "value": "案例 #3439640（day 132）：高分假阳（人工洗清，当时模型分 0.42）。金额 $36.70（$10-50），ProductCD=C，visa/debit，邮箱域=gmail.com，设备=缺失，card1=9633, addr1=284。实体历史：prior 23 笔、成熟欺诈率 0%；设备 fan-out 332。", "window": [11560573, 11560573], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3512001#4（断言强度=supported）
finding：相似案例对照：4 条中 3 条为人工洗清的高分假阳（均 ProductCD=S/visa/credit/~$200），与本笔画像高度一致；唯一确认欺诈案例(CASE_003)的关键区别是其 card1 成熟欺诈率 100%，而本笔 card1 仅 2.33%——本笔更贴近假阳簇而非欺诈簇。
证据：
  - CASE_001: {"fact_id": "CASE_001", "type": "case", "entity": "case=3421835", "value": "案例 #3421835（day 126）：高分假阳（人工洗清，当时模型分 0.37）。金额 $200.00（$100-300），ProductCD=S，visa/credit，邮箱域=缺失，设备=Windows，card1=5409, addr1=330。实体历史：prior 14 笔、成熟欺诈率 0%；设备 fan-out 9。", "window": [10989278, 10989278], "label_based": true}
  - CASE_002: {"fact_id": "CASE_002", "type": "case", "entity": "case=3374785", "value": "案例 #3374785（day 111）：高分假阳（人工洗清，当时模型分 0.17）。金额 $200.00（$100-300），ProductCD=S，visa/credit，邮箱域=缺失，设备=Windows，card1=5409, addr1=330。实体历史：prior 12 笔、成熟欺诈率 0%；设备 fan-out 8。", "window": [9734644, 9734644], "label_based": true}
  - CASE_003: {"fact_id": "CASE_003", "type": "case", "entity": "case=3280739", "value": "案例 #3280739（day 82）：确认欺诈（拒付举报）。金额 $200.00（$100-300），ProductCD=S，visa/credit，邮箱域=缺失，设备=缺失，card1=3698, addr1=330。实体历史：prior 14 笔、成熟欺诈率 100%；设备 fan-out 5。", "window": [7250157, 7250157], "label_based": true}
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=2795", "value": 0.0233, "window": [0, 11989700], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3512903#6（断言强度=supported）
finding：上游 GBDT 分 p=0.7514 与独立证据（实体欺诈史、团伙分、相似确认案例）方向一致，不构成冲突，予以采信。
证据：
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=4504", "value": 0.2352, "window": [0, 12006482], "label_based": true}
  - GRAPH_007: {"fact_id": "GRAPH_007", "type": "gang_score", "entity": "card1=4504", "value": 1.0, "window": [0, 12006482], "label_based": true}
  - CASE_001: {"fact_id": "CASE_001", "type": "case", "entity": "case=3284027", "value": "案例 #3284027（day 83）：确认欺诈（拒付举报）。金额 $44.23（$10-50），ProductCD=C，mastercard/credit，邮箱域=gmail.com，设备=HUAWEI Y360-U23 Build/HUAWEIY360-U23，card1=4504。实体历史：prior 0 笔、无成熟标签；设备 fan-out 56。", "window": [7335934, 7335934], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3515431#0（断言强度=confirmed）
finding：The transaction fields themselves are benign: $59 amount, ProductCD=W, visa/debit, aol.com email. Field-level base rates are near or below the ~3.5% population baseline (aol.com 2.19%, visa 3.47%), so field values alone do not indicate fraud.
证据：
  - TXN_000: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3515431", "value": 59.0, "window": [13900184, 13900184], "label_based": false}
  - TXN_001: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3515431", "value": "W", "window": [13900184, 13900184], "label_based": false}
  - TXN_003: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3515431", "value": "visa", "window": [13900184, 13900184], "label_based": false}
  - TXN_004: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3515431", "value": "debit", "window": [13900184, 13900184], "label_based": false}
  - TXN_006: {"fact_id": "TXN_006", "type": "txn_field:P_emaildomain", "entity": "txn=3515431", "value": "aol.com", "window": [13900184, 13900184], "label_based": false}
  - STAT_000: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "P_emaildomain=aol.com", "value": 0.0219, "window": [0, 12085784], "support_n": 22420, "label_based": true}
  - STAT_001: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "card4=visa", "value": 0.0347, "window": [0, 12085784], "support_n": 305798, "label_based": true}
reasoning_valid: Y
overclaim: N

## 3516184#0（断言强度=confirmed）
finding：本笔金额仅 $21.03，属极小额，即便欺诈损失也远低于误拦成本，拦截性价比低。
证据：
  - TXN_000: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3516184", "value": 21.026, "window": [13915509, 13915509], "label_based": false}
reasoning_valid: Y
overclaim: N

## 3519778#0（断言强度=confirmed）
finding：交易本身特征温和：金额仅 $175，ProductCD=R，visa/debit，desktop/Windows，gmail 邮箱——无极端异常字段。
证据：
  - TXN_000: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3519778", "value": 175.0, "window": [14046842, 14046842], "label_based": false}
  - TXN_001: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3519778", "value": "R", "window": [14046842, 14046842], "label_based": false}
  - TXN_003: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3519778", "value": "visa", "window": [14046842, 14046842], "label_based": false}
  - TXN_004: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3519778", "value": "debit", "window": [14046842, 14046842], "label_based": false}
  - TXN_007: {"fact_id": "TXN_007", "type": "txn_field:DeviceType", "entity": "txn=3519778", "value": "desktop", "window": [14046842, 14046842], "label_based": false}
  - TXN_008: {"fact_id": "TXN_008", "type": "txn_field:DeviceInfo", "entity": "txn=3519778", "value": "Windows", "window": [14046842, 14046842], "label_based": false}
  - TXN_006: {"fact_id": "TXN_006", "type": "txn_field:P_emaildomain", "entity": "txn=3519778", "value": "gmail.com", "window": [14046842, 14046842], "label_based": false}
reasoning_valid: Y
overclaim: N

## 3523036#3（断言强度=supported）
finding：本笔字段与该实体多起确认欺诈案例高度一致：ProductCD=C、visa/credit、gmail.com、小额、同 card1=3643（案例 #3418676、#3207589、#3308995 均为拒付确认欺诈）。
证据：
  - TXN_001: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3523036", "value": "C", "window": [14132529, 14132529], "label_based": false}
  - TXN_003: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3523036", "value": "visa", "window": [14132529, 14132529], "label_based": false}
  - TXN_004: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3523036", "value": "credit", "window": [14132529, 14132529], "label_based": false}
  - TXN_005: {"fact_id": "TXN_005", "type": "txn_field:P_emaildomain", "entity": "txn=3523036", "value": "gmail.com", "window": [14132529, 14132529], "label_based": false}
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3418676", "value": "案例 #3418676（day 125）：确认欺诈（拒付举报）。金额 $74.14（$50-100），ProductCD=C，visa/credit，邮箱域=gmail.com，设备=SM-J700M Build/MMB29K，card1=3643。实体历史：prior 0 笔、无成熟标签；设备 fan-out 10。", "window": [10897346, 10897346], "label_based": true}
  - CASE_001: {"fact_id": "CASE_001", "type": "case", "entity": "case=3207589", "value": "案例 #3207589（day 58）：确认欺诈（拒付举报）。金额 $33.35（$10-50），ProductCD=C，visa/credit，邮箱域=gmail.com，设备=Moto G (4) Build/NPJ25.93-14.5，card1=3643。实体历史：prior 0 笔、无成熟标签；设备 fan-out 5。", "window": [5179980, 5179980], "label_based": true}
  - CASE_003: {"fact_id": "CASE_003", "type": "case", "entity": "case=3308995", "value": "案例 #3308995（day 91）：确认欺诈（拒付举报）。金额 $86.70（$50-100），ProductCD=C，visa/credit，邮箱域=gmail.com，设备=缺失，card1=3643。实体历史：prior 0 笔、无成熟标签；设备 fan-out 6。", "window": [8007254, 8007254], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3526727#0（断言强度=confirmed）
finding：交易金额仅 $50（debit/visa，ProductCD=R，comcast.net，Windows desktop），属小额消费，即便误判损失有限。
证据：
  - TXN_000: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3526727", "value": 50.0, "window": [14233816, 14233816], "label_based": false}
  - TXN_001: {"fact_id": "TXN_001", "type": "txn_field:ProductCD", "entity": "txn=3526727", "value": "R", "window": [14233816, 14233816], "label_based": false}
  - TXN_003: {"fact_id": "TXN_003", "type": "txn_field:card4", "entity": "txn=3526727", "value": "visa", "window": [14233816, 14233816], "label_based": false}
  - TXN_004: {"fact_id": "TXN_004", "type": "txn_field:card6", "entity": "txn=3526727", "value": "debit", "window": [14233816, 14233816], "label_based": false}
  - TXN_006: {"fact_id": "TXN_006", "type": "txn_field:P_emaildomain", "entity": "txn=3526727", "value": "comcast.net", "window": [14233816, 14233816], "label_based": false}
  - TXN_007: {"fact_id": "TXN_007", "type": "txn_field:DeviceType", "entity": "txn=3526727", "value": "desktop", "window": [14233816, 14233816], "label_based": false}
reasoning_valid: Y
overclaim: N

## 3529881#3（断言强度=supported）
finding：无风控规则命中；相似案例中三笔同 card1 家族的高分案例均被人工洗清为假阳，唯一确认欺诈案例(#3464518)画像不同（prior 仅 2 笔、成熟欺诈率 0%），与本笔长历史低欺诈率账户不匹配。
证据：
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3464518", "value": "案例 #3464518（day 142）：确认欺诈（拒付举报）。金额 $107.95（$100-300），ProductCD=W，visa/debit，邮箱域=缺失，设备=缺失，card1=2157, addr1=220。实体历史：prior 2 笔、成熟欺诈率 0%；设备 fan-out 11。", "window": [12358419, 12358419], "label_based": true}
  - CASE_001: {"fact_id": "CASE_001", "type": "case", "entity": "case=3378408", "value": "案例 #3378408（day 112）：高分假阳（人工洗清，当时模型分 0.24）。金额 $280.00（$100-300），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=2157, addr1=512。实体历史：prior 5 笔、无成熟标签；设备 fan-out 10。", "window": [9828377, 9828377], "label_based": true}
  - CASE_002: {"fact_id": "CASE_002", "type": "case", "entity": "case=3378405", "value": "案例 #3378405（day 112）：高分假阳（人工洗清，当时模型分 0.15）。金额 $265.00（$100-300），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=2157, addr1=512。实体历史：prior 4 笔、无成熟标签；设备 fan-out 10。", "window": [9828298, 9828298], "label_based": true}
  - CASE_003: {"fact_id": "CASE_003", "type": "case", "entity": "case=3378325", "value": "案例 #3378325（day 112）：高分假阳（人工洗清，当时模型分 0.18）。金额 $280.00（$100-300），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=2157, addr1=512。实体历史：prior 3 笔、无成熟标签；设备 fan-out 10。", "window": [9827163, 9827163], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3531229#1（断言强度=confirmed）
finding：card1=11849 实体成熟欺诈率 18.18%，card1+addr1 组合达 25%，均远高于全体基率约 3.5%，实体级风险显著。
证据：
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=11849", "value": 0.1818, "window": [0, 12534410], "label_based": true}
  - GRAPH_006: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|addr1", "value": 0.25, "window": [0, 12534410], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3531514#4（断言强度=supported）
finding：上游 GBDT 风险分 p=0.0010（低）与本次独立证据一致，无冲突。
证据：
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=2518", "value": 0.0, "window": [0, 12542342], "label_based": true}
  - GRAPH_008: {"fact_id": "GRAPH_008", "type": "gang_score", "entity": "card1=2518", "value": 0.0, "window": [0, 12542342], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3534425#0（断言强度=supported）
finding：上游 GBDT 分 p=0.9914 与实体证据严重冲突：实体 card1=7170 历史 70 笔、成熟欺诈率仅 1.47%（接近 3.5% 基率），gang_score=0.0，无团伙形态。模型分不被证据支持。
证据：
  - GRAPH_000: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=7170", "value": 70, "window": [0, 14442360], "label_based": false}
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=7170", "value": 0.0147, "window": [0, 12627960], "label_based": true}
  - GRAPH_009: {"fact_id": "GRAPH_009", "type": "gang_score", "entity": "card1=7170", "value": 0.0, "window": [0, 12627960], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3535590#3（断言强度=supported）
finding：命中高危规则：R_PRIOR_FRAUD_CARD1_EMAIL（欺诈率 19.11%，5.4x）、R_ProductCD_C（11.27%，3.2x）、R_NEW_ENTITY（card1+addr1 无历史，7.78%）。ProductCD=C 类别历史欺诈率 11.39%，远高于基率。
证据：
  - RULE_000: {"fact_id": "RULE_000", "type": "rule", "entity": "rule=R_PRIOR_FRAUD_CARD1_EMAIL", "value": "实体欺诈史：card1+邮箱（card1_email_prior_fraud_rate>=0.1）：训练窗 [0,125)天 内该模式欺诈率 19.11%，为基率 3.51% 的 5.4 倍（触发 16,801 笔，其中欺诈 3,210）。", "window": [86400, 10886394], "support_n": 16801, "label_based": true}
  - RULE_001: {"fact_id": "RULE_001", "type": "rule", "entity": "rule=R_ProductCD_C", "value": "高危取值 ProductCD=C（ProductCD=C）：训练窗 [0,125)天 内该模式欺诈率 11.27%，为基率 3.51% 的 3.2 倍（触发 51,857 笔，其中欺诈 5,844）。", "window": [86400, 10886394], "support_n": 51857, "label_based": true}
  - RULE_002: {"fact_id": "RULE_002", "type": "rule", "entity": "rule=R_NEW_ENTITY", "value": "新实体首现（card1+addr1 无历史）（card1_addr1_prior_cnt=0）：训练窗 [0,125)天 内该模式欺诈率 7.78%，为基率 3.51% 的 2.2 倍（触发 82,512 笔，其中欺诈 6,417）。", "window": [86400, 10886394], "support_n": 82512, "label_based": true}
  - GRAPH_001: {"fact_id": "GRAPH_001", "type": "prior_cnt", "entity": "card1=4461|addr1", "value": 0, "window": [0, 14491195], "label_based": false}
  - STAT_000: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=C", "value": 0.1139, "window": [0, 12676795], "support_n": 57657, "label_based": true}
reasoning_valid: Y
overclaim: N

## 3537070#0（断言强度=confirmed）
finding：交易金额极小（$26.95），经济止损意义有限，误拦成本($25)接近交易额本身。
证据：
  - TXN_000: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3537070", "value": 26.95, "window": [14518560, 14518560], "label_based": false}
reasoning_valid: Y
overclaim: N

## 3537316#4（断言强度=supported）
finding：CONFLICT with GBDT p=0.8242: all 4 structurally-similar retrieved cases sharing this exact fingerprint (card1=1976, ProductCD=C, visa/credit, gmail, $100-300 amt) were human-cleared false positives — including one at model score 0.65. This suggests this specific pattern is prone to false positives despite the entity's overall fraud history.
证据：
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3389587", "value": "案例 #3389587（day 116）：高分假阳（人工洗清，当时模型分 0.24）。金额 $203.62（$100-300），ProductCD=C，visa/credit，邮箱域=gmail.com，设备=缺失，card1=1976。实体历史：prior 0 笔、无成熟标签；设备 fan-out 74。", "window": [10166957, 10166957], "label_based": true}
  - CASE_001: {"fact_id": "CASE_001", "type": "case", "entity": "case=3376899", "value": "案例 #3376899（day 112）：高分假阳（人工洗清，当时模型分 0.18）。金额 $283.91（$100-300），ProductCD=C，visa/credit，邮箱域=gmail.com，设备=缺失，card1=1976。实体历史：prior 0 笔、无成熟标签；设备 fan-out 69。", "window": [9772135, 9772135], "label_based": true}
  - CASE_002: {"fact_id": "CASE_002", "type": "case", "entity": "case=3389146", "value": "案例 #3389146（day 116）：高分假阳（人工洗清，当时模型分 0.15）。金额 $407.23（$300-1k），ProductCD=C，visa/credit，邮箱域=gmail.com，设备=M4 SS4458 Build/MMB29M，card1=1976。实体历史：prior 0 笔、无成熟标签；设备 fan-out 74。", "window": [10156316, 10156316], "label_based": true}
  - CASE_003: {"fact_id": "CASE_003", "type": "case", "entity": "case=3388952", "value": "案例 #3388952（day 116）：高分假阳（人工洗清，当时模型分 0.65）。金额 $407.23（$300-1k），ProductCD=C，visa/credit，邮箱域=gmail.com，设备=M4 SS4458 Build/MMB29M，card1=1976。实体历史：prior 0 笔、无成熟标签；设备 fan-out 73。", "window": [10135789, 10135789], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3537557#0（断言强度=confirmed）
finding：交易金额极小（$16.23），单独拦截的期望收益低于误拦成本（$25），金额本身不构成拒绝理由。
证据：
  - TXN_000: {"fact_id": "TXN_000", "type": "txn_field:TransactionAmt", "entity": "txn=3537557", "value": 16.231, "window": [14541234, 14541234], "label_based": false}
reasoning_valid: Y
overclaim: N

## 3544856#5（断言强度=supported）
finding：上游 GBDT 分 p=0.6788 与独立证据方向一致（未冲突）：实体级欺诈史与团伙特征均支持高风险判定。
证据：
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=14276", "value": 0.2822, "window": [0, 12947267], "label_based": true}
  - GRAPH_008: {"fact_id": "GRAPH_008", "type": "gang_score", "entity": "card1=14276", "value": 1.0, "window": [0, 12947267], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3548749#3（断言强度=tentative）
finding：与上游冲突：GBDT p=0.0608 偏低，但实体欺诈史+扇出+与确认欺诈簇的强匹配指向更高风险，证据强于该分数所示。
证据：
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=7664", "value": 0.0429, "window": [0, 13053071], "label_based": true}
  - GRAPH_007: {"fact_id": "GRAPH_007", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.0699, "window": [0, 13053071], "label_based": true}
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3110711", "value": "案例 #3110711（day 27）：确认欺诈（拒付举报）。金额 $49.00（$10-50），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=7664, addr1=264。实体历史：prior 495 笔、成熟欺诈率 0%；设备 fan-out 21。", "window": [2426331, 2426331], "label_based": true}
  - CASE_001: {"fact_id": "CASE_001", "type": "case", "entity": "case=3088738", "value": "案例 #3088738（day 22）：确认欺诈（拒付举报）。金额 $59.00（$50-100），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=7664, addr1=264。实体历史：prior 401 笔、成熟欺诈率 0%；设备 fan-out 13。", "window": [2050432, 2050432], "label_based": true}
  - CASE_002: {"fact_id": "CASE_002", "type": "case", "entity": "case=3176734", "value": "案例 #3176734（day 48）：确认欺诈（拒付举报）。金额 $107.95（$100-300），ProductCD=W，visa/debit，邮箱域=gmail.com，设备=缺失，card1=7664, addr1=264。实体历史：prior 802 笔、成熟欺诈率 4%；设备 fan-out 23。", "window": [4235047, 4235047], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3552038#1（断言强度=confirmed）
finding：关联卡 card1=13832 呈现团伙特征：gang_score 高达 0.939，设备扇出 249（异常高），地区/邮箱扇出 17/16，且该卡的成熟欺诈率 9.39% 约为基率(3.5%)的 2.7 倍。
证据：
  - GRAPH_007: {"fact_id": "GRAPH_007", "type": "gang_score", "entity": "card1=13832", "value": 0.939, "window": [0, 13147786], "label_based": true}
  - GRAPH_004: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=13832", "value": 249, "window": [0, 14962186], "label_based": false}
  - GRAPH_002: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=13832", "value": 17, "window": [0, 14962186], "label_based": false}
  - GRAPH_003: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=13832", "value": 16, "window": [0, 14962186], "label_based": false}
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=13832", "value": 0.0939, "window": [0, 13147786], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3552830#1（断言强度=confirmed）
finding：该 card1 的成熟（已过21天拒付窗）欺诈率显著高于基率：card1 单体 16.67%、card1+设备 31.58%、card1+邮箱 10.53%，均远超全体基率约 3.5%。
证据：
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=17284", "value": 0.1667, "window": [0, 13191634], "label_based": true}
  - GRAPH_007: {"fact_id": "GRAPH_007", "type": "prior_fraud_rate", "entity": "card1|device", "value": 0.3158, "window": [0, 13191634], "label_based": true}
  - GRAPH_006: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.1053, "window": [0, 13191634], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3552830#4（断言强度=supported）
finding：对照信号：邮箱域 gmail.com 欺诈率仅 4.38%，接近基率，非风险来源；card1+addr1 为新实体首现（prior_cnt=0），风险中等（7.78%，2.2x）。
证据：
  - STAT_001: {"fact_id": "STAT_001", "type": "fraud_rate", "entity": "P_emaildomain=gmail.com", "value": 0.0438, "window": [0, 13191634], "support_n": 194717, "label_based": true}
  - GRAPH_001: {"fact_id": "GRAPH_001", "type": "prior_cnt", "entity": "card1=17284|addr1", "value": 0, "window": [0, 15006034], "label_based": false}
  - RULE_004: {"fact_id": "RULE_004", "type": "rule", "entity": "rule=R_NEW_ENTITY", "value": "新实体首现（card1+addr1 无历史）（card1_addr1_prior_cnt=0）：训练窗 [0,125)天 内该模式欺诈率 7.78%，为基率 3.51% 的 2.2 倍（触发 82,512 笔，其中欺诈 6,417）。", "window": [86400, 10886394], "support_n": 82512, "label_based": true}
reasoning_valid: Y
overclaim: N

## 3552965#4（断言强度=supported）
finding：与 GBDT 分数存在冲突：上游 p=0.5443 相对该实体 36.9% 的成熟欺诈率与满分 gang_score 明显偏低（低估），实证证据支持比模型分更高的风险。
证据：
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=3643", "value": 0.369, "window": [0, 13193832], "label_based": true}
  - GRAPH_007: {"fact_id": "GRAPH_007", "type": "gang_score", "entity": "card1=3643", "value": 1.0, "window": [0, 13193832], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3557367#4（断言强度=supported）
finding：未命中任何风控规则；上游 GBDT p=0.0050(低分)与本次证据(字段近基率、实体成熟、无规则命中)方向一致，无实质冲突。
证据：
  - STAT_000: {"fact_id": "STAT_000", "type": "fraud_rate", "entity": "ProductCD=W", "value": 0.0206, "window": [0, 13312248], "support_n": 375772, "label_based": true}
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=7207", "value": 0.0297, "window": [0, 13312248], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3560198#4（断言强度=supported）
finding：对照冲突：4 个相似案例中有 3 个（含模型分 0.96 的一例）被人工洗清为假阳（CASE_000/001/002），仅 1 例确认欺诈（CASE_003）。但被洗清案例的关键区别是当时该实体 prior_cnt=0、无成熟标签；而本笔实体已累积 152 笔交易且成熟欺诈率 32.87%，主体画像已显著恶化，不可直接沿用'假阳'结论。上游 GBDT 分 p=0.9438 与本次证据方向一致，但结论主要建立在实体级团伙证据而非模型分。
证据：
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3380462", "value": "案例 #3380462（day 113）：高分假阳（人工洗清，当时模型分 0.70）。金额 $67.37（$50-100），ProductCD=C，visa/credit，邮箱域=hotmail.com，设备=缺失，card1=5009。实体历史：prior 0 笔、无成熟标签；设备 fan-out 21。", "window": [9874150, 9874150], "label_based": true}
  - CASE_002: {"fact_id": "CASE_002", "type": "case", "entity": "case=3450522", "value": "案例 #3450522（day 136）：高分假阳（人工洗清，当时模型分 0.96）。金额 $3.57（$1-10），ProductCD=C，visa/credit，邮箱域=hotmail.com，设备=rv:33.0，card1=5009。实体历史：prior 0 笔、无成熟标签；设备 fan-out 25。", "window": [11918749, 11918749], "label_based": true}
  - CASE_003: {"fact_id": "CASE_003", "type": "case", "entity": "case=3207851", "value": "案例 #3207851（day 59）：确认欺诈（拒付举报）。金额 $65.44（$50-100），ProductCD=C，visa/credit，邮箱域=gmail.com，设备=Windows，card1=5009。实体历史：prior 0 笔、无成熟标签；设备 fan-out 15。", "window": [5184700, 5184700], "label_based": true}
  - GRAPH_000: {"fact_id": "GRAPH_000", "type": "prior_cnt", "entity": "card1=5009", "value": 152, "window": [0, 15216444], "label_based": false}
  - GRAPH_005: {"fact_id": "GRAPH_005", "type": "prior_fraud_rate", "entity": "card1=5009", "value": 0.3287, "window": [0, 13402044], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3563867#1（断言强度=supported）
finding：该卡扇出为 22 地区/11 邮箱/7 设备，对 67 笔活跃卡属正常范围，且卡-地址与卡-邮箱历史欺诈率均为 0%，不构成团伙扇出特征
证据：
  - GRAPH_002: {"fact_id": "GRAPH_002", "type": "fanout_addr1", "entity": "card1=10165", "value": 22, "window": [0, 15349847], "label_based": false}
  - GRAPH_003: {"fact_id": "GRAPH_003", "type": "fanout_email", "entity": "card1=10165", "value": 11, "window": [0, 15349847], "label_based": false}
  - GRAPH_004: {"fact_id": "GRAPH_004", "type": "fanout_device", "entity": "card1=10165", "value": 7, "window": [0, 15349847], "label_based": false}
  - GRAPH_006: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|addr1", "value": 0.0, "window": [0, 13535447], "label_based": true}
  - GRAPH_007: {"fact_id": "GRAPH_007", "type": "prior_fraud_rate", "entity": "card1|email", "value": 0.0, "window": [0, 13535447], "label_based": true}
reasoning_valid: Y
overclaim: N

## 3575604#2（断言强度=supported）
finding：3 条确认欺诈案例虽同 card1=7826，但均落在 addr1=387（其一 ProductCD=H），与本笔 addr1=325 不同；本笔 card1+addr1(325) 组合有 1120 笔历史、组合欺诈率 4.59%（仅略高于基率）。欺诈聚集在另一地区组合，本笔地区非高危聚集点。
证据：
  - CASE_000: {"fact_id": "CASE_000", "type": "case", "entity": "case=3294062", "value": "案例 #3294062（day 87）：确认欺诈（拒付举报）。金额 $29.50（$10-50），ProductCD=W，mastercard/debit，邮箱域=缺失，设备=缺失，card1=7826, addr1=325。实体历史：prior 546 笔、成熟欺诈率 1%；设备 fan-out 17。", "window": [7619738, 7619738], "label_based": true}
  - CASE_001: {"fact_id": "CASE_001", "type": "case", "entity": "case=2998397", "value": "案例 #2998397（day 2）：确认欺诈（拒付举报）。金额 $117.00（$100-300），ProductCD=W，mastercard/debit，邮箱域=gmail.com，设备=缺失，card1=7826, addr1=387。实体历史：prior 16 笔、无成熟标签；设备 fan-out 1。", "window": [336739, 336739], "label_based": true}
  - CASE_002: {"fact_id": "CASE_002", "type": "case", "entity": "case=3467140", "value": "案例 #3467140（day 143）：确认欺诈（拒付举报）。金额 $25.00（$10-50），ProductCD=H，mastercard/debit，邮箱域=anonymous.com，设备=Windows，card1=7826, addr1=387。实体历史：prior 630 笔、成熟欺诈率 2%；设备 fan-out 20。", "window": [12483504, 12483504], "label_based": true}
  - TXN_005: {"fact_id": "TXN_005", "type": "txn_field:addr1", "entity": "txn=3575604", "value": 325.0, "window": [15770484, 15770484], "label_based": false}
  - GRAPH_001: {"fact_id": "GRAPH_001", "type": "prior_cnt", "entity": "card1=7826|addr1", "value": 1120, "window": [0, 15770484], "label_based": false}
  - GRAPH_006: {"fact_id": "GRAPH_006", "type": "prior_fraud_rate", "entity": "card1|addr1", "value": 0.0459, "window": [0, 13956084], "label_based": true}
reasoning_valid: N
overclaim: N
