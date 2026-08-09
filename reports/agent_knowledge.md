# Agent 知识库（施工顺序2）—— 规则库 + 案例库 + 结构化检索

时间纪律：规则统计窗 **[0,125) 天**（=146−21，对所有 test 交易满足 <= t−21d）；案例池 ⊆ [0,146) 天，检索时再按 dt <= as_of − 21d 过滤。

## 规则库：候选 48 → 准入 15（lift>=1.5 且 n>=500 且欺诈数>=30）

| rule_id | 模式 | 欺诈率 | lift | 触发 n | 欺诈 n |
|---|---|---|---|---|---|
| R_PRIOR_FRAUD_CARD1_DEVICE | card1_device_prior_fraud_rate>=0.1 | 21.04% | 6.0× | 4,920 | 1,035 |
| R_PRIOR_FRAUD_CARD1_EMAIL | card1_email_prior_fraud_rate>=0.1 | 19.11% | 5.4× | 16,801 | 3,210 |
| R_PRIOR_FRAUD_CARD1 | card1_prior_fraud_rate>=0.1 | 17.32% | 4.9× | 21,568 | 3,735 |
| R_PRIOR_FRAUD_CARD1_ADDR1 | card1_addr1_prior_fraud_rate>=0.1 | 13.49% | 3.8× | 9,065 | 1,223 |
| R_ProductCD_C | ProductCD=C | 11.27% | 3.2× | 51,857 | 5,844 |
| R_DeviceType_mobile | DeviceType=mobile | 9.88% | 2.8× | 42,771 | 4,226 |
| R_P_emaildomain_outlook_com | P_emaildomain=outlook.com | 9.35% | 2.7× | 3,699 | 346 |
| R_C_LOW_AMT | ProductCD=C 且 TransactionAmt<10.0 | 8.54% | 2.4× | 4,112 | 351 |
| R_NEW_ENTITY | card1_addr1_prior_cnt=0 | 7.78% | 2.2× | 82,512 | 6,417 |
| R_HAS_IDENTITY | id_01 存在 | 7.44% | 2.1× | 112,859 | 8,397 |
| R_card6_credit | card6=credit | 6.65% | 1.9× | 112,717 | 7,501 |
| R_card4_discover | card4=discover | 6.49% | 1.9× | 4,929 | 320 |
| R_ProductCD_S | ProductCD=S | 6.11% | 1.7× | 7,449 | 455 |
| R_DeviceType_desktop | DeviceType=desktop | 6.09% | 1.7× | 67,488 | 4,113 |
| R_P_emaildomain_hotmail_com | P_emaildomain=hotmail.com | 5.32% | 1.5× | 34,076 | 1,813 |

<details><summary>被数据拒收的候选（诚实记录：规则是准入的，不是编的）</summary>

- R_ProductCD_W：lift 不足（lift=0.59, n=312579, fraud=6435）
- R_ProductCD_R：lift 不足（lift=1.02, n=30680, fraud=1100）
- R_ProductCD_H：lift 不足（lift=1.31, n=28560, fraud=1316）
- R_card4_visa：lift 不足（lift=0.99, n=281222, fraud=9768）
- R_card4_mastercard：lift 不足（lift=1.00, n=137285, fraud=4842）
- R_card4_american express：lift 不足（lift=0.80, n=6860, fraud=194）
- R_card6_debit：lift 不足（lift=0.68, n=317537, fraud=7625）
- R_P_emaildomain_gmail_com：lift 不足（lift=1.25, n=166389, fraud=7317）
- R_P_emaildomain_yahoo_com：lift 不足（lift=0.66, n=73677, fraud=1716）
- R_P_emaildomain_anonymous_com：lift 不足（lift=0.66, n=27686, fraud=645）
- R_P_emaildomain_aol_com：lift 不足（lift=0.60, n=20604, fraud=437）
- R_P_emaildomain_comcast_net：lift 不足（lift=0.64, n=5956, fraud=134）
- R_P_emaildomain_icloud_com：lift 不足（lift=0.89, n=4537, fraud=142）
- R_P_emaildomain_msn_com：lift 不足（lift=0.45, n=3077, fraud=49）
- R_P_emaildomain_att_net：lift 不足（lift=0.12, n=2837, fraud=12）
- R_P_emaildomain_live_com：lift 不足（lift=0.99, n=2278, fraud=79）
- R_P_emaildomain_sbcglobal_net：lift 不足（lift=0.14, n=2223, fraud=11）
- R_P_emaildomain_verizon_net：lift 不足（lift=0.16, n=2077, fraud=12）
- R_P_emaildomain_ymail_com：lift 不足（lift=0.63, n=1749, fraud=39）
- R_P_emaildomain_bellsouth_net：lift 不足（lift=0.71, n=1447, fraud=36）
- R_P_emaildomain_me_com：lift 不足（lift=0.36, n=1187, fraud=15）
- R_P_emaildomain_yahoo_com_mx：lift 不足（lift=0.36, n=1177, fraud=15）
- R_P_emaildomain_cox_net：lift 不足（lift=0.52, n=1038, fraud=19）
- R_P_emaildomain_optonline_net：lift 不足（lift=0.26, n=779, fraud=7）
- R_P_emaildomain_charter_net：lift 不足（lift=0.48, n=658, fraud=11）
- R_P_emaildomain_live_com_mx：lift 不足（lift=0.79, n=578, fraud=16）
- R_AMT_MICRO：触发样本不足（lift=15.65, n=40, fraud=22）
- R_AMT_P99：lift 不足（lift=0.73, n=4312, fraud=111）
- R_FANOUT_DEVICE：lift 不足（lift=1.47, n=175856, fraud=9113）
- R_FANOUT_EMAIL：lift 不足（lift=1.09, n=250743, fraud=9631）
- R_FANOUT_ADDR：lift 不足（lift=1.06, n=264541, fraud=9864）
- R_VELOCITY：lift 不足（lift=0.52, n=4312, fraud=79）
- R_NO_IDENTITY：lift 不足（lift=0.60, n=318266, fraud=6753）
</details>

## 案例库构成
- 正例 1,500（拒付确认；day 0~145，按月分层）；负例 1,500（模型高分假阳，分 0.14~0.99，day 104~145——负例只在有模型分的 [104,146) 挖，与「假阳来自模型上线后」的现实语义一致）。
- 诚实边界（1.2）：正例只含**被举报出来的**欺诈 = 选择性偏差样本（⑤ 在 RAG 层的翻版）；标签有传播性，同实体多案例非独立作案。

## 检索 sanity（3 笔 test 交易）

### 有实体欺诈史的 test 欺诈（txn 3475207, day 146, $54.00, isFraud=1）
- 命中规则 1 条：R_PRIOR_FRAUD_CARD1_ADDR1
- top-4 案例（sim | 结局）：#3257207（20.8 | 欺诈）；#3142027（20.8 | 欺诈）；#3367927（14.5 | 假阳）；#3367925（14.5 | 假阳）
- 时间纪律：✅ 全部 <= as_of − 21d

### 小额端 test 欺诈（txn 3531459, day 165, $2.23, isFraud=1）
- 命中规则 6 条：R_ProductCD_C, R_DeviceType_mobile, R_C_LOW_AMT, R_NEW_ENTITY, R_HAS_IDENTITY, R_card6_credit
- top-4 案例（sim | 结局）：#3469614（7.0 | 假阳）；#3469612（7.0 | 假阳）；#3463605（7.0 | 假阳）；#3426946（7.0 | 假阳）
- 时间纪律：✅ 全部 <= as_of − 21d

### 普通 test 正常交易（txn 3474937, day 146, $36.95, isFraud=0）
- 命中规则 0 条：（无）
- top-4 案例（sim | 结局）：#3387466（19.2 | 欺诈）；#3043538（11.5 | 欺诈）；#3417415（8.0 | 欺诈）；#3384999（8.0 | 假阳）
- 时间纪律：✅ 全部 <= as_of − 21d

## 待决项
- 向量粗排层（BGE/Chroma，拍板稿 1.3）本步未装：结构化相似是 1.4 拍板的主通道，20 条量级规则命中用触发器即可。等步骤4 管道跑通、看检索质量再决定是否为案例卡加向量粗排。
