-- ============================================================================
-- 小星型模型（IEEE-CIS 反欺诈）
--   事实表 1 张（交易，粒度 = 一笔交易）+ 维表 5 张 + 特征事实表 1 张。
--   目的：把现有 pandas 流水线用数仓的方式表达一遍——**不新增任何特征**。
-- ============================================================================

-- ---------- 维表：卡 ----------
CREATE OR REPLACE TABLE dim_card AS
SELECT ROW_NUMBER() OVER (ORDER BY card1, card4, card6) AS card_sk,
       card1, card4, card6
FROM (SELECT DISTINCT card1, card4, card6 FROM raw_txn);

-- ---------- 维表：地址 ----------
CREATE OR REPLACE TABLE dim_addr AS
SELECT ROW_NUMBER() OVER (ORDER BY addr1, addr2) AS addr_sk, addr1, addr2
FROM (SELECT DISTINCT addr1, addr2 FROM raw_txn);

-- ---------- 维表：邮箱域 ----------
CREATE OR REPLACE TABLE dim_email AS
SELECT ROW_NUMBER() OVER (ORDER BY p_emaildomain, r_emaildomain) AS email_sk,
       p_emaildomain, r_emaildomain
FROM (SELECT DISTINCT P_emaildomain AS p_emaildomain,
                      R_emaildomain AS r_emaildomain FROM raw_txn);

-- ---------- 维表：设备 ----------
CREATE OR REPLACE TABLE dim_device AS
SELECT ROW_NUMBER() OVER (ORDER BY device_info, device_type) AS device_sk,
       device_info, device_type
FROM (SELECT DISTINCT DeviceInfo AS device_info, DeviceType AS device_type FROM raw_txn);

-- ---------- 维表：产品 ----------
CREATE OR REPLACE TABLE dim_product AS
SELECT ROW_NUMBER() OVER (ORDER BY product_cd) AS product_sk, product_cd
FROM (SELECT DISTINCT ProductCD AS product_cd FROM raw_txn);

-- ---------- 维表：日期 ----------
-- TransactionDT 是「相对起点的秒数」，不是真实日历时间；day 从 0 起。
CREATE OR REPLACE TABLE dim_date AS
SELECT d AS date_sk, d AS day, d / 7 AS week_idx, d % 7 AS dow_idx
FROM (SELECT DISTINCT CAST(TransactionDT / 86400 AS BIGINT)
                      - (SELECT MIN(CAST(TransactionDT / 86400 AS BIGINT)) FROM raw_txn) AS d
      FROM raw_txn);

-- ---------- 事实表：交易 ----------
CREATE OR REPLACE TABLE fact_transaction AS
SELECT t.TransactionID                       AS transaction_id,
       t.TransactionDT                       AS dt,
       CAST(t.TransactionDT / 86400 AS BIGINT)
         - (SELECT MIN(CAST(TransactionDT / 86400 AS BIGINT)) FROM raw_txn) AS date_sk,
       c.card_sk, a.addr_sk, e.email_sk, dv.device_sk, p.product_sk,
       t.TransactionAmt                      AS amt,
       t.isFraud                             AS is_fraud
FROM raw_txn t
LEFT JOIN dim_card    c  ON t.card1 IS NOT DISTINCT FROM c.card1
                        AND t.card4 IS NOT DISTINCT FROM c.card4
                        AND t.card6 IS NOT DISTINCT FROM c.card6
LEFT JOIN dim_addr    a  ON t.addr1 IS NOT DISTINCT FROM a.addr1
                        AND t.addr2 IS NOT DISTINCT FROM a.addr2
LEFT JOIN dim_email   e  ON t.P_emaildomain IS NOT DISTINCT FROM e.p_emaildomain
                        AND t.R_emaildomain IS NOT DISTINCT FROM e.r_emaildomain
LEFT JOIN dim_device  dv ON t.DeviceInfo IS NOT DISTINCT FROM dv.device_info
                        AND t.DeviceType IS NOT DISTINCT FROM dv.device_type
LEFT JOIN dim_product p  ON t.ProductCD IS NOT DISTINCT FROM p.product_cd;
-- 注：维表连接一律用 IS NOT DISTINCT FROM 而非 =，否则 NULL 维度值会连不上、
--     事实行拿到 NULL 代理键。IEEE-CIS 的 addr/email/device 缺失率很高，这一点是必须的。
