-- ============================================================================
-- 图特征（与 src/features/graph_features.py **逐列等价**，不新增任何特征）
--
-- 全部要点都在两处窗口帧的**类型差异**上——它就是本项目「两层防泄漏」的 SQL 表达：
--
--   结构型（prior_cnt / fan-out）：只要求「在本行之前」   → ROWS  帧（按位置）
--   标签型（obs_cnt / obs_fraud）：还要求「标签已成熟」   → RANGE 帧（按 dt 取值）
--
-- 为什么必须分别用 ROWS 和 RANGE（不是风格问题——负对照实测会差 166 行）：
--   * pandas 版 prior_cnt 取的是**组内位置**（`prior_cnt[s+i] = i`），
--     dt 并列时，排序靠前的那行**算作**后一行的 prior。
--     （全局有 17,191 行 dt 并列，但只有**同组内**并列才造成差异：
--       实测用错帧型 card1 差 166 行、card1_addr1 差 72、card1_email 差 77、card1_device 差 4。）
--     → 对应 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING，
--       且 ORDER BY 必须带与 pandas lexsort 相同的稳定次序（dt, transaction_id）。
--       用 RANGE 会把并列行整体排除，结果不一致。
--   * pandas 版 obs_cnt 取的是 `dt <= t − embargo` 的**取值条件**（two-pointer 的 thr）。
--     → 对应 RANGE BETWEEN UNBOUNDED PRECEDING AND <embargo> PRECEDING。
--       并列行按取值处理，与位置无关。
--   （pandas 里 obs 还有个 `p < i` 的位置上界；embargo > 0 时它恒不生效——
--     dt[j] <= t−embargo < t 已蕴含 j 在前——故两者等价。embargo=0 时会不等价。）
--
-- NA 语义（对齐 codes_group）：**分组键为 NULL 的行各自独立成组**，不塌成一个巨型组。
--   pandas 侧曾因此出过 bug（NA 组合键塌成巨型组），这里用 COALESCE(key, 唯一串) 复刻。
--   复合键在 pandas 里是字符串相加、任一段为 NA 则整体为 NA；SQL 的 || 同样传播 NULL，语义天然一致。
-- ============================================================================

CREATE OR REPLACE TABLE feat_base AS
SELECT
    t.transaction_id,
    t.dt,
    t.is_fraud,
    -- 四个实体键（与 pandas 的 keys 字典逐字对应）
    CAST(c.card1 AS VARCHAR)                                            AS k_card1,
    CAST(c.card1 AS VARCHAR) || '|' || CAST(a.addr1 AS VARCHAR)         AS k_card1_addr1,
    CAST(c.card1 AS VARCHAR) || '|' || e.p_emaildomain                  AS k_card1_email,
    CAST(c.card1 AS VARCHAR) || '|' || dv.device_info                   AS k_card1_device,
    -- fan-out 的 other 维度（NULL 不计入 distinct）
    CAST(a.addr1 AS VARCHAR)                                            AS o_addr1,
    e.p_emaildomain                                                     AS o_email,
    dv.device_info                                                      AS o_device
FROM fact_transaction t
JOIN dim_card    c  ON t.card_sk    = c.card_sk
JOIN dim_addr    a  ON t.addr_sk    = a.addr_sk
JOIN dim_email   e  ON t.email_sk   = e.email_sk
JOIN dim_device  dv ON t.device_sk  = dv.device_sk;

-- 分组键：NULL 各自独立成组（\x00 前缀保证不与真实键串碰撞）
CREATE OR REPLACE TABLE feat_keyed AS
SELECT *,
       COALESCE(k_card1,        '\x00NA#' || transaction_id) AS g_card1,
       COALESCE(k_card1_addr1,  '\x00NA#' || transaction_id) AS g_card1_addr1,
       COALESCE(k_card1_email,  '\x00NA#' || transaction_id) AS g_card1_email,
       COALESCE(k_card1_device, '\x00NA#' || transaction_id) AS g_card1_device
FROM feat_base;

-- ---------------------------------------------------------------- 时间因果聚合
-- ${EMBARGO_SECS} 由构建脚本注入，保持与 graph_features.py 的 EMBARGO_DAYS 同源。
CREATE OR REPLACE TABLE fact_graph_feature AS
WITH agg AS (
    SELECT
        transaction_id,

        -- ---- card1 ----
        COUNT(*)        OVER (PARTITION BY g_card1 ORDER BY dt, transaction_id
                              ROWS  BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS card1_prior_cnt,
        COUNT(*)        OVER (PARTITION BY g_card1 ORDER BY dt
                              RANGE BETWEEN UNBOUNDED PRECEDING AND ${EMBARGO_SECS} PRECEDING) AS card1_obs_cnt,
        COALESCE(SUM(is_fraud) OVER (PARTITION BY g_card1 ORDER BY dt
                              RANGE BETWEEN UNBOUNDED PRECEDING AND ${EMBARGO_SECS} PRECEDING), 0) AS card1_prior_fraud_cnt,

        -- ---- card1_addr1 ----
        COUNT(*)        OVER (PARTITION BY g_card1_addr1 ORDER BY dt, transaction_id
                              ROWS  BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS card1_addr1_prior_cnt,
        COUNT(*)        OVER (PARTITION BY g_card1_addr1 ORDER BY dt
                              RANGE BETWEEN UNBOUNDED PRECEDING AND ${EMBARGO_SECS} PRECEDING) AS card1_addr1_obs_cnt,
        COALESCE(SUM(is_fraud) OVER (PARTITION BY g_card1_addr1 ORDER BY dt
                              RANGE BETWEEN UNBOUNDED PRECEDING AND ${EMBARGO_SECS} PRECEDING), 0) AS card1_addr1_prior_fraud_cnt,

        -- ---- card1_email ----
        COUNT(*)        OVER (PARTITION BY g_card1_email ORDER BY dt, transaction_id
                              ROWS  BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS card1_email_prior_cnt,
        COUNT(*)        OVER (PARTITION BY g_card1_email ORDER BY dt
                              RANGE BETWEEN UNBOUNDED PRECEDING AND ${EMBARGO_SECS} PRECEDING) AS card1_email_obs_cnt,
        COALESCE(SUM(is_fraud) OVER (PARTITION BY g_card1_email ORDER BY dt
                              RANGE BETWEEN UNBOUNDED PRECEDING AND ${EMBARGO_SECS} PRECEDING), 0) AS card1_email_prior_fraud_cnt,

        -- ---- card1_device ----
        COUNT(*)        OVER (PARTITION BY g_card1_device ORDER BY dt, transaction_id
                              ROWS  BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS card1_device_prior_cnt,
        COUNT(*)        OVER (PARTITION BY g_card1_device ORDER BY dt
                              RANGE BETWEEN UNBOUNDED PRECEDING AND ${EMBARGO_SECS} PRECEDING) AS card1_device_obs_cnt,
        COALESCE(SUM(is_fraud) OVER (PARTITION BY g_card1_device ORDER BY dt
                              RANGE BETWEEN UNBOUNDED PRECEDING AND ${EMBARGO_SECS} PRECEDING), 0) AS card1_device_prior_fraud_cnt
    FROM feat_keyed
),
-- fan-out：组内「之前见过的 distinct other 数」。
-- 窗口函数没有 running distinct count，用**首次出现标记**代替：
--   对每个 (card1, other) 只在其首次出现那行打 1，再对 card1 组内在本行之前的标记求和，
--   即为本行之前见过的 distinct other 数。NULL 的 other 不计（与 prior_nunique 跳过 -1 一致）。
firsts AS (
    SELECT transaction_id, dt, g_card1,
           CASE WHEN o_addr1 IS NOT NULL AND ROW_NUMBER() OVER (
                    PARTITION BY g_card1, o_addr1 ORDER BY dt, transaction_id) = 1
                THEN 1 ELSE 0 END AS f_addr1,
           CASE WHEN o_email IS NOT NULL AND ROW_NUMBER() OVER (
                    PARTITION BY g_card1, o_email ORDER BY dt, transaction_id) = 1
                THEN 1 ELSE 0 END AS f_email,
           CASE WHEN o_device IS NOT NULL AND ROW_NUMBER() OVER (
                    PARTITION BY g_card1, o_device ORDER BY dt, transaction_id) = 1
                THEN 1 ELSE 0 END AS f_device
    FROM feat_keyed
),
fanout AS (
    SELECT transaction_id,
           COALESCE(SUM(f_addr1)  OVER w, 0) AS card1_fanout_addr1,
           COALESCE(SUM(f_email)  OVER w, 0) AS card1_fanout_email,
           COALESCE(SUM(f_device) OVER w, 0) AS card1_fanout_device
    FROM firsts
    WINDOW w AS (PARTITION BY g_card1 ORDER BY dt, transaction_id
                 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
)
SELECT a.transaction_id AS "TransactionID",
       a.card1_prior_cnt, a.card1_prior_fraud_cnt,
       CASE WHEN a.card1_obs_cnt > 0
            THEN a.card1_prior_fraud_cnt::DOUBLE / a.card1_obs_cnt END AS card1_prior_fraud_rate,
       a.card1_addr1_prior_cnt, a.card1_addr1_prior_fraud_cnt,
       CASE WHEN a.card1_addr1_obs_cnt > 0
            THEN a.card1_addr1_prior_fraud_cnt::DOUBLE / a.card1_addr1_obs_cnt END AS card1_addr1_prior_fraud_rate,
       a.card1_email_prior_cnt, a.card1_email_prior_fraud_cnt,
       CASE WHEN a.card1_email_obs_cnt > 0
            THEN a.card1_email_prior_fraud_cnt::DOUBLE / a.card1_email_obs_cnt END AS card1_email_prior_fraud_rate,
       a.card1_device_prior_cnt, a.card1_device_prior_fraud_cnt,
       CASE WHEN a.card1_device_obs_cnt > 0
            THEN a.card1_device_prior_fraud_cnt::DOUBLE / a.card1_device_obs_cnt END AS card1_device_prior_fraud_rate,
       f.card1_fanout_addr1, f.card1_fanout_email, f.card1_fanout_device
FROM agg a JOIN fanout f USING (transaction_id)
ORDER BY a.transaction_id;
