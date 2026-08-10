"""在线特征存储：双时间戳事件日志 + 点查聚合。

## 它修的是什么
`/score` 此前接受 `transaction_id`、去离线表里查一个算好的 `p`——**那不是打分，是查表**。
本模块让服务能对**没见过的交易**真正算出特征、真正打分。

## 双时间戳

| 字段 | 含义 |
|---|---|
| `event_time` | 交易**发生**的时刻 |
| `label_time` | 标签**变为已知**的时刻；**可以为 NULL** |

结构型特征只看 `event_time`；标签型特征还要求 `label_time <= t`——
于是 **embargo 不再是一条要记得执行的纪律，而是数据的可见性属性**：
你没法读到未来的标签，因为它当时还不在可见集里。

> **诚实边界（重要）**：本数据集**没有**真实的拒付时间戳，
> 这里的 `label_time = event_time + 21 天` 是**合成的常数平移**。
> 因此双时间戳日志与离线的 `RANGE ... 21d PRECEDING` 帧**同构**，不含新信息——
> 回放对账通过**证明不了时间模型是对的**，只证明两条代码路径算的是同一个东西。
> 真实系统有真的 `label_time`（变长延迟），那时 21 天这个参数会消失、被真实延迟取代。

## `label_time` 为 NULL 的那一块 = 选择性偏差的表面积
未被复核就放行的交易，**永远拿不到标签**（`label_time = NULL`）。
把它建成 nullable 之后，`prior_fraud_rate` 的分母天然只含被复核过的样本——
这条项目里反复讨论的偏差，在存储层是一个**可以指着看的字段**，不是一段解释。

## 三种时间语义，在线必须分开实现（否则静默算错）

| 特征 | 语义 | 在线条件 |
|---|---|---|
| `prior_cnt` / `fan_out` | **位置**（谁排在前面） | `event_time < t` **或** `event_time = t 且 id < 本笔 id` |
| `prior_fraud_cnt` / `_rate` | **取值** + 标签成熟 | `label_time <= t` |
| `velocity_*` | **取值**（落在过去 N 秒内） | `t − W <= event_time < t` |

> ⚠️ 第一行的 `id` tiebreak 是**离线特征定义里的一个人造依赖**：
> 离线用 `lexsort` 的稳定性（即 `TransactionID` 顺序）打破 `dt` 并列。
> **真实系统里同一秒到达的两笔谁在前是到达顺序，不是 ID**——`TransactionID` 在线没有对应物。
> 这里传入 `tiebreak_id` 只是为了能与离线逐行对账；它是**对账的需要**，不是设计。

用法：见 `src/eval/online_replay.py`
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EMBARGO_SECS = 21 * 86_400

# 实体键：与离线 graph_features 完全一致（组合键更特异，稀有共享才是团伙信号）
ENTITY_KEYS = {
    "card1": ["card1"],
    "card1_addr1": ["card1", "addr1"],
    "card1_email": ["card1", "p_emaildomain"],
    "card1_device": ["card1", "device_info"],
}
FANOUT_COLS = {"addr1": "addr1", "email": "p_emaildomain", "device": "device_info"}
VELOCITY_WINDOWS = {"1h": 3600, "6h": 6 * 3600, "24h": 24 * 3600}


class FeatureStore:
    """双时间戳事件日志。**方案 A：不做任何预聚合，每次点查全扫。**

    先做最笨的实现并测出延迟，超预算再谈分层——
    「全扫太慢」在测量之前只是一句未经验证的断言，不该用它来驱动架构。
    """

    def __init__(self, db_path=":memory:"):
        import duckdb
        self.con = duckdb.connect(db_path)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS events (
                transaction_id BIGINT,
                event_time     BIGINT NOT NULL,
                label_time     BIGINT,          -- NULL = 未复核，标签永远不会到
                is_fraud       TINYINT,         -- label_time IS NULL 时无意义
                card1          VARCHAR,
                addr1          VARCHAR,
                p_emaildomain  VARCHAR,
                device_info    VARCHAR
            )""")

    # ── 写入 ────────────────────────────────────────────────────────────
    def append_frame(self, df, label_delay=EMBARGO_SECS, labeled_mask=None):
        """批量灌入历史。

        `labeled_mask=None` → 全部有标签（与离线一致，用于对账）。
        传入掩码则未命中的行 `label_time = NULL`，用于演示选择性偏差。
        """
        import numpy as np
        import pandas as pd
        e = pd.DataFrame({
            "transaction_id": df["TransactionID"].to_numpy(),
            "event_time": df["TransactionDT"].to_numpy(),
            "is_fraud": df["isFraud"].to_numpy().astype("int8"),
        })
        lt = e["event_time"].to_numpy() + label_delay
        if labeled_mask is not None:
            lt = np.where(np.asarray(labeled_mask), lt, np.nan)
        e["label_time"] = lt
        for col, src in (("card1", "card1"), ("addr1", "addr1"),
                         ("p_emaildomain", "P_emaildomain"), ("device_info", "DeviceInfo")):
            e[col] = df[src].astype("string").to_numpy()
        self.con.register("_incoming", e)
        self.con.execute("""INSERT INTO events
            SELECT transaction_id, event_time, label_time, is_fraud,
                   card1, addr1, p_emaildomain, device_info FROM _incoming""")
        self.con.unregister("_incoming")
        return len(e)

    # ── 读取 ────────────────────────────────────────────────────────────
    def get_features(self, txn, tiebreak_id=None, use_tiebreak=True):
        """算出 27 列在线特征。

        `txn` 需含 card1 / addr1 / P_emaildomain / DeviceInfo / TransactionDT。
        `use_tiebreak=False` 是**负对照开关**：去掉 ID tiebreak，让位置语义退化成取值语义。
        """
        t = int(txn["TransactionDT"])
        tid = int(tiebreak_id if tiebreak_id is not None else txn.get("TransactionID", 0))
        vals = {"card1": _s(txn.get("card1")), "addr1": _s(txn.get("addr1")),
                "p_emaildomain": _s(txn.get("P_emaildomain")),
                "device_info": _s(txn.get("DeviceInfo"))}
        out = {}
        for kname, cols in ENTITY_KEYS.items():
            # **缺失键各自成组**：任一列为 NULL → 该实体在历史中无同组行（与离线 codes_group 一致）
            if any(vals[c] is None for c in cols):
                out[f"{kname}_prior_cnt"] = 0
                out[f"{kname}_prior_fraud_cnt"] = 0
                out[f"{kname}_prior_fraud_rate"] = None
                for w in VELOCITY_WINDOWS:
                    out[f"{kname}_velocity_{w}"] = 0
                continue
            where_key = " AND ".join(f"{c} = ?" for c in cols)
            args = [vals[c] for c in cols]
            pos = ("(event_time < ? OR (event_time = ? AND transaction_id < ?))"
                   if use_tiebreak else "event_time < ?")
            pos_args = [t, t, tid] if use_tiebreak else [t]
            out[f"{kname}_prior_cnt"] = self.con.execute(
                f"SELECT count(*) FROM events WHERE {where_key} AND {pos}",
                args + pos_args).fetchone()[0]
            # 标签型：取值语义 + 标签必须已成熟；label_time IS NULL 的永不参与
            oc, of = self.con.execute(
                f"SELECT count(*), coalesce(sum(is_fraud), 0) FROM events "
                f"WHERE {where_key} AND label_time IS NOT NULL AND label_time <= ?",
                args + [t]).fetchone()
            out[f"{kname}_prior_fraud_cnt"] = int(of)
            # **obs_cnt = 0 → NULL，不是 0**：与离线 np.where(oc>0, ..., nan) 对齐
            out[f"{kname}_prior_fraud_rate"] = (of / oc) if oc > 0 else None
            for w, secs in VELOCITY_WINDOWS.items():
                out[f"{kname}_velocity_{w}"] = self.con.execute(
                    f"SELECT count(*) FROM events WHERE {where_key} "
                    f"AND event_time >= ? AND event_time < ?",
                    args + [t - secs, t]).fetchone()[0]
        # fan-out：card1 组内、本行之前见过的 distinct other 数（NULL 的 other 不计）
        for fname, col in FANOUT_COLS.items():
            if vals["card1"] is None:
                out[f"card1_fanout_{fname}"] = 0
                continue
            pos = ("(event_time < ? OR (event_time = ? AND transaction_id < ?))"
                   if use_tiebreak else "event_time < ?")
            pos_args = [t, t, tid] if use_tiebreak else [t]
            out[f"card1_fanout_{fname}"] = self.con.execute(
                f"SELECT count(DISTINCT {col}) FROM events "
                f"WHERE card1 = ? AND {col} IS NOT NULL AND {pos}",
                [vals["card1"]] + pos_args).fetchone()[0]
        return out

    def entity_history_len(self, txn):
        """本笔所属 card1 的历史长度 —— 延迟要按它分桶报，不能只报一个 p95。"""
        c = _s(txn.get("card1"))
        if c is None:
            return 0
        return self.con.execute(
            "SELECT count(*) FROM events WHERE card1 = ? AND event_time < ?",
            [c, int(txn["TransactionDT"])]).fetchone()[0]

    def close(self):
        self.con.close()


def _s(v):
    """统一成 VARCHAR 或 None；NaN / 空串一律视作缺失（与离线 astype('string') 对齐）。"""
    import pandas as pd
    if v is None or (isinstance(v, float) and pd.isna(v)) or v is pd.NA:
        return None
    s = str(v)
    return None if s in ("", "nan", "<NA>", "None") else s


FEATURE_COLUMNS = (
    [f"{k}_{suf}" for k in ENTITY_KEYS for suf in
     ("prior_cnt", "prior_fraud_cnt", "prior_fraud_rate")]
    + [f"card1_fanout_{f}" for f in FANOUT_COLS]
    + [f"{k}_velocity_{w}" for k in ENTITY_KEYS for w in VELOCITY_WINDOWS]
)
