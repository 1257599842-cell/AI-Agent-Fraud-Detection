"""时间因果图特征（硬点① + ⑥ 素材）。

按 graph_eda 结论：IEEE-CIS 无细粒度设备/账户 ID（addr1=地区、DeviceInfo=OS、邮箱域全公共），
唯 card1 有特异性 → 用**组合键**造"稀有共享实体"：card1、card1+addr1、card1+邮箱、card1+设备。

两层防泄漏（加固二，图特征版硬点②）：
  - 结构型（prior_count / fan-out）：只用**该交易时间点之前**的边聚合（two-pointer，只看 DT 更早的行）。
  - 标签型（prior_fraud_rate）：邻居 isFraud 只取 DT ≤ t−EMBARGO（拒付延迟，"之前"再往前挪 21 天）。

每个实体键 K：
  {K}_prior_cnt         —— 之前同 K 的交易数（度/velocity；结构型）
  {K}_prior_fraud_cnt   —— 之前同 K 且已确认(embargo)的欺诈数（标签型）
  {K}_prior_fraud_rate  —— 上两者之比（无历史则 NaN）
card1 的 fan-out（结构型）：prior 见过的 distinct addr1 / 邮箱 / 设备数（团伙扩散信号）。

用法：python -m src.features.graph_features
产出：data/processed/graph_features.parquet（TransactionID + 图特征列）
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARQUET = PROJECT_ROOT / "data" / "processed" / "train_merged.parquet"

# embargo 天数可由环境变量覆盖（泄漏审计用：跑 21 / 60 对比）
EMBARGO_DAYS = int(os.environ.get("GRAPH_EMBARGO_DAYS", "21"))
EMBARGO_SECS = EMBARGO_DAYS * 86_400
_suffix = "" if EMBARGO_DAYS == 21 else f"_e{EMBARGO_DAYS}"
OUT = PROJECT_ROOT / "data" / "processed" / f"graph_features{_suffix}.parquet"
BASE_COLS = ["card1", "addr1", "P_emaildomain", "DeviceInfo"]


def causal_prior_stats(dt, fraud, group_codes, embargo):
    """按 group 分组、组内按时间：返回每行的
       prior_cnt（组内更早行数）、obs_cnt（DT≤t−emb 的更早行数）、obs_fraud（其中的欺诈数）。
    two-pointer，O(n)。"""
    n = len(dt)
    order = np.lexsort((dt, group_codes))          # 先按 group、再按 dt 升序
    g, d, f = group_codes[order], dt[order], fraud[order].astype(np.int64)
    prior_cnt = np.empty(n, np.int64)
    obs_cnt = np.empty(n, np.int64)
    obs_fraud = np.empty(n, np.int64)
    bounds = np.flatnonzero(np.diff(g)) + 1
    seg_s = np.concatenate(([0], bounds))
    seg_e = np.concatenate((bounds, [n]))
    for s, e in zip(seg_s, seg_e):
        dd, ff = d[s:e], f[s:e]
        cumf = np.concatenate(([0], np.cumsum(ff)))   # 前缀：cumf[k]=[0,k) 的欺诈数
        p = 0
        for i in range(len(dd)):
            prior_cnt[s + i] = i
            thr = dd[i] - embargo
            while p < i and dd[p] <= thr:
                p += 1
            obs_cnt[s + i] = p
            obs_fraud[s + i] = cumf[p]
    inv = np.empty(n, np.int64)
    inv[order] = np.arange(n)
    return prior_cnt[inv], obs_cnt[inv], obs_fraud[inv]


def prior_nunique(dt, card_codes, other_codes):
    """card1 组内、按时间：每行之前见过的 distinct other 值数（fan-out，结构型）。
    other 为 NA（code=-1）的不计入 distinct。"""
    n = len(dt)
    order = np.lexsort((dt, card_codes))
    g, oc = card_codes[order], other_codes[order]
    out = np.empty(n, np.int64)
    bounds = np.flatnonzero(np.diff(g)) + 1
    seg_s = np.concatenate(([0], bounds))
    seg_e = np.concatenate((bounds, [n]))
    for s, e in zip(seg_s, seg_e):
        seen = set()
        for i in range(s, e):
            out[i] = len(seen)
            if oc[i] != -1:            # NA other 不算一个实体
                seen.add(oc[i])
    inv = np.empty(n, np.int64)
    inv[order] = np.arange(n)
    return out[inv]


def codes_group(series):
    """实体分组用：factorize；**NA 每行各自成独立组**（不共享，避免缺失值塌成一个巨型组）。"""
    c, _ = pd.factorize(series, use_na_sentinel=True)
    c = c.astype(np.int64)
    na = c == -1
    if na.any():
        base = int(c.max()) + 1
        c[na] = np.arange(base, base + int(na.sum()))
    return c


def codes_plain(series):
    """fan-out 的 other 用：factorize，NA → -1（在 prior_nunique 里被跳过）。"""
    c, _ = pd.factorize(series, use_na_sentinel=True)
    return c.astype(np.int64)


def main():
    print("读取列 …")
    df = pd.read_parquet(PARQUET, columns=["TransactionID", "TransactionDT", "isFraud"] + BASE_COLS)
    dt = df["TransactionDT"].to_numpy()
    fraud = df["isFraud"].to_numpy()
    n = len(df)
    print(f"  {n:,} 行；EMBARGO={EMBARGO_SECS//86400} 天")

    card_s = df["card1"].astype("string")
    addr_s = df["addr1"].astype("string")
    email_s = df["P_emaildomain"].astype("string")
    dev_s = df["DeviceInfo"].astype("string")

    # 实体键：单 card1 + 三组合（组合更特异，稀有共享=团伙信号）
    keys = {
        "card1": card_s,
        "card1_addr1": card_s + "|" + addr_s,
        "card1_email": card_s + "|" + email_s,
        "card1_device": card_s + "|" + dev_s,
    }

    out = pd.DataFrame({"TransactionID": df["TransactionID"].to_numpy()})
    for name, ks in keys.items():
        gc = codes_group(ks)
        n_real = int(pd.factorize(ks, use_na_sentinel=True)[0].max()) + 1  # 真实实体数（不含 NA 独立组）
        pc, oc, of = causal_prior_stats(dt, fraud, gc, EMBARGO_SECS)
        rate = np.where(oc > 0, of / np.maximum(oc, 1), np.nan)
        out[f"{name}_prior_cnt"] = pc
        out[f"{name}_prior_fraud_cnt"] = of
        out[f"{name}_prior_fraud_rate"] = rate
        print(f"  {name}: 真实体≈{n_real:,}，prior_cnt 中位 {np.median(pc):.0f}/均值 {pc.mean():.1f}，有欺诈历史行占比 {(of>0).mean():.2%}")

    # card1 fan-out（结构型）
    card_c = codes_group(card_s)
    for other, os_ in [("addr1", addr_s), ("email", email_s), ("device", dev_s)]:
        out[f"card1_fanout_{other}"] = prior_nunique(dt, card_c, codes_plain(os_))
    print(f"  card1 fan-out: addr1/email/device 之前 distinct 数已算")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\n✅ {out.shape[1]-1} 个图特征 → {OUT.relative_to(PROJECT_ROOT)}")
    print("   列：" + ", ".join(c for c in out.columns if c != "TransactionID"))


if __name__ == "__main__":
    main()
