"""Velocity 特征：同一实体在**短时间窗内**的交易笔数。

## 为什么补这一个
逐笔期望成本框架有一个可证的盲区：**低分 + 微额的合流处**。
一笔 $5、模型给 0.01 的交易，五档算下来最优动作永远是放行——
因为任何干预的固定成本（$0.50 的验证摩擦、$5 的人工复核）都超过整笔暴露本身。

盲区的成因是**口径**：逐笔决策只看这一笔值多少钱。
而试卡（card testing）的价值不在本笔，在于**为后续大额盗刷验证卡的活性**——
它是一个跨笔现象，逐笔口径按定义就看不见。

velocity 是最小的跨笔信号：**同一张卡 1 小时内刷了几次。**

## 时间纪律：结构型，但需要 RANGE 帧
本项目此前把两层纪律与两种窗口帧一一对应过：结构型→`ROWS`、标签型→`RANGE`+embargo。
**velocity 是这个对应关系的反例**：

  · 它是**结构型**的（只数交易笔数，不看任何标签）→ **不需要 embargo**；
  · 但它必须用 **`RANGE`**（按 `TransactionDT` 取值判「是否落在过去 1 小时内」），
    `ROWS` 数的是行的位置，答不了「过去一小时」这个问题。

→ **帧型（位置 vs 取值）与 embargo（标签是否成熟）是两件正交的事。**
   此前的一一对应只是「现有特征恰好如此」，不是规律。

用法：python -m src.features.velocity_features
产出：data/processed/velocity_features.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MERGED = ROOT / "data" / "processed" / "train_merged.parquet"
OUT = ROOT / "data" / "processed" / "velocity_features.parquet"
REPORT = ROOT / "reports" / "velocity_features.md"

HOUR = 3600
WINDOWS = {"1h": 1 * HOUR, "6h": 6 * HOUR, "24h": 24 * HOUR}
# 实体键：与图特征同一套组合键（稀有共享实体才是团伙信号，gmail 这类公共值无意义）
KEYS = {
    "card1": ["card1"],
    "card1_addr1": ["card1", "addr1"],
    "card1_email": ["card1", "P_emaildomain"],
    "card1_device": ["card1", "DeviceInfo"],
}


def velocity_counts(dt, codes, window_secs):
    """每行：同组内落在 `[t − window, t)` 的**先前**交易笔数。

    严格开区间右端（`< t`，不含本行），所以不会数进本笔自己；
    同一时刻的并列交易也不计入——**「同时」不是「之前」**。

    实现：按 (组, 时间) 排序后，组内用两个 searchsorted 定位窗口边界，
    O(n log n)，不用逐行循环。
    """
    order = np.lexsort((dt, codes))
    dt_s, code_s = dt[order], codes[order]
    out_s = np.zeros(len(dt), dtype=np.int32)
    # 组边界
    bounds = np.flatnonzero(np.diff(code_s)) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [len(dt)]))
    for s, e in zip(starts, ends):
        t = dt_s[s:e]
        # 左边界：第一个 >= t − window 的位置；右边界：第一个 >= t 的位置
        lo = np.searchsorted(t, t - window_secs, side="left")
        hi = np.searchsorted(t, t, side="left")
        out_s[s:e] = hi - lo
    out = np.empty_like(out_s)
    out[order] = out_s
    return out


def codes_of(df, cols):
    """把组合键编码成整数；**缺失值各自独立成组**，不塌成一个巨型组。"""
    key = df[cols[0]].astype("string")
    for c in cols[1:]:
        key = key + "\x00" + df[c].astype("string")
    # 任一列缺失 → 该行自成一组（用行号保证唯一）
    na = df[cols].isna().any(axis=1).to_numpy()
    codes = pd.factorize(key, use_na_sentinel=False)[0].astype(np.int64)
    codes[na] = codes.max() + 1 + np.arange(na.sum())
    return codes


def main():
    need = sorted({c for cols in KEYS.values() for c in cols}
                  | {"TransactionID", "TransactionDT", "isFraud", "TransactionAmt"})
    df = pd.read_parquet(MERGED, columns=need)
    dt = df["TransactionDT"].to_numpy()
    print(f"{len(df):,} 行；窗口 {list(WINDOWS)}；实体键 {list(KEYS)}")

    out = pd.DataFrame({"TransactionID": df["TransactionID"].to_numpy()})
    for kname, cols in KEYS.items():
        codes = codes_of(df, cols)
        for wname, secs in WINDOWS.items():
            col = f"{kname}_velocity_{wname}"
            out[col] = velocity_counts(dt, codes, secs)
        print(f"  {kname:<14} 组数 {len(np.unique(codes)):>7,}　"
              + "　".join(f"{w}: 均值 {out[f'{kname}_velocity_{w}'].mean():.2f} "
                          f"/ 最大 {out[f'{kname}_velocity_{w}'].max()}"
                          for w in WINDOWS))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    _write_report(df, out)
    print(f"\n✅ {len(out.columns) - 1} 列 → {OUT.relative_to(ROOT)}")


def _write_report(df, vel):
    from src.report_io import write_report
    y = df["isFraud"].to_numpy().astype(bool)
    base = y.mean()
    L = ["# Velocity 特征：同一实体短窗内的交易笔数\n",
         "> **补这一个是为了盯住逐笔口径的盲区**：低分 + 微额。",
         "> 试卡的价值不在本笔，在于为后续大额盗刷验证卡的活性——**跨笔现象，逐笔口径按定义看不见**。\n",
         "## 时间纪律：结构型，但用 RANGE 帧\n",
         "velocity 只数交易笔数、不读任何标签 → **结构型，不需要 embargo**；",
         "但它必须按 `TransactionDT` **取值**判断「是否落在过去 N 小时内」→ **必须用 `RANGE`**。\n",
         "> 本项目此前把「结构型→`ROWS`、标签型→`RANGE`+embargo」当成一一对应。",
         "> **velocity 是这个对应的反例**：帧型（位置 vs 取值）与 embargo（标签是否成熟）",
         "> 是两件**正交**的事，此前的对应只是现有特征恰好如此。\n",
         "口径：每行数的是落在 `[t − window, t)` 的**先前**交易，",
         "右端开区间——不含本笔，也不含同一时刻的并列交易（**「同时」不是「之前」**）。\n",
         f"全体欺诈率基率 **{base:.3%}**。\n",
         "## 各特征的判别力（全窗，仅供观察；准入筛选在训练窗做）\n",
         "| 特征 | 均值 | P99 | 最大 | 取值 ≥2 的占比 | ≥2 组的欺诈率 | lift |",
         "|---|---|---|---|---|---|---|"]
    for c in [c for c in vel.columns if c != "TransactionID"]:
        v = vel[c].to_numpy()
        m = v >= 2
        fr = y[m].mean() if m.sum() else float("nan")
        L.append(f"| `{c}` | {v.mean():.2f} | {np.percentile(v, 99):.0f} | {v.max()} | "
                 f"{m.mean():.2%} | {fr:.2%} | {fr / base:.2f}× |")
    L += ["", "> 上表按 `≥2`（即「短窗内不止这一笔」）粗分，只为看信号有没有方向。",
          "> **是否进规则库由 `src/model/velocity_rules.py` 按统一准入门槛判定**",
          "> （lift ≥ 1.5 · 触发 ≥ 500 · 欺诈数 ≥ 30，且统计量只在 `[0,125)` 训练窗算）。\n"]
    write_report(REPORT, "\n".join(L))


if __name__ == "__main__":
    main()
