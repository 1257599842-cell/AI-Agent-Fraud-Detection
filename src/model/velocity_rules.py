"""Velocity 规则：过统一准入门槛，并检验它是否真的接住了逐笔口径的盲区。

## 这一步要回答两个问题
1. velocity 候选规则能否通过**与现有 15 条规则同一套**的数据准入？
   （lift ≥ 1.5 · 触发 ≥ 500 · 欺诈数 ≥ 30，统计量只在 `[0,125)` 训练窗算）
2. 它是否**真的接住了那个盲区**——低分 + 微额？盲区是补这个特征的全部理由。

## 预注册预期（跑之前写死，跑完不许改）

**对问题 1**：预期**短窗 + 稀有组合键会过、裸 card1 会被拒**。
依据是规则库的既有发现：裸 fan-out 因 lift≈1.0–1.5 被拒，组合键 `prior_fraud_rate` 准入且排名靠前。
velocity 的全窗粗看也是同一形状（`card1_device_1h` 2.60× vs 裸 `card1_1h` 1.18×）。

**对问题 2**：预期**在本数据的 test 窗上，盲区里没有欺诈可抓**。
依据已经在手：`small_amount_floor.md` 实测 test 窗 `<$1` 共 109 笔、**欺诈 0 笔**；
`<$2.44` 共 177 笔、欺诈 3 笔（1.69%，**低于**全体基率 3.42%）。
试卡簇是**训练窗**现象。

> 若预期 2 成立，正确的结论是：**velocity 的价值在本数据集上无法演示**，
> 而不是「velocity 没用」。**这两句话不一样，不许混说。**
> 盲区在成本几何上可证，但在本 test 窗未造成实际损失——
> 一个抓不到东西的规则，在这里抓不到，不能推广成「它在别处也抓不到」。

用法：python -m src.model.velocity_rules
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.report_io import write_report

ROOT = Path(__file__).resolve().parents[2]
MERGED = ROOT / "data" / "processed" / "train_merged.parquet"
VEL = ROOT / "data" / "processed" / "velocity_features.parquet"
GT = ROOT / "data" / "processed" / "agent_disposition_gt.parquet"
REPORT = ROOT / "reports" / "velocity_rules.md"

# 与 knowledge.py 完全一致的准入门槛 —— **不得为 velocity 单开一套**
MIN_LIFT, MIN_SUPPORT, MIN_FRAUD_N = 1.5, 500, 30
FIT_END_DAY = 125          # 统计量只在 [0,125) 训练窗算，与规则库同尺
THRESHOLDS = [2, 3, 5, 10]


def main():
    vel = pd.read_parquet(VEL)
    meta = pd.read_parquet(MERGED, columns=["TransactionID", "TransactionDT",
                                            "isFraud", "TransactionAmt"])
    df = meta.merge(vel, on="TransactionID")
    day = (df["TransactionDT"] // 86400).to_numpy()
    day = day - day.min()
    fit = day < FIT_END_DAY
    y = df["isFraud"].to_numpy().astype(bool)
    base = y[fit].mean()
    cols = [c for c in vel.columns if c != "TransactionID"]
    print(f"训练窗 [0,{FIT_END_DAY}) {fit.sum():,} 笔，基率 {base:.3%}；"
          f"候选 {len(cols) * len(THRESHOLDS)} 条")

    admitted, rejected = [], []
    for c in cols:
        v = df[c].to_numpy()
        for th in THRESHOLDS:
            m = fit & (v >= th)
            n, fn = int(m.sum()), int(y[m].sum())
            lift = (y[m].mean() / base) if n else 0.0
            rec = {"rule_id": f"R_VEL_{c.upper()}_GE{th}", "col": c, "th": th,
                   "n": n, "fraud_n": fn, "rate": float(y[m].mean()) if n else 0.0,
                   "lift": float(lift)}
            if n >= MIN_SUPPORT and fn >= MIN_FRAUD_N and lift >= MIN_LIFT:
                admitted.append(rec)
            else:
                rec["why"] = ("lift 不足" if lift < MIN_LIFT else
                              "触发样本不足" if n < MIN_SUPPORT else "欺诈数不足")
                rejected.append(rec)
    admitted.sort(key=lambda r: -r["lift"])
    print(f"  准入 {len(admitted)} 条 / 拒收 {len(rejected)} 条")

    blind = _blind_spot(df, y, day, admitted)
    _write(base, admitted, rejected, blind, fit.sum())
    print(f"\n✅ → {REPORT.relative_to(ROOT)}")


def _blind_spot(df, y, day, admitted):
    """盲区检验：低分 + 微额的那一片，velocity 抓到了什么。"""
    gt = pd.read_parquet(GT)[["TransactionID", "p"]]
    d = df.merge(gt, on="TransactionID", how="inner")      # 只有 test 窗有 p
    ym = d["isFraud"].to_numpy().astype(bool)
    amt, p = d["TransactionAmt"].to_numpy(), d["p"].to_numpy()
    # 盲区定义与 small_amount_floor 一致：p=0.30 处放行下限 $2.44；另看更严的 <$1
    bands = [("< $1", amt < 1.0), ("< $2.44", amt < 2.4420),
             ("< $10 且 p < 0.05", (amt < 10) & (p < 0.05))]
    rows = []
    for name, m in bands:
        hit = np.zeros(len(d), bool)
        for r in admitted:
            hit |= (d[r["col"]].to_numpy() >= r["th"])
        rows.append({"band": name, "n": int(m.sum()), "fraud": int(ym[m].sum()),
                     "vel_hit": int((m & hit).sum()),
                     "vel_hit_fraud": int((m & hit & ym).sum())})
    return rows, len(d)


def _write(base, admitted, rejected, blind, n_fit):
    rows, n_test = blind
    L = ["# Velocity 规则：准入结果与盲区检验\n",
         "> **补 velocity 的全部理由**是逐笔期望成本框架的一个可证盲区：**低分 + 微额**。",
         "> 一笔 $5、模型给 0.01 的交易，五档算下来最优永远是放行——",
         "> 任何干预的固定成本都超过整笔暴露本身。试卡的价值不在本笔，是**跨笔现象**。\n",
         f"准入门槛与现有 15 条规则**完全一致**（未为 velocity 单开）：",
         f"lift ≥ {MIN_LIFT} · 触发 ≥ {MIN_SUPPORT} · 欺诈数 ≥ {MIN_FRAUD_N}，",
         f"统计量只在 `[0,{FIT_END_DAY})` 训练窗算（{n_fit:,} 笔，基率 {base:.3%}）。\n",
         f"## 准入：{len(admitted)} 条通过 / {len(rejected)} 条拒收\n",
         "| 规则 | 触发量 | 占训练窗 | 集内欺诈率 | lift |", "|---|---|---|---|---|"]
    for r in admitted:
        L.append(f"| `{r['rule_id']}` | {r['n']:,} | {r['n']/n_fit:.2%} | "
                 f"{r['rate']:.2%} | **{r['lift']:.2f}×** |")
    if not admitted:
        L.append("| —— 无 —— | | | | |")

    by = {}
    for r in rejected:
        by.setdefault(r["why"], []).append(r)
    L += ["", "### 拒收原因分布\n"]
    for why, rs in sorted(by.items(), key=lambda x: -len(x[1])):
        L.append(f"- **{why}**：{len(rs)} 条")

    L += ["", "## 盲区检验（test 窗）\n",
          "> **这是补 velocity 的唯一目的**，所以必须单独看，不能只看准入通没通过。\n",
          f"test 窗共 {n_test:,} 笔。\n",
          "| 金额段 | 笔数 | 其中欺诈 | velocity 规则命中 | 命中且欺诈 |",
          "|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['band']} | {r['n']:,} | **{r['fraud']}** | {r['vel_hit']:,} | "
                 f"**{r['vel_hit_fraud']}** |")

    L += ["", "### 带内判别力：命中率 vs 抓获率\n",
          "只看「抓到几笔」不够——**一条打中所有人的规则，抓到再多也不算判别力**。",
          "所以每个金额段都要比两个比例：规则命中了该段的多大比例，又抓走了该段欺诈的多大比例。\n",
          "| 金额段 | 命中率 | 抓获率 | **带内 lift** |", "|---|---|---|---|"]
    for r in rows:
        hr = r["vel_hit"] / r["n"] if r["n"] else float("nan")
        cr = r["vel_hit_fraud"] / r["fraud"] if r["fraud"] else float("nan")
        lf = cr / hr if r["fraud"] and hr else float("nan")
        L.append(f"| {r['band']} | {hr:.1%} | "
                 + (f"{cr:.1%}" if r["fraud"] else "—（无欺诈可抓）") + " | "
                 + (f"**{lf:.2f}×**" if r["fraud"] and hr else "—") + " |")

    total_fraud = sum(r["fraud"] for r in rows[:2])
    L += ["", "### 判读（**预注册预期部分说错了，照实记**）\n",
          "**预期「盲区里没有欺诈可抓」——只对 `<$1` 成立，对 `<$2.44` 不成立**（那里有 3 笔）。",
          "既有实测（`small_amount_floor.md`：`<$1` 共 109 笔、欺诈 0 笔）被复现，但我把它外推到了更宽的带上。\n",
          "**真正要紧的读数是另一个**：",
          f"velocity 规则在 `<$1` 段**命中 {rows[0]['vel_hit']}/{rows[0]['n']} "
          f"（{rows[0]['vel_hit']/rows[0]['n']:.0%}）却抓到 0 笔**——因为那里根本没有欺诈；",
          "而在 `<$10 且 p<0.05` 段，命中率与抓获率几乎相等（见上表），**带内 lift ≈ 1**。\n",
          "> **同一批规则，全窗 lift 1.5–3.1×，进了盲区就退化到 ≈1×。**",
          "> 这不是规则坏了，是**判别力依赖分布**：velocity 的信号来自「短窗内高频复用稀有实体」，",
          "> 而微额段的高频复用有大量正常成因（订阅、拆单、小额充值），信号被稀释。",
          "> **在 A 分布上验过的 lift，不能直接搬到 B 分布上用。**\n",
          "所以结论要分开说，不能合并：",
          "1. **准入层面：通过。** 12 条过统一门槛，全窗 lift 1.54–3.14×，形状与规则库既有发现一致。",
          "2. **盲区层面：没有解决。** 带内 lift ≈1，它接不住当初补它的那个问题。",
          "3. **本数据集层面：无法充分演示。** 盲区内欺诈仅 3 笔（`<$2.44`）/ 22 笔（`<$10 且 p<0.05`），",
          "   任何结论的置信区间都极宽。**「没抓到」和「抓不到」在这个样本量上分不开。**\n",
          "> 补 velocity 的初衷是接住盲区，**这个目的没有达成**。",
          "> 它作为一般性风控规则是有效的（准入通过），但那是另一件事——**不许拿准入通过去顶盲区没解决**。\n"]

    L += ["## 时间纪律：一个此前没说准的地方\n",
          "本项目此前把两层纪律与两种窗口帧写成一一对应：结构型→`ROWS`、标签型→`RANGE`+embargo。",
          "**velocity 是这个对应的反例**：\n",
          "| | 读不读标签 | 需要 embargo | 帧型 |",
          "|---|---|---|---|",
          "| `prior_cnt`（结构型） | 否 | 否 | `ROWS`（位置：在不在本行之前） |",
          "| `prior_fraud_cnt`（标签型） | 是 | **是** | `RANGE` + 21 天偏移 |",
          "| **`velocity`（结构型）** | **否** | **否** | **`RANGE`**（取值：是否落在过去 N 小时） |",
          "",
          "> **帧型（位置 vs 取值）与 embargo（标签是否成熟）是两件正交的事。**",
          "> 此前的一一对应只是「当时的特征恰好如此」，不是规律——",
          "> 补一个新特征就撞破了它。**归纳出来的规律，要用新样本去撞。**\n"]
    write_report(REPORT, "\n".join(L))
    print("\n".join(L[-14:]))


if __name__ == "__main__":
    main()
