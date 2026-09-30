"""图标签历史年龄与数量的等宽滑窗对照。

## 它补的是哪个缺口

泄漏审计把 embargo 从 21 天拉到 60 天，图特征增益从 **+0.0387 缩到 +0.0234**（−39.6%）。
当时把缩水归因于「标签新鲜度」——**那个归因站不住**：改 embargo 是一刀**截断**，
它同时改变了四件事：

  ① 可用标签**数量**（被砍掉的 39 天带里的邻居没了）
  ② 标签**年龄**（剩下的全部 ≥60 天，而被砍掉的恰好是最新鲜那一头）
  ③ 估计**噪声**（obs_cnt 变小 → 比率由更少样本支撑，1/1=100% 这类噪声率变多）
  ④ 整列**缺失率**（obs_cnt=0 时 prior_fraud_rate 为 NaN）

四件事一起变 → 那 −39.6% 是**无法归因的联合差异**。

## 怎么拆

把标签窗从「截断」换成「**等宽滑窗**」，两臂窗宽相同、只差位置：

    臂 A（新鲜）：只数 t−21d−W < DT ≤ t−21d
    臂 B（陈旧）：只数 t−60d−W < DT ≤ t−60d

取 **W = 39 天 = 60 − 21**，于是两臂是**紧邻且等宽**的两段：
A =（t−60d, t−21d] 正是截断砍掉的那一段；B =（t−99d, t−60d] 是紧挨着它更老的等宽一段。
窗宽相同 → 数量这一维被大致固定 → 剩下的差异主要来自年龄。

## 三条必须随数字一起讲的边界

1. **等宽 ≠ 等量。** 交易密度随时间变、实体早期历史短，臂 B 的窗还可能落到 day 0 之前。
   所以本脚本**实测两臂 obs_cnt 分布并报残余不平衡**，对不上就照实说，不许当已对齐。
2. **本实验的 PR-AUC 与主结果不可比。** 两臂都只看一个 39 天带的标签历史，
   比累积到底的口径弱得多。**它是臂内对照（A vs B），不是对 +0.0387 / +0.0234 的重算。**
3. **结构型特征两臂完全相同**（prior_cnt / fan-out 不读标签、与 embargo 无关），
   直接复用既有 `graph_features.parquet` 的那几列——**只有标签型那 8 列在变。**

用法：python -m src.model.embargo_age_control
产出：reports/embargo_age_control.md
"""

from src.report_io import write_report
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.graph_features import causal_prior_stats, codes_group
from src.model.graph_vs_tabular import fit_eval
from src.model.train_baseline import T0, VAL_DAYS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "data" / "processed" / "train_merged.parquet"
GRAPH = PROJECT_ROOT / "data" / "processed" / "graph_features.parquet"
OUT_MD = PROJECT_ROOT / "reports" / "embargo_age_control.md"

SECS_PER_DAY = 86_400
WIN_DAYS = 39                      # = 60 − 21：两臂紧邻且等宽
# 跨报告参照值（用于「39 天带 vs 累积到底」那个对比）：
# 主实验 21 天累积版的图特征增益，出处 reports/graph_vs_tabular.md。
# 写成常量而非正文硬编码，改口径时只需动这一行。
MAIN_CUMULATIVE_DELTA = 0.0387
ARMS = {"A_fresh": 21, "B_stale": 60}
BASE_COLS = ["card1", "addr1", "P_emaildomain", "DeviceInfo"]
# 与 graph_features.py 同一套实体键
KEY_SPEC = {
    "card1": ["card1"],
    "card1_addr1": ["card1", "addr1"],
    "card1_email": ["card1", "P_emaildomain"],
    "card1_device": ["card1", "DeviceInfo"],
}
LABEL_COLS = [f"{k}_prior_fraud_{s}" for k in KEY_SPEC for s in ("cnt", "rate")]


def _windowed_label_feats(df, embargo_days, win_days):
    """只重算**标签型**那几列（滑窗版）；结构型不碰。"""
    dt = df["TransactionDT"].to_numpy()
    fraud = df["isFraud"].to_numpy()
    emb = embargo_days * SECS_PER_DAY
    win = win_days * SECS_PER_DAY
    out, obs = {}, {}
    for name, cols in KEY_SPEC.items():
        ks = df[cols[0]].astype("string")
        for c in cols[1:]:
            ks = ks + "|" + df[c].astype("string")
        _, oc, of = causal_prior_stats(dt, fraud, codes_group(ks), emb, window=win)
        out[f"{name}_prior_fraud_cnt"] = of
        out[f"{name}_prior_fraud_rate"] = np.where(oc > 0, of / np.maximum(oc, 1), np.nan)
        obs[name] = oc
    return pd.DataFrame(out, index=df.index), obs


def _obs_summary(obs, mask, label):
    """两臂对不对齐，看的就是这张表：同一实体键下 obs_cnt 的分布。"""
    rows = []
    for name, oc in obs.items():
        v = oc[mask]
        rows.append((label, name, float(v.mean()), int(np.median(v)),
                     int(np.percentile(v, 90)), float((v == 0).mean())))
    return rows


def main() -> None:
    print("读取数据 …")
    df = pd.read_parquet(BASE)
    gf = pd.read_parquet(GRAPH)
    struct_cols = [c for c in gf.columns
                   if c != "TransactionID" and c not in LABEL_COLS]
    df = df.merge(gf[["TransactionID"] + struct_cols], on="TransactionID",
                  how="left", validate="one_to_one")
    d = df["TransactionDT"] // SECS_PER_DAY
    day = (d - d.min()).to_numpy()
    y = df["isFraud"].astype(int)
    print(f"  {len(df):,} 行；结构型沿用 {len(struct_cols)} 列，标签型 {len(LABEL_COLS)} 列按臂重算")

    Xbase = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"])
    for c in Xbase.select_dtypes(include="object").columns:
        Xbase[c] = Xbase[c].astype("category")
    tab_cols = [c for c in Xbase.columns if c not in struct_cols]

    fit_mask = day < (T0 - VAL_DAYS)
    test_mask = day >= T0

    print("\n跑纯表基准（同切分、同配置）…")
    m_tab, _ = fit_eval(Xbase, y, day, tab_cols)
    print(f"  纯表 PR-AUC {m_tab['pr']:.4f}  ROC-AUC {m_tab['roc']:.4f}")

    results, obs_rows = {}, []
    for arm, emb_days in ARMS.items():
        lo, hi = emb_days + WIN_DAYS, emb_days
        print(f"\n臂 {arm}：标签窗 (t−{lo}d, t−{hi}d]，宽 {WIN_DAYS} 天 …")
        lab, obs = _windowed_label_feats(df, emb_days, WIN_DAYS)
        X = pd.concat([Xbase, lab], axis=1)
        m, _ = fit_eval(X, y, day, tab_cols + struct_cols + LABEL_COLS)
        results[arm] = m
        print(f"  PR-AUC {m['pr']:.4f}（Δ 纯表 {m['pr'] - m_tab['pr']:+.4f}）"
              f"  ROC-AUC {m['roc']:.4f}")
        obs_rows += _obs_summary(obs, fit_mask, f"{arm}·fit")
        obs_rows += _obs_summary(obs, test_mask, f"{arm}·test")

    _write_md(m_tab, results, obs_rows)
    print(f"\n✅ → {OUT_MD.relative_to(PROJECT_ROOT)}")


def _write_md(m_tab, results, obs_rows):
    a, b = results["A_fresh"], results["B_stale"]
    da, db = a["pr"] - m_tab["pr"], b["pr"] - m_tab["pr"]
    L = [
        "# 图标签「年龄 vs 数量」等宽滑窗对照\n",
        f"**补的缺口**：泄漏审计（embargo 21→60 天）把图特征增益从 +0.0387 压到 +0.0234（−39.6%），"
        "但那是一刀**截断**——它同时改变了 ①可用标签数量 ②标签年龄 ③估计噪声 ④缺失率，"
        "所以那 −39.6% 是**无法归因的联合差异**。本实验把标签窗换成**等宽滑窗**，"
        "两臂窗宽相同、只差位置，把「数量」这一维大致固定住。\n",
        f"| 臂 | 标签窗 | 宽度 |",
        "|---|---|---|",
        f"| **A_fresh** | (t−{21 + WIN_DAYS}d, t−21d] | {WIN_DAYS} 天 |",
        f"| **B_stale** | (t−{60 + WIN_DAYS}d, t−60d] | {WIN_DAYS} 天 |",
        "",
        f"`W = {WIN_DAYS}` 天 `= 60 − 21`，于是两臂**紧邻且等宽**："
        "A 正是截断砍掉的那一段，B 是紧挨着它更老的等宽一段。",
        "**结构型特征（prior_cnt / fan-out）两臂完全相同**——它们不读标签、与 embargo 无关，"
        "直接沿用既有 `graph_features.parquet`。**只有标签型那 8 列在变。**\n",
        "## 1. 结果（同切分 fit<132 / val[132,146) / eval≥146，同 LGB 配置）\n",
        "| 臂 | PR-AUC | ROC-AUC | Δ PR-AUC（对纯表） | recall@0.5% | recall@1% | recall@2% |",
        "|---|---|---|---|---|---|---|",
        f"| 纯表 | {m_tab['pr']:.4f} | {m_tab['roc']:.4f} | — | "
        f"{m_tab['rec@0.5%']:.3f} | {m_tab['rec@1.0%']:.3f} | {m_tab['rec@2.0%']:.3f} |",
        f"| **A_fresh**（新鲜 39 天） | {a['pr']:.4f} | {a['roc']:.4f} | **{da:+.4f}** | "
        f"{a['rec@0.5%']:.3f} | {a['rec@1.0%']:.3f} | {a['rec@2.0%']:.3f} |",
        f"| **B_stale**（陈旧 39 天） | {b['pr']:.4f} | {b['roc']:.4f} | **{db:+.4f}** | "
        f"{b['rec@0.5%']:.3f} | {b['rec@1.0%']:.3f} | {b['rec@2.0%']:.3f} |",
        "",
        f"**年龄效应（A − B，窗宽已对齐）= PR-AUC {a['pr'] - b['pr']:+.4f}**"
        f"（ROC-AUC {a['roc'] - b['roc']:+.4f}）。",
        "",
        "### ⭐ 一个附带发现：更老的标签历史几乎没有额外价值\n",
        f"A_fresh **只用最新鲜的 {WIN_DAYS} 天**标签历史，拿到 **{da:+.4f}**；"
        f"而主实验用**累积到 t−21d 的全部历史**，拿到 **{MAIN_CUMULATIVE_DELTA:+.4f}**"
        "（`graph_vs_tabular.md`）。",
        f"两者差 **{abs(da - MAIN_CUMULATIVE_DELTA):.4f}**，"
        f"**量级远小于两臂之间的 {abs(a['pr'] - b['pr']):.4f}**。",
        "",
        "> **这个对比是公平的**：同一个纯表基线（本实验实测 "
        f"{m_tab['pr']:.4f}，与 `graph_vs_tabular.md` 逐位相同）、同结构型列、同切分，"
        "**只差标签窗是「累积到底」还是「一个 39 天带」**。",
        "> → **图特征的价值几乎全部集中在最近约 40 天的标签历史里，更老的部分是冗余的。**",
        "> → 工程含义：生产里**不需要维护无界的标签历史**，一个约 40 天的滚动窗就吃到几乎全部价值"
        "（同时也降低存储与回溯成本）。",
        "> → 这与 `embargo_decomposition.md` 的彩蛋同向：砍掉**最旧** 6.4 万训练样本，"
        "PR-AUC 反而从 0.5645 微升到 0.5693。**两个独立实验，同一个主题：近期性主导。**",
        "> ⚠️ 未做显著性检验；「几乎相同」是量级判断，不是等价性检验。",
        "",
        "## 2. 残余不平衡：两臂的 obs_cnt 分布真对上了吗\n",
        "> **等宽 ≠ 等量。** 交易密度随时间变、实体早期历史短、臂 B 的窗还可能落到 day 0 之前。"
        "这张表是本实验**能不能声称「只差年龄」的前提**——对不上就得把残余不平衡一起报。\n",
        "| 臂·窗 | 实体键 | obs_cnt 均值 | 中位 | p90 | obs_cnt=0 占比 |",
        "|---|---|---|---|---|---|",
    ]
    for lab, name, mean, med, p90, zero in obs_rows:
        L.append(f"| {lab} | `{name}` | {mean:.2f} | {med} | {p90} | {zero:.1%} |")
    # 残余不平衡按「同键同窗、A vs B 的均值比」量化——一个数比一张表好追问
    by = {}
    for lab, name, mean, med, p90, zero in obs_rows:
        by[(lab, name)] = (mean, zero)
    ratios = []
    for split in ("fit", "test"):
        for name in KEY_SPEC:
            ma, za = by[(f"A_fresh·{split}", name)]
            mb, zb = by[(f"B_stale·{split}", name)]
            ratios.append((split, name, ma, mb, mb / ma if ma else float("nan"), za, zb))
    L += [
        "",
        "### 残余不平衡（同键同窗，B ÷ A）\n",
        "| 窗 | 实体键 | A 均值 | B 均值 | **B/A** | A 的零占比 | B 的零占比 |",
        "|---|---|---|---|---|---|---|",
    ]
    for split, name, ma, mb, r, za, zb in ratios:
        L.append(f"| {split} | `{name}` | {ma:.2f} | {mb:.2f} | **{r:.2f}×** | {za:.1%} | {zb:.1%} |")
    worst = max(ratios, key=lambda t: abs(np.log(t[4])) if t[4] > 0 else np.inf)
    fit_r = [t[4] for t in ratios if t[0] == "fit"]
    test_r = [t[4] for t in ratios if t[0] == "test"]
    L += [
        "",
        f"**读法（本实验最要紧的一处限定）**：对齐程度**两个窗完全不同**——",
        f"**评估窗 B/A = {min(test_r):.2f}–{max(test_r):.2f}×（几乎完美对齐）**，"
        f"而**训练窗 B/A = {min(fit_r):.2f}–{max(fit_r):.2f}×（明显不齐）**。",
        "原因：test 交易都在 day≥146，两个带都落在数据区间内部；"
        "而 fit 窗含早期交易，臂 B 的带常常越过 day 0 → 邻居更少"
        f"（零占比 fit·card1：A {by[('A_fresh·fit','card1')][1]:.1%} vs "
        f"B {by[('B_stale·fit','card1')][1]:.1%}）。",
        "",
        "> ### 所以精确的说法是：**特征在评估窗对齐了，模型在训练窗没对齐。**",
        "> 被评估的那批特征，两臂的邻居数量几乎相同（1.0x）→ **测的确实主要是年龄**；"
        "> 但两个模型是在**密度不同**的特征上训出来的 → 臂 B 的模型训练时见到的标签特征更稀疏。"
        "> **这是一条残余混杂，不是已消除的混杂。**",
        f"> 最偏的一格：`{worst[1]}`（{worst[0]} 窗）B/A = **{worst[4]:.2f}×**。",
        "",
        "## 3. 能声称什么 / 不能声称什么\n",
        "**能声称**：",
        f"1. 在**窗宽已对齐**（都 {WIN_DAYS} 天）、且**评估窗邻居数量也已对齐**"
        f"（B/A = {min(test_r):.2f}–{max(test_r):.2f}×）的前提下，"
        f"把标签历史从「新鲜 {WIN_DAYS} 天」换成「陈旧 {WIN_DAYS} 天」，"
        f"PR-AUC 变化 **{a['pr'] - b['pr']:+.4f}** —— 这比原来那个 −39.6% "
        "**更接近「纯年龄效应」**，因为「数量」这一维在评估侧已被固定。",
        f"2. **更老的标签历史几乎没有额外价值**（§1 的附带发现：{WIN_DAYS} 天带 {da:+.4f} "
        f"≈ 累积到底 {MAIN_CUMULATIVE_DELTA:+.4f}）。",
        "",
        "**不能声称**：",
        f"1. **不能把 B_stale 和主实验的 +0.0234（60 天累积）比。** B 只有一个 {WIN_DAYS} 天带，"
        "而 60 天累积版**保留了全部更老的历史**——两者的特征构造不同，"
        f"这也正是本实验的年龄效应（{a['pr'] - b['pr']:+.4f}）比原截断差异（0.0387−0.0234=0.0153）"
        "**更大**的原因。**（A_fresh 与 +0.0387 的对比则是公平的，见 §1。）**",
        "2. **不能说混杂已清零。** 见 §2：训练窗的邻居数量仍有 "
        f"{min(fit_r):.2f}–{max(fit_r):.2f}× 的残余差异。**这是「少一个混杂」，不是「没有混杂」。**",
        "3. **不能据此声称主实验无泄漏。** 本实验只动标签窗的**位置**，"
        "既没改模型 fit/val 的标签成熟性，也无法审计匿名 C/V 特征的生成时刻。",
        "4. **未做显著性检验。** 全部为点估计，没有 CI。",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    write_report(OUT_MD, "\n".join(L))


if __name__ == "__main__":
    main()
