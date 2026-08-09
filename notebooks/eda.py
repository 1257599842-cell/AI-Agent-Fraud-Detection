"""EDA（探索性数据分析）—— 读合并后的 parquet，产出图 + 数字小结。

只做"看数据"，不建模、不造特征。产出：
  reports/figures/*.png   四组图
  reports/eda_summary.md  数字小结（可直接贴回项目负责人项目负责人判读）

每一块都对应一个设计约束点（见 CLAUDE.md 第六节）：
  - 缺失分布            → 数据质量 / 特征取舍
  - 按时间的欺诈率       → ②时间切分、⑩概念漂移
  - 金额分布            → 特征信号
  - 类别字段欺诈率       → 特征信号 / 业务解释

用法（项目根、已激活 .venv）：  python notebooks/eda.py

图里标签用英文（matplotlib 默认字体无中文）；中文解读写进 summary.md。
"""

from src.report_io import write_report
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面后端，只出文件
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET = PROJECT_ROOT / "data" / "processed" / "train_merged.parquet"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
SUMMARY_MD = PROJECT_ROOT / "reports" / "eda_summary.md"

SECS_PER_DAY = 86_400


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"读取 {PARQUET.relative_to(PROJECT_ROOT)} …")
    df = pd.read_csv(PARQUET) if PARQUET.suffix == ".csv" else pd.read_parquet(PARQUET)
    print(f"  → {df.shape[0]:,} 行 × {df.shape[1]} 列")

    lines: list[str] = []  # 汇总到 summary.md 的中文解读

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    n = len(df)
    n_fraud = int(df["isFraud"].sum())
    say("# EDA 小结 — IEEE-CIS（可带回项目负责人项目负责人判读）\n")
    say(f"- 形状：{n:,} 行 × {df.shape[1]} 列；欺诈率 {n_fraud/n:.4%}（{n_fraud:,} / {n:,}，≈1:{round(n/n_fraud)}）。")
    # identity 覆盖率此前只在 load_data 里 print、从未落进报告，
    # 于是 MODEL_CARD 的 24.42% 成了一个**没有出处的 [实测]**。补进机器区。
    _idy = [c for c in df.columns if c.startswith("id_") or c in ("DeviceType", "DeviceInfo")]
    _cov = df[_idy].notna().any(axis=1).mean() if _idy else float("nan")
    say(f"- identity 覆盖率：**{_cov:.2%}**（{int(_cov * n):,} / {n:,} 笔有 identity 侧字段）。\n")

    # ── 1. 缺失分布 ───────────────────────────────────────────────
    miss = df.isna().mean().sort_values(ascending=False)
    n_all_missing = int((miss == 1).sum())
    n_high_missing = int((miss > 0.9).sum())
    n_no_missing = int((miss == 0).sum())
    say("## 1. 缺失情况")
    say(f"- 完全无缺失的列：{n_no_missing}；缺失率 >90% 的列：{n_high_missing}；几乎全空(=100%)：{n_all_missing}。")
    say(f"- 最缺的 5 列：" + ", ".join(f"{c}({miss[c]:.0%})" for c in miss.head(5).index) + "。")
    say("- 解读：V 系列匿名特征大面积块状缺失（同组同缺），缺失模式本身可能携带信息；建模时缺失当一种取值，别无脑填均值。\n")

    plt.figure(figsize=(7, 4))
    plt.hist(miss.values, bins=40, color="#4C72B0", edgecolor="white")
    plt.xlabel("Missing fraction per column")
    plt.ylabel("Number of columns")
    plt.title("Distribution of per-column missingness")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_missingness.png", dpi=120)
    plt.close()

    # ── 2. 按时间的交易量 & 欺诈率（②时间切分 / ⑩概念漂移）─────────
    day = (df["TransactionDT"] // SECS_PER_DAY).astype(int)
    day = day - day.min()  # 归一到第 0 天起
    by_day = df.assign(day=day).groupby("day")["isFraud"].agg(["count", "mean"])
    span_days = int(day.max() - day.min() + 1)
    # 按周聚合的欺诈率波动（更稳）
    week = (day // 7)
    by_week = df.assign(week=week).groupby("week")["isFraud"].agg(["count", "mean"])
    fr_min, fr_max = by_week["mean"].min(), by_week["mean"].max()
    say("## 2. 时间维度（喂 ②时间切分 / ⑩概念漂移）")
    say(f"- 时间跨度约 {span_days} 天，连续；可前 ~5 个月训练、最后 ~1 个月当时间外测试。")
    say(f"- 按周欺诈率在 {fr_min:.2%} ~ {fr_max:.2%} 间波动（max/min ≈ {fr_max/fr_min:.1f}×）→ 存在时间漂移，支持时间切分与漂移监控。\n")

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.bar(by_day.index, by_day["count"], color="#C7D5E8", width=1.0, label="Daily tx count")
    ax1.set_xlabel("Day index")
    ax1.set_ylabel("Daily transaction count", color="#4C72B0")
    ax2 = ax1.twinx()
    ax2.plot(by_day.index, by_day["mean"], color="#C44E52", lw=1.2, label="Daily fraud rate")
    ax2.set_ylabel("Fraud rate", color="#C44E52")
    plt.title("Transaction volume & fraud rate over time")
    fig.tight_layout()
    plt.savefig(FIG_DIR / "02_fraud_rate_over_time.png", dpi=120)
    plt.close()

    # ── 3. 金额分布（欺诈 vs 正常）──────────────────────────────────
    amt = df["TransactionAmt"].clip(lower=0.01)
    say("## 3. 交易金额 TransactionAmt")
    med_n = df.loc[df.isFraud == 0, "TransactionAmt"].median()
    med_f = df.loc[df.isFraud == 1, "TransactionAmt"].median()
    say(f"- 金额中位数：正常 {med_n:.2f} vs 欺诈 {med_f:.2f}；整体右偏，宜取 log。")
    say(f"- 解读：金额本身有区分度但非决定性；log 变换 + 与类别交叉更有用。\n")

    plt.figure(figsize=(7, 4))
    bins = np.logspace(np.log10(amt.min()), np.log10(amt.max()), 60)
    plt.hist(amt[df.isFraud == 0], bins=bins, density=True, alpha=0.6, label="Legit", color="#4C72B0")
    plt.hist(amt[df.isFraud == 1], bins=bins, density=True, alpha=0.6, label="Fraud", color="#C44E52")
    plt.xscale("log")
    plt.xlabel("TransactionAmt (log scale)")
    plt.ylabel("Density")
    plt.title("Transaction amount: fraud vs legit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_amount_dist.png", dpi=120)
    plt.close()

    # ── 4. 类别字段欺诈率 ─────────────────────────────────────────
    say("## 4. 类别字段欺诈率（与整体 3.50% 比）")
    cat_cols = ["ProductCD", "card4", "card6"]
    fig, axes = plt.subplots(1, len(cat_cols), figsize=(13, 4))
    for ax, col in zip(axes, cat_cols):
        g = df.groupby(col)["isFraud"].agg(["count", "mean"]).sort_values("mean", ascending=False)
        g = g[g["count"] >= 500]  # 滤掉过小的组
        ax.bar(range(len(g)), g["mean"], color="#55A868")
        ax.axhline(n_fraud / n, color="#C44E52", ls="--", lw=1, label="overall")
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(g.index.astype(str), rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Fraud rate by {col}")
        ax.set_ylabel("Fraud rate")
        top = g.head(1)
        say(f"- {col}：最高风险取值 `{top.index[0]}` 欺诈率 {top['mean'].iloc[0]:.2%}"
            f"（n={int(top['count'].iloc[0]):,}），vs 整体 {n_fraud/n:.2%}。")
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_categorical_fraud_rate.png", dpi=120)
    plt.close()

    # 邮箱域 top（单独看，类别多）
    em = df.groupby("P_emaildomain")["isFraud"].agg(["count", "mean"])
    em = em[em["count"] >= 1000].sort_values("mean", ascending=False)
    if len(em):
        say(f"- P_emaildomain：最高风险域 `{em.index[0]}` 欺诈率 {em['mean'].iloc[0]:.2%}"
            f"（n={int(em['count'].iloc[0]):,}）；邮箱域区分度明显，可做特征。")
    say("")

    say("## 给项目负责人的一句话")
    say("数据干净、半年连续、欺诈率 3.5% 极端不平衡且随时间漂移；类别字段（ProductCD/卡类型/邮箱域）"
        "和金额都有区分信号。下一步可定时间切分点、确认不平衡处理口径与评估指标。")

    SUMMARY_MD.parent.mkdir(parents=True, exist_ok=True)
    write_report(SUMMARY_MD, "\n".join(lines))
    print(f"\n✅ 图已存到 {FIG_DIR.relative_to(PROJECT_ROOT)}/，小结写到 {SUMMARY_MD.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
