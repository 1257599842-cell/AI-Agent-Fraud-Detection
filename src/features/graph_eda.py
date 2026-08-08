"""图连边键度分布速览（防守点① 加固一：连边强度过滤）。

网页版指示：先看各候选连边键的度分布，排除/降权超高频公共值（如 gmail.com），
团伙的本质是共享**稀有实体**，不是"都用 gmail"。本脚本给出决策依据：
  - 每个键：唯一值数、最大度、度分布分位、top 值及其占比与欺诈率。
  - 用来定强度过滤阈值：哪些键/值该排除或只在稀有值时连边。

用法：python -m src.features.graph_eda
产出：控制台报告 + reports/graph_eda.md
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARQUET = PROJECT_ROOT / "data" / "processed" / "train_merged.parquet"
OUT_MD = PROJECT_ROOT / "reports" / "graph_eda.md"

KEYS = ["card1", "addr1", "DeviceInfo", "P_emaildomain"]


def describe_key(df, key, overall_fraud):
    vc = df[key].value_counts(dropna=True)
    n_txn = int(vc.sum())
    # top-1% 高频值吃掉多少交易量
    top1pct_n = max(1, int(len(vc) * 0.01))
    share_top1pct = vc.head(top1pct_n).sum() / n_txn
    # 单值最大度 + 分位
    deg = vc.values
    # 最高频 5 个值的欺诈率（看公共值是不是噪声）
    top_rows = []
    for val in vc.head(5).index:
        m = df[key] == val
        top_rows.append((val, int(m.sum()), float(df.loc[m, "isFraud"].mean())))
    return {
        "key": key, "n_unique": int(vc.shape[0]), "n_txn_nonnull": n_txn,
        "max_deg": int(deg.max()), "p50": int(np.percentile(deg, 50)),
        "p99": int(np.percentile(deg, 99)), "share_top1pct": share_top1pct,
        "top_rows": top_rows,
    }


def main():
    df = pd.read_parquet(PARQUET, columns=["card1", "addr1", "DeviceInfo", "P_emaildomain", "isFraud"])
    overall = df["isFraud"].mean()
    print(f"整体欺诈率 {overall:.3%}\n")
    stats = [describe_key(df, k, overall) for k in KEYS]

    for s in stats:
        print(f"=== {s['key']} ===")
        print(f"  唯一值 {s['n_unique']:,}，最大度 {s['max_deg']:,}，度 p50={s['p50']} p99={s['p99']}，"
              f"top1% 高频值吃掉 {s['share_top1pct']:.1%} 交易量")
        for val, cnt, fr in s["top_rows"]:
            flag = "  ⚠️公共值(欺诈率≈整体，噪声)" if abs(fr - overall) < 0.015 and cnt > 5000 else ""
            print(f"    {str(val)[:28]:28s} n={cnt:>7,} 欺诈率={fr:.2%}{flag}")
        print()

    _write_md(stats, overall)
    print(f"✅ {OUT_MD.relative_to(PROJECT_ROOT)}")


def _write_md(stats, overall):
    L = ["# 图连边键度分布（① 加固一：强度过滤依据）\n",
         f"整体欺诈率 {overall:.3%}。团伙=共享稀有实体；下面找出该排除/降权的高频公共值。\n",
         "| 键 | 唯一值 | 最大度 | 度 p50 | 度 p99 | top1%值吃掉交易量 |",
         "|----|--------|--------|--------|--------|------------------|"]
    for s in stats:
        L.append(f"| {s['key']} | {s['n_unique']:,} | {s['max_deg']:,} | {s['p50']} | {s['p99']} | {s['share_top1pct']:.1%} |")
    L.append("\n## 各键 top-5 高频值（看是否公共噪声）\n")
    for s in stats:
        L.append(f"**{s['key']}**：")
        for val, cnt, fr in s["top_rows"]:
            L.append(f"- `{str(val)[:30]}` n={cnt:,}，欺诈率 {fr:.2%}")
        L.append("")
    L += [
        "## 强度过滤结论（定阈值）",
        "- 最大度极高、且高频值欺诈率≈整体的键（典型 P_emaildomain 的 gmail/hotmail）= 公共噪声 → **构图时排除这些高频值**，或只在共享**稀有值**（频次低于阈值）时连边。",
        "- 具体标识键（card1/addr1/DeviceInfo/card1+addr1）保留，但度特征对超高频 hub 值要么截断要么单独标记。",
        "- 下一步 `graph_features.py` 按此过滤后再建时间因果图特征。",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
