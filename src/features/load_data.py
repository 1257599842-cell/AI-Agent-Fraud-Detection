"""数据加载 + 合并（IEEE-CIS Fraud Detection）。

职责（只做这些，不做特征工程、不建模）：
  1. 读 train_transaction.csv（主表，含 isFraud 标签）和 train_identity.csv（身份/设备补充）。
  2. 按 TransactionID 做 left join 合并：以交易主表为准，把 identity 贴上去。
     —— 用 left 而非 inner：只有约 24% 交易有 identity，inner 会丢掉其余 76% 的真实交易（含欺诈），
        人为制造样本偏差。left 保留全部交易，没 identity 的字段留 NaN；"有无 identity"本身可作特征。
  3. 落盘成 parquet（读得快、保留 dtype）。
  4. 打印体检报告：形状 / 内存 / isFraud 欺诈率 / identity 覆盖率 / 时间跨度。

用法（在项目根目录、已激活 .venv 下）：
    python -m src.features.load_data
或直接：
    python src/features/load_data.py
"""

from pathlib import Path

import pandas as pd

# 项目根 = 本文件向上三级（src/features/load_data.py -> 项目根）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MERGED_PARQUET = DATA_PROCESSED / "train_merged.parquet"


def load_merged(save: bool = True) -> pd.DataFrame:
    """读两表 → 按 TransactionID left join → （可选）落盘 parquet → 返回合并后的 DataFrame。"""
    txn_path = DATA_RAW / "train_transaction.csv"
    idy_path = DATA_RAW / "train_identity.csv"

    print(f"[1/4] 读主表  {txn_path.name} …")
    txn = pd.read_csv(txn_path)
    print(f"      → {txn.shape[0]:,} 行 × {txn.shape[1]} 列")

    print(f"[2/4] 读身份表 {idy_path.name} …")
    idy = pd.read_csv(idy_path)
    print(f"      → {idy.shape[0]:,} 行 × {idy.shape[1]} 列")

    print("[3/4] 按 TransactionID 做 left join 合并 …")
    merged = txn.merge(idy, on="TransactionID", how="left")
    print(f"      → 合并后 {merged.shape[0]:,} 行 × {merged.shape[1]} 列")

    if save:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        print(f"[4/4] 落盘 parquet → {MERGED_PARQUET.relative_to(PROJECT_ROOT)} …")
        merged.to_parquet(MERGED_PARQUET, index=False)
        print(f"      → 已写出（{MERGED_PARQUET.stat().st_size / 1e6:.1f} MB）")

    _report(merged, idy)
    return merged


def _report(merged: pd.DataFrame, idy: pd.DataFrame) -> None:
    """打印体检报告：欺诈率 / identity 覆盖率 / 内存 / 时间跨度。"""
    n = len(merged)
    n_fraud = int(merged["isFraud"].sum())
    fraud_rate = n_fraud / n

    # identity 覆盖率：合并后凡是 identity 列有值的行（用 id_01 是否非空近似判定）
    n_with_idy = merged["TransactionID"].isin(idy["TransactionID"]).sum()

    mem_gb = merged.memory_usage(deep=True).sum() / 1e9

    # TransactionDT 是相对秒数（相对某基准时刻），看跨度即可
    dt_min, dt_max = merged["TransactionDT"].min(), merged["TransactionDT"].max()
    span_days = (dt_max - dt_min) / 86400

    print("\n" + "=" * 52)
    print("体检报告（合并后）")
    print("=" * 52)
    print(f"  样本数            : {n:,}")
    print(f"  列数              : {merged.shape[1]}")
    print(f"  欺诈样本 isFraud=1 : {n_fraud:,}")
    print(f"  欺诈率            : {fraud_rate:.4%}  (≈ 1 : {round(1/fraud_rate)})")
    print(f"  含 identity 的交易 : {n_with_idy:,}  ({n_with_idy/n:.2%})")
    print(f"  内存占用(deep)     : {mem_gb:.2f} GB")
    print(f"  TransactionDT 跨度 : {span_days:.1f} 天  (min={dt_min}, max={dt_max})")
    print("=" * 52)


if __name__ == "__main__":
    load_merged(save=True)
