"""LightGBM baseline —— 硬点② 时间切分 + 自然分布。

严格按项目负责人 2026-06-30 拍板的"已定建模口径"（见 PROGRESS.md）：
  - 时间切分：test 固定 [t0, 末]=[146,181]；train = day < (t0 - embargo)。
      embargo=0  → train [0,145]（简单切，第一组数）
      embargo=21 → train [0,124]，空窗 [125,145] 丢弃（模拟标签延迟）
      两轮 AUC 之差 = "线下→线上乐观 gap"。
  - 自然分布：不重采样、不 scale_pos_weight。
  - 类别：object 列 → category，用 LightGBM 原生类别处理（纯 label 映射，无目标编码 → 无泄漏）。
  - 缺失：LightGBM 原生处理，不填均值。
  - 早停 val：从训练窗末尾切 14 天（在 test 之前，合法）。
  - 指标：PR-AUC / ROC-AUC / precision@{0.5,1,2}% / recall@同档 / ECE / Brier。

用法（项目根、已激活 .venv）：  python -m src.model.train_baseline
产出：控制台对比表 + reports/baseline_metrics.md
"""

from src.report_io import write_report
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARQUET = PROJECT_ROOT / "data" / "processed" / "train_merged.parquet"
OUT_MD = PROJECT_ROOT / "reports" / "baseline_metrics.md"

SECS_PER_DAY = 86_400
T0 = 146          # 测试集起始日（含），test = [146, 末]
VAL_DAYS = 14     # 训练窗末尾留作早停验证的天数
CAP_FRACS = [0.005, 0.01, 0.02]   # 复核容量：按分数取 top X%
SEED = 42

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",          # 早停用 auc（稳）；PR-AUC 等我们在 test 上自己算
    "learning_rate": 0.05,
    "num_leaves": 128,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 100,
    "verbose": -1,
    "seed": SEED,
    "n_jobs": -1,
}


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error（等宽分箱）。"""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += (m.sum() / len(p)) * abs(y[m].mean() - p[m].mean())
    return e


def prec_recall_at_topk(y: np.ndarray, p: np.ndarray, frac: float):
    """固定复核容量 = 按分数取 top frac：返回 precision@容量、recall@容量、k。"""
    k = max(1, int(len(p) * frac))
    top = np.argsort(-p)[:k]
    tp = float(y[top].sum())
    return tp / k, tp / float(y.sum()), k


def prepare() -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """读 parquet，构造特征矩阵 X、标签 y、每行的 day。object 列转 category。"""
    df = pd.read_parquet(PARQUET)
    day = (df["TransactionDT"] // SECS_PER_DAY).astype(int)
    day = (day - day.min()).to_numpy()
    y = df["isFraud"].astype(int)
    X = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"])
    # object → category（纯 label 映射，全量做不构成目标泄漏；类别在各切片间一致）
    obj_cols = X.select_dtypes(include="object").columns
    for c in obj_cols:
        X[c] = X[c].astype("category")
    print(f"  特征 {X.shape[1]} 列（其中类别列 {len(obj_cols)}），样本 {len(X):,}")
    return X, y, day


def run_once(X, y, day, embargo: int) -> dict:
    """按给定 embargo 训练并在 test=[T0,末] 上评估，返回指标 dict。"""
    train_end = T0 - embargo                       # 训练上界（不含）
    train_mask = day < (train_end - VAL_DAYS)      # 早停用 fit
    val_mask = (day >= train_end - VAL_DAYS) & (day < train_end)
    test_mask = day >= T0

    dtr = lgb.Dataset(X[train_mask], label=y[train_mask])
    dval = lgb.Dataset(X[val_mask], label=y[val_mask], reference=dtr)

    booster = lgb.train(
        LGB_PARAMS,
        dtr,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    yte = y[test_mask].to_numpy()
    pte = booster.predict(X[test_mask], num_iteration=booster.best_iteration)

    m = {
        "embargo": embargo,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "fraud_rate_train": float(y[train_mask].mean()),
        "fraud_rate_test": float(yte.mean()),
        "best_iter": int(booster.best_iteration),
        "pr_auc": float(average_precision_score(yte, pte)),
        "roc_auc": float(roc_auc_score(yte, pte)),
        "ece": float(ece(yte, pte)),
        "brier": float(brier_score_loss(yte, pte)),
    }
    for f in CAP_FRACS:
        prec, rec, k = prec_recall_at_topk(yte, pte, f)
        m[f"prec@{f:.1%}"] = prec
        m[f"recall@{f:.1%}"] = rec
        m[f"k@{f:.1%}"] = k
    return m


def main() -> None:
    print("读取数据 + 构造特征 …")
    X, y, day = prepare()

    results = []
    for embargo in (0, 21):
        print(f"\n=== 训练 embargo={embargo} 天 "
              f"(train day<{T0 - embargo}, test day>={T0}) ===")
        m = run_once(X, y, day, embargo)
        results.append(m)
        print(f"  best_iter={m['best_iter']}  n_train={m['n_train']:,}  n_test={m['n_test']:,}")
        print(f"  PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}  "
              f"ECE={m['ece']:.4f}  Brier={m['brier']:.4f}")
        for f in CAP_FRACS:
            print(f"    top {f:.1%} (k={m[f'k@{f:.1%}']:,}): "
                  f"precision={m[f'prec@{f:.1%}']:.3f}  recall={m[f'recall@{f:.1%}']:.3f}")

    _write_md(results)
    r0, r21 = results
    print("\n=== embargo 效应（后 − 前，负=变差）===")
    print(f"  PR-AUC : {r0['pr_auc']:.4f} → {r21['pr_auc']:.4f}  (Δ {r21['pr_auc']-r0['pr_auc']:+.4f})")
    print(f"  ROC-AUC: {r0['roc_auc']:.4f} → {r21['roc_auc']:.4f}  (Δ {r21['roc_auc']-r0['roc_auc']:+.4f})")
    print(f"\n✅ 写出 {OUT_MD.relative_to(PROJECT_ROOT)}")


def _write_md(results: list[dict]) -> None:
    r0, r21 = results
    L = ["# Baseline 指标 — LightGBM 自然分布 + 时间切分（硬点②）\n",
         "口径见 PROGRESS.md「已定建模口径」。test 固定 [146,181]，自然分布，原生类别/缺失。\n",
         "## 两轮对比（embargo 0 vs 21 天）\n",
         "> Δ = embargo效应（后−前）。AUC/precision/recall 负=变差；ECE/Brier 正=变差。\n",
         "| 指标 | 无 embargo | +21天 embargo | Δ(embargo效应) |",
         "|------|-----------|--------------|------------|",
         f"| PR-AUC | {r0['pr_auc']:.4f} | {r21['pr_auc']:.4f} | {r21['pr_auc']-r0['pr_auc']:+.4f} |",
         f"| ROC-AUC | {r0['roc_auc']:.4f} | {r21['roc_auc']:.4f} | {r21['roc_auc']-r0['roc_auc']:+.4f} |",
         f"| ECE | {r0['ece']:.4f} | {r21['ece']:.4f} | {r21['ece']-r0['ece']:+.4f} |",
         f"| Brier | {r0['brier']:.4f} | {r21['brier']:.4f} | {r21['brier']-r0['brier']:+.4f} |"]
    for f in CAP_FRACS:
        kf = f"prec@{f:.1%}"; rf = f"recall@{f:.1%}"
        L.append(f"| precision@top{f:.1%} | {r0[kf]:.3f} | {r21[kf]:.3f} | {r21[kf]-r0[kf]:+.3f} |")
        L.append(f"| recall@top{f:.1%} | {r0[rf]:.3f} | {r21[rf]:.3f} | {r21[rf]-r0[rf]:+.3f} |")
    L += ["",
          f"- 训练量：无 embargo n_train={r0['n_train']:,}；+21天 n_train={r21['n_train']:,}（少 {r0['n_train']-r21['n_train']:,}）。",
          f"- 训练期欺诈率 {r0['fraud_rate_train']:.3%} vs 测试期 {r0['fraud_rate_test']:.3%}（聚合近乎持平、略降；"
          f"EDA 看到的漂移是周度波动/分布型，非测试窗的单调抬升 —— 需带回项目负责人重定调）。",
          f"- test 样本 {r0['n_test']:,}。",
          "",
          "## 填入 README ▢",
          f"- baseline PR-AUC / ROC-AUC（无 embargo）：{r0['pr_auc']:.4f} / {r0['roc_auc']:.4f}",
          f"- 线下→线上乐观 gap（PR-AUC）：{r0['pr_auc']:.4f} → {r21['pr_auc']:.4f}（Δ {r21['pr_auc']-r0['pr_auc']:+.4f}，embargo 后掉了）",
          f"- recall@容量（=有效拦截率，top1%，无 embargo）：{r0['recall@1.0%']:.3f}",
          "",
          "> 校准意外：全局 ECE 仅 {:.4f}（很低）—— LightGBM logloss 训练全局已较准。但 3.4% 基率下等宽 ECE".format(r0['ece']),
          "> 被 p≈0 的易负样本主导，**决策区间（top1~2%）的校准仍需用可靠性图单独验**。embargo 使 ECE {:.4f}→{:.4f}".format(r0['ece'], r21['ece']),
          "> 变差，支持近窗重校准。硬点③口径改为「决策区间验证校准 + 漂移后近窗重校准」，待项目负责人确认。"]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    write_report(OUT_MD, "\n".join(L))


if __name__ == "__main__":
    main()
