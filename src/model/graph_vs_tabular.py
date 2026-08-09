"""纯表 vs 表+图 干净对照（防守点⑥ 的量化依据）。

同 baseline 时间切分（fit<132/val[132,146)/test≥146）、同 LGB 配置，只差图特征。
产出 delta（PR-AUC / ROC-AUC / recall@容量）——即"轻量图特征带来了 X"，
是 ⑥"为何图特征而非 GNN"的量化前提。**delta 可能≈0，那是诚实结论，不硬凑。**

用法：python -m src.model.graph_vs_tabular
产出：控制台对比 + reports/graph_vs_tabular.md
"""

from src.report_io import write_report
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.model.train_baseline import CAP_FRACS, LGB_PARAMS, T0, VAL_DAYS, prec_recall_at_topk

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = PROJECT_ROOT / "data" / "processed" / "train_merged.parquet"
# 图特征文件可由环境变量覆盖（泄漏审计：指向 21/60 天版本）
GRAPH = Path(os.environ.get("GRAPH_FILE", PROJECT_ROOT / "data" / "processed" / "graph_features.parquet"))
# md 按图文件的 embargo 后缀命名，避免审计运行覆盖主结果
_gsuffix = GRAPH.stem.replace("graph_features", "")
OUT_MD = PROJECT_ROOT / "reports" / f"graph_vs_tabular{_gsuffix}.md"

SECS_PER_DAY = 86_400


def load():
    df = pd.read_parquet(BASE)
    gf = pd.read_parquet(GRAPH)
    graph_cols = [c for c in gf.columns if c != "TransactionID"]
    df = df.merge(gf, on="TransactionID", how="left")
    day = ((df["TransactionDT"] // SECS_PER_DAY) - (df["TransactionDT"] // SECS_PER_DAY).min()).to_numpy()
    y = df["isFraud"].astype(int)
    X = df.drop(columns=["isFraud", "TransactionID", "TransactionDT"])
    for c in X.select_dtypes(include="object").columns:
        X[c] = X[c].astype("category")
    return X, y, day, graph_cols


def fit_eval(X, y, day, cols):
    fit = day < (T0 - VAL_DAYS)
    val = (day >= T0 - VAL_DAYS) & (day < T0)
    test = day >= T0
    dtr = lgb.Dataset(X.loc[fit, cols], label=y[fit])
    dval = lgb.Dataset(X.loc[val, cols], label=y[val], reference=dtr)
    booster = lgb.train(
        LGB_PARAMS, dtr, num_boost_round=2000, valid_sets=[dval],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    yte = y[test].to_numpy()
    p = booster.predict(X.loc[test, cols], num_iteration=booster.best_iteration)
    m = {"pr": average_precision_score(yte, p), "roc": roc_auc_score(yte, p),
         "best_iter": booster.best_iteration}
    for f in CAP_FRACS:
        _, rec, _ = prec_recall_at_topk(yte, p, f)
        m[f"rec@{f:.1%}"] = rec
    return m, booster


def main():
    print("读取 base + graph，合并 …")
    X, y, day, graph_cols = load()
    tab_cols = [c for c in X.columns if c not in graph_cols]
    print(f"  表特征 {len(tab_cols)}，图特征 {len(graph_cols)}")

    print("\nA. 纯表 …")
    A, _ = fit_eval(X, y, day, tab_cols)
    print(f"   PR-AUC {A['pr']:.4f}  ROC-AUC {A['roc']:.4f}  recall@1% {A['rec@1.0%']:.3f}")
    print("B. 表+图 …")
    B, bst = fit_eval(X, y, day, tab_cols + graph_cols)
    print(f"   PR-AUC {B['pr']:.4f}  ROC-AUC {B['roc']:.4f}  recall@1% {B['rec@1.0%']:.3f}")

    print(f"\n=== delta（表+图 − 纯表）===")
    print(f"  PR-AUC {B['pr']-A['pr']:+.4f}   ROC-AUC {B['roc']-A['roc']:+.4f}   "
          f"recall@1% {B['rec@1.0%']-A['rec@1.0%']:+.3f}   recall@2% {B['rec@2.0%']-A['rec@2.0%']:+.3f}")

    # 图特征在 B 模型里的重要性 + 排名
    imp = pd.Series(bst.feature_importance(importance_type="gain"),
                    index=tab_cols + graph_cols).sort_values(ascending=False)
    ranks = {c: int(np.where(imp.index == c)[0][0]) + 1 for c in graph_cols}
    total = len(imp)
    print(f"\n图特征重要性排名（共 {total} 特征）：")
    for c in graph_cols:
        print(f"  #{ranks[c]:>3}/{total}  gain={imp[c]:>12.0f}  {c}")

    _write_md(A, B, graph_cols, ranks, total, imp)
    print(f"\n✅ {OUT_MD.relative_to(PROJECT_ROOT)}")


def _write_md(A, B, graph_cols, ranks, total, imp):
    dpr, droc = B["pr"] - A["pr"], B["roc"] - A["roc"]
    best_gc = min(graph_cols, key=lambda c: ranks[c])
    L = [
        "# 纯表 vs 表+图 干净对照（防守点⑥ 量化依据）\n",
        "同切分（fit<132/test≥146）、同 LGB 配置，只差图特征。**delta 可能≈0 = 诚实结论（图信号已被匿名特征吸收）。**\n",
        "## 结果（test）\n",
        "| 臂 | PR-AUC | ROC-AUC | recall@0.5% | recall@1% | recall@2% |",
        "|----|--------|---------|-------------|-----------|-----------|",
        f"| 纯表 | {A['pr']:.4f} | {A['roc']:.4f} | {A['rec@0.5%']:.3f} | {A['rec@1.0%']:.3f} | {A['rec@2.0%']:.3f} |",
        f"| 表+图 | {B['pr']:.4f} | {B['roc']:.4f} | {B['rec@0.5%']:.3f} | {B['rec@1.0%']:.3f} | {B['rec@2.0%']:.3f} |",
        f"| **delta** | **{dpr:+.4f}** | **{droc:+.4f}** | {B['rec@0.5%']-A['rec@0.5%']:+.3f} | {B['rec@1.0%']-A['rec@1.0%']:+.3f} | {B['rec@2.0%']-A['rec@2.0%']:+.3f} |",
        "",
        f"## 图特征重要性（共 {total} 特征）\n",
        "| 图特征 | gain 排名 | gain |",
        "|--------|-----------|------|",
    ]
    for c in sorted(graph_cols, key=lambda c: ranks[c]):
        L.append(f"| {c} | #{ranks[c]}/{total} | {imp[c]:.0f} |")
    L += [
        "",
        "## 结论（按实际数字，接 ⑥）",
        f"- 表+图相对纯表：PR-AUC {dpr:+.4f}、ROC-AUC {droc:+.4f}。最有用的图特征是 `{best_gc}`（排名 #{ranks[best_gc]}）。",
        "- 若 delta 微弱：印证「图信号已被 C1-C14/V 匿名计数特征吸收」→ **用轻量图特征（度/prior 欺诈率/fan-out）而非 GNN**：轻量特征已吃掉大部分图价值，GNN 的边际增量不值其训练/推理/服务复杂度。",
        "- 时间因果：结构型（prior_cnt/fan-out）只用 t 之前的边；标签型（prior_fraud_rate）邻居标签留 21 天 embargo——图特征版防守点②，标签+结构两层泄漏都防住。",
        "- 团伙叙事（喂②风控/④业务安全两张皮）：组合键（card1+邮箱/设备）的 prior_fraud_rate 与 fan-out 是「共享稀有实体」的团伙信号。",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    write_report(OUT_MD, "\n".join(L))


if __name__ == "__main__":
    main()
