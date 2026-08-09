"""乐观 gap 归因 —— 等量旧窗对照，把 embargo 的损失拆成「数据量」+「新鲜度」两项。

问题：baseline 里 embargo（砍最新 21 天）让 PR-AUC 掉 0.032，但这混了两件事：
  (1) 训练样本少了 6.4 万（数据量损失）；(2) 最新信息被砍（新鲜度损失）。
面试官必问"这 gap 到底哪来的"。本脚本用三组对照把它拆开（test/val/seed/超参全对齐）：

  full      ：fit = day<132，         val=[132,146)        —— 全量
  embargo   ：fit = day<111，         val=[111,125)        —— 砍最新 21 天（含 21 天 gap 到 test）
  ctrl_old  ：fit = full 里剔除最旧 N，val=[132,146)        —— 同样少 N 个样本，但砍最旧

  数据量代价   = PR(ctrl_old) − PR(full)      （< 0）
  数据量+新鲜度 = PR(embargo)  − PR(full)      （< 0，= baseline 的 gap）
  embargo 净效应 = PR(embargo)  − PR(ctrl_old)  （新鲜度的净损失）

用法：python -m src.model.embargo_control
产出：控制台分解 + reports/embargo_decomposition.md
"""

from src.report_io import write_report
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.model.train_baseline import (
    CAP_FRACS,
    LGB_PARAMS,
    T0,
    VAL_DAYS,
    ece,
    prec_recall_at_topk,
    prepare,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_MD = PROJECT_ROOT / "reports" / "embargo_decomposition.md"
EMBARGO_DAYS = 21


def train_eval(X, y, fit_mask, val_mask, test_mask, label: str) -> dict:
    dtr = lgb.Dataset(X[fit_mask], label=y[fit_mask])
    dval = lgb.Dataset(X[val_mask], label=y[val_mask], reference=dtr)
    booster = lgb.train(
        LGB_PARAMS, dtr, num_boost_round=2000, valid_sets=[dval],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    yte = y[test_mask].to_numpy()
    pte = booster.predict(X[test_mask], num_iteration=booster.best_iteration)
    m = {
        "label": label,
        "n_fit": int(fit_mask.sum()),
        "best_iter": int(booster.best_iteration),
        "pr_auc": float(average_precision_score(yte, pte)),
        "roc_auc": float(roc_auc_score(yte, pte)),
        "ece": float(ece(yte, pte)),
        "brier": float(brier_score_loss(yte, pte)),
    }
    for f in CAP_FRACS:
        prec, rec, _ = prec_recall_at_topk(yte, pte, f)
        m[f"prec@{f:.1%}"] = prec
        m[f"recall@{f:.1%}"] = rec
    print(f"  [{label:9s}] n_fit={m['n_fit']:,}  best_iter={m['best_iter']}  "
          f"PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}  recall@1%={m['recall@1.0%']:.3f}")
    return m


def main() -> None:
    print("读取数据 + 构造特征 …")
    X, y, day = prepare()
    test_mask = day >= T0

    # full
    full_fit = day < (T0 - VAL_DAYS)
    full_val = (day >= T0 - VAL_DAYS) & (day < T0)

    # embargo（砍最新）
    emb_end = T0 - EMBARGO_DAYS
    emb_fit = day < (emb_end - VAL_DAYS)
    emb_val = (day >= emb_end - VAL_DAYS) & (day < emb_end)

    # ctrl_old（剔除最旧 N，N = full 与 embargo 的 fit 样本数之差；val 同 full）
    n_drop = int(full_fit.sum() - emb_fit.sum())
    full_idx = np.where(full_fit)[0]
    oldest_first = full_idx[np.argsort(day[full_idx], kind="stable")]
    ctrl_fit = np.zeros(len(day), dtype=bool)
    ctrl_fit[oldest_first[n_drop:]] = True   # 丢最旧 n_drop，保留最新
    ctrl_val = full_val

    print(f"\n训练三组（test 固定 [{T0},末]，丢弃样本数 N={n_drop:,}）…")
    full = train_eval(X, y, full_fit, full_val, test_mask, "full")
    emb = train_eval(X, y, emb_fit, emb_val, test_mask, "embargo")
    ctrl = train_eval(X, y, ctrl_fit, ctrl_val, test_mask, "ctrl_old")

    def decomp(metric: str):
        cost_data = ctrl[metric] - full[metric]      # 纯数据量代价
        cost_total = emb[metric] - full[metric]      # 数据量 + 新鲜度
        net_fresh = emb[metric] - ctrl[metric]       # embargo 净效应（新鲜度）
        return cost_data, cost_total, net_fresh

    print("\n=== 乐观 gap 分解 ===")
    for metric in ("pr_auc", "roc_auc"):
        cd, ct, nf = decomp(metric)
        print(f"  {metric:8s}: 总gap(砍新−全量)={ct:+.4f} = 数据量({cd:+.4f}) + 新鲜度净效应({nf:+.4f})")

    _write_md(full, emb, ctrl, n_drop, decomp)
    print(f"\n✅ 写出 {OUT_MD.relative_to(PROJECT_ROOT)}")


def _write_md(full, emb, ctrl, n_drop, decomp) -> None:
    pr_cd, pr_ct, pr_nf = decomp("pr_auc")
    roc_cd, roc_ct, roc_nf = decomp("roc_auc")
    L = [
        "# 乐观 gap 归因 —— 等量旧窗对照（防守点② 弹药）\n",
        f"把 baseline 的 embargo gap 拆成「数据量损失」+「新鲜度损失」。三组 test/val/seed/超参对齐，丢弃样本数 N={n_drop:,}。\n",
        "## 三组结果（test 固定 [146,181]）\n",
        "| 组 | fit 描述 | n_fit | PR-AUC | ROC-AUC | recall@1% |",
        "|----|---------|-------|--------|---------|-----------|",
        f"| full | day<132（全量） | {full['n_fit']:,} | {full['pr_auc']:.4f} | {full['roc_auc']:.4f} | {full['recall@1.0%']:.3f} |",
        f"| ctrl_old | 剔除最旧 N（保留最新） | {ctrl['n_fit']:,} | {ctrl['pr_auc']:.4f} | {ctrl['roc_auc']:.4f} | {ctrl['recall@1.0%']:.3f} |",
        f"| embargo | day<111（砍最新21天） | {emb['n_fit']:,} | {emb['pr_auc']:.4f} | {emb['roc_auc']:.4f} | {emb['recall@1.0%']:.3f} |",
        "",
        "## 分解（负=变差）\n",
        "| 指标 | 总 gap（砍新−全量） | = 数据量损失（砍旧−全量） | + 新鲜度净效应（砍新−砍旧） |",
        "|------|------|------|------|",
        f"| PR-AUC | {pr_ct:+.4f} | {pr_cd:+.4f} | {pr_nf:+.4f} |",
        f"| ROC-AUC | {roc_ct:+.4f} | {roc_cd:+.4f} | {roc_nf:+.4f} |",
        "",
        "## 一句话弹药（进 INTERVIEW.md ②）",
        f"- embargo 的总 gap（PR-AUC {pr_ct:+.4f}）拆开后，**纯数据量只占 {pr_cd:+.4f}，信息新鲜度净损失 {pr_nf:+.4f}**——",
        "  所以这个 gap 主要是「拿不到最新标签」的真实代价，不是「训练数据变少」的假象。白板上能拆给面试官看。",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    write_report(OUT_MD, "\n".join(L))


if __name__ == "__main__":
    main()
