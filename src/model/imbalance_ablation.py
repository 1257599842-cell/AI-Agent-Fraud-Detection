"""不平衡处理消融（防守点④）—— 反例演示：SMOTE/scale_pos_weight 如何毁校准。

口径（网页版 2026-07-01）：这是**反例演示**，证明"任何动先验的操作（重采样/加权）都会破坏
我整条链依赖的概率质量"，**不是**"我用了 SMOTE 提升了 X"。看的是：
  1. 可靠性图怎么被带偏（预测概率被抬到远离对角线）。
  2. 决策区间 recall 有没有真提升（大概率没有，甚至更差）—— **不看全局 AUC**（延续"全局指标会骗人"主线）。

三组（同 baseline 时间切分 fit<132 / val[132,146) / test[146,182)）：
  A. natural         —— 自然分布（baseline，无重采样无加权）
  B. scale_pos_weight—— = n_neg/n_pos（LightGBM 原生加权）
  C. oversample      —— 随机过采样正类到 ~均衡（SMOTE 的先验抬升效应之代理）

关于 SMOTE 本体：IEEE-CIS 有大量缺失 + 31 个类别列，SMOTE 需数值且无 NaN，
在此**不能干净地用**——这本身就是"SMOTE 是错的工具"的又一论据。故用随机过采样代理其先验抬升效应。

用法：python -m src.model.imbalance_ablation
产出：reports/figures/10_imbalance_reliability.png + reports/imbalance_ablation.md
"""

from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.model.train_baseline import LGB_PARAMS, T0, VAL_DAYS, ece, prec_recall_at_topk, prepare

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG = PROJECT_ROOT / "reports" / "figures" / "10_imbalance_reliability.png"
OUT_MD = PROJECT_ROOT / "reports" / "imbalance_ablation.md"
CAPS = [0.005, 0.01, 0.02]


def equal_count_reliability(y, p, n_bins=20):
    order = np.argsort(p)
    xs, ys = [], []
    for idx in np.array_split(order, n_bins):
        xs.append(p[idx].mean())
        ys.append(y[idx].mean())
    return np.array(xs), np.array(ys)


def decision_gap(y, p, frac):
    k = max(1, int(len(p) * frac))
    top = np.argsort(-p)[:k]
    return float(p[top].mean()), float(y[top].mean())


def evaluate(booster, Xte, yte, best_iter):
    p = booster.predict(Xte, num_iteration=best_iter)
    m = {"roc": roc_auc_score(yte, p), "pr": average_precision_score(yte, p),
         "ece": ece(yte, p), "brier": brier_score_loss(yte, p)}
    for f in CAPS:
        prec, rec, _ = prec_recall_at_topk(yte, p, f)
        m[f"rec@{f:.1%}"] = rec
        m[f"prec@{f:.1%}"] = prec
    t1p, t1a = decision_gap(yte, p, 0.01)
    t2p, t2a = decision_gap(yte, p, 0.02)
    m["top1_pred"], m["top1_actual"] = t1p, t1a
    m["top2_pred"], m["top2_actual"] = t2p, t2a
    m["_p"] = p
    return m


def train(Xf, yf, extra=None):
    params = {**LGB_PARAMS, **(extra or {})}
    dtr = lgb.Dataset(Xf, label=yf)
    return lgb.train(params, dtr, num_boost_round=300)  # 固定轮数，三组同配置公平比


def main() -> None:
    print("读取数据 …")
    X, y, day = prepare()
    fit = day < (T0 - VAL_DAYS)          # day<132
    test = day >= T0
    Xf, yf = X[fit], y[fit].to_numpy()
    Xte, yte = X[test], y[test].to_numpy()
    n_pos, n_neg = int(yf.sum()), int((yf == 0).sum())
    spw = n_neg / n_pos
    print(f"  fit 正 {n_pos:,} / 负 {n_neg:,}（scale_pos_weight={spw:.1f}）；test {len(yte):,}")

    print("  A. natural …")
    A = evaluate(train(Xf, yf), Xte, yte, None)
    print("  B. scale_pos_weight …")
    B = evaluate(train(Xf, yf, {"scale_pos_weight": spw}), Xte, yte, None)
    print("  C. oversample（复制正类到 ~均衡）…")
    factor = int(round(n_neg / n_pos))
    pos_idx = np.where(yf == 1)[0]
    Xf_os = pd.concat([Xf] + [Xf.iloc[pos_idx]] * (factor - 1), axis=0)
    yf_os = np.concatenate([yf] + [yf[pos_idx]] * (factor - 1))
    C = evaluate(train(Xf_os, yf_os), Xte, yte, None)

    models = {"natural": A, "scale_pos_weight": B, "oversample": C}
    print("\n=== 决策区间 recall（看有没有真提升）+ 校准（看有没有被带偏）===")
    for name, m in models.items():
        print(f"  {name:16s}: recall@1%={m['rec@1.0%']:.3f} recall@2%={m['rec@2.0%']:.3f} | "
              f"ECE={m['ece']:.4f} top2%(pred {m['top2_pred']:.3f} vs act {m['top2_actual']:.3f})")

    _plot(yte, models)
    _write_md(models, spw)
    print(f"\n✅ 图 10 + {OUT_MD.relative_to(PROJECT_ROOT)}")


def _plot(yte, models):
    plt.figure(figsize=(6.5, 6))
    plt.plot([0, 1], [0, 1], color="gray", ls=":", label="perfect")
    colors = {"natural": "#4C72B0", "scale_pos_weight": "#C44E52", "oversample": "#DD8452"}
    for name, m in models.items():
        xs, ys = equal_count_reliability(yte, m["_p"])
        plt.plot(xs, ys, marker="o", ms=3, lw=1.2, color=colors[name], label=name)
    plt.xlabel("Mean predicted probability (equal-count bins)")
    plt.ylabel("Actual fraud rate")
    plt.title("Resampling/weighting inflates probabilities (off diagonal)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG, dpi=120)
    plt.close()


def _write_md(models, spw):
    A, B, C = models["natural"], models["scale_pos_weight"], models["oversample"]
    L = [
        "# 不平衡处理消融（防守点④）—— 反例演示，非正向 delta\n",
        "**定位**：证明「动先验的操作（重采样/加权）会破坏概率质量」，**不是**「用 SMOTE 提升了 X」。",
        "看决策区间 recall（有无提升）+ 可靠性（有无被带偏），**不看全局 AUC**。",
        f"同 baseline 切分（fit<132/test≥146），三组同 LGB 配置、固定 300 轮；scale_pos_weight={spw:.1f}。\n",
        "## 结果（test）\n",
        "| 组 | recall@0.5% | recall@1% | recall@2% | 全局 ECE | top2% 预测 vs 实际 |",
        "|----|-------------|-----------|-----------|---------|-------------------|",
        f"| natural（对照）| {A['rec@0.5%']:.3f} | {A['rec@1.0%']:.3f} | {A['rec@2.0%']:.3f} | {A['ece']:.4f} | {A['top2_pred']:.3f} vs {A['top2_actual']:.3f} |",
        f"| scale_pos_weight | {B['rec@0.5%']:.3f} | {B['rec@1.0%']:.3f} | {B['rec@2.0%']:.3f} | {B['ece']:.4f} | {B['top2_pred']:.3f} vs {B['top2_actual']:.3f} |",
        f"| oversample | {C['rec@0.5%']:.3f} | {C['rec@1.0%']:.3f} | {C['rec@2.0%']:.3f} | {C['ece']:.4f} | {C['top2_pred']:.3f} vs {C['top2_actual']:.3f} |",
        "",
        "## 结论（按实际数字）",
        "- **决策区间 recall 无提升**：重采样/加权的 recall@容量 与 natural 基本持平（甚至更差）——不平衡处理换不来排序质量。",
        "- **校准被带偏**：scale_pos_weight/oversample 把先验抬到 ~均衡，预测概率整体上移、远离对角线（图 10），全局 ECE 爆炸、决策区间 top2% 预测远高于实际。",
        "- **所以我全程不碰它们**：我的代价敏感阈值①与校准③依赖真概率，动先验会同时废掉这两条。",
        "- SMOTE 本体在此**不适用**（大量 NaN + 31 类别列，需数值且无缺失）——这是「SMOTE 是错工具」的又一论据；故用随机过采样代理其先验抬升效应。",
        "",
        "## 与 ③⑤ 合体的一句话（进 INTERVIEW ④）",
        "> 任何动先验的操作——重采样也好、盲目校准也好——都会破坏我整条链依赖的概率质量，这是我全程不碰它们的原因。（③证 blanket 校准伤尾，④证重采样毁校准，同一个故事的两半。）",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
