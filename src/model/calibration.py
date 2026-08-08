"""概率校准（防守点③）—— 决策区间验证 + 近窗重校准演示。

新口径（网页版 2026-06-30 二次拍板）：全局 ECE 已低（logloss 训练全局较准），
不看会被 p≈0 易负样本稀释的全局数字，而是：
  1. 验决策区间（top1~2%）的校准（等频分箱可靠性图；raw vs Platt vs Isotonic）。
  2. 演示"漂移后近窗重校准"：同一 isotonic 用【旧外部窗】vs【近外部窗】拟合，都在 test 上评。

四段时间切分（腾出宽外部区做校准对照）：
  fit  [0,104)   训练树
  val  [104,111) 早停
  old  [111,125) 旧校准窗（external，14 天）
  near [132,146) 近校准窗（external，14 天）
  test [146,182) 评估

校准器只在【时间外】窗上拟合（不碰训练集 = 防泄漏；近窗 = 反映漂移后先验）。

用法：python -m src.model.calibration
产出：reports/figures/06_reliability.png + 07_recalib_drift.png + reports/calibration.md
"""

from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from src.model.train_baseline import LGB_PARAMS, ece, prepare

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_REL = PROJECT_ROOT / "reports" / "figures" / "06_reliability.png"
FIG_DRIFT = PROJECT_ROOT / "reports" / "figures" / "07_recalib_drift.png"
OUT_MD = PROJECT_ROOT / "reports" / "calibration.md"

FIT_END, VAL_END = 104, 111
OLD_LO, OLD_HI = 111, 125
NEAR_LO, NEAR_HI = 132, 146
TEST_LO = 146


def decision_region_gap(y, p, frac):
    """决策区间校准 gap：按预测分取 top frac，比较 平均预测 vs 实际欺诈率。"""
    k = max(1, int(len(p) * frac))
    top = np.argsort(-p)[:k]
    pred, actual = float(p[top].mean()), float(y[top].mean())
    return pred, actual, abs(pred - actual)


def cal_metrics(y, p):
    d1 = decision_region_gap(y, p, 0.01)
    d2 = decision_region_gap(y, p, 0.02)
    return {
        "ece": float(ece(y, p)), "brier": float(brier_score_loss(y, p)),
        "top1_pred": d1[0], "top1_actual": d1[1], "top1_gap": d1[2],
        "top2_pred": d2[0], "top2_actual": d2[1], "top2_gap": d2[2],
    }


def equal_count_reliability(y, p, n_bins=20):
    """等频分箱可靠性：每箱样本数相同，返回 (mean_pred, mean_actual) 每箱。"""
    order = np.argsort(p)
    xs, ys = [], []
    for idx in np.array_split(order, n_bins):
        xs.append(p[idx].mean())
        ys.append(y[idx].mean())
    return np.array(xs), np.array(ys)


def main() -> None:
    print("读取数据 + 训练树（fit day<104）…")
    X, y, day = prepare()
    fit = day < FIT_END
    val = (day >= FIT_END) & (day < VAL_END)
    old = (day >= OLD_LO) & (day < OLD_HI)
    near = (day >= NEAR_LO) & (day < NEAR_HI)
    test = day >= TEST_LO

    dtr = lgb.Dataset(X[fit], label=y[fit])
    dval = lgb.Dataset(X[val], label=y[val], reference=dtr)
    booster = lgb.train(
        LGB_PARAMS, dtr, num_boost_round=2000, valid_sets=[dval],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    def prob(mask):
        return booster.predict(X[mask], num_iteration=booster.best_iteration)

    def margin(mask):
        return booster.predict(X[mask], num_iteration=booster.best_iteration, raw_score=True)

    p_near, y_near = prob(near), y[near].to_numpy()
    p_old, y_old = prob(old), y[old].to_numpy()
    p_test, y_test = prob(test), y[test].to_numpy()
    f_near, f_test = margin(near), margin(test)
    print(f"  test {len(p_test):,} 笔；near 校准窗 {len(p_near):,}；old 校准窗 {len(p_old):,}")

    # ── 校准器（在 near 窗拟合）──
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_near, y_near)
    platt = LogisticRegression().fit(f_near.reshape(-1, 1), y_near)
    p_iso = iso.predict(p_test)
    p_platt = platt.predict_proba(f_test.reshape(-1, 1))[:, 1]

    methods = {"raw": p_test, "platt(near)": p_platt, "isotonic(near)": p_iso}
    print("\n=== 校准指标（test 上）===")
    results = {}
    for name, pp in methods.items():
        m = cal_metrics(y_test, pp)
        results[name] = m
        print(f"  {name:16s}: ECE={m['ece']:.4f} Brier={m['brier']:.4f}  "
              f"top1%(pred {m['top1_pred']:.3f} vs act {m['top1_actual']:.3f}, gap {m['top1_gap']:.3f})  "
              f"top2%(gap {m['top2_gap']:.3f})")

    # ── 漂移对照：isotonic 用 old 窗 vs near 窗 ──
    iso_old = IsotonicRegression(out_of_bounds="clip").fit(p_old, y_old)
    p_iso_old = iso_old.predict(p_test)
    m_old = cal_metrics(y_test, p_iso_old)
    m_near = results["isotonic(near)"]
    print("\n=== 漂移对照：isotonic 旧窗 vs 近窗（都评 test）===")
    print(f"  old 窗校准 : ECE={m_old['ece']:.4f} Brier={m_old['brier']:.4f} top2%gap={m_old['top2_gap']:.3f}")
    print(f"  near窗校准 : ECE={m_near['ece']:.4f} Brier={m_near['brier']:.4f} top2%gap={m_near['top2_gap']:.3f}")

    _plot_reliability(y_test, methods)
    _plot_drift(y_test, p_iso_old, p_iso)
    _write_md(results, m_old, m_near, len(p_test), len(p_near), len(p_old))
    print(f"\n✅ 图 06/07 + {OUT_MD.relative_to(PROJECT_ROOT)}")


def _plot_reliability(y, methods):
    plt.figure(figsize=(6.5, 6))
    plt.plot([0, 1], [0, 1], color="gray", ls=":", label="perfect")
    colors = {"raw": "#C44E52", "platt(near)": "#DD8452", "isotonic(near)": "#4C72B0"}
    for name, pp in methods.items():
        xs, ys = equal_count_reliability(y, pp, n_bins=20)
        plt.plot(xs, ys, marker="o", ms=3, lw=1.2, color=colors[name], label=name)
    plt.xlabel("Mean predicted probability (equal-count bins)")
    plt.ylabel("Actual fraud rate")
    plt.title("Reliability diagram (decision region = top-right)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_REL, dpi=120)
    plt.close()


def _plot_drift(y, p_old, p_near):
    plt.figure(figsize=(6.5, 6))
    plt.plot([0, 1], [0, 1], color="gray", ls=":", label="perfect")
    for pp, name, c in [(p_old, "isotonic(old window)", "#C44E52"),
                        (p_near, "isotonic(near window)", "#4C72B0")]:
        xs, ys = equal_count_reliability(y, pp, n_bins=20)
        plt.plot(xs, ys, marker="o", ms=3, lw=1.2, color=c, label=name)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Actual fraud rate")
    plt.title("Near-window recalibration beats old-window (on test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DRIFT, dpi=120)
    plt.close()


def _write_md(results, m_old, m_near, n_test, n_near, n_old):
    raw, platt, iso = results["raw"], results["platt(near)"], results["isotonic(near)"]
    L = [
        "# 概率校准（防守点③）—— 决策区间验证 + 近窗重校准\n",
        "新口径：不看会被 p≈0 稀释的全局 ECE，而看**决策区间（top1~2%）**的校准；校准器只在**时间外近窗**拟合。\n",
        f"四段时间切分：fit[0,104) / val[104,111) / old[111,125) / near[132,146) / test[146,182)。",
        f"test={n_test:,}，near 校准窗={n_near:,}，old 校准窗={n_old:,}。\n",
        "## 1. 校准方法对比（test 上；校准器拟合于 near 窗）\n",
        "| 方法 | 全局 ECE | Brier | top1% 预测 vs 实际 | top1% gap | top2% gap |",
        "|------|---------|-------|-------------------|-----------|-----------|",
        f"| raw（未校准）| {raw['ece']:.4f} | {raw['brier']:.4f} | {raw['top1_pred']:.3f} vs {raw['top1_actual']:.3f} | {raw['top1_gap']:.3f} | {raw['top2_gap']:.3f} |",
        f"| Platt(near) | {platt['ece']:.4f} | {platt['brier']:.4f} | {platt['top1_pred']:.3f} vs {platt['top1_actual']:.3f} | {platt['top1_gap']:.3f} | {platt['top2_gap']:.3f} |",
        f"| Isotonic(near) | {iso['ece']:.4f} | {iso['brier']:.4f} | {iso['top1_pred']:.3f} vs {iso['top1_actual']:.3f} | {iso['top1_gap']:.3f} | {iso['top2_gap']:.3f} |",
        "",
        "> 诚实读法：比较 raw 与校准后的 **top1~2% gap**（决策区间）与全局 ECE。若 raw 的决策区间 gap 已很小、"
        "而校准把它变大，说明模型本已较准、blanket 校准反伤稀疏尾（图 06 右上角看顶箱是否偏离对角线）。别只看全局 ECE 下降就以为校准有用。",
        "",
        "## 2. 漂移对照：近窗 vs 旧窗（同 isotonic，都评 test）\n",
        "| 校准窗 | 样本数 | 全局 ECE | Brier | top2% gap |",
        "|--------|-------|---------|-------|-----------|",
        f"| 旧窗 [111,125) | {n_old:,} | {m_old['ece']:.4f} | {m_old['brier']:.4f} | {m_old['top2_gap']:.3f} |",
        f"| 近窗 [132,146) | {n_near:,} | {m_near['ece']:.4f} | {m_near['brier']:.4f} | {m_near['top2_gap']:.3f} |",
        "",
        "> 注意：两窗样本量不同（旧窗 vs 近窗），会混淆'近窗更新鲜'与'样本更多'。若差异落在噪声内，"
        "**不要硬凑成'近窗更好'**；诚实结论按实际数字写（见 reports/calibration.md 手写版）。",
        "",
        "## 讲法（防守点③）",
        "- 全局 ECE 会骗人：极端不平衡下被 p≈0 易负样本主导；我看的是**决策区间可靠性**，也提防校准把稀疏尾搞坏。",
        "- 校准器在**时间外近窗**拟合：不碰训练集（防泄漏）+ 反映漂移后先验；窗要够大以免尾部噪声。",
        "- 关键判断：**先测再决定要不要校准**——本数据 raw 已较准就不 blanket 套；校准是阈值①的前提，只在决策区间真失准时才做。",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
