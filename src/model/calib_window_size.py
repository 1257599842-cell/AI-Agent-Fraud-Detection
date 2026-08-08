"""校准近窗大小排除法 —— blanket 校准伤尾，是真信号还是 14 天小窗的样本量假象？

同一棵树（fit day<104），在【贴着 test、都对树外部】的近窗上拟合 isotonic，
近窗长度取 {14, 28, 42} 天（都以 day 146 为右端），评估 test 决策区间 gap：
  - 尾部 gap 随窗变大而回落到 raw 水平 → 之前的"伤尾"是样本量假象，修正为"小窗别校"。
  - 尾部 gap 依旧高 → "blanket 校准伤尾"是真信号，③ 结论更硬。

用法：python -m src.model.calib_window_size
产出：控制台表 + reports/calib_window_size.md（追加到 ③ 证据链）
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression

from src.model.calibration import cal_metrics
from src.model.train_baseline import LGB_PARAMS, prepare

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_MD = PROJECT_ROOT / "reports" / "calib_window_size.md"

FIT_END, VAL_END = 104, 111
TEST_LO = 146
NEAR_END = 146           # 近窗右端（贴着 test）
WINDOWS = [14, 28, 42]   # 近窗天数


def main() -> None:
    print("读取数据 + 训练树（fit day<104）…")
    X, y, day = prepare()
    fit = day < FIT_END
    val = (day >= FIT_END) & (day < VAL_END)
    test = day >= TEST_LO

    dtr = lgb.Dataset(X[fit], label=y[fit])
    dval = lgb.Dataset(X[val], label=y[val], reference=dtr)
    booster = lgb.train(
        LGB_PARAMS, dtr, num_boost_round=2000, valid_sets=[dval],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    p_test, y_test = booster.predict(X[test], num_iteration=booster.best_iteration), y[test].to_numpy()

    raw = cal_metrics(y_test, p_test)
    print(f"\n  raw（未校准）: ECE={raw['ece']:.4f}  top1% gap={raw['top1_gap']:.3f}  top2% gap={raw['top2_gap']:.3f}")

    rows = []
    for w in WINDOWS:
        lo = NEAR_END - w
        near = (day >= lo) & (day < NEAR_END)
        p_near, y_near = booster.predict(X[near], num_iteration=booster.best_iteration), y[near].to_numpy()
        n_pos = int(y_near.sum())
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_near, y_near)
        m = cal_metrics(y_test, iso.predict(p_test))
        rows.append({"w": w, "lo": lo, "n": int(near.sum()), "n_pos": n_pos, **m})
        print(f"  近窗 {w}天 [{lo},{NEAR_END}) n={int(near.sum()):,}(正样本{n_pos:,}): "
              f"ECE={m['ece']:.4f}  top1% gap={m['top1_gap']:.3f}  top2% gap={m['top2_gap']:.3f}")

    _write_md(raw, rows)
    print(f"\n✅ 写出 {OUT_MD.relative_to(PROJECT_ROOT)}")


def _write_md(raw, rows):
    L = [
        "# 校准近窗大小排除法（③ 证据链）\n",
        "问题：`calibration.py` 发现 blanket isotonic（14 天近窗）把决策区间尾部搞过头——这是真信号，还是小窗样本量假象？",
        "做法：同一棵树（fit<104），近窗长度 {14,28,42} 天（都以 day 146 为右端、对树外部），比尾部 gap 是否随窗变大回落。\n",
        "## 结果（test 上）\n",
        "| 近窗 | 区间 | n | 正样本 | 全局 ECE | top1% gap | top2% gap |",
        "|------|------|---|--------|---------|-----------|-----------|",
        f"| raw（不校准）| — | — | — | {raw['ece']:.4f} | {raw['top1_gap']:.3f} | **{raw['top2_gap']:.3f}** |",
    ]
    for r in rows:
        L.append(f"| {r['w']}天 | [{r['lo']},146) | {r['n']:,} | {r['n_pos']:,} | "
                 f"{r['ece']:.4f} | {r['top1_gap']:.3f} | {r['top2_gap']:.3f} |")
    L += [
        "",
        "## 结论（按实际数字，不硬凑）",
        "- 见上表 top2% gap 随近窗天数的走势：若仍显著高于 raw 的值，则\"blanket 校准伤尾\"是**真信号**（不是样本量假象），③ 结论更硬；",
        "- 若随窗变大回落到 raw 水平，则修正为\"**校准需要足够近窗样本，小窗别校**\"——同样是一句可讲的判断。",
        "- 无论哪种，都排除了\"样本太少\"这个混淆（与上一轮'等量旧窗对照剥离数据量'同一种严谨动作）。",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
