"""概念漂移监控演示（硬点⑩）—— 冻结模型 + 滚动窗 AUC 衰减 + 监控触发器。

做法：在早期数据上训一个模型并冻结，在后续每个时间窗上算 AUC（不重训），
观察 AUC 随时间的走势；设一个触发器：窗口 ROC-AUC 跌破 (基线−δ) 就告警"该重训/重校准"。
配合 ⑩ 的口径：漂移是波动/分布型，监控要盯**分窗表现衰减**而非整体欺诈率。

用法：python -m src.eval.drift_monitor
产出：reports/figures/08_drift_monitor.png + reports/drift_monitor.md
"""

from src.report_io import write_report
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.model.train_baseline import LGB_PARAMS, prepare

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG = PROJECT_ROOT / "reports" / "figures" / "08_drift_monitor.png"
OUT_MD = PROJECT_ROOT / "reports" / "drift_monitor.md"

TRAIN_END = 83          # fit day<83
VAL_END = 90            # 早停 val [83,90)
WINDOW = 14             # 滚动窗天数
DELTA = 0.02            # 触发阈值：ROC-AUC 跌破 基线−0.02 告警


def main() -> None:
    print("读取数据 + 训练冻结模型（fit day<83）…")
    X, y, day = prepare()
    fit = day < TRAIN_END
    val = (day >= TRAIN_END) & (day < VAL_END)
    dtr = lgb.Dataset(X[fit], label=y[fit])
    dval = lgb.Dataset(X[val], label=y[val], reference=dtr)
    booster = lgb.train(
        LGB_PARAMS, dtr, num_boost_round=2000, valid_sets=[dval],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    day_max = int(day.max())
    wins = []
    for lo in range(VAL_END, day_max + 1, WINDOW):
        hi = lo + WINDOW
        mask = (day >= lo) & (day < hi)
        yy = y[mask].to_numpy()
        if mask.sum() < 500 or yy.sum() < 10:
            continue
        pp = booster.predict(X[mask], num_iteration=booster.best_iteration)
        wins.append({
            "lo": lo, "hi": hi, "n": int(mask.sum()), "fraud_rate": float(yy.mean()),
            "roc": float(roc_auc_score(yy, pp)), "pr": float(average_precision_score(yy, pp)),
        })

    base = wins[0]["roc"]           # 基线 = 第一个（最近训练期）窗口的 ROC-AUC
    trigger = base - DELTA
    print(f"\n基线 ROC-AUC（首窗）={base:.4f}，触发线={trigger:.4f}（跌破即告警）")
    n_alarm = 0
    for w in wins:
        alarm = w["roc"] < trigger
        n_alarm += alarm
        print(f"  [{w['lo']:>3},{w['hi']:>3}) n={w['n']:>6,} 欺诈率={w['fraud_rate']:.2%} "
              f"ROC={w['roc']:.4f} PR={w['pr']:.4f} {'⚠️ ALARM' if alarm else ''}")

    _plot(wins, base, trigger)
    _write_md(wins, base, trigger, n_alarm)
    print(f"\n✅ 图 08 + {OUT_MD.relative_to(PROJECT_ROOT)}")


def _plot(wins, base, trigger):
    xs = [w["lo"] for w in wins]
    roc = [w["roc"] for w in wins]
    fr = [w["fraud_rate"] for w in wins]
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(xs, roc, marker="o", color="#4C72B0", label="Window ROC-AUC")
    ax1.axhline(base, color="#55A868", ls=":", lw=1, label=f"baseline {base:.3f}")
    ax1.axhline(trigger, color="#C44E52", ls="--", lw=1.2, label=f"trigger {trigger:.3f}")
    for w in wins:
        if w["roc"] < trigger:
            ax1.scatter([w["lo"]], [w["roc"]], color="#C44E52", zorder=5, s=40)
    ax1.set_xlabel("Window start day")
    ax1.set_ylabel("ROC-AUC", color="#4C72B0")
    ax2 = ax1.twinx()
    ax2.plot(xs, fr, marker="s", ms=3, color="#8172B3", alpha=0.5, label="Fraud rate")
    ax2.set_ylabel("Fraud rate", color="#8172B3")
    ax1.set_title("Frozen model: windowed ROC-AUC decay + retrain trigger")
    ax1.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG, dpi=120)
    plt.close()


def _write_md(wins, base, trigger, n_alarm):
    last = wins[-1]
    decay = last["roc"] - base
    L = [
        "# 概念漂移监控演示（硬点⑩）\n",
        "冻结早期模型（fit day<83），在后续每 14 天窗上算 AUC（不重训），看衰减 + 监控触发器。",
        f"基线 ROC-AUC（首窗）={base:.4f}，触发线=基线−{DELTA}={trigger:.4f}，共 {len(wins)} 个窗、{n_alarm} 个触发告警。\n",
        "## 滚动窗 AUC\n",
        "| 窗口(day) | n | 欺诈率 | ROC-AUC | PR-AUC | 告警 |",
        "|-----------|---|--------|---------|--------|------|",
    ]
    for w in wins:
        L.append(f"| [{w['lo']},{w['hi']}) | {w['n']:,} | {w['fraud_rate']:.2%} | "
                 f"{w['roc']:.4f} | {w['pr']:.4f} | {'⚠️' if w['roc'] < trigger else ''} |")
    L += [
        "",
        "## 结论（按实际数字）",
        f"- 末窗 ROC-AUC {last['roc']:.4f} vs 基线 {base:.4f}（Δ {decay:+.4f}）。",
        "- 监控盯**分窗表现衰减**（而非整体欺诈率）：欺诈率列波动明显但聚合稳，印证 ⑩「波动/分布型漂移」。",
        "- 触发器：窗口 ROC-AUC 跌破 基线−δ 即告警「该重训/近窗重校准」——这就是 ③⑩ 那条控制回路的监控端。",
        # 判读要引用「PR 比 ROC 更敏感」，那就把两者的波动幅度**由机器算出来**，
        # 免得人写区自己去减——减错了也没人发现。
        f"- 波动幅度：ROC-AUC {max(w['roc'] for w in wins) - min(w['roc'] for w in wins):.4f}"
        f"、PR-AUC **{max(w['pr'] for w in wins) - min(w['pr'] for w in wins):.4f}**"
        f"（PR 区间 [{min(w['pr'] for w in wins):.4f}, {max(w['pr'] for w in wins):.4f}]）。",
        f"- **PR-AUC 相对基线的最大落差 {wins[0]['pr'] - min(w['pr'] for w in wins):.4f}**"
        f"（基线 {wins[0]['pr']:.4f} → 最低 {min(w['pr'] for w in wins):.4f}），"
        f"同期 ROC 落差仅 {wins[0]['roc'] - min(w['roc'] for w in wins):.4f}"
        f" —— 正类监控该盯 PR。",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    write_report(OUT_MD, "\n".join(L))


if __name__ == "__main__":
    main()
