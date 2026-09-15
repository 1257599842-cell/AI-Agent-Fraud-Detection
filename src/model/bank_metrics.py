"""银行风控口径的指标：KS、十等分表、跨时间稳定性。

## 为什么单独做这一组
PR-AUC / ROC-AUC 是机器学习圈的通用语言；**KS 和十等分表是银行风控的通用语言**。
同一个模型、同一份分数，换一套口径表述，面向的读者就不同。

## 三样产出
1. **KS 的 delta**：纯表 → 表+图，与 `PR-AUC 0.5645→0.6032` 并列、同源同窗。
2. **十等分表**：按分数降序分十组，逐组好坏样本数、累计占比、逐组 KS。
   这是银行评分卡的标准交付物。
3. **训练窗 KS vs 测试窗 KS**：银行不只看 KS 高低，更看**跨时间稳不稳**。

## 口径纪律（两条，都踩过坑）
- **分数必须与 PR-AUC 同源**：读 `gvt_scores.parquet`，那是 `graph_vs_tabular.py`
  **同一次运行**落盘的两臂分数。不重训、不另跑——重训一次就多一份可能不同源的数字。
- **无 embargo 口径**：与对外报告的 `0.5645→0.6032` 一致（fit<132 / val[132,146) / test≥146）。
  不与 embargo 版混用。

## KS 的定义
KS = max(累计坏样本占比 − 累计好样本占比)，等价于 ROC 曲线上 TPR − FPR 的最大值。

用法：python -m src.model.bank_metrics
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.report_io import write_report

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "data" / "processed" / "gvt_scores.parquet"
REPORT = ROOT / "reports" / "bank_metrics.md"
N_BINS = 10


def ks(y, p):
    """KS 统计量 + 取到最大值的那个分数切点。

    按分数降序累计：坏样本累计占比 − 好样本累计占比，取最大值。
    与 `max(TPR − FPR)` 等价——这里直接按定义算，便于同时给出十等分表。
    """
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, dtype=float)
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    n_bad, n_good = ys.sum(), (~ys).sum()
    cum_bad = np.cumsum(ys) / n_bad
    cum_good = np.cumsum(~ys) / n_good
    d = cum_bad - cum_good
    i = int(np.argmax(d))
    return float(d[i]), float(ps[i]), int(i + 1)


def decile_table(y, p, n_bins=N_BINS):
    """十等分表：按分数降序等频分组，逐组好坏、累计占比、逐组 KS。"""
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, dtype=float)
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    n = len(ys)
    edges = [int(round(n * k / n_bins)) for k in range(n_bins + 1)]
    n_bad, n_good = ys.sum(), (~ys).sum()
    rows, cb, cg = [], 0, 0
    for k in range(n_bins):
        s, e = edges[k], edges[k + 1]
        seg = ys[s:e]
        bad, good = int(seg.sum()), int((~seg).sum())
        cb += bad
        cg += good
        rows.append({
            "组": k + 1, "分数区间": f"{ps[e-1]:.5f} – {ps[s]:.5f}",
            "样本": e - s, "坏": bad, "好": good,
            "组内坏率": bad / max(e - s, 1),
            "累计坏占比": cb / n_bad, "累计好占比": cg / n_good,
            "KS": cb / n_bad - cg / n_good,
        })
    return pd.DataFrame(rows)


def main():
    if not SCORES.exists():
        raise SystemExit(f"缺分数文件 {SCORES.name}，先跑 python -m src.model.graph_vs_tabular")
    d = pd.read_parquet(SCORES)
    te = d[d["split"] == "test"]
    fit = d[d["split"] == "fit"]
    y_te = te["isFraud"].to_numpy()

    ks_tab, cut_tab, _ = ks(y_te, te["p_tab"])
    ks_gra, cut_gra, _ = ks(y_te, te["p_graph"])
    ks_fit, _, _ = ks(fit["isFraud"].to_numpy(), fit["p_graph"])
    dec = decile_table(y_te, te["p_graph"])

    print(f"测试窗 {len(te):,} 笔，坏样本 {int(y_te.sum()):,}（{y_te.mean():.2%}）")
    print(f"  KS 纯表   {ks_tab:.4f}")
    print(f"  KS 表+图  {ks_gra:.4f}   delta {ks_gra - ks_tab:+.4f}")
    print(f"  KS 训练窗 {ks_fit:.4f}（样本内）  测试窗 {ks_gra:.4f}  "
          f"差 {ks_fit - ks_gra:+.4f}")

    _write(te, fit, ks_tab, ks_gra, ks_fit, cut_gra, dec)
    print(f"\n✅ → {REPORT.relative_to(ROOT)}")


def _write(te, fit, ks_tab, ks_gra, ks_fit, cut, dec):
    y = te["isFraud"].to_numpy()
    L = [
        "# 银行风控口径：KS · 十等分表 · 跨时间稳定性\n",
        "> 同一个模型、同一份分数，换一套口径表述。",
        "> PR-AUC / ROC-AUC 是机器学习圈的通用语言，**KS 与十等分表是银行风控的通用语言**。\n",
        "**口径**：与 `graph_vs_tabular.md` 的 `PR-AUC 0.5645 → 0.6032` **同源同窗**——",
        "读的是那一次运行落盘的两臂分数（`gvt_scores.parquet`），**未重训、未另跑**。",
        "时间切分 `fit < day 132 / val [132,146) / test ≥ 146`，**无 embargo**。\n",
        f"测试窗 **{len(te):,}** 笔，坏样本 **{int(y.sum()):,}**（{y.mean():.2%}）。\n",
        "## 1. KS 的 delta（纯表 → 表+图）\n",
        "| 模型 | KS | PR-AUC | ROC-AUC |", "|---|---|---|---|",
        f"| 纯表（431 特征） | {ks_tab:.4f} | 0.5645 | 0.9138 |",
        f"| 表+图（+15 图特征） | **{ks_gra:.4f}** | 0.6032 | 0.9306 |",
        f"| **delta** | **{ks_gra - ks_tab:+.4f}** | +0.0387 | +0.0168 |",
        "",
        f"> KS 由 **{ks_tab:.3f}** 升至 **{ks_gra:.3f}**，取到最大值的分数切点为 `p = {cut:.5f}`。\n",
        "### ⚠️ 这个 KS 高于信贷评分卡的常见区间，必须说明\n",
        "银行做授信评分卡的人，日常见到的 KS 在 **0.30–0.40**；",
        f"本项目 KS **{ks_gra:.2f}**，第一反应会是「是不是有泄漏」。原因是**场景不同，不可横向比**：\n",
        "- **交易反欺诈**：欺诈是**短期行为事件**，信号强且集中（同卡短窗高频、设备扇出、"
        "组合键历史欺诈率），一笔交易的证据密度远高于一个借款人的还款前景。",
        "- **信贷违约**：长周期结果，信号弱且被大量噪声稀释。\n",
        "> 两者的 KS 不能横向比较。",
        "> **另有泄漏审计支撑**：把标签隔离期从 21 天延长到 **60 天**，图特征增益仍为正",
        "> （**+0.039 → +0.023**），缩水的部分是**数据新鲜度**而非泄漏（见 `graph_leak_audit.md`）。\n",
        "## 2. 十等分表（表+图，测试窗）\n",
        "按模型分**降序**等频分十组。这是银行评分卡的标准交付物。\n",
        "| 组 | 分数区间 | 样本 | 坏 | 好 | 组内坏率 | 累计坏占比 | 累计好占比 | KS |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in dec.iterrows():
        star = " **←KS 最大**" if abs(r["KS"] - dec["KS"].max()) < 1e-12 else ""
        L.append(f"| {int(r['组'])} | {r['分数区间']} | {int(r['样本']):,} | {int(r['坏']):,} | "
                 f"{int(r['好']):,} | {r['组内坏率']:.2%} | {r['累计坏占比']:.2%} | "
                 f"{r['累计好占比']:.2%} | {r['KS']:.4f}{star} |")
    top = dec.iloc[0]
    L += ["",
          f"- **头部集中度**：第 1 组（前 10%）抓到 **{top['累计坏占比']:.1%}** 的坏样本，"
          f"组内坏率 **{top['组内坏率']:.2%}**（全窗基率 {y.mean():.2%}，"
          f"**lift {top['组内坏率']/y.mean():.1f}×**）。",
          f"- KS 在第 **{int(dec.loc[dec['KS'].idxmax(), '组'])}** 组取到最大值 "
          f"**{dec['KS'].max():.4f}**。\n",
          "## 3. 跨时间稳定性：训练窗 KS vs 测试窗 KS\n",
          "> 银行不只看 KS 高低，更看**跨时间稳不稳**——一个训练窗很高、测试窗掉一半的模型，",
          "> 上线必然翻车。这一节是本项目「时间切分 + 21 天标签隔离期」那套做法在银行语境下的翻译。\n",
          "| 窗口 | 区间 | 样本 | KS |", "|---|---|---|---|",
          f"| 训练窗（样本内） | day < 132 | {len(fit):,} | {ks_fit:.4f} |",
          f"| **测试窗（时间外）** | day ≥ 146 | {len(te):,} | **{ks_gra:.4f}** |",
          f"| **差** | | | **{ks_fit - ks_gra:+.4f}** |",
          "",
          "> 训练窗上的分数是**样本内**的，天然偏乐观——正因如此它才是有用的对照：",
          "> 两窗之差就是「样本内乐观」加「时间外漂移」的合计。",
          "> 本项目另有两条独立证据把这个差拆开：",
          "> **时间外乐观 gap**（PR-AUC 0.5645 → 0.5323，`baseline_metrics.md`）与",
          "> **gap 归因**（数据量损失 ≈0、新鲜度损失 −0.037，`embargo_decomposition.md`）。\n",
          "## 口径与限制\n",
          "- KS 与十等分表**只换表述、不换模型**：同一次运行的同一份分数。",
          "- 训练窗 KS 为**样本内**指标，不可与测试窗 KS 直接比高低，只用于看差距大小。",
          "- 本节全部为**无 embargo** 口径。embargo 版的对照见 `baseline_metrics.md`。\n"]
    write_report(REPORT, "\n".join(L))


if __name__ == "__main__":
    main()
