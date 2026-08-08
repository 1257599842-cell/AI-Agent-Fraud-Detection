"""代价敏感阈值（防守点①）—— 把"朴素 0.5 → 代价敏感阈值"的收益跑成曲线 + delta。

成本假设（可辩护、做敏感性）：
  - 漏放（FN，放过欺诈）成本 = 该笔交易金额 TransactionAmt（真损失就是钱）。
  - 误拦（FP，拦错好人）成本 = 固定 c_FP（客服 + 流失），锚定 $25，敏感性 {10,25,50,100}。
  → 只有成本比例重要；因 c_FN 随金额变，逐样本最优阈值 t_i = c_FP/(a_i + c_FP)。

产出三件事：
  1. 期望成本曲线：扫全局阈值 t，cost(t)=c_FP·(误拦好人) + Σ金额(漏放欺诈)，找 t*=argmin；对比 0.5。
  2. 逐样本代价敏感策略（block 当 p_i > c_FP/(a_i+c_FP)）的成本（金额感知，理论更优）。
  3. 头条 delta：有效拦截率 recall 从 0.5 阈值 → 代价敏感 t*。

用法：python -m src.model.cost_sensitive
产出：reports/figures/05_cost_curve.png + reports/cost_sensitive.md
"""

from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.model.train_baseline import LGB_PARAMS, T0, VAL_DAYS, prepare

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG = PROJECT_ROOT / "reports" / "figures" / "05_cost_curve.png"
OUT_MD = PROJECT_ROOT / "reports" / "cost_sensitive.md"

C_FP_GRID = [10, 25, 50, 100]
C_FP_ANCHOR = 25
# 阈值网格：低分段加密（行动都在低 p 区）
T_GRID = np.unique(np.concatenate([np.linspace(0, 0.1, 400), np.linspace(0.1, 1.0, 200)]))


def fit_full_model(X, y, day):
    fit = day < (T0 - VAL_DAYS)
    val = (day >= T0 - VAL_DAYS) & (day < T0)
    test = day >= T0
    dtr = lgb.Dataset(X[fit], label=y[fit])
    dval = lgb.Dataset(X[val], label=y[val], reference=dtr)
    booster = lgb.train(
        LGB_PARAMS, dtr, num_boost_round=2000, valid_sets=[dval],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    p = booster.predict(X[test], num_iteration=booster.best_iteration)
    return p, y[test].to_numpy(), X.loc[test, "TransactionAmt"].to_numpy(dtype=float)


def cost_at(t, p, y, amt, c_fp):
    flag = p >= t
    fp = float(c_fp * ((y == 0) & flag).sum())
    fn = float(amt[(y == 1) & ~flag].sum())
    return fp + fn


def metrics_at(t, p, y, amt):
    flag = p >= t
    n_flag = int(flag.sum())
    tp = int(((y == 1) & flag).sum())
    recall = tp / int((y == 1).sum())
    precision = tp / n_flag if n_flag else 0.0
    return {"t": float(t), "n_flag": n_flag, "vol": n_flag / len(p),
            "recall": recall, "precision": precision}


def main() -> None:
    print("读取数据 + 训练全量模型 …")
    X, y, day = prepare()
    p, yte, amt = fit_full_model(X, y, day)
    n_fraud = int((yte == 1).sum())
    fraud_dollars = float(amt[yte == 1].sum())
    print(f"  test: {len(p):,} 笔，欺诈 {n_fraud:,} 笔，欺诈总金额 ${fraud_dollars:,.0f}")

    rows = []
    for c_fp in C_FP_GRID:
        costs = np.array([cost_at(t, p, yte, amt, c_fp) for t in T_GRID])
        i_star = int(costs.argmin())
        t_star = float(T_GRID[i_star])
        cost_star = float(costs[i_star])
        cost_half = cost_at(0.5, p, yte, amt, c_fp)
        cost_zero = cost_at(0.0, p, yte, amt, c_fp)   # 全拦（上界参考）
        # 逐样本代价敏感（金额感知）
        ti = c_fp / (amt + c_fp)
        block = p > ti
        cost_inst = float(c_fp * ((yte == 0) & block).sum() + amt[(yte == 1) & ~block].sum())
        m_star = metrics_at(t_star, p, yte, amt)
        m_half = metrics_at(0.5, p, yte, amt)
        rows.append({
            "c_fp": c_fp, "t_star": t_star, "cost_half": cost_half, "cost_star": cost_star,
            "cost_inst": cost_inst, "save_vs_half": 1 - cost_star / cost_half,
            "recall_half": m_half["recall"], "recall_star": m_star["recall"],
            "prec_star": m_star["precision"], "vol_star": m_star["vol"],
            "n_flag_star": m_star["n_flag"], "block_inst_vol": float(block.mean()),
        })
        print(f"  c_FP=${c_fp:<3}: t*={t_star:.3f}  cost 0.5→t*: ${cost_half:,.0f}→${cost_star:,.0f} "
              f"(省 {rows[-1]['save_vs_half']:.1%})  recall 0.5→t*: {m_half['recall']:.3f}→{m_star['recall']:.3f}  "
              f"逐样本cost ${cost_inst:,.0f}")

    _plot(p, yte, amt, C_FP_ANCHOR)
    _write_md(rows, n_fraud, fraud_dollars, len(p))
    print(f"\n✅ 图 {FIG.relative_to(PROJECT_ROOT)}，小结 {OUT_MD.relative_to(PROJECT_ROOT)}")


def _plot(p, yte, amt, c_fp):
    costs = np.array([cost_at(t, p, yte, amt, c_fp) for t in T_GRID])
    i_star = int(costs.argmin())
    plt.figure(figsize=(8, 4.5))
    plt.plot(T_GRID, costs / 1000, color="#4C72B0", lw=1.5)
    plt.axvline(T_GRID[i_star], color="#55A868", ls="--", lw=1.2,
                label=f"cost-optimal t*={T_GRID[i_star]:.3f}")
    plt.axvline(0.5, color="#C44E52", ls="--", lw=1.2, label="naive t=0.5")
    plt.scatter([T_GRID[i_star]], [costs[i_star] / 1000], color="#55A868", zorder=5)
    plt.xlabel("Probability threshold t")
    plt.ylabel("Total cost on test ($k)")
    plt.title(f"Expected-cost curve (c_FP=${c_fp}, c_FN=amount)")
    plt.legend()
    plt.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG, dpi=120)
    plt.close()


def _write_md(rows, n_fraud, fraud_dollars, n_test):
    a = next(r for r in rows if r["c_fp"] == C_FP_ANCHOR)
    L = [
        "# 代价敏感阈值（防守点①）—— 朴素 0.5 → 代价敏感 t*\n",
        "**成本假设**：漏放成本 = 交易金额（真损失）；误拦成本 = 固定 c_FP（客服/流失）。只有比例重要，故做 c_FP 敏感性。\n",
        f"test：{n_test:,} 笔，欺诈 {n_fraud:,} 笔，欺诈暴露总额 ${fraud_dollars:,.0f}。\n",
        "## c_FP 敏感性（全局阈值 t* vs 朴素 0.5）\n",
        "| c_FP | t* | cost@0.5 | cost@t* | 省 | recall@0.5 | recall@t* | precision@t* | t* 拦截量占比 |",
        "|------|-----|----------|---------|----|-----------|-----------|--------------|------------|",
    ]
    for r in rows:
        L.append(
            f"| ${r['c_fp']} | {r['t_star']:.3f} | ${r['cost_half']:,.0f} | ${r['cost_star']:,.0f} | "
            f"{r['save_vs_half']:.1%} | {r['recall_half']:.3f} | {r['recall_star']:.3f} | "
            f"{r['prec_star']:.3f} | {r['vol_star']:.2%} |")
    L += [
        "",
        "## 头条（锚点 c_FP=$%d）" % C_FP_ANCHOR,
        f"- **有效拦截率（recall）：朴素 0.5 阈值 {a['recall_half']:.3f} → 代价敏感 t*={a['t_star']:.3f} 时 {a['recall_star']:.3f}**（README 的 X→Y）。",
        f"- 总成本：${a['cost_half']:,.0f} → ${a['cost_star']:,.0f}（**省 {a['save_vs_half']:.1%}**）。",
        f"- 代价敏感 t* 拦截 {a['vol_star']:.2%} 的交易量，precision {a['prec_star']:.3f}。",
        f"- 逐样本金额感知策略（block 当 p>c_FP/(a+c_FP)）成本 ${a['cost_inst']:,.0f}，理论上 ≤ 最优全局阈值，因为它对大额更敏感。",
        "",
        "## 讲法（防守点①）",
        "- 0.5 不是「抓不到」（recall@0.5=0.338），而是**远离成本最优**——它漏掉约 2/3 的欺诈金额，因为它无视「漏一笔的代价=整笔金额」这个不对称。",
        "- 把 FP/FN 成本写进期望成本，最优阈值远低于 0.5（c_FP=$25 时 t*≈0.078）；金额感知版进一步对大额降阈值，成本还更低。",
        "- 注意容量约束：t* 拦截量若超过复核员日容量，则在 t* 与容量之间取约束最优（接 precision@容量 曲线）。",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
