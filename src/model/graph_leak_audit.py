"""图特征增益的泄漏审计：21 天 vs 60 天 embargo 的对比表 —— **由代码生成**。

## 来历
这张两行表此前是**人工把两次运行的结果抄到一起**的（写在 `graph_vs_tabular.md` 里），
而它承载的结论「**增益缩水但不崩塌 → 是真 repeat-offender 信号，不是泄漏**」
被 README / INTERVIEW / MODEL_CARD 引用 5 次以上。

手抄的表意味着：任一次重新生成都不会更新它，而**没人会发现它过时了**。
现在改成机器生成 —— **事实归机器，判读归人**：
表由本模块产出，「缩但不崩」那句解释留在人写区（`<!-- HUMAN:BEGIN -->`）。

## 做法
同一套 `graph_vs_tabular` 的训练/评估流程跑两遍，只换图特征文件：
  21 天 → `graph_features.parquet`
  60 天 → `graph_features_e60.parquet`（3× 更保守的标签延迟假设）
两轮都用同一切分、同一 LGB 配置，**只差 embargo**，所以 delta 之差可归因于标签新鲜度。

开销约 90s（训练两次 ×2 臂），属 `heavy` 档。

用法：python -m src.model.graph_leak_audit
"""

from pathlib import Path

from src.report_io import write_report

ROOT = Path(__file__).resolve().parents[2]
OUT_MD = ROOT / "reports" / "graph_leak_audit.md"
VARIANTS = [("21 天", "graph_features.parquet"),
            ("60 天", "graph_features_e60.parquet")]


def run_one(graph_file):
    """跑一次纯表 vs 表+图，返回两臂指标。**不落盘**，只回值。"""
    import os
    os.environ["GRAPH_FILE"] = str(ROOT / "data" / "processed" / graph_file)
    import importlib
    import src.model.graph_vs_tabular as g
    importlib.reload(g)                      # GRAPH 是模块级常量，必须重载才生效
    X, y, day, graph_cols = g.load()
    tab = [c for c in X.columns if c not in graph_cols]
    A, _ = g.fit_eval(X, y, day, tab)          # fit_eval 返回 (指标, booster)
    B, _ = g.fit_eval(X, y, day, list(X.columns))
    return A, B


def main():
    rows, results = [], {}
    for label, gf in VARIANTS:
        p = ROOT / "data" / "processed" / gf
        if not p.exists():
            raise SystemExit(f"缺图特征文件：{p}　（60 天版由 graph_features.py 的 embargo 参数产出）")
        print(f"跑 {label} embargo（{gf}）…")
        A, B = run_one(gf)
        results[label] = (A, B)
        rows.append((label, B["pr"] - A["pr"], B["roc"] - A["roc"], A["pr"], B["pr"]))

    d21, d60 = rows[0][1], rows[1][1]
    L = ["# 图特征增益的泄漏审计：21 天 vs 60 天 embargo\n",
         "> **为什么必做**：图特征里的 `prior_fraud_rate` 是**标签衍生**的，",
         "> 而这类特征「意外地强」正是泄漏高发区。把 embargo 拉长 3×，",
         "> 看增益是**崩塌**（说明原先在偷看未来标签）还是**存活**（说明是真信号）。\n",
         "> 两轮同切分、同 LGB 配置，**只差图特征文件的 embargo**——",
         "> 所以两个 delta 之差可归因于标签新鲜度，而非其他因素。\n",
         "## 对比\n",
         "| embargo | 纯表 PR-AUC | 表+图 PR-AUC | **PR-AUC delta** | ROC-AUC delta |",
         "|---|---|---|---|---|"]
    for label, dpr, droc, a, b in rows:
        L.append(f"| {label} | {a:.4f} | {b:.4f} | **{dpr:+.4f}** | {droc:+.4f} |")
    shrink = (d21 - d60) / d21 if d21 else float("nan")
    L += ["",
          f"- 增益由 **{d21:+.4f}** 变为 **{d60:+.4f}**，缩水 **{shrink:.1%}**。",
          f"- **判读条件（预注册）**：若 60 天仍显著为正 → 增益是真 repeat-offender 信号，"
          "缩掉的那部分是「新鲜度」；若塌到 ≈0 或转负 → 原增益主要来自标签泄漏。",
          f"- 本次落在**{'前者' if d60 > 0.5 * d21 else '后者' if d60 <= 0 else '中间地带'}**"
          f"（60 天增益为 {d60:+.4f}，是 21 天的 {d60 / d21:.0%}）。",
          "",
          "> 两层时间因果在两轮里都成立：结构型（`prior_cnt`/`fan_out`）只用 t 之前的边；",
          "> 标签型（`prior_fraud_rate`）邻居标签留满 embargo。**本审计动的只是后者的长度。**\n"]
    write_report(OUT_MD, "\n".join(L))
    print("\n".join(L))
    print(f"\n✅ → {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
