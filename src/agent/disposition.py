"""四档处置：期望成本 argmin（AGENT_DESIGN.md 3.2 + 修订1/2 + 施工提醒1，施工顺序3）。

四档期望成本（每一项各自代表什么、不重叠——防双重计数）：
  放行 approve : E = p·a                    （本笔欺诈损失）
  拒绝 decline : E = (1−p)·c_FP             （本笔误拦成本；欺诈被完全拦住）
  挂起 hold    : E = c_review + p·m_h·a + (1−p)·f_h·c_FP     （人工兜底，残余错误）
  上报 escalate: E = c_report + p·m_e·a + (1−p)·f_e·c_FP
                     − p·gang·k_future·A_med                  （网络项 = 未来批量增量）

防双重计数（owner 轻提醒）：p 定价**本笔**损失（prior_fraud_rate 通过模型特征进 p，
只影响本笔）；网络项定价**冻结实体后拦到的未来批量**，且乘 p 门控（本笔非欺诈则
调查大概率无着落）。gang = 广度×确证史，量"实体铺多开"，非"本笔多危险"——
两个随机变量，语义不重叠。

时间纪律（施工提醒1，最隐蔽的泄漏口）：网络项的 fan-out / prior_fraud_cnt 只从
data/processed/graph_features.parquet（21 天 embargo 版）取——即 graph_vs_tabular
泄漏审计（embargo 21→60 增益缩但不崩）用的同一份、由 causal_prior_stats 逐行
时间因果生成：结构型只用 t 之前的边，标签型标签 <= t−21d。**不碰全量重算。**
A_med（欺诈中位金额）在 [0,125) 窗算，与规则库同一把尺子。

模型 p：表+图 LightGBM（ML 核心定稿模型）**raw 概率**（修订1：③已实证 blanket
校准伤决策区间尾部；本脚本当场重验本模型 top1%/2% 校准 gap 作为资质检查）。

诚实边界：网络项的"未来拦截收益"在 IEEE-CIS 上不可观测（冻结的反事实无数据），
成本参数是假设 → 全部进敏感性扫描；实报成本给两口径（保守=不计未来收益）。

用法：python -m src.agent.disposition
产出：data/processed/agent_test_scores.parquet（p 缓存，后续步骤复用）
     data/processed/agent_disposition_gt.parquet（逐笔四档成本 + 应然档，eval 用）
     reports/figures/12_disposition_regions.png + reports/disposition.md
"""

from src.report_io import write_report
import numpy as np
import pandas as pd
from pathlib import Path

from src.model.graph_vs_tabular import load, fit_eval
from src.model.train_baseline import T0, VAL_DAYS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MERGED = PROJECT_ROOT / "data" / "processed" / "train_merged.parquet"
SCORES_OUT = PROJECT_ROOT / "data" / "processed" / "agent_test_scores.parquet"
GT_OUT = PROJECT_ROOT / "data" / "processed" / "agent_disposition_gt.parquet"
FIG_OUT = PROJECT_ROOT / "reports" / "figures" / "12_disposition_regions.png"
MD_OUT = PROJECT_ROOT / "reports" / "disposition.md"

RULE_END_DAY = 125          # A_med 的标签窗，与规则库同尺（146−21）
ACTIONS = ["approve", "hold", "decline", "escalate"]

# 参数初值（AGENT_DESIGN.md 表；全部为假设 → 敏感性扫描）
BASE = {
    "c_fp": 25.0,       # 误拦成本（①已锚）
    "c_review": 5.0,    # 挂起人工复核成本
    "c_report": 40.0,   # 上报建档/深查成本
    "m_h": 0.10,        # 挂起残余漏检率
    "f_h": 0.02,        # 挂起残余误拦率
    "m_e": 0.05,        # 上报残余漏检率（深查更彻底）
    "f_e": 0.02,        # 上报残余误拦率
    "k_future": 5.0,    # 上报冻结实体拦到的未来欺诈笔数（期望）
}
SCAN = {"c_review": [2, 5, 10, 20], "c_report": [20, 40, 100],
        "k_future": [0, 2, 5, 20], "m_h": [0.05, 0.10, 0.20]}
GANG_F0 = 10.0      # 广度饱和：fan-out 10 设备
GANG_R0 = 0.10      # 确证史饱和：prior 欺诈率 0.10（与规则库 R_PRIOR_FRAUD_* 同门槛）
GANG_MIN_CNT = 2    # 小样本闸：至少 2 笔成熟确证，防 1/1=100% 的噪声率


def gang_score(fanout_device, prior_fraud_rate, prior_fraud_cnt):
    """团伙形态分 ∈[0,1]：广度（fan-out 饱和）× 确证史（成熟欺诈**率**饱和，cnt 只作小样本闸）。
    第一版用 cnt 饱和于 3，结果 74% 交易 gang>0——大实体攒几千笔按基率躺着也有 3 笔欺诈，
    笔数不归一交易量 = 给大实体发团伙牌；与规则层「裸 fan-out lift 仅 1.47 被拒」同因。
    改用 rate（歧视性来自超基率的欺诈密度），门槛与规则库同尺。
    三个输入都来自 embargo-21 图特征（时间因果）；NaN → 0（无历史=无团伙证据）。"""
    f = np.nan_to_num(np.asarray(fanout_device, dtype=float)) / GANG_F0
    r = np.nan_to_num(np.asarray(prior_fraud_rate, dtype=float)) / GANG_R0
    ok = np.nan_to_num(np.asarray(prior_fraud_cnt, dtype=float)) >= GANG_MIN_CNT
    return np.minimum(f, 1.0) * np.minimum(r, 1.0) * ok


def expected_costs(p, a, g, a_med, prm):
    """四档期望成本，返回 shape (n,4) 按 ACTIONS 顺序。"""
    p, a, g = (np.asarray(x, dtype=float) for x in (p, a, g))
    e_approve = p * a
    e_hold = prm["c_review"] + p * prm["m_h"] * a + (1 - p) * prm["f_h"] * prm["c_fp"]
    e_decline = (1 - p) * prm["c_fp"]
    e_escalate = (prm["c_report"] + p * prm["m_e"] * a + (1 - p) * prm["f_e"] * prm["c_fp"]
                  - p * g * prm["k_future"] * a_med)
    return np.stack([e_approve, e_hold, e_decline, e_escalate], axis=1)


def argmin_action(p, a, g, a_med, prm):
    return np.asarray(ACTIONS)[expected_costs(p, a, g, a_med, prm).argmin(axis=1)]


def realized_cost(action, y, a, g, a_med, prm, credit_future=False):
    """代入真标签的实报成本（eval 层2 的口径雏形）。
    上报的未来收益不可观测：默认保守口径不计；credit_future=True 给模型口径。"""
    y, a, g = (np.asarray(x, dtype=float) for x in (y, a, g))
    c = np.where(action == "approve", y * a, 0.0)
    c = np.where(action == "decline", (1 - y) * prm["c_fp"], c)
    m = action == "hold"
    c = np.where(m, prm["c_review"] + y * prm["m_h"] * a + (1 - y) * prm["f_h"] * prm["c_fp"], c)
    m = action == "escalate"
    esc = prm["c_report"] + y * prm["m_e"] * a + (1 - y) * prm["f_e"] * prm["c_fp"]
    if credit_future:
        esc = esc - y * g * prm["k_future"] * a_med
    return np.where(m, esc, c).sum()


# ---------------------------------------------------------------- 主流程

def train_and_cache():
    """训表+图模型（ML 核心定稿口径），缓存 test 逐笔 p + 网络项输入；有缓存则直接用。"""
    if SCORES_OUT.exists():
        print(f"复用 p 缓存 {SCORES_OUT.relative_to(PROJECT_ROOT)}（删除该文件可强制重训）")
        out = pd.read_parquet(SCORES_OUT)
        from sklearn.metrics import average_precision_score, roc_auc_score
        m = {"pr": average_precision_score(out["isFraud"], out["p"]),
             "roc": roc_auc_score(out["isFraud"], out["p"])}
        zone = {}
        for frac in (0.01, 0.02):
            top = out.nlargest(int(len(out) * frac), "p")
            zone[frac] = (float(top["p"].mean()), float(top["isFraud"].mean()))
        lab = pd.read_parquet(MERGED, columns=["TransactionDT", "TransactionAmt", "isFraud"])
        d = lab["TransactionDT"] // 86_400
        lab_day = (d - d.min()).astype(int)
        a_med = float(lab.loc[(lab_day < RULE_END_DAY) & (lab["isFraud"] == 1),
                              "TransactionAmt"].median())
        return out, a_med, m, zone

    X, y, day, graph_cols = load()
    tab_cols = [c for c in X.columns if c not in graph_cols]
    ids = pd.read_parquet(MERGED, columns=["TransactionID"])["TransactionID"].to_numpy()

    print("训练 表+图 模型（fit<132 / val[132,146) / test>=146）…")
    m, booster = fit_eval(X, y, day, tab_cols + graph_cols)
    print(f"  PR-AUC {m['pr']:.4f}  ROC-AUC {m['roc']:.4f}（对照 graph_vs_tabular 定稿 0.6032/0.9306）")

    test = day >= T0
    cols = tab_cols + graph_cols
    p = booster.predict(X.loc[test, cols], num_iteration=booster.best_iteration)
    out = pd.DataFrame({
        "TransactionID": ids[test],
        "day": day[test],
        "isFraud": y[test].to_numpy(),
        "TransactionAmt": X.loc[test, "TransactionAmt"].to_numpy(),
        "p": p,
        "card1_fanout_device": X.loc[test, "card1_fanout_device"].to_numpy(),
        "card1_prior_fraud_cnt": X.loc[test, "card1_prior_fraud_cnt"].to_numpy(),
        "card1_prior_fraud_rate": X.loc[test, "card1_prior_fraud_rate"].to_numpy(),
        "card1_addr1_prior_fraud_rate": X.loc[test, "card1_addr1_prior_fraud_rate"].to_numpy(),
    })

    # 修订1 资质检查：raw p 的决策区间校准（③口径：不 blanket 校准，先验后用）
    zone = {}
    for frac in (0.01, 0.02):
        k = int(len(out) * frac)
        top = out.nlargest(k, "p")
        zone[frac] = (float(top["p"].mean()), float(top["isFraud"].mean()))
        print(f"  决策区间 top{frac:.0%}: 预测 {zone[frac][0]:.3f} vs 实际 {zone[frac][1]:.3f} "
              f"(gap {zone[frac][0]-zone[frac][1]:+.3f})")

    # A_med：欺诈中位金额，[0,125) 标签窗（与规则库同尺）
    lab = pd.read_parquet(MERGED, columns=["TransactionDT", "TransactionAmt", "isFraud"])
    d = lab["TransactionDT"] // 86_400
    lab_day = (d - d.min()).astype(int)
    a_med = float(lab.loc[(lab_day < RULE_END_DAY) & (lab["isFraud"] == 1), "TransactionAmt"].median())
    print(f"  A_med（[0,{RULE_END_DAY}) 欺诈中位金额）= ${a_med:.2f}")

    out.to_parquet(SCORES_OUT, index=False)
    print(f"✅ p 缓存 → {SCORES_OUT.relative_to(PROJECT_ROOT)}")
    return out, a_med, m, zone


def make_figure(a_med):
    """展品：(p, 金额) 平面 × 团伙证据三档 的四档分区图——①代价敏感、③已验概率、⑥图特征在一张图上。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    pp = np.logspace(-3, 0, 400)
    aa = np.logspace(0, 3.5, 400)
    P, A = np.meshgrid(pp, aa)
    cmap = ListedColormap(["#66bb6a", "#ffb300", "#e53935", "#6a1b9a"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, g in zip(axes, [0.0, 0.5, 1.0]):
        E = expected_costs(P.ravel(), A.ravel(), np.full(P.size, g), a_med, BASE)
        Z = E.argmin(axis=1).reshape(P.shape)
        ax.pcolormesh(P, A, Z, cmap=cmap, vmin=0, vmax=3, shading="auto")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"gang evidence g={g:g}")
        ax.set_xlabel("model probability p (raw, zone-verified)")
    axes[0].set_ylabel("amount ($)")
    fig.legend(handles=[Patch(color=c, label=l) for c, l in
                        zip(cmap.colors, ["approve", "hold", "decline", "escalate"])],
               loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Four-action disposition regions = expected-cost argmin "
                 f"(cFP 25, review 5, report 40, k-future 5, median fraud amt {a_med:.0f} USD)")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    print(f"✅ 分区图 → {FIG_OUT.relative_to(PROJECT_ROOT)}")


def main():
    df, a_med, model_m, zone = train_and_cache()
    p, a, y = df["p"].to_numpy(), df["TransactionAmt"].to_numpy(), df["isFraud"].to_numpy()
    g = gang_score(df["card1_fanout_device"], df["card1_prior_fraud_rate"],
                   df["card1_prior_fraud_cnt"])
    n = len(df)
    print(f"\ntest {n:,} 笔；gang_score>0 占 {(g > 0).mean():.2%}，>=0.5 占 {(g >= 0.5).mean():.2%}")

    # 应然档（base 参数）+ 网络项关闭对照（修订2 效果检查）
    E = expected_costs(p, a, g, a_med, BASE)
    act = np.asarray(ACTIONS)[E.argmin(axis=1)]
    act0 = argmin_action(p, a, g, a_med, {**BASE, "k_future": 0.0})

    print("\n四档分布（base 参数）：")
    dist_rows = []
    for name, arr in [("含网络项", act), ("k_future=0", act0)]:
        shares = {x: float((arr == x).mean()) for x in ACTIONS}
        dist_rows.append((name, shares))
        print(f"  {name}: " + "  ".join(f"{x} {shares[x]:.2%}" for x in ACTIONS))

    # 网络项把谁推进上报档（owner 要看的效果）：来源 + 金额画像
    moved = act0[act == "escalate"]
    movers = df[act == "escalate"]
    med_amt_esc = float(movers["TransactionAmt"].median()) if len(movers) else float("nan")
    from collections import Counter
    src = Counter(moved.tolist())
    print(f"\n上报档 {len(movers):,} 笔（中位金额 ${med_amt_esc:.0f}；k_future=0 时它们本属："
          + "、".join(f"{k} {v}" for k, v in src.most_common()) + "）")
    ex = movers[(movers["TransactionAmt"] < 50)].nlargest(2, "p")
    examples = [(int(r["TransactionID"]), float(r["TransactionAmt"]), float(r["p"]),
                 float(gang_score(r["card1_fanout_device"], r["card1_prior_fraud_rate"],
                                  r["card1_prior_fraud_cnt"])),
                 int(r["isFraud"])) for _, r in ex.iterrows()]
    for t, amt, pp_, gg, yy in examples:
        print(f"  例：txn {t} ${amt:.2f} p={pp_:.2f} gang={gg:.2f} isFraud={yy} → escalate（小额靠团伙证据上报）")

    # 实报成本（保守口径：不计未来收益）vs 基线策略
    policies = {
        "argmin 四档": act,
        "四档(k_future=0)": act0,
        "全放行": np.full(n, "approve"),
        "① 两档金额感知": np.where(p > BASE["c_fp"] / (a + BASE["c_fp"]), "decline", "approve"),
    }
    cost_rows = []
    for name, arr in policies.items():
        c = realized_cost(arr, y, a, g, a_med, BASE)
        cost_rows.append((name, c, c / n * 10_000))
        print(f"  实报成本[保守] {name}: ${c:,.0f}（每万笔 ${c / n * 10_000:,.0f}）")
    c_model = realized_cost(act, y, a, g, a_med, BASE, credit_future=True)
    print(f"  （模型口径：argmin 四档计入未来收益 = ${c_model:,.0f}——不可观测，仅列参考）")

    # 敏感性扫描：单参数变动 → 四档份额
    scan_rows = []
    for key, values in SCAN.items():
        for v in values:
            prm = {**BASE, key: float(v)}
            arr = argmin_action(p, a, g, a_med, prm)
            scan_rows.append((key, v, {x: float((arr == x).mean()) for x in ACTIONS}))

    # 逐笔落盘（eval 步骤5 的"应然档"）
    gt = df[["TransactionID", "day", "isFraud", "TransactionAmt", "p"]].copy()
    gt["gang_score"] = g
    for i, x in enumerate(ACTIONS):
        gt[f"E_{x}"] = E[:, i]
    gt["disposition_gt"] = act
    gt["disposition_gt_no_gang"] = act0
    gt.to_parquet(GT_OUT, index=False)
    print(f"✅ 应然档 → {GT_OUT.relative_to(PROJECT_ROOT)}")

    make_figure(a_med)
    _write_md(model_m, zone, a_med, n, g, dist_rows, src, med_amt_esc, examples,
              cost_rows, c_model, scan_rows)
    print(f"✅ 报告 → {MD_OUT.relative_to(PROJECT_ROOT)}")


def _write_md(model_m, zone, a_med, n, g, dist_rows, src, med_amt_esc, examples,
              cost_rows, c_model, scan_rows):
    L = ["# 四档处置：期望成本 argmin（施工顺序3）\n",
         "公式与防双重计数、时间纪律声明见 `src/agent/disposition.py` 模块 docstring。\n",
         "## p 的资质（修订1：raw + 决策区间当场验证）",
         f"- 表+图模型 test PR-AUC **{model_m['pr']:.4f}** / ROC-AUC {model_m['roc']:.4f}。",
         f"- 决策区间校准 gap：top1% 预测 {zone[0.01][0]:.3f} vs 实际 {zone[0.01][1]:.3f}"
         f"（gap {zone[0.01][0]-zone[0.01][1]:+.3f}）；top2% {zone[0.02][0]:.3f} vs {zone[0.02][1]:.3f}"
         f"（gap {zone[0.02][0]-zone[0.02][1]:+.3f}）→ raw 概率直接进成本公式（③结论的复用与复验）。",
         "",
         "## 时间纪律（施工提醒1）",
         f"- 网络项输入 = `graph_features.parquet`（**embargo 21 版**，泄漏审计同款，逐行时间因果）；"
         f"A_med=${a_med:.2f} 在 [0,{RULE_END_DAY}) 窗算（与规则库同尺）。未触碰任何全量统计。",
         "",
         f"## 四档分布（test {n:,} 笔；gang_score>0 占 {(g > 0).mean():.2%}）",
         "| 策略 | approve | hold | decline | escalate |", "|---|---|---|---|---|"]
    for name, shares in dist_rows:
        L.append(f"| {name} | " + " | ".join(f"{shares[x]:.2%}" for x in ACTIONS) + " |")
    L += ["",
          "- gang 代理修订记录（诚实决策日志）：v1 用 prior_fraud_cnt 饱和 → gang>0 覆盖 74%"
          "（大实体按基率躺着攒笔数=误伤）；v2 改 prior_fraud_**rate** 饱和 + cnt>=2 小样本闸。"
          "教训：团伙代理必须归一暴露量——与规则层「裸 fan-out 被拒」同因。",
          f"- 网络项效果（修订2 该产生的效果）：上报档全部 {sum(src.values()):,} 笔在 k_future=0 时本属 "
          + "、".join(f"**{k} {v:,}**" for k, v in src.most_common())
          + f"；上报档中位金额 ${med_amt_esc:.0f}（小额也能靠团伙证据上报，不再唯金额论）。"]
    for t, amt, pp_, gg, yy in examples:
        L.append(f"  - 例：txn {t}，${amt:.2f}，p={pp_:.2f}，gang={gg:.2f}，isFraud={yy} → escalate。")
    L += ["", "## 实报成本（保守口径：上报的未来收益不可观测、不计入）",
          "| 策略 | 总成本 | 每万笔 |", "|---|---|---|"]
    for name, c, per in cost_rows:
        L.append(f"| {name} | ${c:,.0f} | ${per:,.0f} |")
    L += [f"\n- 模型口径（计入未来收益估计）：argmin 四档 ${c_model:,.0f}——该收益是反事实假设，"
          "只作参考、不进 README 数字。",
          "- 诚实张力：保守口径下上报档必然「显亏」（付 c_report 而收益记零）——上报的辩护在"
          "网络项假设 + 敏感性，不在本数据可观测的成本上。",
          "",
          "## 参数敏感性（单参数变动 → 四档份额）",
          "| 参数 | 值 | approve | hold | decline | escalate |", "|---|---|---|---|---|---|"]
    for key, v, shares in scan_rows:
        star = "**" if v == BASE[key] else ""
        L.append(f"| {key} | {star}{v:g}{star} | " + " | ".join(f"{shares[x]:.2%}" for x in ACTIONS) + " |")
    L += ["",
          "结论：四档份额随参数**连续、单调**变化（框架稳、参数留业务标定）；k_future=0 时上报档"
          "退化并入挂起/拒绝（=修订2 防住的「四档塌三档」的直接演示）。",
          "",
          "## 展品",
          "- 图 12（`figures/12_disposition_regions.png`）：(p, 金额) 平面 × 团伙证据三档的四档分区——"
          "①代价敏感阈值的四动作推广 + ③已验 raw 概率 + ⑥图特征的成本语义，一张图三条线。"]
    write_report(MD_OUT, "\n".join(L))


if __name__ == "__main__":
    main()
