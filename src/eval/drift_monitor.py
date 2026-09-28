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

# δ 标定（本轮新增）：`0.02` 是拍的，而两个指标的自然波动幅度差一个量级
# （ROC 全幅约 0.04、PR 约 0.16）——同一个 δ 在两者上的严厉程度完全不同。
# 所以不再共用一个常数，而是**按各指标自己的抽样噪声定 δ**：
#   δ = Z_SIGMA × SE(基线 − 某窗)  ，SE 由窗内自助法估计。
# 注意这只覆盖**抽样噪声**；若跨窗实测波动显著超过抽样噪声，说明还有真实的
# 分布波动成分，那时抽样 SE 就不足以单独定 δ —— 报告里会把这个比值算出来。
BOOT_B = 500            # 每窗自助重抽次数
Z_SIGMA = 2.0           # δ 取几个 SE（2σ）
PERSIST = 2             # 持续性规则：连续几窗跌破才告警
SEED_BOOT = 42          # 自助法种子（固定，保证报告可复现）

# ── 合成漂移灵敏度（**构造场景**，用来测「检出率」这一半）────────────────────
# 真数据这 6 个窗里**没有一次真退化事件**，所以只能标定「不误报」，测不出「多久能发现」。
# MODEL_CARD 明文允许：**明确标注的合成漂移可用于检验检测器，但不得冒充真实退化。**
#
# 注入方式：把模型分与均匀噪声按 α 掺合  p' = (1−α)·p + α·u ，u~U(0,1)。
#   · α=0 → 原分；α=1 → 纯噪声。α 直接刻画「排序能力损失了多少」。
#   · 只动分数、不动标签 —— 保留 ground truth，评估口径不被污染。
#   · ROC/PR 都是**基于排序**的指标，所以掺噪破坏校准无妨。
# 注入从第 ONSET_IDX 个**评估窗**（不含基线窗）起生效，模拟「某时点之后开始退化」。
DRIFT_ALPHAS = [0.0, 0.05, 0.10, 0.20, 0.40]
ONSET_IDX = 2           # 评估窗下标（0 = 基线窗之后的第一个窗）
SEED_DRIFT = 7


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
    psi_edges = None                 # PSI 的分箱切点：**只在基准窗定一次**
    for lo in range(VAL_END, day_max + 1, WINDOW):
        hi = lo + WINDOW
        mask = (day >= lo) & (day < hi)
        yy = y[mask].to_numpy()
        if mask.sum() < 500 or yy.sum() < 10:
            continue
        pp = booster.predict(X[mask], num_iteration=booster.best_iteration)
        if psi_edges is None:        # 首窗 = 基准，切点由它确定，后续窗一律沿用
            psi_edges, base_ratio = _psi_reference(pp)
            psi = 0.0
        else:
            psi = _psi(pp, psi_edges, base_ratio)
        # 数据在 day_max 截断，最后一个窗往往装不满 14 天。不标出来的话，
        # 「末窗 ROC 略高于基线」会被当成同尺度的回升——而它是在更少样本上算的、更噪。
        covered = min(hi, day_max + 1) - lo
        se_roc, se_pr = _boot_se(yy, pp)
        wins.append({
            "lo": lo, "hi": hi, "n": int(mask.sum()), "fraud_rate": float(yy.mean()),
            "roc": float(roc_auc_score(yy, pp)), "pr": float(average_precision_score(yy, pp)),
            "psi": float(psi), "days": int(covered), "partial": covered < WINDOW,
            "se_roc": se_roc, "se_pr": se_pr,
            "_y": yy, "_p": pp,          # 留着给合成漂移注入用（不进报告）
        })

    # 历史口径：δ=0.02 拍在 ROC 上。**保留它是为了可追溯，不代表它被选中**——
    # 口径已定调为「两指标各自标定、并列上报，不选定触发指标」，理由见 _write_md。
    base = wins[0]["roc"]
    trigger = base - DELTA
    print(f"\n[历史口径·仅追溯] ROC 基线={base:.4f}，δ={DELTA} → 触发线={trigger:.4f}")
    n_alarm = 0
    for w in wins:
        alarm = w["roc"] < trigger
        n_alarm += alarm
        print(f"  [{w['lo']:>3},{w['hi']:>3}) n={w['n']:>6,} 欺诈率={w['fraud_rate']:.2%} "
              f"ROC={w['roc']:.4f} PR={w['pr']:.4f} {'⚠️ ALARM' if alarm else ''}")

    cal = _calibrate_delta(wins)
    print(f"\n=== δ 标定（{Z_SIGMA:g}σ，自助 B={BOOT_B}）——两指标并列，不选定 ===")
    for k in ("roc", "pr"):
        c = cal[k]
        print(f"  {k.upper():3s}: 基线 {c['base']:.4f}  SE_基线 {c['se_base']:.4f}  "
              f"SE_典型 {c['se_typ']:.4f}  δ={c['delta']:.4f}  触发线 {c['trigger']:.4f}  "
              f"单窗告警 {c['n_alarm']}/{c['n_eval']}  连续{PERSIST}窗告警 {c['n_persist']}")
        print(f"       跨窗实测SD {c['sd_obs']:.4f} vs 抽样SE {c['se_typ']:.4f} "
              f"→ 超量波动比 {c['excess']:.2f}×；最大落差/SE = {c['drop_z']:.1f}")

    sens = _synthetic_sensitivity(wins, cal)
    print(f"\n=== 合成漂移灵敏度（构造场景，α 从第 {ONSET_IDX} 个评估窗起注入）===")
    for r in sens:
        print(f"  α={r['alpha']:.2f}  ROC {r['roc_min']:.4f}({r['roc_drop']:+.4f}) "
              f"告警{r['roc_alarm']}/{r['n_post']} 新增{r['roc_new']} 首检{r['roc_first'] or '—'}"
              f"  |  PR {r['pr_min']:.4f}({r['pr_drop']:+.4f}) "
              f"告警{r['pr_alarm']}/{r['n_post']} 新增{r['pr_new']} 首检{r['pr_first'] or '—'}")

    _plot(wins, base, trigger)
    _write_md(wins, base, trigger, n_alarm, day_max, cal, sens)
    print(f"\n✅ 图 08 + {OUT_MD.relative_to(PROJECT_ROOT)}")


def _boot_se(yy, pp, B=BOOT_B):
    """窗内自助法估 ROC-AUC / PR-AUC 的抽样标准误。

    为什么需要它：`δ=0.02` 是拍的，而 ROC 天然落在 0.88–0.92 的窄带、PR 在 0.45–0.62 的
    宽带——同一个 δ 对两者的严厉程度差一个量级。**δ 必须按各指标自己的噪声定。**
    重抽单位是交易（每笔独立评分，不存在实体聚类），与 gang 效度那里按实体重抽不同。
    """
    rng = np.random.default_rng(SEED_BOOT)
    n = len(yy)
    rocs, prs = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        ys, ps = yy[idx], pp[idx]
        if ys.sum() == 0 or ys.sum() == len(ys):   # 全同类时两个指标都无定义
            continue
        rocs.append(roc_auc_score(ys, ps))
        prs.append(average_precision_score(ys, ps))
    f = (lambda a: float(np.std(a, ddof=1)) if len(a) > 1 else float("nan"))
    return f(rocs), f(prs)


def _calibrate_delta(wins):
    """按抽样噪声给每个指标各自标定 δ，并报「超量波动」与「落差信噪比」。

    δ 比较的是**两个估计之差**（基线 vs 某窗），所以用 SE(差) = sqrt(SE_基线² + SE_窗²)。
    `excess` = 跨窗实测 SD ÷ 典型抽样 SE：>1 说明窗间差异超出了抽样噪声能解释的范围
    （即存在真实的分布波动），那时**抽样 SE 不足以单独定 δ**——必须靠持续性规则或更宽的 δ。
    `drop_z` = 相对基线的最大落差 ÷ SE(差)：这是「谁更敏感」的**噪声归一化**版本，
    比直接比 0.0905 vs 0.0105 公平——后者混了两个指标不同的量纲。
    """
    out = {}
    for key, se_key in (("roc", "se_roc"), ("pr", "se_pr")):
        vals = [w[key] for w in wins]
        ses = [w[se_key] for w in wins]
        base, se_base = vals[0], ses[0]
        rest, rest_se = vals[1:], ses[1:]
        se_typ = float(np.median(rest_se))
        se_diff = float(np.sqrt(se_base ** 2 + se_typ ** 2))
        delta = Z_SIGMA * se_diff
        trigger = base - delta
        flags = [v < trigger for v in rest]
        n_persist = sum(1 for i in range(len(flags))
                        if all(flags[max(0, i - PERSIST + 1):i + 1]) and i + 1 >= PERSIST)
        drop = base - min(rest)
        out[key] = {
            "base": base, "se_base": se_base, "se_typ": se_typ, "se_diff": se_diff,
            "delta": delta, "trigger": trigger,
            "n_alarm": sum(flags), "n_eval": len(rest), "n_persist": n_persist,
            "sd_obs": float(np.std(rest, ddof=1)),
            "excess": float(np.std(rest, ddof=1)) / se_typ if se_typ else float("nan"),
            "drop": drop, "drop_z": drop / se_diff if se_diff else float("nan"),
            "flags": flags,
        }
    return out


def _synthetic_sensitivity(wins, cal):
    """**构造场景**：注入已知强度的排序退化，测检测器多久能发现。

    这补的是 δ 标定测不了的那一半——真数据没有退化事件，所以「不误报」能标定、
    「检出率」不能。本函数用 `p' = (1−α)·p + α·u` 造一个**已知强度**的退化，
    然后用**同一次运行里标定出来的 δ 与持续性规则**去检，看：
      · 多大的 α 才会触发（灵敏度下限）
      · 从注入开始到首次告警差几个窗（检出延迟）

    ⚠️ 这不是「我经历过真实退化」的证据。α 是我设的，退化形态是我造的。
    它能声称的只有：**在这种形态、这个强度下，我的检测器有/没有反应。**
    """
    post = wins[1:]                                  # 基线窗不注入（它定义基线）

    def run_alpha(alpha):
        # 每个 α 用同一个种子重开 → 各 α 之间注入的噪声序列相同，
        # 差异只来自 α 本身（否则 α 之间的比较会混进不同的随机抽取）。
        rng = np.random.default_rng(SEED_DRIFT)
        rocs, prs = [], []
        for i, w in enumerate(post):
            p = w["_p"]
            u = rng.random(len(p))                   # 无论是否注入都抽，保持序列对齐
            if alpha > 0 and i >= ONSET_IDX:
                p = (1 - alpha) * p + alpha * u
            rocs.append(float(roc_auc_score(w["_y"], p)))
            prs.append(float(average_precision_score(w["_y"], p)))
        return {"roc": rocs, "pr": prs}

    # ⚠️ α=0 的告警必须先算出来当参照：PR 在**注入之前**就已经在 3/6 个窗告警，
    # 如果直接报「本 α 下告警几次」，就分不清是**检出了注入**还是**本来就在响**。
    # 所以额外报「**新增**告警」= 本 α 告警 且 α=0 不告警 —— 那才是注入的效应。
    # （同一条纪律在本项目出现过多次：判别力必须相对既有基线算，不能拿绝对命中数当功劳。）
    ref = run_alpha(0.0)
    ref_flags = {k: [v < cal[k]["trigger"] for v in ref[k]] for k in ("roc", "pr")}

    out = []
    for alpha in DRIFT_ALPHAS:
        vals_by = run_alpha(alpha)
        row = {"alpha": alpha, "n_post": len(post)}
        for key in ("roc", "pr"):
            c, vals = cal[key], vals_by[key]
            flags = [v < c["trigger"] for v in vals]          # 用标定出的 δ，不用拍的 0.02
            new = [f and not r for f, r in zip(flags, ref_flags[key])]
            persist = [i for i in range(len(flags))
                       if i + 1 >= PERSIST and all(flags[i - PERSIST + 1:i + 1])]
            first_new = next((i for i, f in enumerate(new) if f), None)
            row[f"{key}_min"] = min(vals)
            row[f"{key}_drop"] = min(vals) - c["base"]
            row[f"{key}_alarm"] = sum(flags)
            row[f"{key}_new"] = sum(new)
            row[f"{key}_persist"] = len(persist)
            # 首个**新增**告警：报「注入开始后第几个窗」——1 = 注入当窗即发现
            row[f"{key}_first"] = (first_new - ONSET_IDX + 1) if (
                first_new is not None and first_new >= ONSET_IDX) else None
        out.append(row)
    return out


PSI_BINS = 10
PSI_WARN, PSI_ALERT = 0.10, 0.25   # 银行常用阈值：<0.1 稳定 / 0.1–0.25 轻微 / >0.25 显著


def _psi_reference(p, n_bins=PSI_BINS):
    """用**基准窗**的分数定分箱切点与各箱占比。

    切点只在基准窗定一次、后续窗沿用——**若每个窗各自等频分箱，占比恒等于 1/n，
    PSI 永远是 0**，这个监控就成了摆设。
    与本项目「目标编码只在训练集拟合」是同一条纪律：**参照系不能跟着被测对象一起动。**
    """
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    ratio = np.histogram(p, bins=edges)[0] / len(p)
    return edges, np.maximum(ratio, 1e-6)      # 防 log(0)


def _psi(p, edges, base_ratio):
    """PSI = Σ (当前占比 − 基准占比) × ln(当前占比 / 基准占比)。"""
    cur = np.histogram(p, bins=edges)[0] / len(p)
    cur = np.maximum(cur, 1e-6)
    return float(np.sum((cur - base_ratio) * np.log(cur / base_ratio)))


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


def _write_md(wins, base, trigger, n_alarm, day_max, cal, sens):
    last = wins[-1]
    decay = last["roc"] - base
    L = [
        "# 概念漂移监控演示（硬点⑩）\n",
        "冻结早期模型（fit day<83），在后续每 14 天窗上算 AUC（不重训），看衰减 + 监控触发器。",
        f"基线 ROC-AUC（首窗）={base:.4f}，触发线=基线−{DELTA}={trigger:.4f}，共 {len(wins)} 个窗、{n_alarm} 个触发告警。\n",
        "## 滚动窗 AUC\n",
        "| 窗口(day) | 实覆盖天数 | n | 欺诈率 | ROC-AUC | PR-AUC | PSI | 告警 |",
        "|-----------|-----------|---|--------|---------|--------|-----|------|",
    ]
    for w in wins:
        flag = "⚠️" if w["roc"] < trigger else ""
        if w["psi"] >= PSI_ALERT:
            flag += "📊显著"
        elif w["psi"] >= PSI_WARN:
            flag += "📊轻微"
        psi_txt = "基准" if w is wins[0] else f"{w['psi']:.4f}"
        days_txt = f"**{w['days']}（不满）**" if w["partial"] else f"{w['days']}"
        L.append(f"| [{w['lo']},{w['hi']}) | {days_txt} | {w['n']:,} | {w['fraud_rate']:.2%} | "
                 f"{w['roc']:.4f} | {w['pr']:.4f} | {psi_txt} | {flag} |")
    partial = [w for w in wins if w["partial"]]
    if partial:
        full_n = [w["n"] for w in wins if not w["partial"]]
        L.append("")
        for w in partial:
            L.append(
                f"> ⚠️ **窗 [{w['lo']},{w['hi']}) 不完整**：数据在 day {day_max} 截断，"
                f"实际只覆盖 **{w['days']} / {WINDOW}** 天、**{w['n']:,}** 笔"
                f"（完整窗 {min(full_n):,}–{max(full_n):,} 笔）。"
                f"它的 ROC/PR 建立在更少样本上、估计更噪——"
                f"**不得拿它的读数（如「末窗高于基线」）当同尺度结论。**")
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
        f"- **PSI（模型分分布稳定性）**：各窗相对首窗（基准）的 "
        f"PSI 最大 **{max(w['psi'] for w in wins):.4f}**、末窗 {wins[-1]['psi']:.4f}"
        f"（银行阈值：<0.10 稳定 / 0.10–0.25 轻微 / >0.25 显著）。"
        f"分箱切点**只在基准窗定一次**，后续窗沿用——"
        f"每窗各自等频分箱会让占比恒为 1/n、PSI 恒等于 0，监控就成了摆设。",
        f"- **PR-AUC 相对基线的最大落差 {wins[0]['pr'] - min(w['pr'] for w in wins):.4f}**"
        f"（基线 {wins[0]['pr']:.4f} → 最低 {min(w['pr'] for w in wins):.4f}），"
        f"同期 ROC 落差仅 {wins[0]['roc'] - min(w['roc'] for w in wins):.4f}"
        f" —— 正类监控该盯 PR。",
    ]
    r, p = cal["roc"], cal["pr"]
    L += [
        "",
        f"## δ 标定：按各指标自己的抽样噪声定阈值（{Z_SIGMA:g}σ，窗内自助 B={BOOT_B}）\n",
        f"`DELTA={DELTA}` 是拍的常数，而两个指标的自然波动幅度差一个量级"
        f"（ROC 全幅 {max(w['roc'] for w in wins) - min(w['roc'] for w in wins):.4f}、"
        f"PR {max(w['pr'] for w in wins) - min(w['pr'] for w in wins):.4f}）"
        f"——**同一个 δ 对两者的严厉程度完全不同**。下表按各指标自身噪声重新标定。\n",
        "| 指标 | 基线 | SE(基线) | SE(典型窗) | SE(差) | **δ** | 触发线 | 单窗告警 | 连续2窗告警 |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| ROC-AUC | {r['base']:.4f} | {r['se_base']:.4f} | {r['se_typ']:.4f} | {r['se_diff']:.4f} | "
        f"**{r['delta']:.4f}** | {r['trigger']:.4f} | {r['n_alarm']}/{r['n_eval']} | {r['n_persist']} |",
        f"| PR-AUC | {p['base']:.4f} | {p['se_base']:.4f} | {p['se_typ']:.4f} | {p['se_diff']:.4f} | "
        f"**{p['delta']:.4f}** | {p['trigger']:.4f} | {p['n_alarm']}/{p['n_eval']} | {p['n_persist']} |",
        "",
        "### 噪声归一化后的「谁更敏感」\n",
        "直接比 PR 落差 vs ROC 落差是**混了量纲**的。除以各自的 SE(差) 才公平：\n",
        "| 指标 | 相对基线最大落差 | ÷ SE(差) = **落差信噪比** |",
        "|---|---|---|",
        f"| ROC-AUC | {r['drop']:.4f} | **{r['drop_z']:.1f}** |",
        f"| PR-AUC | {p['drop']:.4f} | **{p['drop_z']:.1f}** |",
        "",
        "### 超量波动：抽样 SE 够不够定 δ\n",
        "| 指标 | 跨窗实测 SD | 典型抽样 SE | **超量波动比** |",
        "|---|---|---|---|",
        f"| ROC-AUC | {r['sd_obs']:.4f} | {r['se_typ']:.4f} | **{r['excess']:.2f}×** |",
        f"| PR-AUC | {p['sd_obs']:.4f} | {p['se_typ']:.4f} | **{p['excess']:.2f}×** |",
        "",
        "> **超量波动比 > 1** 表示窗间差异超出抽样噪声能解释的范围——存在**真实的分布波动**成分。"
        "此时抽样 SE **不足以单独定 δ**：按 2σ 抽样噪声设线会把正常波动也判成告警。"
        "本项目对此的处置是**持续性规则**（连续 "
        f"{PERSIST} 窗跌破才告警），上表最后一列即该规则下的告警数。",
        "",
        "> ⚠️ **本节只标定了「不误报」这一半。** 这 6 个评估窗里**没有一次真退化事件**，"
        "所以**检出率（真退化来了多久能发现）在真数据上无法测量**——下一节用"
        "**明确标注的构造场景**补这一半。",
        "",
        "> ### 为什么不选定触发指标\n"
        "> **没有真退化事件 → 任何触发器的检出率都无法在真数据上验证 → 选定一个就等于声称验证过它。**\n"
        "> 保留 ROC 已不可辩护（落差 1.5 SE，本数据上没有分辨力）；直接换 PR 则要辩护"
        "「6 个窗响 3 次」而那 3 次**证明不了是对的**（超量波动比 4.64× 说明含真实分布波动，"
        "而分布波动 ≠ 模型退化——本实验无干预、无结局标签，二者分不开）。\n"
        "> → **两指标各自标定、并列上报，不选定。** 上方 `δ=0.02` 那一档是**历史拍定值，"
        "保留仅为可追溯，不代表被选中**。",
        "",
        "## 合成漂移灵敏度（**构造场景**，补「检出率」这一半）\n",
        f"注入方式：`p' = (1−α)·p + α·u`，`u~U(0,1)`，**只动分数不动标签**"
        f"（保留 ground truth，评估口径不被污染；ROC/PR 都基于排序，掺噪破坏校准无妨）。"
        f"从第 **{ONSET_IDX}** 个评估窗起生效，模拟「某时点之后开始退化」。"
        f"判据用**上一节标定出的 δ**（ROC {cal['roc']['delta']:.4f} / PR {cal['pr']['delta']:.4f}）"
        f"与同一套持续性规则（连续 {PERSIST} 窗）。\n",
        "| α | ROC 最低 | ROC 落差 | ROC 告警 | **ROC 新增** | **ROC 首检** | PR 最低 | PR 落差 | PR 告警 | **PR 新增** | **PR 首检** |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sens:
        L.append(
            f"| **{r['alpha']:.2f}** | {r['roc_min']:.4f} | {r['roc_drop']:+.4f} | "
            f"{r['roc_alarm']}/{r['n_post']} | **{r['roc_new']}** | "
            f"{('第 ' + str(r['roc_first']) + ' 窗') if r['roc_first'] else '—'} | "
            f"{r['pr_min']:.4f} | {r['pr_drop']:+.4f} | "
            f"{r['pr_alarm']}/{r['n_post']} | **{r['pr_new']}** | "
            f"{('第 ' + str(r['pr_first']) + ' 窗') if r['pr_first'] else '—'} |")
    zero = sens[0]
    L += [
        "",
        "> ### ⚠️ 为什么必须看「新增」而不是「告警」这一列\n"
        f"> **PR 在注入之前就已经在 {zero['pr_alarm']}/{zero['n_post']} 个窗告警**"
        f"（ROC 是 {zero['roc_alarm']}/{zero['n_post']}）。"
        "直接看「告警」列，就分不清某个窗是**检出了注入**还是**本来就在响**。\n"
        "> 所以本表以 **α=0 为参照**，另报 **「新增」= 本 α 下告警 且 α=0 下不告警**"
        "——**那才是注入的效应**。「首检」也按首个**新增**告警计。\n"
        "> （执行方第一版把 `ONSET_IDX` 正好设在 PR 既有告警起点上，"
        "导致 α=0 行报出「第 1 窗检出」这种自相矛盾的读数；本列是那处缺陷的修法。"
        "同一条纪律在本项目反复出现：**判别力必须相对既有基线算，不能拿绝对命中数当功劳。**）",
        "",
        f"> **α=0 行是对照**：不注入，`新增` 两列必须都是 **0**"
        f"（实测 ROC {zero['roc_new']} / PR {zero['pr_new']}）——"
        "**这是确认注入装置本身不改变基线的自检**；若非 0，后面每一行都不可信。",
        "> **「首检」= 从注入开始算第几个窗出现新增告警**（1 = 注入当窗即发现）。",
        "",
        "",
        "### ⭐ 两节结论相反，而这恰好是「不选定触发指标」的实证\n",
        f"真数据那一节：**PR 有分辨力（落差 {cal['pr']['drop_z']:.1f} 个 SE）、"
        f"ROC 没有（{cal['roc']['drop_z']:.1f} 个 SE）**。",
        f"本节（注入纯排序退化）：**ROC 在 α={DRIFT_ALPHAS[1]:.2f} 就新增 "
        f"{sens[1]['roc_new']} 个告警、注入当窗即检出；PR 在同一 α 下新增 "
        f"{sens[1]['pr_new']} 个**。",
        "",
        "> **两者不矛盾——它们回答的是两个不同的问题：**\n"
        "> · 真数据那节问「**哪个指标看见了实际发生的那次下滑**」→ 答案是 PR；\n"
        "> · 本节问「**哪个指标的标定后触发器会对注入的排序退化响**」→ 答案是 ROC。\n"
        ">\n"
        "> **机制（可从上面几张表直接读出）**：\n"
        f"> ① PR 的 δ 更宽（{cal['pr']['delta']:.4f} vs ROC {cal['roc']['delta']:.4f}），"
        f"**因为 PR 本身更噪**（SE(差) {cal['pr']['se_diff']:.4f} vs {cal['roc']['se_diff']:.4f}，"
        f"约 {cal['pr']['se_diff'] / cal['roc']['se_diff']:.1f}×）。"
        "**噪声大 → δ 必须宽 → 触发器就钝。**\n"
        f"> ② PR 还处在**饱和**状态：注入前它已在 {zero['pr_alarm']}/{zero['n_post']} 个窗告警，"
        "**已经在响的窗无法再「新增」**，所以注入在那几个窗上不可能被记为检出。\n"
        ">\n"
        "> ### → **「敏感的指标」不等于「敏感的触发器」。**\n"
        "> 指标的**分辨力**（落差 ÷ 自身噪声）和触发器的**灵敏度**（δ 相对注入效应）是两件事，"
        "**因为 δ 正是从噪声里推出来的**——一个指标越敏感地捕捉到真实波动，它的噪声就越大，"
        "标定出的 δ 就越宽，触发器反而越钝。\n"
        "> → **所以「两指标并列上报、不选定」不是和稀泥**：PR 负责**看见**，ROC 的触发器负责**响**，"
        "**任何单一指标在这份数据上都被证明不够**。",
        "",
        "> ### ⚠️ 这一节能声称什么 / 不能声称什么\n"
        "> **能**：在**这种退化形态**（均匀噪声掺合，纯排序能力损失）下，"
        "标定后的检测器需要多大的 α 才有反应、以及从注入到首次**新增**告警差几个窗。\n"
        "> **不能**：① **这不是「我经历过真实退化」的证据**——α 是设定的、形态是构造的，"
        "MODEL_CARD 的纪律是「合成漂移可用于检验检测器，**不得冒充真实退化**」；"
        "② **不能外推到其他漂移形态**（本注入是无结构噪声；真实概念漂移往往是"
        "「某个子群的 P(y|x) 变了」，那种形态检测难度不同）；"
        "③ 仍**未做显著性检验**，单次注入、单一随机种子。",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    write_report(OUT_MD, "\n".join(L))


if __name__ == "__main__":
    main()
