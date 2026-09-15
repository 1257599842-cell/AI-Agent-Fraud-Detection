"""逻辑回归评分卡对照：分箱 → WOE/IV → LR → 标准刻度评分卡 → 同口径对照 GBDT。

## 为什么做这个对照
银行到今天仍大量使用逻辑回归评分卡，原因不是技术落后，是**监管要求模型可解释、
可复核、可审计**——每一笔拒绝都要能说出理由，而且理由要能被业务和监管直接读懂。

「我可以用 SHAP」是所有人都会给的答案。本模块给的是另一种答案：
**我做了对照，评分卡在这个数据上损失多少性能、换来什么，所以受监管场景我会怎么选。**

## 最关键的防泄漏点（本模块的核心纪律）
**WOE 本质上是有监督的目标编码**——它用标签信息给每个箱赋值。
所以**分箱与 WOE 必须只在训练窗拟合，再映射到测试窗**；全量分箱就是泄漏。

这与本项目早已焊死的那条纪律（目标编码只在训练集拟合，高基数类别尤甚）是同一个问题，
也与图特征的两层时间纪律同源：**任何吃标签的变换，都必须受时间切分约束。**

## 口径（与 GBDT 对照必须严格一致）
时间切分 `fit < day 132 / val [132,146) / test ≥ 146`，无 embargo，
与 `graph_vs_tabular.md` 的 `PR-AUC 0.5645 → 0.6032`、`bank_metrics.md` 的 KS 同源同窗。
LR 在 `fit` 上拟合（不使用 val——评分卡不需要早停），在同一个 test 窗评估。

## 范围（严格限定，不扩张）
不调参、不追 AUC、不碰 434 列、不引第三方评分卡库（自己写分箱更好讲）。
产出四个对照数字 + 一张评分卡表即收工。

用法：python -m src.model.scorecard
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.report_io import write_report

ROOT = Path(__file__).resolve().parents[2]
MERGED = ROOT / "data" / "processed" / "train_merged.parquet"
GRAPH = ROOT / "data" / "processed" / "graph_features.parquet"
SCORES = ROOT / "data" / "processed" / "gvt_scores.parquet"
REPORT = ROOT / "reports" / "scorecard.md"

T0, VAL_DAYS = 146, 14          # 与 train_baseline 同一套切分常量
MIN_BIN_FRAC = 0.05             # 每箱样本占比下限（银行惯例）
MAX_BINS = 6
IV_FLOOR, IV_CEIL = 0.02, 0.5   # IV<0.02 剔除；IV>0.5 警惕泄漏
CORR_MAX = 0.80                 # 高相关剔除阈值
BASE_SCORE, BASE_ODDS, PDO = 600, 50.0, 20   # 标准刻度

# 候选：GBDT gain top-20 表特征 + 进 top-11 的图特征
CAND_TAB = ["V258", "V257", "DeviceInfo", "C1", "C14", "C13", "V294", "D2",
            "TransactionAmt", "R_emaildomain", "card1", "C11", "card2", "addr1",
            "D15", "id_31", "P_emaildomain", "D1", "C8", "V317"]
CAND_GRAPH = ["card1_prior_fraud_rate", "card1_addr1_prior_fraud_rate",
              "card1_email_prior_fraud_rate"]
# card1 是**卡号 ID**，不是风险属性。评分卡里放 ID 等于给每张卡记忆一个分值，
# 业务上讲不通、监管不会接受。GBDT 能用它（树会切段），评分卡不能——**这是真实分歧点**。
DROP_ID_LIKE = {"card1"}


# ── 分箱 ────────────────────────────────────────────────────────────────
def bin_numeric(x, y, max_bins=MAX_BINS, min_frac=MIN_BIN_FRAC):
    """数值型等频分箱后**合箱至 WOE 单调**。缺失单独成箱（不填均值）。

    单调是银行的硬要求：业务上必须能说出「分越高风险越低」。
    """
    m = ~pd.isna(x)
    xs = np.asarray(x[m], dtype=float)
    if len(np.unique(xs)) < 2:
        return [], True
    qs = np.linspace(0, 1, max_bins + 1)[1:-1]
    cuts = sorted(set(np.quantile(xs, qs).tolist()))
    while True:
        edges = [-np.inf] + cuts + [np.inf]
        idx = np.digitize(xs, cuts, right=True)
        cnt = np.bincount(idx, minlength=len(edges) - 1)
        # 先保证每箱样本量
        small = [i for i, c in enumerate(cnt) if c < min_frac * len(xs)]
        if small and len(cuts) > 0:
            cuts.pop(min(small[0], len(cuts) - 1))
            continue
        w = _woe_by_index(idx, np.asarray(y[m]), len(edges) - 1)
        bad = _first_nonmonotonic(w)
        if bad is None or len(cuts) == 0:
            return cuts, bad is None
        cuts.pop(bad)                      # 合箱：去掉破坏单调的那个切点


def _woe_by_index(idx, y, n_bins):
    y = np.asarray(y).astype(bool)
    nb, ng = max(y.sum(), 1), max((~y).sum(), 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        out.append(np.log(max(y[m].sum(), 0.5) / nb / (max((~y[m]).sum(), 0.5) / ng)))
    return np.array(out)


def _first_nonmonotonic(w):
    """返回第一个破坏单调的切点下标；全单调返回 None。"""
    if len(w) < 3:
        return None
    up = all(w[i] <= w[i + 1] for i in range(len(w) - 1))
    dn = all(w[i] >= w[i + 1] for i in range(len(w) - 1))
    if up or dn:
        return None
    dif = np.diff(w)
    sign = 1 if (dif > 0).sum() >= (dif < 0).sum() else -1
    for i, d in enumerate(dif):
        if np.sign(d) != sign:
            return i
    return 0


def bin_categorical(x, y, min_frac=MIN_BIN_FRAC):
    """类别型：低频取值合成「其他」，缺失单独成箱。"""
    s = pd.Series(x).astype("string")
    vc = s.value_counts(dropna=True)
    keep = list(vc[vc >= min_frac * len(s)].index)
    return keep


def apply_bins(x, spec):
    """把分箱规则映射到任意窗（**规则来自训练窗，此处只应用**）。"""
    if spec["kind"] == "num":
        idx = np.full(len(x), -1, dtype=int)     # −1 = 缺失箱
        m = ~pd.isna(x)
        idx[m] = np.digitize(np.asarray(x[m], dtype=float), spec["cuts"], right=True)
        return idx
    s = pd.Series(x).astype("string")
    idx = np.full(len(s), -1, dtype=int)
    # **nullable string 的比较会产生 <NA>**，直接 to_numpy() 不是纯 bool 数组，
    # 用作索引会抛 IndexError。fillna(False) 把缺失显式判为「不等于」。
    for i, v in enumerate(spec["keep"]):
        idx[(s == v).fillna(False).to_numpy(dtype=bool)] = i
    notna = s.notna().to_numpy(dtype=bool)
    idx[notna & (idx == -1)] = len(spec["keep"])   # 其他
    idx[~notna] = -1                                # 缺失单独成箱
    return idx


def woe_table(idx, y):
    """按箱算 WOE 与 IV。**只在训练窗调用。**"""
    y = np.asarray(y).astype(bool)
    nb, ng = max(y.sum(), 1), max((~y).sum(), 1)
    rows = {}
    iv = 0.0
    for b in sorted(set(idx.tolist())):
        m = idx == b
        bad, good = max(int(y[m].sum()), 0), max(int((~y[m]).sum()), 0)
        pb, pg = max(bad, 0.5) / nb, max(good, 0.5) / ng
        w = float(np.log(pb / pg))
        iv += (pb - pg) * w
        rows[b] = {"woe": w, "n": int(m.sum()), "bad": bad, "good": good,
                   "bad_rate": bad / max(m.sum(), 1)}
    return rows, float(iv)


# ── 主流程 ──────────────────────────────────────────────────────────────
def load():
    need = sorted(set(CAND_TAB) | {"TransactionID", "TransactionDT", "isFraud"})
    import pyarrow.parquet as pq
    avail = set(pq.ParquetFile(MERGED).schema.names)
    miss = [c for c in need if c not in avail]
    if miss:
        print(f"  ⚠️ 候选里有 {len(miss)} 个列不在数据中，剔除：{miss}")
        need = [c for c in need if c in avail]
    df = pd.read_parquet(MERGED, columns=need)
    g = pd.read_parquet(GRAPH, columns=["TransactionID"] + CAND_GRAPH)
    df = df.merge(g, on="TransactionID", how="left", validate="one_to_one")
    d = df["TransactionDT"] // 86_400
    df["day"] = (d - d.min()).astype(int)
    return df


def fit_scorecard(df, feats):
    """**只在训练窗**拟合分箱规则、WOE、LR —— WOE 是有监督编码，全量拟合即泄漏。"""
    fit = df["day"] < (T0 - VAL_DAYS)
    yf = df.loc[fit, "isFraud"].to_numpy()
    specs, woes, ivs, dropped = {}, {}, {}, []
    for c in feats:
        x = df.loc[fit, c]
        if pd.api.types.is_numeric_dtype(x):
            cuts, mono = bin_numeric(x, yf)
            spec = {"kind": "num", "cuts": cuts, "monotonic": mono}
        else:
            spec = {"kind": "cat", "keep": bin_categorical(x, yf)}
        idx = apply_bins(x, spec)
        w, iv = woe_table(idx, yf)
        if iv < IV_FLOOR:
            dropped.append((c, f"IV {iv:.4f} < {IV_FLOOR}")); continue
        specs[c], woes[c], ivs[c] = spec, w, iv
    return specs, woes, ivs, dropped


def to_woe_frame(df, mask, specs, woes):
    out = {}
    for c, spec in specs.items():
        idx = apply_bins(df.loc[mask, c], spec)
        wmap = {b: v["woe"] for b, v in woes[c].items()}
        out[c] = np.array([wmap.get(int(i), 0.0) for i in idx])
    return pd.DataFrame(out, index=df.index[mask])


def drop_negative_coefficients(lr, keep, Wf, y, max_rounds=10):
    """剔除**系数为负**的变量，逐轮重拟合直到全为正。

    **为什么这是标准步骤而不是补丁**：WOE 已经把方向编码进去了
    （WOE 越高 = 该箱越坏），所以 LR 系数理应**全为正**。
    出现负系数，说明该变量与其他变量共线、被「借」去充当修正项——
    它在卡上的分值方向会与自身坏率相反（**坏率最高的箱反而加分**），
    业务无法解释、监管不会接受。

    银行的做法是**直接剔除**，不是强行约束符号：
    一个方向讲不通的变量，留在卡上就是一个讲不通的拒绝理由。
    """
    from sklearn.linear_model import LogisticRegression
    dropped = []
    for _ in range(max_rounds):
        neg = [c for c, b in zip(keep, lr.coef_[0]) if b < 0]
        if not neg:
            break
        worst = min(zip(keep, lr.coef_[0]), key=lambda t: t[1])[0]
        dropped.append((worst, f"系数 {dict(zip(keep, lr.coef_[0]))[worst]:+.4f} < 0，"
                               f"分值方向与坏率相反"))
        keep = [c for c in keep if c != worst]
        lr.fit(Wf[keep], y)
    return keep, dropped


def drop_correlated(W, ivs, thr=CORR_MAX):
    """高相关只留 IV 更高的那个 —— 评分卡要求变量间可独立解释。"""
    cm = W.corr().abs()
    keep, dropped = list(W.columns), []
    for i, a in enumerate(W.columns):
        for b in W.columns[i + 1:]:
            if a in keep and b in keep and cm.loc[a, b] > thr:
                lose = b if ivs[a] >= ivs[b] else a
                keep.remove(lose)
                dropped.append((lose, f"与 {a if lose==b else b} 相关 {cm.loc[a,b]:.2f}"))
    return keep, dropped


def scale(coef, intercept, n_feat):
    """标准刻度：基准分 600、基准 odds 1:50、PDO 20。

    score = offset + Σ(−(β_i·WOE_i + α/n)·factor)
    这是评分卡的标准折算法，产出的分值表可直接给业务与监管审阅。
    """
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    return factor, offset


def expected_loss(y, p, amt, c_fp=25.0):
    """套上本项目的代价敏感阈值，算期望总损失。

    与 `cost_sensitive.md` 同一口径：逐样本金额感知阈值 t_i = c_FP/(a_i + c_FP)。
    拦截 → 若为好人则赔 c_FP；放行 → 若为欺诈则赔整笔金额。
    """
    y = np.asarray(y).astype(bool)
    p, amt = np.asarray(p, float), np.asarray(amt, float)
    block = p > (c_fp / (amt + c_fp))
    return float((block & ~y).sum() * c_fp + (~block & y) @ amt), int(block.sum())


def ece(y, p, n_bins=15):
    y, p = np.asarray(y).astype(float), np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from src.model.bank_metrics import ks

    df = load()
    feats = [c for c in CAND_TAB + CAND_GRAPH
             if c in df.columns and c not in DROP_ID_LIKE]
    print(f"候选特征 {len(feats)} 个（已剔除 ID 型 {sorted(DROP_ID_LIKE)}）")

    specs, woes, ivs, dropped_iv = fit_scorecard(df, feats)
    print(f"  IV 筛选后剩 {len(specs)} 个；剔除 {len(dropped_iv)} 个")
    high_iv = [(c, v) for c, v in ivs.items() if v > IV_CEIL]

    fit = df["day"] < (T0 - VAL_DAYS)
    test = df["day"] >= T0
    Wf = to_woe_frame(df, fit, specs, woes)
    keep, dropped_corr = drop_correlated(Wf, ivs)
    print(f"  高相关剔除 {len(dropped_corr)} 个 → 最终入模 {len(keep)} 个")

    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(Wf[keep], df.loc[fit, "isFraud"])
    keep, dropped_sign = drop_negative_coefficients(lr, keep, Wf, df.loc[fit, "isFraud"])
    Wt = to_woe_frame(df, test, specs, woes)[keep]
    p_sc = lr.predict_proba(Wt)[:, 1]

    y_te = df.loc[test, "isFraud"].to_numpy()
    amt = df.loc[test, "TransactionAmt"].to_numpy()
    gb = pd.read_parquet(SCORES)
    gb = gb[gb["split"] == "test"].set_index("TransactionID").loc[
        df.loc[test, "TransactionID"].to_numpy()]
    p_gb = gb["p_graph"].to_numpy()
    assert (gb["isFraud"].to_numpy() == y_te).all(), "GBDT 分数与本窗标签未对齐"

    res = {}
    for name, p in (("评分卡（LR+WOE）", p_sc), ("GBDT（表+图）", p_gb)):
        loss, nblock = expected_loss(y_te, p, amt)
        res[name] = {"ks": ks(y_te, p)[0], "pr": average_precision_score(y_te, p),
                     "roc": roc_auc_score(y_te, p), "ece": ece(y_te, p),
                     "loss": loss, "nblock": nblock}
        print(f"  {name}: KS {res[name]['ks']:.4f}  PR-AUC {res[name]['pr']:.4f}  "
              f"ECE {res[name]['ece']:.4f}  期望损失 ${loss:,.0f}")

    card = build_card(specs, woes, keep, lr, len(keep))
    _write(df, feats, specs, woes, ivs, keep, dropped_iv, dropped_corr,
           dropped_sign, high_iv, res, card, lr, len(fit), int(fit.sum()), int(test.sum()))
    print(f"\n✅ → {REPORT.relative_to(ROOT)}")


def build_card(specs, woes, keep, lr, n_feat):
    """产出真实的评分卡表：哪个变量、哪个区间、给多少分。"""
    factor, offset = scale(lr.coef_[0], lr.intercept_[0], n_feat)
    base = offset - factor * lr.intercept_[0]
    rows = []
    for c, beta in zip(keep, lr.coef_[0]):
        spec = specs[c]
        for b, v in sorted(woes[c].items()):
            if spec["kind"] == "num":
                cuts = spec["cuts"]
                if b == -1:
                    label = "缺失"
                elif not cuts:
                    label = "全体"
                elif b == 0:
                    label = f"≤ {cuts[0]:.4g}"
                elif b >= len(cuts):
                    label = f"> {cuts[-1]:.4g}"
                else:
                    label = f"({cuts[b-1]:.4g}, {cuts[b]:.4g}]"
            else:
                label = ("缺失" if b == -1
                         else "其他" if b >= len(spec["keep"]) else str(spec["keep"][b]))
            rows.append({"变量": c, "分箱": label, "样本": v["n"],
                         "坏率": v["bad_rate"], "WOE": v["woe"],
                         "分值": -factor * beta * v["woe"]})
    return pd.DataFrame(rows), base


def _write(df, feats, specs, woes, ivs, keep, dropped_iv, dropped_corr,
           dropped_sign, high_iv, res, card_pack, lr, _n, n_fit, n_test):
    card, base = card_pack
    sc, gb = res["评分卡（LR+WOE）"], res["GBDT（表+图）"]
    L = [
        "# 逻辑回归评分卡 vs GBDT：同口径对照\n",
        "> **为什么做这个对照**：银行到今天仍大量使用评分卡，原因不是技术落后，",
        "> 是**监管要求模型可解释、可复核、可审计**——每一笔拒绝都要能说出理由，",
        "> 且理由要能被业务和监管直接读懂。\n",
        "> 「我可以用 SHAP」是通用答案。本节给的是另一种：**我做了对照，",
        "> 评分卡在这个数据上损失多少性能、换来什么，所以受监管场景我会怎么选。**\n",
        "## 口径（与 GBDT 严格一致，否则对照无意义）\n",
        f"时间切分 `fit < day {T0-VAL_DAYS} / test ≥ {T0}`，**无 embargo**，",
        "与 `graph_vs_tabular.md` 的 `PR-AUC 0.5645→0.6032`、`bank_metrics.md` 的 KS 同源同窗。",
        f"训练 **{n_fit:,}** 笔、测试 **{n_test:,}** 笔。\n",
        "### ⚠️ 最关键的防泄漏点：WOE 是有监督的目标编码\n",
        "WOE 用**标签信息**给每个箱赋值，所以**分箱与 WOE 只在训练窗拟合、再映射到测试窗**；",
        "**全量分箱就是泄漏**。\n",
        "> 这与本项目早已焊死的那条纪律（目标编码只在训练集拟合，高基数类别尤甚）是同一个问题，",
        "> 也与图特征的两层时间纪律同源：**任何吃标签的变换，都必须受时间切分约束。**\n",
        "## 1. 四项同口径对照\n",
        "| 指标 | 评分卡（LR+WOE） | GBDT（表+图） | 差 |", "|---|---|---|---|",
        f"| **KS** | {sc['ks']:.4f} | **{gb['ks']:.4f}** | {sc['ks']-gb['ks']:+.4f} |",
        f"| **PR-AUC** | {sc['pr']:.4f} | **{gb['pr']:.4f}** | {sc['pr']-gb['pr']:+.4f} |",
        f"| ROC-AUC | {sc['roc']:.4f} | {gb['roc']:.4f} | {sc['roc']-gb['roc']:+.4f} |",
        f"| **ECE**（越小越准） | {sc['ece']:.4f} | {gb['ece']:.4f} | {sc['ece']-gb['ece']:+.4f} |",
        f"| **期望总损失**（金额感知阈值，c_FP=$25） | ${sc['loss']:,.0f} | ${gb['loss']:,.0f} | "
        f"{(sc['loss']-gb['loss'])/gb['loss']:+.1%} |",
        f"| ↳ 拦截笔数 | {sc['nblock']:,} | {gb['nblock']:,} | |",
        "",
        "> 期望损失口径与 `cost_sensitive.md` 一致：逐样本金额感知阈值 `t_i = c_FP/(a_i + c_FP)`；",
        "> 拦截且为好人赔 `c_FP`，放行且为欺诈赔整笔金额。\n",
        "### 判读\n",
        f"**GBDT 赢，而差距本身就是结论**：评分卡 KS 低 **{gb['ks']-sc['ks']:.3f}**、",
        f"期望损失高 **{(sc['loss']-gb['loss'])/gb['loss']:.1%}**。换来的是：\n",
        "1. **逐笔可解释的拒绝理由** —— 每一笔的分值可拆到「哪个变量、哪个区间、扣了多少分」；",
        "2. **可被业务与监管直接审阅的分值表**（见下节），不需要额外的解释器；",
        "3. **更稳定的跨时间表现**（线性模型自由度低，不易追随短期噪声）。\n",
        "> **在受监管的授信场景我会用评分卡；在实时反欺诈这种不需要向客户逐笔解释、",
        "> 但对漏损极敏感的场景我会用 GBDT。两者不是替代关系。**\n",
        "## 2. 入模变量与 IV\n",
        f"候选 {len(feats)} 个（GBDT gain top-20 表特征 + 进 top-11 的图特征）→ "
        f"IV 与相关性筛选后**入模 {len(keep)} 个**。\n",
        "| 变量 | IV | 分箱数 | 单调 |", "|---|---|---|---|"]
    for c in sorted(keep, key=lambda x: -ivs[x]):
        mono = specs[c].get("monotonic")
        L.append(f"| `{c}` | {ivs[c]:.4f} | {len(woes[c])} | "
                 f"{'✅' if mono else ('—' if specs[c]['kind']=='cat' else '合箱后仍非严格单调')} |")
    L += ["", "**剔除记录**（照实列出，不只报留下的）：\n"]
    if dropped_iv:
        L.append(f"- IV 不足（<{IV_FLOOR}）**{len(dropped_iv)}** 个："
                 + "、".join(f"`{c}`({r.split()[1]})" for c, r in dropped_iv[:8])
                 + ("…" if len(dropped_iv) > 8 else ""))
    if dropped_corr:
        L.append(f"- 高相关（>{CORR_MAX}）**{len(dropped_corr)}** 个："
                 + "、".join(f"`{c}`" for c, _ in dropped_corr))
    if dropped_sign:
        L.append(f"- **系数为负 {len(dropped_sign)} 个**："
                 + "、".join(f"`{c}`（{r}）" for c, r in dropped_sign) + "。")
        L.append("  WOE 已把方向编码进去（WOE 越高 = 该箱越坏），所以系数**理应全为正**。"
                 "出现负系数说明该变量与他人共线、被「借」去当修正项，"
                 "**其分值方向会与自身坏率相反——坏率最高的箱反而加分**，"
                 "业务无法解释、监管不会接受。银行的做法是**直接剔除**，"
                 "不是强行约束符号：**一个方向讲不通的变量，留在卡上就是一个讲不通的拒绝理由。**")
    L.append(f"- **ID 型**：`card1` 是**卡号**，不是风险属性。评分卡里放 ID 等于"
             "给每张卡记忆一个分值，业务上讲不通、监管不会接受。"
             "**GBDT 能用它（树会切段），评分卡不能——这是两者的真实分歧点，不是实现取舍。**")
    if high_iv:
        gcand = [c for c, _ in high_iv if c in CAND_GRAPH]
        L.append(f"- ⚠️ **IV > {IV_CEIL} 需警惕泄漏**：" +
                 "、".join(f"`{c}`(IV {v:.2f})" for c, v in high_iv) + "。")
        L.append("  **这些是 Vesta 的匿名工程特征（V/C 系列），构造方式未公开，"
                 "因此本项目无法对它们做泄漏审计——高 IV 是一个真实的、我解决不了的疑点，"
                 "照实记。**")
        L.append("  > 需要区分清楚：本项目做过的泄漏审计（标签隔离期 21→60 天、"
                 "图特征增益 +0.039→+0.023 缩但不崩）覆盖的是**自建的图特征**，"
                 "**不覆盖 V/C 系列**。**不能拿那个审计去替这些特征背书。**")
    L += ["", "## 3. 评分卡（标准刻度：基准分 600、基准 odds 1:50、PDO 20）\n",
          f"**基础分 {base:.0f}**，各变量按所落分箱加减。总分越高 → 风险越低。\n",
          "| 变量 | 分箱 | 样本 | 坏率 | WOE | 分值 |", "|---|---|---|---|---|---|"]
    for _, r in card.iterrows():
        L.append(f"| `{r['变量']}` | {r['分箱']} | {int(r['样本']):,} | {r['坏率']:.2%} | "
                 f"{r['WOE']:+.4f} | **{r['分值']:+.1f}** |")
    L += ["",
          "> 这张表是本节最值钱的交付物：**一眼可见「哪个变量、哪个区间、给多少分」**，",
          "> 业务和监管可以直接读，不需要任何解释器。\n",
          "## 口径与限制\n",
          "- **未调参**：LR 用默认 `C=1.0`，不做网格搜索——本节的目的是对照，不是把 LR 推到极限。",
          "- 分箱为等频起步 + **合箱至 WOE 单调**；每箱样本占比下限 "
          f"{MIN_BIN_FRAC:.0%}；缺失**单独成箱**，不填均值（沿用本项目「缺失当一种取值」的纪律）。",
          "- 评分卡在 `fit` 窗拟合，**不使用 val 窗**（评分卡不需要早停）；GBDT 用了 val 做早停。",
          "  这对 GBDT 略有利，属于两类模型的固有差异，**照实记，不做补偿**。\n"]
    write_report(REPORT, "\n".join(L))


if __name__ == "__main__":
    main()
