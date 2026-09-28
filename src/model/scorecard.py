"""小规模WOE+LR评分卡对照；保留GBDT主模型。

拟合窗[0,132)，测试窗[146,182)，图标签使用主实验历史延迟特征。
分箱、WOE、IV及变量筛选只读拟合窗；普通箱至少占全拟合窗5%，
缺失单列。单调与系数符号是本对照的解释性约束。
产出四项指标、分值表以及精确分箱/系数/逐笔分数，不调参。
运行：python -m src.model.scorecard
"""

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from src.report_io import write_report

ROOT = Path(__file__).resolve().parents[2]
MERGED = ROOT / "data" / "processed" / "train_merged.parquet"
GRAPH = ROOT / "data" / "processed" / "graph_features.parquet"
SCORES = ROOT / "data" / "processed" / "gvt_scores.parquet"
REPORT = ROOT / "reports" / "scorecard.md"

T0, VAL_DAYS = 146, 14          # 与 train_baseline 同一套切分常量
MIN_BIN_FRAC = 0.05             # 普通箱占训练全窗比例；缺失箱单列并披露
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
# card1为匿名卡相关字段，不假定它是唯一卡号。本最小评分卡不对其数值顺序作业务解释。
DROP_ID_LIKE = {"card1"}


# ── 分箱 ────────────────────────────────────────────────────────────────
def bin_numeric(x, y, max_bins=MAX_BINS, min_frac=MIN_BIN_FRAC):
    """数值型等频分箱后**合箱至 WOE 单调**。缺失单独成箱（不填均值）。

    单调是本对照的建模约束；缺失无自然次序，不参加普通箱单调约束。
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
        small = [i for i, c in enumerate(cnt) if c < min_frac * len(x)]
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
    # 若非空「其他」仍不足5%，再合入最小的保留类别。
    while keep:
        other_n = int(s.notna().sum() - vc.reindex(keep).sum())
        if other_n == 0 or other_n >= min_frac * len(s):
            break
        keep.remove(min(keep, key=lambda value: vc[value]))
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
        if x.notna().sum() < MIN_BIN_FRAC * len(x):
            dropped.append((c, '非缺失总样本不足训练全窗5%，无法建立普通箱')); continue
        if pd.api.types.is_numeric_dtype(x):
            cuts, mono = bin_numeric(x, yf)
            spec = {"kind": "num", "cuts": cuts, "monotonic": mono}
        else:
            spec = {"kind": "cat", "keep": bin_categorical(x, yf)}
        idx = apply_bins(x, spec)
        w, iv = woe_table(idx, yf)
        if any(v['n'] < MIN_BIN_FRAC * len(x) for b, v in w.items() if b != -1):
            raise ValueError(f'{c}普通箱不足训练全窗5%')
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
    """本对照选择逐轮剔除负系数，使分箱风险方向与分值方向一致。

    多变量条件效应可与单变量方向不同；负系数不自动证明共线或泄漏。
    这是本项目的解释性约束，不是普遍监管规则。
    """
    dropped = []
    for _ in range(max_rounds):
        neg = [c for c, b in zip(keep, lr.coef_[0]) if b < 0]
        if not neg:
            break
        worst = min(zip(keep, lr.coef_[0]), key=lambda t: t[1])[0]
        dropped.append((worst, f"系数 {dict(zip(keep, lr.coef_[0]))[worst]:+.4f} < 0，"
                               f"分值方向与坏率相反"))
        keep = [c for c in keep if c != worst]
        if not keep:
            raise ValueError('符号筛选剔除了全部变量')
        lr.fit(Wf[keep], y)
    if not keep or np.any(lr.coef_[0] < 0):
        raise ValueError('未在限定轮次内得到非负系数评分卡')
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
    """标准刻度：基准分 600、基准好:坏 odds 50:1、PDO 20。

    score = offset + Σ(−(β_i·WOE_i + α/n)·factor)
    这是评分卡的标准折算法，产出的分值表可直接给业务与监管审阅。
    """
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    return factor, offset


def expected_loss(y, p, amt, c_fp=25.0):
    """套上预设二动作代价阈值，以真标签计算模型化损失。

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
    from src.model.bank_metrics import ks, validate_scores

    df = load()
    feats = [c for c in CAND_TAB + CAND_GRAPH
             if c in df.columns and c not in DROP_ID_LIKE]
    print(f"候选特征 {len(feats)} 个（已剔除匿名标识候选 {sorted(DROP_ID_LIKE)}）")

    specs, woes, ivs, dropped_iv = fit_scorecard(df, feats)
    print(f"  IV 筛选后剩 {len(specs)} 个；剔除 {len(dropped_iv)} 个")
    high_iv = [(c, v) for c, v in ivs.items() if v > IV_CEIL]

    fit = df["day"] < (T0 - VAL_DAYS)
    test = df["day"] >= T0
    Wf = to_woe_frame(df, fit, specs, woes)
    keep, dropped_corr = drop_correlated(Wf, ivs)
    print(f"  高相关剔除 {len(dropped_corr)} 个 → 符号筛选前剩 {len(keep)} 个")

    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(Wf[keep], df.loc[fit, "isFraud"])
    keep, dropped_sign = drop_negative_coefficients(lr, keep, Wf, df.loc[fit, "isFraud"])
    print(f"  符号筛选后最终入模 {len(keep)} 个")
    Wt = to_woe_frame(df, test, specs, woes)[keep]
    p_sc = lr.predict_proba(Wt)[:, 1]

    y_te = df.loc[test, "isFraud"].to_numpy()
    amt = df.loc[test, "TransactionAmt"].to_numpy()
    gb = pd.read_parquet(SCORES)
    validate_scores(gb)
    if set(gb.loc[gb["split"] == "test", "TransactionID"]) != set(df.loc[test, "TransactionID"]):
        raise ValueError("评分卡与GBDT测试交易集合不同")
    gb = gb[gb["split"] == "test"].set_index("TransactionID").loc[
        df.loc[test, "TransactionID"].to_numpy()]
    p_gb = gb["p_graph"].to_numpy()
    if not np.array_equal(gb["isFraud"].to_numpy(), y_te):
        raise ValueError("GBDT分数与本窗标签未对齐")

    res = {}
    for name, p in (("评分卡（LR+WOE）", p_sc), ("GBDT（表+图）", p_gb)):
        loss, nblock = expected_loss(y_te, p, amt)
        res[name] = {"ks": ks(y_te, p)[0], "pr": average_precision_score(y_te, p),
                     "roc": roc_auc_score(y_te, p), "ece": ece(y_te, p),
                     "loss": loss, "nblock": nblock}
        print(f"  {name}: KS {res[name]['ks']:.4f}  PR-AUC {res[name]['pr']:.4f}  "
              f"ECE {res[name]['ece']:.4f}  模型化损失 ${loss:,.0f}")

    card = build_card(specs, woes, keep, lr, len(keep))
    export_artifacts(df, test, specs, woes, keep, lr, p_sc, card[1])
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
        possible = [-1] + list(range(len(spec.get('cuts', spec.get('keep'))) + 1))
        for b in possible:
            v = woes[c].get(b, {'woe': 0.0, 'n': 0, 'bad_rate': float('nan')})
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


def export_artifacts(df, test, specs, woes, keep, lr, p_sc, base):
    """保存精确分箱/系数和逐笔预测，使评分表可实际回算；不是服务主模型。"""
    factor, offset = scale(None, None, len(keep))
    artifact = {'version': 'scorecard-total-fit-min-bin-v2',
                'base_score': BASE_SCORE, 'pdo': PDO, 'base_good_bad_odds': BASE_ODDS,
                'factor': factor, 'offset': offset, 'intercept': float(lr.intercept_[0]),
                'base_points': float(base), 'features': keep,
                'coefficients': dict(zip(keep, map(float, lr.coef_[0]))),
                'specs': {c: specs[c] for c in keep},
                'woe': {c: {str(b): v['woe'] for b, v in woes[c].items()} for c in keep},
                'unseen_bin_woe': 0.0,
                'input_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (MERGED, GRAPH, SCORES)}}
    (SCORES.parent / 'scorecard_model.json').write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    out = df.loc[test, ['TransactionID', 'day', 'isFraud', 'TransactionAmt']].copy()
    out['p_scorecard'] = p_sc
    out['score'] = offset - factor * lr.decision_function(
        to_woe_frame(df, test, specs, woes)[keep])
    out.to_parquet(SCORES.parent / 'scorecard_test_scores.parquet', index=False)


def _write(df, feats, specs, woes, ivs, keep, dropped_iv, dropped_corr,
           dropped_sign, high_iv, res, card_pack, lr, _n, n_fit, n_test):
    card, base = card_pack
    sc, gb = res['评分卡（LR+WOE）'], res['GBDT（表+图）']
    L = ['# 逻辑回归评分卡 vs GBDT：同窗最小对照\n',
         '分箱、WOE/IV、相关性筛选与LR全部只在fit [0,132)拟合；test [146,182)与主实验逐笔ID、标签对齐。',
         '训练和早停没有统一21天标签成熟隔离。GBDT沿用主实验缓存，评分卡本次重新拟合。',
         f'训练 **{n_fit:,}** 笔；测试 **{n_test:,}** 笔。',
         '本版修正普通箱最小样本的分母：由非缺失样本改为训练全窗；缺失箱单列并披露小样本。',
         '旧版评分卡数字不可与本版混用。候选是历史GBDT重要性列表固定的22列，未根据此次测试表现再选变量。\n',
         '## 1. 四项对照\n',
         '| 指标 | 评分卡 LR+WOE | GBDT 表+图 | 评分卡减GBDT |', '|---|---|---|---|']
    for key, label in [('ks', 'KS'), ('pr', 'PR-AUC（AP）'), ('roc', 'ROC-AUC'), ('ece', 'ECE（15个等宽概率箱）')]:
        L.append(f"| {label} | {sc[key]:.4f} | {gb[key]:.4f} | {sc[key]-gb[key]:+.4f} |")
    L += [f"| 标签实现的模型化总损失 | ${sc['loss']:,.0f} | ${gb['loss']:,.0f} | {(sc['loss']/gb['loss']-1):+.1%} |",
          f"| 拦截笔数 | {sc['nblock']:,} | {gb['nblock']:,} | {sc['nblock']-gb['nblock']:+,} |", '',
          '两臂均使用未经另行校准的概率；ECE仅反映本测试窗所选分箱下的平均偏差，不证明所有工作点可靠。',
          '成本参数c_FP=$25为假设，逐笔阈值 t_i=25/(金额+25)，p>t_i时拦截，等于时放行。',
          '这是二动作（放行/拦截）对照，不是现有四动作/五动作策略收益。',
          '表中损失用真标签计算：误拦好样本×25 + 放过欺诈的金额；不是模型预测的期望损失或真实业务损失。',
          '阈值公式不在测试标签上扫描，但该时段此前已被研究使用，仍是回顾性对照。\n',
          '## 2. 特征与分箱约束\n',
          f'候选 {len(feats)} 列，最终保留 {len(keep)} 列。IV<0.02剔除，WOE相关绝对值>0.80保留IV较高者。',
          '普通数值箱至少占训练全窗5%，按WOE合箱至单调；缺失无自然顺序，独立成箱且豁免5%约束。',
          '类别变量不强加字母顺序单调，低频类别合为其他；训练未见箱采用中性WOE=0并在评分表列出。',
          'WOE=ln(坏样本分布/好样本分布)，零计数用0.5平滑；本对照剔除负LR系数以统一分值方向。',
          '单调与符号约束是本项目的选择，不是所有银行模型一概适用的硬规定。\n',
          '| 变量 | IV | 已观测箱数 | 普通箱WOE单调 |', '|---|---|---|---|']
    for c in sorted(keep, key=lambda c: -ivs[c]):
        mono = '是' if specs[c].get('monotonic') else '不适用（类别）'
        L.append(f'| `{c}` | {ivs[c]:.4f} | {len(woes[c])} | {mono} |')
    L += ['', '### 剔除与警示\n',
          '- `card1`为匿名卡相关字段，无法核实其业务编码含义，本对照不赋予其数值顺序解释；不称其唯一卡号。']
    for c, reason in dropped_iv + dropped_corr + dropped_sign:
        L.append(f'- `{c}`：{reason}。')
    for c, iv in high_iv:
        L.append(f'- 高IV警示 `{c}`={iv:.4f}；' + ('保留' if c in keep else '已剔除') + '。')
    L += ['高IV不是泄漏判决。匿名V/C构造不可核实；自建图标签延迟审计不能替匿名列背书。',
          '这也限制了业务解释：可拆解分值不等于能用自然语言解释匿名变量的真实含义。', '',
          '### 小样本缺失箱（豁免但不可忽略）\n']
    sparse = [(c, w[-1]) for c, w in woes.items() if c in keep and -1 in w and w[-1]['n'] < MIN_BIN_FRAC*n_fit]
    if sparse:
        for c, v in sparse:
            L.append(f"- `{c}`：{v['n']:,}笔，占fit {v['n']/n_fit:.4%}，坏样本{v['bad']}笔；其WOE及分值估计不稳定。")
    else:
        L.append('本次没有低于5%的已观测缺失箱。')
    L += ['', '## 3. 可复算评分卡\n',
          f'基准分{BASE_SCORE}，PDO={PDO}，基准好:坏odds={BASE_ODDS:g}:1。',
          'factor=PDO/ln(2)，offset=600−factor×ln(50)，score=offset−factor×logit(p)。',
          f'基础分 **{base:.6f}**；各箱贡献为 −factor×系数×WOE。总分越高，模型预测风险越低。',
          '当p=1/51时得600分；好:坏odds翻倍加20分。',
          '精确切点、WOE、系数与输入哈希在 `data/processed/scorecard_model.json`；',
          '逐笔概率与未舍入总分在 `data/processed/scorecard_test_scores.parquet`。下表为显示而舍入，实际计算使用精确产物。\n',
          '| 变量 | 分箱 | 样本 | 坏率 | WOE | 分值 |', '|---|---|---|---|---|---|']
    for _, r in card.iterrows():
        rate = f"{r['坏率']:.2%}" if r['样本'] else '未见：中性回退'
        L.append(f"| `{r['变量']}` | {r['分箱']} | {int(r['样本']):,} | {rate} | {r['WOE']:+.4f} | {r['分值']:+.4f} |")
    L += ['', '## 4. 结论及证据边界\n',
          f"本窗评分卡KS比GBDT低 {gb['ks']-sc['ks']:.4f}，模型化损失高 {(sc['loss']/gb['loss']-1):.1%}。",
          '换来的是固定加法结构、精确可拆解的分值表；没有实验支持“评分卡跨时间更稳定”。',
          '未调参（LR C=1、max_iter=1000），没有强评分卡基线或高IV变量消融，不代表评分卡类模型的性能上限。',
          'LR不使用验证窗；GBDT用同一fit窗训练并用val选早停轮次，两臂模型选择预算并不完全相同。',
          '本实验支持保留GBDT作为该反欺诈任务的主模型。授信模型选择还需其自身数据、解释性审查与验证，不能由这份交易实验决定。',
          f"\nGBDT缓存SHA-256：`{hashlib.sha256(SCORES.read_bytes()).hexdigest()}`。"]
    write_report(REPORT, '\n'.join(L))


if __name__ == '__main__':
    main()
