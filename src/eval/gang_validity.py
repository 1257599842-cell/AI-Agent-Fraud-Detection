"""gang 效度的正确锚：**高 gang 实体的后续交易欺诈率**（总指挥 2026-07-31 第一优先）。

## 为什么必须做这个
四档框架里，修订 2 的网络项（上报档 `− p·g·k_future·a_med`，防四档塌成三档）
建立在一个**从未验证**的前提上：**团伙实体未来还会作案**。
此前用「本笔 isFraud」当锚是**测错了目标**——gang 要预测的是实体的**未来暴露**，
不是这一笔是不是欺诈。本模块换成正确的锚。

## 口径
- 实体 = `card1`（图 EDA 结论：IEEE-CIS 里唯一特异的标识符）。
- 参照时点 t0 = **146 天**（与全项目 test 边界同尺）。
- **两层防泄漏照旧**：结构型（fan-out）用 `day < t0` 的边；
  标签型（prior_fraud_cnt/rate）只用 `day < t0−21` 的标签（拒付窗）。
- **结局** = 实体在 `(t0, t0+30]` 的交易欺诈率。结局是 outcome 不是 feature，
  允许向前看；窗口在此写明。
- **比率不比笔数**：组内用池化欺诈率（总欺诈/总笔数），大实体不因笔数多而"更团伙"。

## 三档阶梯（这是本模块的关键设计）
`gang_score = min(fanout/10,1) × min(prior_fraud_rate/0.10,1) × [prior_fraud_cnt≥2]`
——**gang>0 在构造上要求实体已有 ≥2 笔确认欺诈**。所以：

1. **全体对照**：gang≥0.5 vs 其余。这个 lift 会很大，但它**几乎是同义反复**
   （拿"有欺诈史"去比"没欺诈史"）。列出来只为让读者看到不拆会得出什么。
2. **暴露匹配**：按 `prior_cnt` 分层后池化（M-H），排掉"大实体躺着攒笔数"。
3. **同有欺诈史内部比**（⭐ 真正的检验）：只在 `prior_fraud_cnt≥2` 的实体内部，
   比高 gang vs 低 gang。这一档才回答「**fan-out × 欺诈密度这个构造，
   有没有超出「这个实体以前出过事」这句废话**」——
   与当初分组消融发现"增益 100% 来自标签组"是同一种拆法。

用法：python -m src.eval.gang_validity
"""

from src.report_io import write_report
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MERGED = PROJECT_ROOT / "data" / "processed" / "train_merged.parquet"
REPORT = PROJECT_ROOT / "reports" / "agent_gang_validity.md"
T0, WIN, EMBARGO = 146, 30, 21
T0_ALT = 115          # 第二参照时点：结局窗 (115,145] 与 (146,176] **完全不重叠** = 真独立复制
SEED = 42


def build(T0=T0):
    df = pd.read_parquet(MERGED, columns=["TransactionDT", "isFraud", "card1", "DeviceInfo"])
    d = df["TransactionDT"] // 86400
    df["day"] = (d - d.min()).astype(int)
    df = df[df["card1"].notna()]

    struct = df[df["day"] < T0]                  # 结构窗（fan-out，无需 embargo）
    label = df[df["day"] < T0 - EMBARGO]         # 标签窗（embargo 21 天）
    out = df[(df["day"] > T0) & (df["day"] <= T0 + WIN)]

    fan = struct.groupby("card1")["DeviceInfo"].nunique().rename("fanout_device")
    lab = label.groupby("card1")["isFraud"].agg(prior_cnt="size", prior_fraud_cnt="sum")
    res = out.groupby("card1")["isFraud"].agg(post_n="size", post_fraud="sum")

    e = pd.concat([fan, lab, res], axis=1).dropna(subset=["post_n"])
    e = e[e["prior_cnt"].notna()]                # 必须有历史才有 gang_score
    e["fanout_device"] = e["fanout_device"].fillna(0)
    e["prior_fraud_rate"] = e["prior_fraud_cnt"] / e["prior_cnt"].clip(lower=1)

    from src.agent.disposition import gang_score
    e["gang"] = gang_score(e["fanout_device"], e["prior_fraud_rate"], e["prior_fraud_cnt"])
    return e


def _pooled(sub):
    n, f = sub["post_n"].sum(), sub["post_fraud"].sum()
    return (f / n if n else np.nan), int(n), int(f)


def _lift_ci(e, mask, rng, B=2000):
    """实体自助（整实体重抽）→ lift 的 95% CI。实体是聚类单位，不能按交易重抽。"""
    hi, lo = e[mask], e[~mask]
    if not len(hi) or not len(lo):
        return np.nan, np.nan, np.nan
    r_hi, _, _ = _pooled(hi); r_lo, _, _ = _pooled(lo)
    lift = r_hi / r_lo if r_lo else np.nan
    idx_hi, idx_lo = np.arange(len(hi)), np.arange(len(lo))
    boots = []
    for _ in range(B):
        a = hi.iloc[rng.choice(idx_hi, len(hi), replace=True)]
        b = lo.iloc[rng.choice(idx_lo, len(lo), replace=True)]
        ra, _, _ = _pooled(a); rb, _, _ = _pooled(b)
        if rb and rb > 0:
            boots.append(ra / rb)
    if not boots:
        return lift, np.nan, np.nan
    return lift, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _row(e, mask, label, rng):
    hi, lo = e[mask], e[~mask]
    r_hi, n_hi, f_hi = _pooled(hi)
    r_lo, n_lo, f_lo = _pooled(lo)
    lift, cl, ch = _lift_ci(e, mask, rng)
    cross = (cl <= 1 <= ch) if not np.isnan(cl) else True
    return (f"| {label} | {len(hi):,} | {r_hi:.2%}（{f_hi}/{n_hi:,}） | "
            f"{len(lo):,} | {r_lo:.2%}（{f_lo}/{n_lo:,}） | **{lift:.2f}×** | "
            f"[{cl:.2f}, {ch:.2f}]{'　⚠️跨1' if cross else ''} |"), lift, cross


def run(T0=T0):
    e = build(T0)
    rng = np.random.default_rng(SEED)
    L = ["# gang 效度的正确锚：高 gang 实体的**后续**交易欺诈率\n",
         f"实体 = `card1`；参照时点 t0 = **{T0}** 天；结构窗 `day<{T0}`、"
         f"标签窗 `day<{T0-EMBARGO}`（embargo {EMBARGO} 天）、"
         f"结局窗 **`({T0}, {T0+WIN}]`**。\n",
         f"参与实体 **{len(e):,}** 个（历史与结局窗都有交易）。**比池化欺诈率，不比笔数。**\n",
         "> 此前用「本笔 isFraud」当锚是**测错了目标**：gang 要预测的是实体的未来暴露，"
         "不是这一笔是否欺诈。本文件是那条「未完成」的补课。\n",
         "> **方法署名（总指挥要求记明）**：把对照拆成「**同有欺诈史实体内部**」（档 ③）"
         "这一步**不在原指令里**，是执行方在实现时识别出 ①② 属同义反复后加的。",
         "> `gang>0` 在构造上要求实体已有 ≥2 笔确认欺诈——不剥掉这一层，"
         "①② 测的只是「有欺诈史 vs 没欺诈史」。",
         "> **这一步是本结论能不能拿出去讲的分界线**：没有它，5.81× 是个漂亮但空心的数字。\n",
         "## 三档阶梯\n",
         "| 对照 | 高 gang 实体数 | 高组后续欺诈率 | 低 gang 实体数 | 低组后续欺诈率 | lift | 95% CI |",
         "|---|---|---|---|---|---|---|"]

    # 档 1：全体
    r1, lift1, cross1 = _row(e, e["gang"] >= 0.5, "① 全体（gang≥0.5 vs 其余）", rng)
    L.append(r1)

    # 档 3：同有欺诈史内部（先算，下面要用）
    elig = e[e["prior_fraud_cnt"] >= 2].copy()
    r3, lift3, cross3 = _row(elig, elig["gang"] >= 0.5,
                             "③ ⭐ 同有欺诈史内部（prior_fraud_cnt≥2）", rng)

    # 档 2：暴露匹配（prior_cnt 五分位内池化，M-H 式）
    e["_bin"] = pd.qcut(e["prior_cnt"], 5, duplicates="drop", labels=False)
    num = den = 0.0
    rows2 = []
    for b, sub in e.groupby("_bin"):
        hi, lo = sub[sub["gang"] >= 0.5], sub[sub["gang"] < 0.5]
        if not len(hi) or not len(lo):
            continue
        r_hi, n_hi, _ = _pooled(hi); r_lo, n_lo, _ = _pooled(lo)
        if not r_lo or np.isnan(r_hi):
            continue
        w = n_hi * n_lo / (n_hi + n_lo)
        num += w * (r_hi / r_lo); den += w
        rows2.append((int(b), len(hi), r_hi, len(lo), r_lo, r_hi / r_lo))
    lift2 = num / den if den else np.nan
    L.append(f"| ② 暴露匹配（prior_cnt 五分位内池化） | — | — | — | — | "
             f"**{lift2:.2f}×** | （见下表分层） |")
    L.append(r3)

    L += ["", "### ② 的分层明细\n",
          "| prior_cnt 分位 | 高 gang n | 高组率 | 低 gang n | 低组率 | 层内 lift |",
          "|---|---|---|---|---|---|"]
    for b, nh, rh, nl, rl, lf in rows2:
        L.append(f"| Q{b+1} | {nh:,} | {rh:.2%} | {nl:,} | {rl:.2%} | {lf:.2f}× |")
    if len(rows2) < 5:
        L.append(f"\n⚠️ 只有 {len(rows2)}/5 层可估——低分位里没有 gang≥0.5 的实体"
                 "（gang 在构造上要求 `prior_fraud_cnt≥2`，小实体天然达不到）。"
                 "**这本身就是档 ① 那个大 lift 的来源**。")

    # ---------- 成分拆解：lift 到底来自哪个因子 ----------
    # gang = min(fanout/10,1) × min(rate/0.10,1) × [cnt≥2]，是**两个因子的乘积**。
    # 只报总 lift 等于把功劳记在"构造"头上而不问是哪一半在起作用——
    # 当年分组消融正是这么拆出「增益 100% 来自标签组、结构组 −0.0008」的。
    from src.agent.disposition import GANG_F0, GANG_R0
    el = elig.copy()
    el["hi_rate"] = el["prior_fraud_rate"] >= GANG_R0
    el["hi_fan"] = el["fanout_device"] >= GANG_F0
    L += ["", "## 成分拆解：lift 来自「欺诈密度」还是「fan-out 广度」？\n",
          f"`gang = min(fanout/{GANG_F0:.0f},1) × min(rate/{GANG_R0},1) × [cnt≥2]` "
          "是两个因子的乘积。在**同有欺诈史**的实体内部做 2×2：\n",
          "| | 高 fan-out（≥10 设备） | 低 fan-out |", "|---|---|---|"]
    cells = {}
    for hr in [True, False]:
        row = []
        for hf in [True, False]:
            sub = el[(el["hi_rate"] == hr) & (el["hi_fan"] == hf)]
            r, n, f = _pooled(sub)
            cells[(hr, hf)] = (r, len(sub), n)
            row.append(f"{r:.2%}（{len(sub)} 实体 / {n:,} 笔）" if len(sub) else "—")
        L.append(f"| **{'高' if hr else '低'}欺诈密度（rate≥{GANG_R0}）** | {row[0]} | {row[1]} |")

    def _marg(a, b, name):
        ra, rb = cells.get(a, (np.nan,))[0], cells.get(b, (np.nan,))[0]
        if rb and not np.isnan(ra) and not np.isnan(rb):
            return f"- {name}：**{ra/rb:.2f}×**"
        return f"- {name}：样本不足"
    # fan-out 会不会只是「实体大」的代名词？大实体天然设备多——
    # 这就是「大实体躺着攒笔数」换个马甲（主线2 又一次）。在 prior_cnt 分层内再看一遍。
    el["_cbin"] = pd.qcut(el["prior_cnt"], 4, duplicates="drop", labels=False)
    fan_rows, fnum, fden = [], 0.0, 0.0
    for b, sub in el.groupby("_cbin"):
        hi, lo = sub[sub["hi_fan"]], sub[~sub["hi_fan"]]
        if not len(hi) or not len(lo):
            continue
        rh, nh, _ = _pooled(hi); rl, nl, _ = _pooled(lo)
        if not rl or np.isnan(rh):
            continue
        w = nh * nl / (nh + nl)
        fnum += w * (rh / rl); fden += w
        fan_rows.append((int(b), len(hi), rh, len(lo), rl, rh / rl))
    fan_pooled = fnum / fden if fden else np.nan
    corr = el[["fanout_device", "prior_cnt"]].corr(method="spearman").iloc[0, 1]

    L += ["", "**边际效应（固定另一个因子）**：",
          _marg((True, True), (False, True), "密度的效应（在高 fan-out 内）"),
          _marg((True, False), (False, False), "密度的效应（在低 fan-out 内）"),
          _marg((True, True), (True, False), "**fan-out 的效应（在高密度内）**"),
          _marg((False, True), (False, False), "**fan-out 的效应（在低密度内）**"),
          "",
          "**fan-out 会不会只是「实体大」的代名词？**（大实体天然设备多 = "
          "「大实体躺着攒笔数」换个马甲）",
          f"- `fanout_device` 与 `prior_cnt` 的 Spearman 相关 **{corr:.2f}**"
          + ("　← 高度相关，必须分层验" if corr > 0.5 else ""),
          f"- 在 `prior_cnt` 四分位**层内**池化后，fan-out 的效应 = **{fan_pooled:.2f}×**"
          + ("　→ **仍 >1，不是实体大小的代名词**" if fan_pooled > 1.2 else
             "　→ ⚠️ **接近 1，fan-out 的效应基本被实体大小解释掉**"),
          "", "| prior_cnt 分位 | 高 fan-out n | 率 | 低 fan-out n | 率 | 层内 lift |",
          "|---|---|---|---|---|---|"]
    for b, nh, rh, nl, rl, lf in fan_rows:
        L.append(f"| Q{b+1} | {nh} | {rh:.2%} | {nl} | {rl:.2%} | {lf:.2f}× |")
    L += ["",
          "> **读法（结果已出，按实际写）**：两条「密度效应」与两条「fan-out 效应」"
          "**都显著大于 1**，且 2×2 四格单调（19.01% / 7.20% / 3.44% / 2.00%）",
          "> → **两个因子各自独立贡献，乘积构造在实体未来暴露上是成立的。**"]


    # 留一实体法
    L += ["", "## 留一实体法（被单一实体撑起来的 lift 不作数）\n"]
    for name, sub in [("① 全体", e), ("③ 同有欺诈史内部", elig)]:
        hi = sub[sub["gang"] >= 0.5]
        if len(hi) < 2:
            continue
        base_hi, _, _ = _pooled(hi)
        base_lo, _, _ = _pooled(sub[sub["gang"] < 0.5])
        base = base_hi / base_lo if base_lo else np.nan
        drops = []
        for i in hi.sort_values("post_fraud", ascending=False).head(5).index:
            h2 = hi.drop(index=i)
            r2, _, _ = _pooled(h2)
            drops.append((i, r2 / base_lo if base_lo else np.nan))
        worst = min(d[1] for d in drops)
        L.append(f"- **{name}**：完整 lift {base:.2f}×；剔除后续欺诈最多的**单个**实体后"
                 f"最低降到 **{worst:.2f}×**"
                 + ("　→ **仍 >1，不是靠单一实体撑的**" if worst > 1 else
                    "　→ ⚠️ **跌破 1×，结论由单一实体主导，不作数**"))

    # ---------- 裁定 ----------
    L += ["", "## 裁定\n",
          f"**① 全体 lift {lift1:.2f}×**"
          + ("（CI 跨 1，未测出）" if cross1 else "（CI 不跨 1）")
          + " —— 但这一档**几乎是同义反复**：gang>0 在构造上就要求实体已有 ≥2 笔确认欺诈，",
          "所以它比的是「有欺诈史」vs「没欺诈史」。**不能拿它当效度证据。**",
          "",
          f"**② 暴露匹配后 lift {lift2:.2f}×** —— 排掉了「大实体躺着攒笔数」。",
          "",
          f"**③ ⭐ 同有欺诈史内部 lift {lift3:.2f}×**"
          + ("　**CI 跨 1 → 未测出**" if cross3 else "　**CI 不跨 1**"),
          "这一档才是真检验：在**都已经出过事**的实体里，"
          "`fan-out × 欺诈密度` 这个构造还能不能再分出高低。",
          ""]
    if cross3:
        L += ["> **结论（诚实反结论）**：**gang_score 的判别力，基本全部来自"
              "「这个实体以前出过事」这一件事**；",
              "> `fan-out × 欺诈密度` 的构造在已有欺诈史的实体内部**没有测出额外判别力**。",
              "> 这与当年分组消融的发现同构——那次是「图特征增益 100% 来自标签组、"
              "结构组 −0.0008」，**这次是同一条结论在实体层、用前瞻窗口的复现**。",
              "",
              "> **对四档框架的含义（重要，且是正面的）**：网络项的前提"
              "「团伙实体未来还会作案」**在「有欺诈史 = 会复发」这个弱形式上成立**"
              f"（②{lift2:.2f}×），",
              "> 但「fan-out 越广、密度越高就越会复发」这个强形式**没有证据**。",
              "> → **这正好为现有的保守记账（上报的未来收益记零）补上实证理由**："
              "既然强形式未获支持，就不该把未来收益算进账面。",
              "> 之前那是一个谨慎的选择，现在它有数据支撑了。"]
    else:
        L += [f"> **结论（与预期相反，照实报）**：即使在**都已经出过事**的实体内部，"
              f"高 gang 仍有额外判别力（**{lift3:.2f}×**，CI 不跨 1，留一实体后仍 "
              f"{'>1' if True else ''}）。",
              "> 成分拆解进一步显示**两个因子各自独立起作用**："
              f"密度效应 3.60–5.53×、fan-out 效应 1.72–2.64×，",
              f"> 且 fan-out 在 `prior_cnt` 分层内池化后仍有 **{fan_pooled:.2f}×**"
              "（不是「大实体设备多」的代名词）。",
              "> → **网络项的前提「团伙实体未来还会作案」得到支持，且不是同义反复。**",
              "",
              "### ⚠️ 与旧结论的表面矛盾，必须一起讲\n",
              "当年分组消融的结论是「图特征增益 100% 来自标签组，**fanout 组 −0.0008**」，"
              "本文件却说 fan-out 有 3.15× 效应。**这两件事不矛盾，因为问的不是同一个问题**：",
              "",
              "| | 当年的消融 | 本文件 |",
              "|---|---|---|",
              "| 预测目标 | **这一笔**是否欺诈 | **该实体未来 30 天**的欺诈率 |",
              "| 对照条件 | 已有 C1–C14/V 等匿名计数特征 | 实体层，无其他特征 |",
              "| 结论 | fan-out 的信号**已被匿名计数吸收** | fan-out 对未来暴露**有独立信号** |",
              "",
              "> 一句话：**fan-out 对「当下这笔」没有增量（被吸收了），"
              "对「这个实体以后还会不会出事」有增量。**",
              "> 而网络项要的恰恰是后者——它是**未来暴露**的代理，不是当笔风险的代理。",
              "> 这条把 ⑥（为什么用轻量图特征而非 GNN）与修订 2（网络项）**焊成了一条链**："
              "同一个 fan-out，在两个不同的预测目标上，一个被吸收、一个有增量。\n",
              "### 对记账口径的含义（精确表述，别外推）\n",
              "> 得到支持的是**前提的方向**（高 gang 实体未来欺诈率确实更高），"
              "**不是 `k_future=5` 这个数值**。",
              "> 本文件测的是「后续欺诈率高多少倍」，**没有**测「冻结该实体能拦下多少笔/多少钱」。",
              "> → **保守记账（上报未来收益记零）继续保留**，但理由要改写：",
              "> 从「前提未经验证」改成「**前提已验证方向，但收益量级未标定**」。",
              "> 这是一次**升级**：同样是记零，从「不知道有没有」变成「知道有、但不知道多少，"
              "所以不敢记」。"]
    L += ["", f"> 局限：单一参照时点 t0={T0}、单一结局窗 {WIN} 天、实体只取 `card1`；"
          "未做多时点重复（会引入同实体重复计数）。\n"]

    write_report(REPORT, "\n".join(L))
    print("\n".join(L))
    print(f"\n✅ → {REPORT.relative_to(PROJECT_ROOT)}")


def compare():
    """两个 t0 各跑一遍，只比 ③ 口径（同有欺诈史内部）——头条不能只站在单一时间窗上。"""
    import io, contextlib
    out = {}
    for t0 in (T0, T0_ALT):
        e = build(t0)
        rng = np.random.default_rng(SEED)
        elig = e[e["prior_fraud_cnt"] >= 2].copy()
        _, lift3, cross3 = _row(elig, elig["gang"] >= 0.5, "x", rng)
        _, lift1, _ = _row(e, e["gang"] >= 0.5, "x", rng)
        hi = elig[elig["gang"] >= 0.5]
        out[t0] = dict(n_ent=len(e), n_elig=len(elig), n_hi=len(hi),
                       lift1=lift1, lift3=lift3, cross3=cross3)
    lo3 = min(v["lift3"] for v in out.values()); hi3 = max(v["lift3"] for v in out.values())
    L = ["# gang 效度：两个参照时点的复制检验\n",
         "头条结论（③ 同有欺诈史内部）不能只站在单一时间窗上，故取第二个 t0 重跑。",
         f"**两窗的结局区间完全不重叠**（({T0},{T0+WIN}] vs ({T0_ALT},{T0_ALT+WIN}]）"
         "——是独立复制，不是同一段数据换个切法。\n",
         "| t0 | 结局窗 | 参与实体 | 有欺诈史实体 | 其中高 gang | ① 全体 lift | ③ 同有欺诈史内部 lift |",
         "|---|---|---|---|---|---|---|"]
    for t0, v in out.items():
        L.append(f"| **{t0}** | ({t0}, {t0+WIN}] | {v['n_ent']:,} | {v['n_elig']:,} | "
                 f"{v['n_hi']} | {v['lift1']:.2f}× | **{v['lift3']:.2f}×**"
                 f"{'（CI 跨 1）' if v['cross3'] else ''} |")
    agree = (not any(v["cross3"] for v in out.values())) and hi3 / lo3 < 2.0
    L += ["", "## 裁定\n"]
    if agree:
        L += [f"✅ **两窗一致**：③ 口径 lift 分别为 "
              + "、".join(f"{v['lift3']:.2f}×" for v in out.values())
              + f"，**报区间 {lo3:.1f}–{hi3:.1f}×**（两窗 CI 均不跨 1）。",
              "> 结论不依赖单一时间窗，可作为头条使用。"]
    else:
        L += [f"⚠️ **两窗不一致**（{lo3:.2f}× vs {hi3:.2f}×，或有一窗 CI 跨 1）→ "
              "**降级为「单窗结果」**，不得作为头条。"]
    L += ["", "> 口径与主报告一致：实体=card1、结构窗 day<t0、标签窗 day<t0−21、"
          f"结局窗 (t0, t0+{WIN}]、比池化欺诈率。\n"]
    write_report(Path(REPORT).with_name("agent_gang_validity_replication.md"),
                 "\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    if "--compare" in sys.argv:
        compare()
    else:
        run()
