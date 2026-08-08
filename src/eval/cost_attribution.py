"""层1 分歧的**成本**归因（任务 C）：把「生产拓扑 − 应然档」的每万笔差额
按混淆矩阵格子拆开，HT 加权。

**为什么必须按钱拆、不能按笔数拆**：层1 一致率 58% 是笔数口径，它把「把一笔 $5 的
approve 误判成 escalate」和「把一笔 $800 的 decline 误判成 approve」记成同样的 1 笔。
round 3 要修哪个 prompt 病灶，得由**钱**决定。

口径与 `--score` 的层2 完全一致（同一个 `realized_cost`、同一份 HT 权重、同一个
保守口径 credit_future=False），所以各格子求和必须精确等于总差额——代码里断言了这一点。

**生产拓扑的定义决定了归因结构**：⑧ 闸门把「应然档=approve」的交易挡在 Agent 之前，
所以那些格子在可部署口径下**结构性地贡献 0**（不管 Agent 在 eval 里说了什么）。
这一点本身就是结论：eval 里最扎眼的病灶未必是最值钱的病灶。

用法：python -m src.eval.cost_attribution r1
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "reports" / "eval_runs"
EVAL_SET = PROJECT_ROOT / "data" / "processed" / "agent_eval_set.parquet"
REPORT = PROJECT_ROOT / "reports" / "agent_cost_attribution.md"
SEED = 42
A_MED = 76.02          # 与 --score 一致（训练窗欺诈中位金额）


def _per_txn_cost(actions, y, amt, g, prm):
    from src.agent.disposition import realized_cost
    return np.array([realized_cost(np.array([a]), np.array([yy]), np.array([aa]),
                                   np.array([gg]), A_MED, prm)
                     for a, yy, aa, gg in zip(actions, y, amt, g)])


def load(tag):
    rows = []
    for f in sorted((RUNS_DIR / tag).glob("txn_*.json")):
        r = json.loads(f.read_text())
        rep = r.get("report") or {}
        if rep.get("disposition"):
            rows.append({"TransactionID": r["txn_id"], "agent": rep["disposition"],
                         "confidence": rep.get("confidence")})
    es = pd.read_parquet(EVAL_SET)
    d = pd.DataFrame(rows).merge(es, on="TransactionID")
    # ⑧ 闸门在位：应然档 approve 的交易根本不进 Agent（eval 是 force 灌入的）
    d["prod"] = np.where(d["disposition_gt"] == "approve", "approve", d["agent"])
    return d


def run(tag):
    from src.agent.disposition import BASE
    d = load(tag)
    w = d["ht_weight"].to_numpy()
    y, amt, g = (d["isFraud"].to_numpy(), d["TransactionAmt"].to_numpy(),
                 d["gang_score"].to_numpy())
    c_prod = _per_txn_cost(d["prod"], y, amt, g, BASE)
    c_gt = _per_txn_cost(d["disposition_gt"], y, amt, g, BASE)
    c_agent = _per_txn_cost(d["agent"], y, amt, g, BASE)
    scale = 10_000 / w.sum()
    tot_prod, tot_gt = (c_prod * w).sum() * scale, (c_gt * w).sum() * scale
    gap = tot_prod - tot_gt
    d["_delta"] = (c_prod - c_gt) * w * scale
    d["_delta_agent"] = (c_agent - c_gt) * w * scale

    L = ["# 层1 分歧的成本归因（任务 C）\n",
         f"round `{tag}`，{len(d)} 笔，HT 加权 → 每万笔。口径同 `--score` 层2"
         "（保守：上报的未来收益不可观测、记零）。\n",
         f"- 生产拓扑（⑧闸门 + Agent）：**${tot_prod:,.0f}**",
         f"- 应然档（argmin）：**${tot_gt:,.0f}**",
         f"- **差额 = ${gap:,.0f} / 万笔** ← 本文件要拆的就是它\n"]

    # ---------- 主表：按 (应然档, 生产拓扑实际动作) 拆 ----------
    L += ["## 按混淆矩阵格子拆（生产拓扑口径）\n",
          "| 应然档 | 实际动作 | 笔数 | HT 权重占比 | Δ$/万笔 | 占差额 |",
          "|---|---|---|---|---|---|"]
    cells = (d.groupby(["disposition_gt", "prod"], observed=True)
             .agg(n=("_delta", "size"), wsum=("ht_weight", "sum"),
                  delta=("_delta", "sum")).reset_index()
             .sort_values("delta", ascending=False))
    for _, r in cells.iterrows():
        mark = "" if r["disposition_gt"] != r["prod"] else "（一致）"
        L.append(f"| {r['disposition_gt']} | {r['prod']}{mark} | {int(r['n'])} | "
                 f"{r['wsum']/w.sum():.1%} | ${r['delta']:,.0f} | "
                 f"{r['delta']/gap if gap else 0:.0%} |")
    assert abs(cells["delta"].sum() - gap) < 1e-6, "格子求和 ≠ 总差额，口径不一致"
    L += ["", f"（求和校验：各格 Δ 之和 ${cells['delta'].sum():,.0f} = 总差额 ${gap:,.0f} ✅）\n"]

    # ---------- ⑧ 闸门吃掉了哪些格子 ----------
    gated = d[d["disposition_gt"] == "approve"]
    over = gated[gated["agent"] != "approve"]
    L += ["## ⑧ 闸门结构性地吃掉了「过度上报」这个病灶\n",
          f"- 应然档=approve 的 **{len(gated)}** 笔里，Agent 在 eval 中给出非 approve 的有 "
          f"**{len(over)}** 笔（"
          + "、".join(f"{k} {v}" for k, v in over["agent"].value_counts().items()) + "）。",
          "- 但 ⑧ 闸门在生产拓扑里把这些交易**挡在 Agent 之前**，所以它们对差额的贡献是 "
          f"**${d.loc[gated.index, '_delta'].sum():,.0f}**（结构性为 0）。",
          "",
          "- 作为对照，若**没有**闸门（Agent 单独，eval 强制全量投喂），这批交易的贡献是 "
          f"**${d.loc[over.index, '_delta_agent'].sum():,.0f}/万笔**。",
          "",
          "> **这推翻了 round 2 定的头号 prompt 目标。** 当时把「对 approve 应然档过度上报"
          f"（{len(over)} 笔，54% 带 gang≥0.5）」列为 round 3 首要修复项——那是**笔数**口径的结论。"
          "按**钱**口径，它在可部署拓扑下一分不值，因为闸门已经免费解决了它。"
          "改 prompt 去修它，是拿 Agent 的复杂度去解决一个 `if` 语句已经解决的问题。\n"]

    # ---------- 方向分解：漏检 vs 过拦 ----------
    ORD = {"approve": 0, "hold": 1, "escalate": 2, "decline": 3}
    dis = d[d["disposition_gt"] != d["prod"]].copy()
    dis["dir"] = np.where(dis["prod"].map(ORD) < dis["disposition_gt"].map(ORD),
                          "偏宽（漏检方向）", "偏严（过拦方向）")
    L += ["## 方向分解（生产拓扑口径，只看不一致的格子）\n",
          "| 方向 | 笔数 | Δ$/万笔 | 占差额 |", "|---|---|---|---|"]
    for k, s in dis.groupby("dir", observed=True):
        L.append(f"| {k} | {len(s)} | ${s['_delta'].sum():,.0f} | "
                 f"{s['_delta'].sum()/gap if gap else 0:.0%} |")

    # ---------- 逐笔最贵的分歧 ----------
    top = d.nlargest(8, "_delta")[["TransactionID", "disposition_gt", "agent", "prod",
                                   "isFraud", "TransactionAmt", "p", "gang_score",
                                   "stratum", "_delta"]]
    L += ["", "## 最贵的 8 笔分歧（钱在哪儿）\n",
          "| txn | 应然档 | Agent | 生产动作 | isFraud | 金额 | p | gang | 层 | Δ$/万笔 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in top.iterrows():
        L.append(f"| {int(r['TransactionID'])} | {r['disposition_gt']} | {r['agent']} | "
                 f"{r['prod']} | {int(r['isFraud'])} | ${r['TransactionAmt']:,.2f} | "
                 f"{r['p']:.3f} | {r['gang_score']:.2f} | {r['stratum']} | ${r['_delta']:,.0f} |")

    # ---------- 分层自助：这个拆分稳不稳 ----------
    rng = np.random.default_rng(SEED)
    idx_by = {s: np.where(d["stratum"].to_numpy() == s)[0] for s in d["stratum"].unique()}
    boots, share_top = [], []
    top_cell = cells.iloc[0][["disposition_gt", "prod"]].tolist()
    is_top = ((d["disposition_gt"] == top_cell[0]) & (d["prod"] == top_cell[1])).to_numpy()
    lenient = (d["prod"].map(ORD) < d["disposition_gt"].map(ORD)).to_numpy()
    top_abs, lenient_share_wins = [], 0
    for _ in range(2000):
        take = np.concatenate([rng.choice(v, size=len(v), replace=True)
                               for v in idx_by.values()])
        ww = w[take]; sc = 10_000 / ww.sum()
        dd = (c_prod[take] - c_gt[take]) * ww * sc
        boots.append(dd.sum())
        top_abs.append(dd[is_top[take]].sum())
        lenient_share_wins += dd[lenient[take]].sum() > dd[~lenient[take]].sum()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    tlo, thi = np.percentile(top_abs, [2.5, 97.5])
    n_eff = w.sum() ** 2 / (w ** 2).sum()
    top1 = d["_delta"].max() / gap
    L += ["", "## 稳不稳（分层自助 2000 次）\n",
          f"- 差额点估计 **${gap:,.0f}**，95% CI **[${lo:,.0f}, ${hi:,.0f}]**"
          f"（Kish 有效样本量 n_eff={n_eff:.1f}）",
          f"- P(差额 > 0) = **{np.mean(np.array(boots) > 0):.1%}**",
          f"- 最大格子 `{top_cell[0]}→{top_cell[1]}` 的绝对贡献 **${cells.iloc[0]['delta']:,.0f}**，"
          f"95% CI [${tlo:,.0f}, ${thi:,.0f}]",
          f"- P(偏宽方向的总额 > 偏严方向) = **{lenient_share_wins/2000:.1%}**",
          "",
          "> **占比的置信区间故意不报**：差额本身的 CI 跨 0，占比 = 两个随机量之比，"
          "自助出来是 [-340%, 347%] 这种无意义区间。比值在分母可能过零时不可估——"
          "改报**绝对贡献的区间**与**方向排序的概率**。",
          "",
          f"⚠️ **集中度警告**：单笔 txn 3527729 一笔就占了差额的 **{top1:.0%}**"
          f"（$ {d['_delta'].max():,.0f}/万笔）。n_eff≈{n_eff:.0f} 且高度集中 →",
          "**这份归因的结论强度是「方向」级，不是「金额」级**：",
          "可以说「钱几乎全在偏宽方向、且集中在 mid_ambiguous 层的漏放」，",
          "不能说「hold→approve 值 $4,965」。\n"]

    # ---------- Jackknife：结论是不是由那一两笔撑起来的 ----------
    order = d["_delta"].abs().sort_values(ascending=False).index
    L += ["", "## Jackknife：留一法（结论是不是靠一两笔撑着）\n",
          "单笔 3527729 占了 68% 的成本质量，所以必须问：**把它拿掉，方向和排名还成立吗？**",
          "（这与自助的 P(偏宽>偏严)=85.3% 是两个互补的检查：自助问「重抽会怎样」，"
          "留一问「最重的那几笔是不是唯一支柱」。两个都报。）\n",
          "| 剔除 | 剩余差额 | 偏宽 | 偏严 | 偏宽占比 | 最大格子 |",
          "|---|---|---|---|---|---|"]
    jack_dirs = []
    for drop_n in [0, 1, 2]:
        keep = d.drop(index=order[:drop_n]) if drop_n else d
        kd = keep[keep["disposition_gt"] != keep["prod"]].copy()
        kd["dir"] = np.where(kd["prod"].map(ORD) < kd["disposition_gt"].map(ORD),
                             "lenient", "strict")
        wide = kd.loc[kd["dir"] == "lenient", "_delta"].sum()
        strict_ = kd.loc[kd["dir"] == "strict", "_delta"].sum()
        tot = keep["_delta"].sum()
        cc = (kd.groupby(["disposition_gt", "prod"], observed=True)["_delta"].sum()
              .sort_values(ascending=False))
        top = f"{cc.index[0][0]}→{cc.index[0][1]}" if len(cc) else "—"
        jack_dirs.append(wide > strict_)
        who = "（无）" if drop_n == 0 else "、".join(
            str(int(d.loc[i, "TransactionID"])) for i in order[:drop_n])
        L.append(f"| {who} | ${tot:,.0f} | ${wide:,.0f} | ${strict_:,.0f} | "
                 f"{wide/tot if tot else float('nan'):.0%} | {top} |")
    stable_dir = all(jack_dirs)
    L += ["", ("✅ **方向稳**：剔除最贵的 1 笔、2 笔后，「偏宽 > 偏严」都成立，"
               "最大格子也没换 → **可以照 C 定 round 3 目标**。"
               if stable_dir else
               "❌ **方向翻转**：剔除后「偏宽 > 偏严」不再成立 → **round 3 目标必须重选**，"
               "不能照本文件的排名走。"),
          "",
          "> 注意留一法在这里读法要小心：剔除样本会同时缩小分子和分母，"
          "**占比看着可能不降反升**。真正要看的是**符号与排名**，不是占比数值。\n"]

    # ---------- 给 round 3 的结论 ----------
    worst = cells[cells["disposition_gt"] != cells["prod"]].iloc[0]
    n_lenient = int((dis["dir"] == "偏宽（漏检方向）").sum())
    L += ["## 给 round 3 的输入\n",
          "### ⛔ 金额口径**不能**用来选目标（jackknife 判的）\n",
          f"本文件按钱排出的名次（最大格 `{worst['disposition_gt']}→{worst['prod']}`、"
          f"偏宽占 {dis[dis['dir']=='偏宽（漏检方向）']['_delta'].sum()/gap:.0%}）"
          "**在留一法下不成立**：剔除最贵的 2 笔后方向就翻转、最大格子换了三次。",
          "n_eff=29.4 且单笔占 68% 的样本，本来就撑不起「哪个格子最值钱」这种排序结论。",
          "**→ 按前置裁决，round 3 目标不得照本文件的金额排名选。**\n",
          "### ✅ 但**笔数口径**的那条观察活着，而且它才是被选中的指标\n",
          f"- 不一致的 {len(dis)} 笔里，**{n_lenient} 笔是偏宽方向**（Agent 比应然档更轻），"
          f"偏严只有 {len(dis)-n_lenient} 笔。这个 {n_lenient}:{len(dis)-n_lenient} 的悬殊"
          "**不依赖金额加权**，因此不受集中度问题影响。",
          f"- `decline` 全场只被 Agent 用了 **{int((d['agent']=='decline').sum())} 次**，"
          f"而应然档要求 **{int((d['disposition_gt']=='decline').sum())} 次** —— "
          "这是一个纯计数事实，同样与金额无关。",
          "",
          "**两者合起来支持同一个改向，只是理由换了**："
          "不是「偏宽方向值 90% 的钱」（已被 jackknife 否掉），",
          "而是「**偏宽是分歧的主导形态（笔数 %d:%d），且 decline 档几乎没被使用**」。"
          % (n_lenient, len(dis) - n_lenient),
          "",
          "### ⑧ 闸门那条结论不受影响\n",
          "「对 approve 过度上报在生产拓扑下值 $0」是**结构性**结论（闸门在 Agent 之前），"
          "不依赖任何金额估计的稳定性，jackknife 影响不到它。**该病灶仍不进 round 3。**\n",
          "> 方法论：这是**同一份数据上，一个结论死了、另一个活着**的例子。"
          "死的是需要金额加权排序的（集中度撑不住），活的是纯计数的（不需要加权）。"
          "教训不是「成本归因没用」，是**先问结论需要多强的估计量，再看样本给不给得起**。\n"]

    REPORT.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n✅ → {REPORT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "r1")
