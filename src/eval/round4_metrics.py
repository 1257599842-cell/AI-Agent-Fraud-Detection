"""round 4 指标（**预注册**：定义写死在本文件，基线先在 r3 上算出，跑完不许改）。

修的缺陷：owner 人工核验发现的头号形态——**引用单个字段值、却把由它推出的
因果/对比判断标成 `confirmed`**（4 条过度断言里 3 条同形，见 agent_defect_taxonomy.md）。

## 预注册指标（全硬层，不需 judge）

1. **主指标**（纯代码，越小越好）：
   `assertion_strength == "confirmed"` 且 **所引 fact 全为 `txn_field:*`** 且
   **不同 fact 数 ≤ 2** 的 finding 计数。

2. **反向护栏**（纯代码，越小越好）：
   `assertion_strength == "tentative"` 且 **所引 ≥2 条 统计/图 类 fact**（`STAT_*`/`GRAPH_*`/
   `RULE_*`）的 finding 计数。
   **存在理由**：单边指标会奖励过度矫正——prompt 一改就全线降档即可把主指标刷到 0。
   judge 那一轮已经踩过这个坑（过严率降到 17%，代价是少数类召回掉到 0）。
   护栏必须与主指标**同时**报，任何一边恶化都不算改进。

3. **次指标**（小样本手核）：主指标命中里，**真属「由字段推出的判断」**的条数。
   n 小可肉眼核；**必须带否定/边界守卫**（同类错误本项目已栽三次：
   解析器跨块顺延、翻转实验检出器极性、缺陷聚类的否定式套话）。

## 单变量纪律（重要）
round 4 **只改 prompt 一处**。`policy_param` 那条改动（把成本假设升为一等公民 fact）
**不在 round 4 的运行期内生效**——否则证据池同时变了，主指标会因为
「不再全是 txn_field」而下降，那不是 prompt 的功劳。round 3 的教训：一次只改一处。

用法：
  python -m src.eval.round4_metrics --baseline r3    # 跑前：算基线并冻结预注册
  python -m src.eval.round4_metrics --score r3 r4    # 跑后：对比
"""

from src.report_io import write_report
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "reports" / "eval_runs"
PREREG = PROJECT_ROOT / "reports" / "round4_preregistration.md"
REPORT = PROJECT_ROOT / "reports" / "agent_round4_metrics.md"

STAT_LIKE = ("STAT_", "GRAPH_", "RULE_")


def _facts(r):
    return {f["fact_id"]: f for f in r.get("facts", [])}


def _cited(kf, facts):
    return [facts[e] for e in kf.get("evidence_ids", []) if e in facts]


def primary(r, kf):
    """主指标：confirmed + 所引全为 txn_field:* + 不同 fact 数 ≤2。"""
    if kf.get("assertion_strength") != "confirmed":
        return False
    c = _cited(kf, _facts(r))
    if not c:
        return False
    return all(str(f.get("type", "")).startswith("txn_field:") for f in c) and len(c) <= 2


def guardrail(r, kf):
    """反向护栏：tentative 却引 ≥2 条统计/图/规则类 fact（过度降档的信号）。"""
    if kf.get("assertion_strength") != "tentative":
        return False
    ids = [e for e in kf.get("evidence_ids", []) if e in _facts(r)]
    return sum(1 for e in ids if e.startswith(STAT_LIKE)) >= 2


def tally(tag):
    out = {"n_reports": 0, "n_findings": 0, "primary": [], "guard": [],
           "strength": {}}
    for f in sorted((RUNS_DIR / tag).glob("txn_*.json")):
        r = json.loads(f.read_text())
        rep = r.get("report") or {}
        if not rep:
            continue
        out["n_reports"] += 1
        for i, kf in enumerate(rep.get("key_findings", [])):
            out["n_findings"] += 1
            s = kf.get("assertion_strength")
            out["strength"][s] = out["strength"].get(s, 0) + 1
            if primary(r, kf):
                out["primary"].append((r["txn_id"], i, kf.get("finding", "")))
            if guardrail(r, kf):
                out["guard"].append((r["txn_id"], i, kf.get("finding", "")))
    return out


def baseline(tag):
    t = tally(tag)
    L = ["# round 4 预注册（跑前写死，跑完不许改）\n",
         f"基线轮次：`{tag}`（{t['n_reports']} 份报告 / {t['n_findings']} 条 finding）\n",
         "## 指标定义（代码即定义，见 `src/eval/round4_metrics.py`）\n",
         "1. **主指标**（越小越好）：`confirmed` + 所引 fact 全为 `txn_field:*` + 不同 fact 数 ≤2 的 finding 计数。",
         "2. **反向护栏**（越小越好）：`tentative` + 所引 ≥2 条 `STAT_/GRAPH_/RULE_` fact 的 finding 计数。",
         "   —— 防止「一改就全线降档」把主指标刷到 0；**两边必须同时不恶化**才算改进。",
         "3. **次指标**：主指标命中里真属「由字段推出的判断」的条数（小样本手核，带否定/边界守卫）。\n",
         "## 冻结基线\n",
         f"- **主指标 = {len(t['primary'])}** 条",
         f"- **反向护栏 = {len(t['guard'])}** 条",
         f"- 断言强度分布：" + "、".join(f"{k} {v}" for k, v in sorted(t["strength"].items())),
         "",
         "## 判读规则（预先写死）\n",
         "- 主指标下降**且**护栏不上升 → 记为改进；",
         "- 主指标下降**但**护栏上升 → 记为**过度矫正**，不算改进；",
         "- 主指标不动 → **报零**（round 3 已有先例，零也是结果）。\n",
         "## 单变量纪律\n",
         "round 4 **只改 prompt 一处**。`policy_param`（成本假设升为一等公民 fact）"
         "**不在 round 4 运行期内生效**——",
         "否则证据池同时改变，主指标会因「不再全是 `txn_field`」而下降，那不是 prompt 的功劳。\n",
         "## 基线命中明细（供跑后逐条对照）\n"]
    for txn, i, txt in t["primary"]:
        L.append(f"- `{txn}#{i}`：«{txt[:100]}»")
    PREREG.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n✅ 预注册已冻结 → {PREREG.relative_to(PROJECT_ROOT)}")


# 次指标手核用：判断 finding 文本是否含「由字段推出的判断」
# 守卫：否定式（"不构成风险"）与边界（"金额"出现在纯陈述里）都不算
_INFER = re.compile(r"限制|降低|抵消|说明|表明|意味|因此|故|支持|一致|不构成|风险(?:有限|较低)|"
                    r"损失(?:有限|可控)|可控|无需|足以")
_NEG_CTX = re.compile(r"[未无不非]")


def infer_like(text):
    """是否是「由字段推出的判断」而非纯字段陈述。带否定守卫（本项目已栽三次）。"""
    for m in _INFER.finditer(text):
        # 「不构成风险」本身就是一个推论，不排除；只排除被否定的推论动词
        if m.group(0) in ("不构成",):
            return True
        if not _NEG_CTX.search(text[max(0, m.start() - 4):m.start()]):
            return True
    return False


def strength_profile(tag):
    """各断言强度档的**证据画像**：引用 fact 数、引用统计/图类 fact 的比例。

    用途：判断 `assertion_strength` 这个字段在修复后**还有没有判别力**。
    若各档的证据画像趋同，则结论必须写成
    「修复有效，但代价是断言强度字段本身变钝」——而不是只说护栏是单边的。
    """
    prof = {}
    for f in sorted((RUNS_DIR / tag).glob("txn_*.json")):
        r = json.loads(f.read_text())
        rep = r.get("report") or {}
        facts = _facts(r)
        for kf in rep.get("key_findings", []):
            s = kf.get("assertion_strength")
            ids = [e for e in kf.get("evidence_ids", []) if e in facts]
            d = prof.setdefault(s, {"n": 0, "nf": [], "stat": 0, "only_txn": 0})
            d["n"] += 1
            d["nf"].append(len(ids))
            if any(e.startswith(STAT_LIKE) for e in ids):
                d["stat"] += 1
            if ids and all(str(facts[e].get("type", "")).startswith("txn_field:") for e in ids):
                d["only_txn"] += 1
    return prof


def _assoc(prof):
    """强度序数 ↔ 引用 fact 数 的 Spearman 相关（判别力的单值概括）。"""
    order = {"tentative": 0, "supported": 1, "confirmed": 2}
    xs, ys = [], []
    for s, d in prof.items():
        if s in order:
            xs += [order[s]] * len(d["nf"])
            ys += d["nf"]
    if len(set(xs)) < 2:
        return float("nan")
    import scipy.stats as st
    return float(st.spearmanr(xs, ys).statistic)


def score(a, b):
    ta, tb = tally(a), tally(b)
    L = ["# round 4 结果（对照预注册）\n",
         f"| 指标 | {a}（基线） | {b} | 变化 |", "|---|---|---|---|",
         f"| **主指标**（confirmed+纯字段+≤2 fact） | {len(ta['primary'])} | "
         f"**{len(tb['primary'])}** | {len(tb['primary'])-len(ta['primary']):+d} |",
         f"| **反向护栏**（tentative+≥2 统计/图 fact） | {len(ta['guard'])} | "
         f"**{len(tb['guard'])}** | {len(tb['guard'])-len(ta['guard']):+d} |",
         f"| finding 总数 | {ta['n_findings']} | {tb['n_findings']} | "
         f"{tb['n_findings']-ta['n_findings']:+d} |", ""]
    L += ["**断言强度分布**（看有没有全线降档）：", "",
          "| 强度 | " + f"{a} | {b} |", "|---|---|---|"]
    for k in ["confirmed", "supported", "tentative"]:
        L.append(f"| {k} | {ta['strength'].get(k,0)} | {tb['strength'].get(k,0)} |")

    d_main = len(tb["primary"]) - len(ta["primary"])
    d_guard = len(tb["guard"]) - len(ta["guard"])
    L += ["", "## 判读（按预注册规则，不改口径）\n"]
    if d_main < 0 and d_guard <= 0:
        L += [f"✅ **记为改进**：主指标 {d_main:+d}，护栏 {d_guard:+d}（未恶化）。"]
    elif d_main < 0 and d_guard > 0:
        L += [f"⚠️ **记为过度矫正，不算改进**：主指标降了 {d_main:+d}，"
              f"但护栏升了 {d_guard:+d} —— 用全线降档换来的。"]
    elif d_main == 0:
        L += ["**报零**：主指标未动。round 3 已有先例——改了没用同样是结果。"]
    else:
        L += [f"❌ **变差**：主指标 {d_main:+d}。"]

    # ---- 事后分析：预注册之外，明确标注，不改任何指标 ----
    na, nb = ta["n_findings"], tb["n_findings"]
    sh = lambda t, k: t["strength"].get(k, 0) / t["n_findings"]
    L += ["", "## ⚠️ 事后分析：预注册的护栏有一个盲区（照实记，**不改指标**）\n",
          "护栏当初是按「怕它全线降档到 tentative 刷分」设计的。**实际位移不在那个方向**：\n",
          "| 强度 | " + f"{a} 占比 | {b} 占比 | 变化 |", "|---|---|---|---|"]
    for k in ["confirmed", "supported", "tentative"]:
        L.append(f"| {k} | {sh(ta,k):.1%} | {sh(tb,k):.1%} | {(sh(tb,k)-sh(ta,k))*100:+.1f}pp |")
    L += ["",
          f"- 质量**全部涌向 supported**（{sh(ta,'supported'):.1%} → {sh(tb,'supported'):.1%}，"
          f"{(sh(tb,'supported')-sh(ta,'supported'))*100:+.1f}pp），",
          f"  且**两头都在流入**：confirmed {(sh(tb,'confirmed')-sh(ta,'confirmed'))*100:+.1f}pp、"
          f"tentative {(sh(tb,'tentative')-sh(ta,'tentative'))*100:+.1f}pp。",
          "- `confirmed → supported` **正是 prompt 要的**；但 `tentative → supported` 是**升档**，"
          "prompt 没要求过，属于计划外的副作用。",
          "- **我的护栏看不见这个方向**：它只统计 `tentative + ≥2 统计/图 fact`，"
          "所以「大家都往 supported 收敛」时它反而会变好看。",
          "  护栏这次的下降（−8）**有一部分正是 tentative 总数变少带来的**，"
          "不能全记作「没有过度矫正」。",
          "",
          "> **这是预注册本身的缺陷，不是结果的缺陷**：单边护栏只挡住了我预想的那一种作弊方式。",
          "> 按规矩**指标不改、判读不改**；但「记为改进」这句话必须带着这一条一起讲。",
          "> 正确的护栏应当是**双边**的——同时盯「降档」与「升档」，或者直接盯强度分布的整体位移。",
          f"- 另记：finding 总数 {na} → {nb}（{nb/na-1:+.1%}），"
          f"主指标**率** {len(ta['primary'])/na:.2%} → {len(tb['primary'])/nb:.2%}"
          "（率的降幅比绝对计数更大，方向一致）。\n",
          "## 次指标：主指标命中里真属「由字段推出的判断」的条数\n",
          "（带否定/边界守卫；本项目此类错误已栽三次，故守卫与计数一起给出）\n"]

    # ---- 补测：断言强度字段还有没有判别力（描述性统计，不改任何指标）----
    pa, pb = strength_profile(a), strength_profile(b)
    ra, rb = _assoc(pa), _assoc(pb)
    L += ["", "## 补测：`assertion_strength` 各档的证据画像（描述性，指标不改）\n",
          "问的是一个比「护栏是单边的」更要紧的问题：**修完之后，断言强度这个字段还分得开吗？**\n",
          "| 轮次 | 强度 | n | 平均引用 fact 数 | 引用统计/图类 fact 占比 | 只引 txn_field 占比 |",
          "|---|---|---|---|---|---|"]
    for tag, prof in [(a, pa), (b, pb)]:
        for s in ["confirmed", "supported", "tentative"]:
            d = prof.get(s)
            if not d or not d["n"]:
                continue
            L.append(f"| {tag} | {s} | {d['n']} | {np.mean(d['nf']):.2f} | "
                     f"{d['stat']/d['n']:.0%} | {d['only_txn']/d['n']:.0%} |")
    gap = lambda p: (np.mean(p["confirmed"]["nf"]) - np.mean(p["tentative"]["nf"])) \
        if p.get("confirmed") and p.get("tentative") else float("nan")
    ga, gb = gap(pa), gap(pb)
    L += ["",
          f"- **强度序数 ↔ 引用 fact 数 的 Spearman 相关**：`{a}` **{ra:+.3f}** → `{b}` **{rb:+.3f}**",
          f"- confirmed 与 tentative 的平均引用数之差：`{a}` {ga:+.2f} → `{b}` {gb:+.2f}", ""]
    weaker = (not np.isnan(ra)) and (not np.isnan(rb)) and abs(rb) < abs(ra) - 0.03
    if weaker:
        L += ["> ⚠️ **关联变弱 → 结论必须写成「修复有效，但代价是断言强度字段本身变钝」。**",
              "> 主指标那个 25→9 是真的，但它有一部分是靠**把强度整体挤向 supported** 换来的；",
              "> 强度与证据量的对应关系随之松掉——这个字段作为「证据强弱的信号」变得更不好用了。",
              "> **不能只说「我的护栏是单边的」就了事**：那是在说工具的毛病，",
              "> 这一条说的是**修复本身的代价**，量级更大，必须一起讲。"]
    elif not np.isnan(rb) and ra < 0 <= rb:
        L += ["> ✅ **预先写好的「变钝」那一支没有触发；实际结果比它更强：关联的方向被修正了。**", "",
              f"> `{a}` 的相关是 **{ra:+.3f}（负的）**——`confirmed` 引用的 fact 数反而**少于** "
              f"`tentative`（{np.mean(pa['confirmed']['nf']):.2f} vs "
              f"{np.mean(pa['tentative']['nf']):.2f}）。",
              "> **强度与证据量呈反向关系，这本身就是那个缺陷的独立证据**："
              "「拿一个字段值就敢标 confirmed」正是它的成因。",
              f"> `{b}` 修正为 **{rb:+.3f}（正的）**，confirmed 的平均引用数升到 "
              f"{np.mean(pb['confirmed']['nf']):.2f}、高于 tentative 的 "
              f"{np.mean(pb['tentative']['nf']):.2f}。", "",
              f"> 另一处佐证：`{b}` 的 confirmed 里「只引 txn_field」占比反而升到 "
              f"{pb['confirmed']['only_txn']/pb['confirmed']['n']:.0%}"
              f"（`{a}` 为 {pa['confirmed']['only_txn']/pa['confirmed']['n']:.0%}）——",
              "> **这与主指标下降并不矛盾，方向恰好一致**：prompt 明说「字段值本身可以标 confirmed」，",
              "> 所以「罗列多个字段值」留在 confirmed 是**对的**；被赶走的是"
              "「**只引 1–2 个字段却下因果/对比判断**」那一类（主指标正是这么定义的）。",
              "",
              "> → 强度字段**没有变钝，反而变准了**。主指标的下降不是靠把它挤扁换来的。"]
    elif not np.isnan(rb):
        L += ["> ✅ **关联未见削弱** → 强度字段的判别力在修复后保住了，",
              "> 主指标的下降不是靠把强度字段挤扁换来的。"]
    L += ["",
          "> **通则（本项目第三次被单边指标坑）**：judge 的过严率（只测过严、测不了过松）、",
          "> 富集筛子（只放大 judge 认为有问题的）、本轮的预注册护栏（只防降档、不防升档）。",
          "> → **护栏必须双向；只防一个方向的过度矫正等于没防。**\n"]
    for tag, t in [(a, ta), (b, tb)]:
        hits = [(x, i, s) for x, i, s in t["primary"] if infer_like(s)]
        L.append(f"- `{tag}`：主指标 {len(t['primary'])} 条中，判为「由字段推出判断」"
                 f"**{len(hits)}** 条")
        for x, i, s in hits[:6]:
            L.append(f"  - `{x}#{i}`：«{s[:90]}»")
    write_report(REPORT, "\n".join(L))
    print("\n".join(L))
    print(f"\n✅ → {REPORT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    if "--baseline" in sys.argv:
        baseline(sys.argv[sys.argv.index("--baseline") + 1])
    elif "--score" in sys.argv:
        i = sys.argv.index("--score")
        score(sys.argv[i + 1], sys.argv[i + 2])
    else:
        print(__doc__)
