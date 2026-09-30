"""演示页发版前检查（静态契约与真实浏览器公式对账）。

检查页面可离线加载、文字可读、引用完整，以及 JavaScript 与 Python 成本公式一致。
渲染期布局和交互检查由 shoot_demo.py 补充，两者共同进入发布检查。

覆盖六项：
  1. 零外部依赖   —— 断网、file:// 双击必须能开
  2. 投影仪字号   —— 正文 ≥13px、关键数字 ≥20px
  3. 对比度       —— 浅/深两套配色所有文字对 ≥4.5:1（WCAG AA）
  4. 公式一致性   —— 页面 JS 的五档成本 vs 后端 disposition/stepup，逐位比
  5. 数据完整性   —— 案例字段齐全、finding 引用的 fact 不悬空
  6. 动效克制     —— 只允许既定的两处过渡

**查不到的那一类**：渲染期缺陷（曾漏掉「.bar-i 是行内元素导致整根条消失」）。
那类只有真渲染能发现 → 见 src/serving/shoot_demo.py。两个脚本互补，都要跑。

用法：python -m src.serving.check_demo_page
"""

import argparse
import re
import sys
from pathlib import Path

from src.serving.page_contract import external_dependencies

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "reports" / "demo" / "index.html"
MIN_BODY, MIN_KEY, MIN_RATIO = 13, 20, 4.5


def _lum(h):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    c = [(v / 12.92 if v <= .03928 else ((v + .055) / 1.055) ** 2.4) for v in c]
    return .2126 * c[0] + .7152 * c[1] + .0722 * c[2]


def ratio(a, b):
    la, lb = _lum(a), _lum(b)
    return (max(la, lb) + .05) / (min(la, lb) + .05)


def _vars(html, selector):
    """取某个 :root / [data-theme] 块里的 CSS 变量。"""
    m = re.search(re.escape(selector) + r"\{([^}]*)\}", html)
    return dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})", m.group(1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", choices=("webkit", "chromium"), default="webkit")
    parser.add_argument("--channel", help="Chromium 可选本机 Chrome: --channel chrome")
    args = parser.parse_args()
    if args.channel and args.browser != "chromium":
        parser.error("--channel 只适用于 chromium")
    html = PAGE.read_text(encoding="utf-8")
    fails = []

    # 1. 零外部依赖
    ext = external_dependencies(html)
    fails.extend(f"外部依赖：{item}" for item in ext)
    print(f"1. 零外部依赖         {'✅' if not ext else '❌'}  "
          f"自动加载资源/网络调用 {len(ext)}；允许导航到证据页面")

    # 2. 字号
    sizes = [int(x) for x in re.findall(r"font-size:\s*(\d+)px", html)]
    small = [s for s in sizes if s < MIN_BODY]
    if small:
        fails.append(f"字号低于 {MIN_BODY}px：{sorted(set(small))}")
    # 关键数字：成本列、滑块读数、首屏大数
    key = [int(m) for m in re.findall(
        r"\.(?:rowc \.cost|ctl label b|chip \.v)\{[^}]*font-size:\s*(\d+)px", html)]
    bad_key = [s for s in key if s < MIN_KEY]
    if bad_key:
        fails.append(f"关键数字低于 {MIN_KEY}px：{bad_key}")
    print(f"2. 投影仪字号         {'✅' if not small and not bad_key else '❌'}  "
          f"最小 {min(sizes)}px / 关键数字 {sorted(set(key))}")

    # 3. 对比度：正文、次要文字、各档语义色，浅深两套都验
    pairs = [("text", "bg"), ("text", "surface"), ("muted", "bg"), ("muted", "surface"),
             ("accent", "bg"), ("accent", "accent-soft"), ("inverse", "accent"),
             ("ok", "ok-bg"), ("warn", "warn-bg"), ("bad", "bad-bg"), ("esc", "esc-bg")]
    worst = {}
    for theme, sel in [("浅色", ":root"), ("深色", 'html[data-theme="dark"]')]:
        v = _vars(html, sel)
        rs = []
        for fg, bg in pairs:
            if fg in v and bg in v:
                r = ratio(v[fg], v[bg])
                rs.append((r, fg, bg))
                if r < MIN_RATIO:
                    fails.append(f"{theme} 对比度 {fg}/{bg} = {r:.2f} < {MIN_RATIO}")
        worst[theme] = min(rs)
    print(f"3. 对比度 ≥{MIN_RATIO}         "
          f"{'✅' if not [f for f in fails if '对比度' in f] else '❌'}  " +
          "、".join(f"{t} 最低 {r:.2f}({fg}/{bg})" for t, (r, fg, bg) in worst.items()))

    # 4/5 都在真浏览器里取值。
    # 用真实浏览器执行页面，直接取内联 DEMO，支持 WebKit 和 Chromium。
    import numpy as np
    from playwright.sync_api import sync_playwright
    from src.agent.disposition import BASE
    from src.model.stepup import ACTIONS5, STEPUP, costs5
    rng = np.random.default_rng(0)
    ps = rng.uniform(1e-4, .999, 400)
    amts = 10 ** rng.uniform(0, 3.5, 400)
    gangs = rng.choice([0., .5, 1.], 400)

    with sync_playwright() as pw:
        launch = {"channel": args.channel} if args.channel else {}
        b = getattr(pw, args.browser).launch(**launch)
        pg = b.new_page()
        pg.goto(PAGE.as_uri())
        got = np.array(pg.evaluate(
            "P=>P.p.map((_,i)=>{var c=costs(P.p[i],P.a[i],P.g[i]);"
            "return [c.approve,c.stepup,c.hold,c.decline,c.escalate]})",
            {"p": ps.tolist(), "a": amts.tolist(), "g": gangs.tolist()}))
        d = pg.evaluate("DEMO")
        b.close()

    exp = costs5(ps, amts, gangs, 76.02, BASE, STEPUP)
    exp = exp[:, [list(ACTIONS5).index(k)
                  for k in ("approve", "stepup", "hold", "decline", "escalate")]]
    dev = float(np.abs(got - exp).max())
    if dev > 1e-9:
        fails.append(f"公式偏差 {dev:.2e} —— 页面算的和报告里的数会对不上")
    print(f"4. 公式一致性         {'✅' if dev <= 1e-9 else '❌'}  "
          f"400 组随机输入（{args.browser} 实算），最大偏差 {dev:.2e}")

    # 5. 数据完整性
    if True:
        dangling = need = 0
        for c in d["cases"]:
            ids = {f["fact_id"] for f in c.get("facts", [])}
            for kf in (c.get("report") or {}).get("key_findings", []):
                for e in kf.get("evidence_ids", []):
                    need += 1
                    dangling += e not in ids
            for k in ("label", "teaches", "p", "cost_four", "cost_five"):
                if c.get(k) in (None, ""):
                    fails.append(f"案例 {c['key']} 缺字段 {k}")
        if dangling:
            fails.append(f"finding 引用了不存在的 fact × {dangling}")
        print(f"5. 数据完整性         {'✅' if not dangling else '❌'}  "
              f"{len(d['cases'])} 案例 / {need} 处引用，悬空 {dangling}")

    # 6. 动效克制
    tr = re.findall(r"transition:\s*([^;}]+)", html)
    if len(tr) > 2:
        fails.append(f"动效 {len(tr)} 处，超出既定的 2 处：{tr}")
    print(f"6. 动效克制           {'✅' if len(tr) <= 2 else '❌'}  {len(tr)} 处："
          + "、".join(t.strip() for t in tr))

    print()
    if fails:
        print(f"❌ {len(fails)} 项不合规：")
        for f in fails:
            print("   -", f)
        sys.exit(1)
    print(f"✅ 六项全过（{PAGE.stat().st_size // 1024} KB，"
          "断网/file:// 可开）。渲染期缺陷另见 shoot_demo.py。")


if __name__ == "__main__":
    main()
