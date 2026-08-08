"""用 WebKit（Safari 同引擎）给演示页出图 + 实测断网现场的三件事。

**为什么用 WebKit 而不是 Chromium**：owner 现场用 Safari 演示，
用同一引擎渲染才是忠实的检查——Chrome 上好看不代表 Safari 上不塌。

产出 reports/demo/shots/：浅色/深色各一张 1280×720，另加两张整页长图备用。
同时**实测**（此前只能静态推断）：横向滚动、JS 运行时报错、字体回退后的实际渲染。

用法：python -m src.serving.shoot_demo
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "reports" / "demo" / "index.html"
OUT = ROOT / "reports" / "demo" / "shots"
W, H = 1280, 720


def main():
    from playwright.sync_api import sync_playwright
    OUT.mkdir(parents=True, exist_ok=True)
    errors, results = [], []

    with sync_playwright() as pw:
        b = pw.webkit.launch()
        for theme, scene in [("light", "sandbox"), ("dark", "case")]:
            pg = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=2)
            pg.on("console", lambda m: errors.append(f"[console.{m.type}] {m.text}")
                  if m.type == "error" else None)
            pg.on("pageerror", lambda e: errors.append(f"[pageerror] {e}"))
            pg.goto(PAGE.as_uri())            # file:// —— 与现场双击打开完全一致
            pg.wait_for_timeout(400)

            if theme == "dark":
                pg.click("#theme")
                pg.wait_for_timeout(300)

            if scene == "sandbox":
                # p=0.30、金额 $20、gang=无 → 五档 argmin = 加验证，且与第二名分得最开。
                # **别用 p=0.24/$100**：那一格已翻回挂起（$76 才是 stepup），坐标越界一格。
                pg.eval_on_selector("#p", "el=>{el.value='-0.5229';el.dispatchEvent(new Event('input'))}")
                pg.eval_on_selector("#a", "el=>{el.value='1.3010';el.dispatchEvent(new Event('input'))}")
                pg.wait_for_timeout(200)
                # 对准标题而非 section 盒：section 上下 padding 相加 128px，
                # 按盒顶取景会在主图顶上留一条大空带。
                target = pg.query_selector("section:nth-of-type(2) h2")
            else:
                # 案卷：选 p≈0.007 那笔。finding 3、4 由模板默认展开（P0-1/P0-2），
                # 无需点击。对准 finding 3——它供结构型那条，finding 4 紧随其后。
                pg.eval_on_selector_all(
                    ".caseitem", "els=>els.find(e=>e.textContent.includes('0.006987')).click()")
                pg.wait_for_timeout(300)
                target = pg.query_selector("#detail .finding[open]")

            ovf = pg.evaluate("document.documentElement.scrollWidth - window.innerWidth")
            if target:
                # 用 rect+scrollY，不用 offsetTop——offsetTop 是相对 offsetParent 的，
                # finding 嵌在多层定位容器里，按它滚会冲过头把标题切掉。
                pg.evaluate("el=>window.scrollTo(0, el.getBoundingClientRect().top"
                            "+window.scrollY-80)", target)
                pg.wait_for_timeout(250)

            if scene == "case":
                # Markdown 记号必须已转成标签——曾把 ** 原样漏在页面上
                html = pg.eval_on_selector("#detail .teach", "el=>el.innerHTML")
                assert "<b>" in html and "**" not in html, "teaches 的 ** 未转粗体"
                # P0-2 由脚本守住，不靠肉眼看一次：视口内必须同时有
                # 标签型 chip、结构型 chip、以及 embargo 说明行。缺一即拍废。
                seen = pg.evaluate("""() => {
                  const inView = el => { const r = el.getBoundingClientRect();
                    return r.top >= 0 && r.bottom <= window.innerHeight; };
                  const txt = [...document.querySelectorAll('#detail .finding[open] .chipsm')]
                    .filter(inView).map(e => e.textContent);
                  const emb = [...document.querySelectorAll('#detail .embargo')].some(inView);
                  return {lb: txt.some(t => t.includes('标签型')),
                          st: txt.some(t => t.includes('结构型')), emb: emb}; }""")
                assert seen["lb"] and seen["st"], f"P0-2 两型未同屏：{seen}"
                assert seen["emb"], "P0-2 embargo 说明行不在视口内"
            f = OUT / f"{theme}_{scene}_{W}x{H}.png"
            pg.screenshot(path=str(f))
            full = OUT / f"{theme}_full.png"
            pg.screenshot(path=str(full), full_page=True)
            results.append((theme, scene, ovf, f.stat().st_size // 1024,
                            full.stat().st_size // 1024))
            pg.close()
        b.close()

    print(f"{'主题':<7}{'场景':<9}{'横向溢出':>9}{'视口图KB':>10}{'整页图KB':>10}")
    for t, s, o, k1, k2 in results:
        flag = "✅ 0" if o <= 0 else f"❌ {o}px"
        print(f"{t:<8}{s:<10}{flag:>9}{k1:>10}{k2:>10}")
    print()
    if errors:
        print("❌ 运行时报错：")
        for e in dict.fromkeys(errors):
            print("   ", e)
        sys.exit(1)
    print("✅ 无 JS 运行时报错（file:// 直开，与现场双击一致）")
    print(f"✅ 图 → {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
