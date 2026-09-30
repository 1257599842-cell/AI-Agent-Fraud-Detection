"""真实浏览器检查与截图：桌面/手机、两种主题、全部案例、离线交互。

默认 WebKit；已有 Chrome 可用 --browser chromium --channel chrome。
--check-only 不写截图，可用于持续集成。截图是页面真实渲染，不做后期合成。
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "reports" / "demo" / "index.html"
OUT = ROOT / "reports" / "demo" / "shots"


def no_overflow(page, context):
    overflow = page.evaluate("document.documentElement.scrollWidth - innerWidth")
    assert overflow <= 0, f"{context}: 横向溢出 {overflow}px"


def scroll_to(page, selector):
    page.locator(selector).evaluate(
        "el => window.scrollTo(0, el.getBoundingClientRect().top + scrollY - 90)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", choices=("webkit", "chromium"), default="webkit")
    parser.add_argument("--channel", help="使用本机 Chrome: --channel chrome")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.channel and args.browser != "chromium":
        parser.error("--channel 只适用于 chromium")
    from playwright.sync_api import sync_playwright

    if not args.check_only:
        OUT.mkdir(parents=True, exist_ok=True)
    errors, requests = [], []
    with sync_playwright() as pw:
        launch = {"channel": args.channel} if args.channel else {}
        browser = getattr(pw, args.browser).launch(**launch)
        for width in (320, 375, 768, 1280, 1440):
            for theme in ("light", "dark"):
                context = browser.new_context(
                    viewport={"width": width, "height": 900}, offline=True,
                    reduced_motion="reduce", device_scale_factor=1)
                page = context.new_page()
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
                page.on("request", lambda r: requests.append(r.url)
                        if r.url.startswith(("http:", "https:")) else None)
                page.goto(PAGE.as_uri())
                page.wait_for_function("document.querySelectorAll('.caseitem').length === 7")
                if theme == "dark":
                    page.click("#theme")
                no_overflow(page, f"{width}/{theme}/首页")
                if width == 1280 and theme == "light" and not args.check_only:
                    page.screenshot(path=str(OUT / "overview.png"))

                # 所有案例都展开所有证据：长文本不能撑破移动端。
                for i in range(7):
                    page.locator(".caseitem").nth(i).click()
                    page.eval_on_selector_all(".finding", "els => els.forEach(el => el.open = true)")
                    assert page.locator('#cases [aria-pressed="true"]').count() == 1
                    assert page.locator("#detail .teach").inner_text().strip()
                    assert "**" not in page.locator("#detail .teach").inner_text()
                    no_overflow(page, f"{width}/{theme}/案例{i}")

                # 验证四/五动作、关联分和成本条都真能交互。
                assert page.locator("#rows .rowc").count() == 4
                page.click('#tiers [data-tiers="5"]')
                assert page.locator("#rows .rowc").count() == 5
                page.eval_on_selector("#p", "el => {el.value='-0.52'; el.dispatchEvent(new Event('input'))}")
                page.eval_on_selector("#a", "el => {el.value='1.30'; el.dispatchEvent(new Event('input'))}")
                assert page.locator("#rows .win").get_attribute("data-action") == "stepup"
                page.click('#tiers [data-tiers="4"]')
                assert page.locator("#rows .win").get_attribute("data-action") == "hold"
                page.click('#gseg [data-g="1"]')
                assert not page.locator("#negative-note").is_hidden()
                assert page.locator("#rows .win").get_attribute("data-action") == "escalate"
                fills = page.locator(".bar-i").evaluate_all(
                    "els => els.map(el => ({w: el.getBoundingClientRect().width, h: el.getBoundingClientRect().height}))")
                assert all(x["w"] > 0 and x["h"] > 0 for x in fills), "成本条塌陷"
                no_overflow(page, f"{width}/{theme}/沙盘")

                page.click('#fbseg [data-m="degraded"]')
                assert "degraded" in page.locator("#fb").inner_text()
                assert "$0.0000" in page.locator("#fb").inner_text()
                page.click('#fbseg [data-m="llm"]')
                assert "llm" in page.locator("#fb").inner_text()
                no_overflow(page, f"{width}/{theme}/降级")

                if width == 1280 and not args.check_only:
                    page.set_viewport_size({"width": 1280, "height": 720})
                    if theme == "light":
                        page.click('#gseg [data-g="0"]')
                        scroll_to(page, "#sandbox h2")
                        page.screenshot(path=str(OUT / "light_sandbox_1280x720.png"))
                    else:
                        page.locator('.caseitem').first.click()
                        page.locator(".finding").nth(2).locator("summary").click()
                        scroll_to(page, "#detail .finding[open]")
                        page.screenshot(path=str(OUT / "dark_case_1280x720.png"))
                print(f"✅ {width}px / {theme}: 7 案例、四/五动作、负成本提示、降级、无横向溢出")
                context.close()
        browser.close()
    assert not errors, f"浏览器错误：{errors}"
    assert not requests, f"离线页面发起了远程请求：{requests}"
    print("✅ 无 JS 错误、无远程资源请求；全部交互在 offline 模式下通过")
    if not args.check_only:
        print("✅ 已更新 overview.png / light_sandbox_1280x720.png / dark_case_1280x720.png")


if __name__ == "__main__":
    main()
