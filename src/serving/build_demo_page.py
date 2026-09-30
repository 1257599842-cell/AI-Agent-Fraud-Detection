"""DEMO-1 第二交付物：把数据**内联**进单文件 `reports/demo/index.html`。

**为什么内联而不是 fetch**：`file://` 下 fetch 会被 CORS 拦、页面直接空白，
而「双击就能开」是断网现场唯一可靠的启动方式。**零外部依赖**：
无 CDN、无 Google Fonts、无构建工具、无前端框架——只用系统字体栈与手写 CSS。

用法：python -m src.serving.build_demo_page
"""

import argparse
import json
from pathlib import Path

from src.serving.page_contract import external_dependencies

ROOT = Path(__file__).resolve().parents[2]
TPL = ROOT / "reports" / "demo" / "_template.html"
DATA = ROOT / "reports" / "demo" / "demo_data.json"
OUT = ROOT / "reports" / "demo" / "index.html"
# GitHub Pages 从 docs/ 提供服务。**同一份构建产物写两处**，而不是手动拷贝——
# 手拷的那份迟早和源版本不一致，且没人会发现（本项目已在报告上栽过一次）。
PAGES = ROOT / "docs" / "index.html"


def render():
    """只读模板和归档，不写文件、不访问网络。"""
    data = json.loads(DATA.read_text(encoding="utf-8"))
    # </script> 必须转义，否则会提前闭合脚本块
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    template = TPL.read_text(encoding="utf-8")
    assert template.count("/*__DEMO_DATA__*/") == 1, "模板必须且只能有一个数据占位符"
    html = template.replace(
        "/*__DEMO_DATA__*/", "var DEMO = " + payload + ";")
    assert not external_dependencies(html), "检测到外部加载依赖"
    return data, html


def build(check=False):
    data, html = render()
    if check:
        stale = [str(p.relative_to(ROOT)) for p in (OUT, PAGES)
                 if not p.exists() or p.read_bytes() != html.encode("utf-8")]
        if not (PAGES.parent / ".nojekyll").exists():
            stale.append("docs/.nojekyll")
        if stale:
            raise SystemExit("构建产物过期：" + ", ".join(stale)
                             + "；运行 python -m src.serving.build_demo_page")
        print("✅ 模板、数据、本地页面与 Pages 产物一致（只读检查）")
        return
    OUT.write_text(html, encoding="utf-8")
    PAGES.parent.mkdir(parents=True, exist_ok=True)
    PAGES.write_text(html, encoding="utf-8")
    # Jekyll 默认会忽略下划线开头的文件并重新处理页面；.nojekyll 让它原样发布
    (PAGES.parent / ".nojekyll").write_text("", encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    assert OUT.read_bytes() == PAGES.read_bytes(), "两份产物不一致"
    print(f"✅ 单文件页面 → {OUT.relative_to(ROOT)}（{kb:.0f} KB，零外部依赖）")
    print(f"   同一份 → {PAGES.relative_to(ROOT)}（GitHub Pages，逐字节相同）")
    print(f"   内联案例 {len(data['cases'])} 笔；双击即可打开（file:// 可用）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只读检查构建产物是否过期")
    build(check=parser.parse_args().check)
