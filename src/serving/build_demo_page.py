"""DEMO-1 第二交付物：把数据**内联**进单文件 `reports/demo/index.html`。

**为什么内联而不是 fetch**：`file://` 下 fetch 会被 CORS 拦、页面直接空白，
而「双击就能开」是断网现场唯一可靠的启动方式。**零外部依赖**：
无 CDN、无 Google Fonts、无构建工具、无前端框架——只用系统字体栈与手写 CSS。

用法：python -m src.serving.build_demo_page
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TPL = ROOT / "reports" / "demo" / "_template.html"
DATA = ROOT / "reports" / "demo" / "demo_data.json"
OUT = ROOT / "reports" / "demo" / "index.html"
# GitHub Pages 从 docs/ 提供服务。**同一份构建产物写两处**，而不是手动拷贝——
# 手拷的那份迟早和源版本不一致，且没人会发现（本项目已在报告上栽过一次）。
PAGES = ROOT / "docs" / "index.html"


def build():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    # </script> 必须转义，否则会提前闭合脚本块
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = TPL.read_text(encoding="utf-8").replace(
        "/*__DEMO_DATA__*/", "var DEMO = " + payload + ";")
    assert "fetch(" not in html and "cdn" not in html.lower(), "检测到外部依赖或 fetch"
    assert "fonts.googleapis" not in html, "检测到 Google Fonts"
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
    build()
