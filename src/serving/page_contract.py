"""页面可导航到证据，但渲染和交互不能依赖远程资源。"""

import re
from html.parser import HTMLParser


class PageLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.resources = []
        self.links = []
        self.ids = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])
        for name in ("src", "srcset", "poster", "data", "href"):
            value = attrs.get(name, "")
            if name == "href" and tag == "a":
                continue
            if re.search(r"(?:https?:)?//", value, re.I):
                self.resources.append(f"{tag}[{name}]={value}")

    handle_startendtag = handle_starttag


def external_dependencies(html):
    """返回自动加载资源/网络调用；普通 a[href] 不算渲染依赖。"""
    parsed = PageLinks()
    parsed.feed(html)
    problems = parsed.resources.copy()
    for name, pattern in {
        "network call": r"\bfetch\s*\(|XMLHttpRequest|\bWebSocket\s*\(",
        "CSS import": r"@import\b",
        "CSS remote resource": r"url\(\s*['\"]?(?:https?:)?//",
    }.items():
        if re.search(pattern, html, re.I):
            problems.append(name)
    return problems
