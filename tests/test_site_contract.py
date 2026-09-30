"""公共展示契约：来源可打开、产物可复建、外链不会变成加载依赖。"""

import re
import unittest
from pathlib import Path
from urllib.parse import unquote, urlsplit

from src.serving.build_demo_page import render
from src.serving.page_contract import PageLinks, external_dependencies

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "https://github.com/1257599842-cell/AI-Agent-Fraud-Detection/blob/main/"


class TestSiteContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.html = render()
        cls.links = PageLinks()
        cls.links.feed(cls.html)

    def test_generated_page_is_fresh(self):
        for rel in ("docs/index.html", "reports/demo/index.html"):
            self.assertEqual((ROOT / rel).read_bytes(), self.html.encode("utf-8"))

    def test_unique_ids_and_valid_section_links(self):
        self.assertEqual(len(self.links.ids), len(set(self.links.ids)))
        for link in self.links.links:
            if link.startswith("#"):
                self.assertIn(link[1:], self.links.ids)

    def test_public_evidence_links_exist(self):
        source_links = [link for link in self.links.links if link.startswith(PREFIX)]
        self.assertGreaterEqual(len(source_links), 15)
        for link in source_links:
            self.assertTrue((ROOT / unquote(link[len(PREFIX):])).is_file(), link)
        for case in self.data["cases"]:
            self.assertTrue((ROOT / case["source_file"]).is_file())

    def test_reading_guides_have_no_broken_relative_links(self):
        paths = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
        for path in paths:
            rel = path.relative_to(ROOT)
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text()):
                if target.startswith(("http:", "https:", "#")):
                    continue
                dest = path.parent / unquote(urlsplit(target).path)
                self.assertTrue(dest.exists(), f"{rel}: {target}")

    def test_readme_leads_with_results_not_a_screenshot(self):
        readme = (ROOT / "README.md").read_text()
        self.assertLess(readme.index("## 主要结果"), readme.index("## 系统设计"))
        prefix = readme.split("## 主要结果", 1)[0]
        self.assertNotIn("reports/demo/shots/", prefix)
        self.assertIn("actions/workflows/verify.yml/badge.svg?branch=main", prefix)
        self.assertIn("<details>", readme)

    def test_historical_documents_have_a_separate_home(self):
        paths = {
            "AGENT_DESIGN.md": "design",
            "AUDIT_REPORT.md": "audits",
            "AUDIT_CHECKS.json": "audits",
            "BANK_EVIDENCE_REVIEW.md": "audits",
            "MODEL_CARD_SOURCES.json": "audits",
        }
        index = (ROOT / "docs/README.md").read_text()
        for name, section in paths.items():
            self.assertFalse((ROOT / name).exists(), name)
            self.assertTrue((ROOT / "docs" / section / name).is_file(), name)
            self.assertIn(f"{section}/{name}", index)

    def test_navigation_is_allowed_but_remote_resources_are_not(self):
        self.assertEqual(external_dependencies('<a href="https://example.org">source</a>'), [])
        for fragment in (
            '<script src="https://example.org/app.js"></script>',
            '<link rel="stylesheet" href="//example.org/theme.css">',
            '<img srcset="https://example.org/one.png 1x">',
            '<style>body{background:url(//example.org/bg.png)}</style>',
            '<script>fetch("https://example.org/data")</script>',
        ):
            with self.subTest(fragment=fragment):
                self.assertTrue(external_dependencies(fragment))

    def test_review_notes_do_not_rewrite_historical_payload(self):
        self.assertIn("var NOTES=", self.html)
        self.assertIn("历史报告摘要（原文）", self.html)
        self.assertIn("var gang=0,tiers=4", self.html)
        self.assertIn("不能直接部署为真实最优策略", self.html)
        self.assertIn("不是新的盲测", self.html)

    def test_headline_metrics_match_source(self):
        report = (ROOT / "reports/graph_vs_tabular.md").read_text()
        for score in ("0.5645", "0.6032", "0.9138", "0.9306"):
            self.assertIn(score, report)
            self.assertIn(score, self.html)
        self.assertIn("14.2%", self.html)
        self.assertIn("13.9%", self.html)

    def test_project_heading_is_descriptive(self):
        self.assertIn('id="hero-title">交易反欺诈', self.html)
        self.assertIn("风险建模与辅助调查", self.html)
        self.assertIn("主实验设置", self.html)
        for old_copy in ("分数之后", "把决策说清楚", "FRAUD / LAB",
                         "Risk models. Decisions. Evidence."):
            self.assertNotIn(old_copy, self.html)


if __name__ == "__main__":
    unittest.main()
