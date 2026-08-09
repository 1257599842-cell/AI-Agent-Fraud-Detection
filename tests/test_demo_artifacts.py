"""演示产物：`demo_data.json` / `index.html` / 截图与动图。

**分工**：需要真渲染的检查（对比度、横向溢出、JS 报错、行内元素塌陷）在
`src/serving/check_demo_page.py` 与 `shoot_demo.py`；本文件只做**不需要浏览器**的部分，
这样它能进任何 CI、秒级跑完。

最值钱的一条是 `TestStoredArgminStillMatchesCode`：
演示数据包里存着每笔的四档/五档 argmin，它必须与**今天的代码**算出来的一致。
若成本公式被改动而数据包没重建，页面上给评审者看的档位就会与报告里的对不上——
**而这种偏离不会报错**。
"""

import json
import re
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEMO_JSON = ROOT / "reports" / "demo" / "demo_data.json"
PAGE = ROOT / "reports" / "demo" / "index.html"
SHOTS = ROOT / "reports" / "demo" / "shots"
A_MED = 76.02


def _load():
    if not DEMO_JSON.exists():
        raise unittest.SkipTest("demo_data.json 不存在")
    return json.loads(DEMO_JSON.read_text(encoding="utf-8"))


class TestStoredArgminStillMatchesCode(unittest.TestCase):
    """数据包里冻结的档位 vs 今天代码算的档位。"""

    @classmethod
    def setUpClass(cls):
        cls.doc = _load()

    def test_four_tier_argmin_reproduces(self):
        from src.agent.disposition import BASE, argmin_action
        for c in self.doc["cases"]:
            with self.subTest(case=c["key"]):
                got = argmin_action([c["p"]], [c["fields"]["TransactionAmt"]],
                                    [c["gang_score"]], A_MED, BASE)[0]
                self.assertEqual(got, c["cost_four"]["argmin"])

    def test_five_tier_argmin_reproduces(self):
        from src.agent.disposition import BASE
        from src.model.stepup import STEPUP, argmin5
        for c in self.doc["cases"]:
            with self.subTest(case=c["key"]):
                got = argmin5(np.array([c["p"]]),
                              np.array([c["fields"]["TransactionAmt"]]),
                              np.array([c["gang_score"]]), A_MED, BASE, STEPUP)[0]
                self.assertEqual(got, c["cost_five"]["argmin"])

    def test_stored_costs_match_at_displayed_precision(self):
        """页面显示到 2 位小数，所以这里按**半分钱**判——那是观众能看见的粒度。"""
        from src.agent.disposition import BASE
        from src.model.stepup import STEPUP, costs5
        for c in self.doc["cases"]:
            want = costs5(np.array([c["p"]]),
                          np.array([c["fields"]["TransactionAmt"]]),
                          np.array([c["gang_score"]]), A_MED, BASE, STEPUP)[0]
            with self.subTest(case=c["key"]):
                np.testing.assert_allclose(c["cost_five"]["costs"], want, atol=0.005)

    def test_known_rounding_gap_stays_bounded(self):
        """**已知且有界的偏差**，不是把容差调松了事。

        `build_demo_data.py` 存的成本用**未舍入**的 p/gang 计算，
        而 JSON 里的 `p`/`gang_score` 字段是 `round(_, 6)` / `round(_, 4)` 后的值。
        于是**拿这份 JSON 自己的字段重算，对不上它自己存的成本**——差约 1e-3。

        显示到 2 位小数看不出，故封版时未改数据管道。
        但这个缺口要**被记住而不是被容差掩盖**：若哪天它涨到 0.005 以上，
        就会开始影响页面显示，这条测试届时会炸。
        """
        from src.agent.disposition import BASE
        from src.model.stepup import STEPUP, costs5
        worst = 0.0
        for c in self.doc["cases"]:
            want = costs5(np.array([c["p"]]),
                          np.array([c["fields"]["TransactionAmt"]]),
                          np.array([c["gang_score"]]), A_MED, BASE, STEPUP)[0]
            worst = max(worst, float(np.max(np.abs(np.array(c["cost_five"]["costs"]) - want))))
        self.assertLess(worst, 0.005, f"舍入缺口已涨到 {worst:.4f}，开始影响显示")
        self.assertGreater(worst, 0.0, "缺口消失了——说明数据管道已改，请更新本测试的说明")


class TestDemoDataIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = _load()

    def test_no_dangling_evidence_reference(self):
        """finding 引用的 fact 必须存在于本笔证据池。"""
        for c in self.doc["cases"]:
            ids = {f["fact_id"] for f in c.get("facts", [])}
            for kf in (c.get("report") or {}).get("key_findings", []):
                for e in kf.get("evidence_ids", []):
                    with self.subTest(case=c["key"], fact=e):
                        self.assertIn(e, ids)

    def test_every_case_carries_its_teaching_point(self):
        """每笔都要说明「用来展示什么」——没有它的案例在现场没有用途。"""
        for c in self.doc["cases"]:
            with self.subTest(case=c["key"]):
                for k in ("label", "teaches", "p", "cost_four", "cost_five"):
                    self.assertTrue(c.get(k) not in (None, ""), f"缺 {k}")

    def test_lowp_gang_keeps_its_caveat(self):
        """那笔 p≈0.007 **不能**被用来讲「网络项生效」，caveat 必须在。

        它讲的是「Agent 的证据发现是对的、算术不是」；
        网络项生效的证据在 gang_escalate 与聚合层面。两者混讲一问就穿。
        """
        c = next(x for x in self.doc["cases"] if x["key"] == "lowp_gang")
        self.assertTrue(c.get("caveat"))
        self.assertEqual(c["cost_four"]["argmin"], "approve",
                         "这笔的公式判定必须仍是放行，否则 caveat 的前提就没了")

    def test_params_are_labelled_as_assumptions(self):
        """成本参数**全部为假设值**，页面必须照实标注，不得让人误以为是实测。"""
        for k, v in self.doc["params"]["four_tier"].items():
            with self.subTest(param=k):
                self.assertEqual(v["status"], "[假设]")
        for k, v in self.doc["params"]["stepup"].items():
            with self.subTest(param=k):
                self.assertEqual(v["status"], "[假设]")

    def test_offline_banner_is_present(self):
        m = self.doc["meta"]
        self.assertIn("离线演示", m["banner"])
        self.assertNotIn("生产级", m["positioning"])


class TestPageIsSelfContained(unittest.TestCase):
    """断网、`file://` 双击即开——这是演示能不能进行的前提。"""

    @classmethod
    def setUpClass(cls):
        if not PAGE.exists():
            raise unittest.SkipTest("index.html 未生成")
        cls.html = PAGE.read_text(encoding="utf-8")

    def test_no_external_resource_of_any_kind(self):
        checks = {
            "http(s) 外链": r'(?:src|href)\s*=\s*["\']https?://',
            "协议相对链接": r'(?:src|href)\s*=\s*["\']//',
            "fetch / XHR": r"\bfetch\s*\(|XMLHttpRequest",
            "@import": r"@import",
            "CDN": r"cdn",
            "Google Fonts": r"fonts\.(?:googleapis|gstatic)",
        }
        for name, pat in checks.items():
            with self.subTest(check=name):
                self.assertEqual(re.findall(pat, self.html, re.I), [])

    def test_data_is_inlined_not_fetched(self):
        """`file://` 下 fetch 会被 CORS 拦、页面全白——数据必须内联。"""
        self.assertIn("var DEMO", self.html)

    def test_projector_font_floor(self):
        sizes = [int(x) for x in re.findall(r"font-size:\s*(\d+)px", self.html)]
        self.assertTrue(sizes)
        self.assertGreaterEqual(min(sizes), 13)

    def test_key_numbers_are_large_enough(self):
        """成本列 / 滑块读数 / 首屏大数 ≥20px，投影仪后排要能看清。"""
        key = [int(m) for m in re.findall(
            r"\.(?:rowc \.cost|ctl label b|chip \.v)\{[^}]*font-size:\s*(\d+)px", self.html)]
        self.assertTrue(key, "没找到关键数字的字号规则——选择器可能被改名了")
        self.assertGreaterEqual(min(key), 20)

    def test_inline_bar_fill_is_blockified(self):
        """`.bar-i` 必须 `display:block`。

        它是 `<span>`；**行内非替换元素的 width/height 会被浏览器忽略**，
        漏掉这一条会让五档成本条整根消失——而静态检查（对比度/外链/字号）一条都查不出。
        """
        m = re.search(r"\.rowc \.bar-i\{([^}]*)\}", self.html)
        self.assertIsNotNone(m, "找不到 .bar-i 规则")
        self.assertIn("display:block", m.group(1).replace(" ", ""))

    def test_animation_budget_is_respected(self):
        self.assertLessEqual(len(re.findall(r"transition:\s*[^;}]+", self.html)), 2)


class TestShippedFigures(unittest.TestCase):
    """README 首屏引用的图必须存在且不超预算。"""

    def test_screenshots_exist(self):
        for name in ("light_sandbox_1280x720.png", "dark_case_1280x720.png"):
            with self.subTest(fig=name):
                self.assertTrue((SHOTS / name).exists())

    def test_flip_gif_within_budget(self):
        gif = SHOTS / "sandbox_flip.gif"
        if not gif.exists():
            self.skipTest("动图未生成")
        self.assertLess(gif.stat().st_size, 3 * 1024 * 1024)

    def test_readme_references_resolve(self):
        """README 里的图片相对路径必须真的指到文件（GitHub 区分大小写）。"""
        readme = ROOT / "README.md"
        if not readme.exists():
            self.skipTest("README 不存在")
        for rel in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme.read_text(encoding="utf-8")):
            if rel.startswith("http"):
                continue
            with self.subTest(path=rel):
                self.assertTrue((ROOT / rel).exists(), f"README 引用了不存在的图：{rel}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
