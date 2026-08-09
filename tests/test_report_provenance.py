"""`reports/` 只能由代码写 —— 由测试执法，不靠人记。

裁定（2026-08-09）：`reports/` 是构建产物，**手工编辑就此废止**。
依据：README / INTERVIEW / MODEL_CARD / REHEARSAL 的每个数都回源到 `reports/`；
若 `reports/` 可手编，第 3 条硬规矩（**每个数字都有可回的源**）就失去锚点。

本文件把这条裁定变成会自动拦住的东西：
  · 每份报告**机器区**的 `sha256` 记在 `reports/_manifest.json`（人写区不参与）；
  · **手编 → 哈希对不上 → 测试红**；
  · 要改内容只能改生成器、重跑、再 `--update` 刷新清单。

真正的「重新生成 = 逐字节相同」对拍在
`python -m src.eval.report_manifest --verify-rerun`（有副作用，故不进快速测试集）。
本文件只做无副作用的哈希核对，秒级完成。
"""

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MANIFEST = REPORTS / "_manifest.json"


def sha(p):
    """**只哈希机器区**——人写区归人，改判读不该让测试变红。

    过严的检查等于没有检查：一条天天误报的断言，最后一定被关掉。
    """
    from src.report_io import split_report
    machine, _ = split_report(p.read_text(encoding="utf-8"))
    return hashlib.sha256(machine.encode("utf-8")).hexdigest()


class TestReportProvenance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MANIFEST.exists():
            raise unittest.SkipTest("清单未生成：python -m src.eval.report_manifest --update")
        cls.man = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_every_listed_report_matches_its_hash(self):
        """任何手工编辑都会在这里现形。"""
        for name, rec in sorted(self.man.items()):
            p = REPORTS / name
            with self.subTest(report=name):
                self.assertTrue(p.exists(), f"{name} 缺失")
                self.assertEqual(
                    sha(p), rec["sha256_machine_only"],
                    f"\n{name} 与清单不符——它被手编过，或生成器改了没重跑。"
                    f"\n重生成：{rec['command']}"
                    f"\n刷新清单：python -m src.eval.report_manifest --update")

    def test_every_entry_records_a_runnable_command(self):
        """出处不是「大概是某个脚本生成的」，得是一条能直接复制运行的命令。"""
        for name, rec in sorted(self.man.items()):
            with self.subTest(report=name):
                # 允许 `VAR=... python -m ...`：环境变量也是口径的一部分
                # （e60 那份就是靠 GRAPH_FILE 切换的，命令里不带就重跑成 21 天）
                self.assertRegex(rec["command"], r"^(\w+=\S+ )*python -m (src|notebooks)\.")
                self.assertIn(rec["tier"], ("cheap", "heavy", "api", "frozen"))

    def test_manifest_and_generator_table_agree(self):
        """清单与 `GENERATORS` 表不得脱节——脱节意味着有报告没人认领。"""
        from src.eval.report_manifest import GENERATORS
        self.assertEqual(set(self.man), set(GENERATORS) & set(self.man))
        missing = [n for n in GENERATORS if not (REPORTS / n).exists()]
        self.assertEqual(missing, [], f"清单登记了但文件不存在：{missing}")


class TestNoSilentCliDefaults(unittest.TestCase):
    """参数是**口径**的一部分，缺参必须报错而不是顶默认值。

    事故：`round3_metrics` 曾写作 `run(args or ["r1"])`，裸跑顶上单轮口径，
    把两轮对比整段覆盖，**退出码仍是 0**。
    """

    ENTRYPOINTS = ["round3_metrics", "evidence_vs_decision", "self_awareness",
                   "cost_attribution", "disposition_sensitivity"]

    def test_no_module_falls_back_to_a_default_tag(self):
        """**先剥注释再匹配。**

        初版直接在源码文本上搜 `or ["`，结果命中了注释里那句历史说明
        「曾写作 `run(args or ["r1"])`」——**匹配到文本而非语义**，
        与本项目烧过的极性 bug（「未命中…三类 N」被当成命中）同一家族。
        """
        import re
        pat = re.compile(r"or \[[\"']|if len\(sys\.argv\) > 1 else")
        for m in self.ENTRYPOINTS:
            src = (ROOT / "src" / "eval" / f"{m}.py").read_text(encoding="utf-8")
            tail = src[src.rindex("if __name__"):]
            code = "\n".join(l for l in tail.splitlines()
                             if not l.lstrip().startswith("#"))
            with self.subTest(module=m):
                self.assertIsNone(pat.search(code),
                                  f"{m} 的入口仍在顶默认值——缺参必须报错")
                self.assertIn("require_tags", code)

    def test_round3_report_kept_the_two_round_comparison(self):
        """committed 的报告必须是**两轮对比**口径。

        单轮与两轮写同一个文件名，所以口径退化不会报错、只会让内容悄悄变少。
        这条测试盯的就是那次退化。
        """
        p = REPORTS / "agent_round3_metrics.md"
        if not p.exists():
            self.skipTest("报告不存在")
        text = p.read_text(encoding="utf-8")
        self.assertIn("| r3 |", text.replace(" r3 ", " r3 "),
                      "报告里没有 r3 列——很可能是被单轮口径覆盖了")
        self.assertIn("McNemar", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
