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
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MANIFEST = REPORTS / "_manifest.json"


def sha_full(p):
    """整文件哈希 —— 冻结件用它（冻结件没有机器区/人写区之分）。"""
    return hashlib.sha256(p.read_bytes()).hexdigest()


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
        """任何手工编辑都会在这里现形。**四档各有各的哈希口径**：

        报告 → 只算机器区；冻结件 → 整文件；归档目录 → 目录聚合树哈希。
        用同一把尺量三种东西，是上一版测试红成一片的原因。
        """
        from src.eval.report_manifest import tree_sha
        for name, rec in sorted(self.man.items()):
            tier = rec.get("tier")
            with self.subTest(entry=name, tier=tier):
                if tier == "archive":
                    d = ROOT / name.rstrip("/")
                    self.assertTrue(d.is_dir(), f"{name} 归档目录缺失")
                    self.assertEqual(tree_sha(d)[0], rec["sha256_tree"],
                                     f"{name} 归档内容变了——原始返回不该被改动")
                elif tier == "frozen":
                    self.assertEqual(sha_full(ROOT / name), rec["sha256"],
                                     f"{name} 是冻结件，内容变了即毁掉其时间声明")
                else:
                    p = REPORTS / name
                    self.assertTrue(p.exists(), f"{name} 缺失")
                    self.assertEqual(
                        sha(p), rec["sha256_machine_only"],
                        f"\n{name} 与清单不符——它被手编过，或生成器改了没重跑。"
                        f"\n重生成：{rec['command']}"
                        f"\n刷新清单：python -m src.eval.report_manifest --update")

    def test_every_entry_records_a_runnable_command(self):
        """出处不是「大概是某个脚本生成的」，得是一条能直接复制运行的命令。

        归档与冻结件例外：它们**本来就不该被重跑**，命令位写的是「为什么不可复现」。
        """
        for name, rec in sorted(self.man.items()):
            if rec.get("tier") in ("archive", "frozen"):
                with self.subTest(entry=name):
                    self.assertTrue(rec.get("why"), "归档/冻结件必须写明理由")
                continue
            with self.subTest(report=name):
                # 允许 `VAR=... python -m ...`：环境变量也是口径的一部分
                # （e60 那份就是靠 GRAPH_FILE 切换的，命令里不带就重跑成 21 天）
                self.assertRegex(rec["command"], r"^(\w+=\S+ )*python -m (src|notebooks)\.")
                self.assertIn(rec["tier"], ("cheap", "heavy", "api", "frozen"))

    def test_manifest_and_generator_table_agree(self):
        """清单与 `GENERATORS` 表不得脱节——脱节意味着有报告没人认领。"""
        from src.eval.report_manifest import GENERATORS
        reports_only = {n for n, r in self.man.items()
                        if r.get("tier") not in ("archive", "frozen")}
        self.assertEqual(reports_only, set(GENERATORS) & reports_only)
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


class TestFrozenArtifacts(unittest.TestCase):
    """承载**时间声明**的文件：盲标表 / 预注册 / 评分前 rubric。

    判据一句话：**该文件是否声称写于某时点之前、是否承载「当时不知道结果」。**
    重写它毁掉的不是内容，是那个声明本身——而声明正是这些数字的全部效力来源。

    此前只有 `chmod 444` 拦着，**是权限位拦住的、不是设计拦住的**；
    权限位会在 clone / 打包 / 复制时丢失。现在判断在代码里。
    """

    def test_write_report_refuses_every_frozen_path(self):
        from src.report_io import FROZEN, FrozenArtifactError, write_report
        self.assertGreaterEqual(len(FROZEN), 6, "冻结件登记数下降了——是不是被误删？")
        for rel in FROZEN:
            with self.subTest(artifact=rel):
                with self.assertRaises(FrozenArtifactError):
                    write_report(ROOT / rel, "覆盖尝试")

    def test_every_frozen_entry_says_what_breaks(self):
        """理由要写「动了会毁掉什么」，不是「别动它」——后者不能让人自己判断。"""
        from src.report_io import FROZEN
        for rel, (when, why) in FROZEN.items():
            with self.subTest(artifact=rel):
                self.assertRegex(when, r"^20\d\d-\d\d-\d\d$")
                self.assertGreater(len(why), 25, "理由太短，多半只写了『别动』")

    def test_frozen_files_still_match_recorded_digest(self):
        from src.report_io import frozen_digests
        for rel, rec in frozen_digests().items():
            with self.subTest(artifact=rel):
                self.assertEqual(sha_full(ROOT / rel), rec["sha256"])

    @classmethod
    def setUpClass(cls):
        if MANIFEST.exists():
            cls.man = json.loads(MANIFEST.read_text(encoding="utf-8"))


class TestManifestMetadataIsItselfChecked(unittest.TestCase):
    """**执法工具自己也要被执法。**

    登记的生成器名曾有 1 处是错的、3 处指向不存在的文件——
    一个自己没被核实过的检查器比没有检查器更坏：它让人以为已经查过了。
    """

    def test_no_generator_entry_is_broken(self):
        from src.eval.report_manifest import validate_generators
        problems, _ = validate_generators()
        self.assertEqual(problems, [], f"清单登记的生成器有问题：{problems}")

    def test_unverified_entries_are_labelled_not_blank(self):
        """无法断言的必须标 `[未验证]`——留空看起来像通过。"""
        if not MANIFEST.exists():
            self.skipTest("清单未生成")
        man = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for name, rec in man.items():
            if rec.get("tier") in ("archive", "frozen"):
                continue
            with self.subTest(report=name):
                self.assertIn("generator_verified", rec)
                self.assertIn(rec["generator_verified"], (True, "[未验证]"))


class TestModelCardIsNotStale(unittest.TestCase):
    """**文档过期要被自动抓到，不靠人记。**

    来历：`§13「已推演但未实现」` 两次留着已经做成的能力（step-up 拖了 5 天、
    在线特征服务拖了 1 天），表头的版本与日期也停在 5 天前。

    根因是结构性的：**`§13` 是全文唯一一个「做成了要回来删东西」的清单**，
    其余章节都是往里加——所以只有它会系统性地漏。
    项目负责人为此定了「本节随能力落地逐条重划」的规矩；本测试是那条规矩的执行端。

    判据：文档日期不得早于它所引用的任何一份报告的**生成时间**。
    报告是机器写的、动过就会更新 mtime；文档是人写的、改不改全凭记得。
    **拿会自动更新的那个，去卡不会自动更新的那个。**
    """

    HEADER_DATE = re.compile(r"^\|\s*文档日期\s*\|\s*(\d{4}-\d{2}-\d{2})\s*\|", re.M)

    @classmethod
    def setUpClass(cls):
        cls.card = ROOT / "MODEL_CARD.md"
        if not cls.card.exists():
            raise unittest.SkipTest("MODEL_CARD.md 不存在")
        cls.text = cls.card.read_text(encoding="utf-8")

    def test_header_declares_a_document_date(self):
        self.assertIsNotNone(self.HEADER_DATE.search(self.text),
                             "表头缺「文档日期」——没有它就无从判断是否过期")

    def test_document_date_is_not_older_than_its_sources(self):
        """引用的报告比文档新 → 文档在讲一份已经变过的数据。"""
        import datetime as dt
        doc_day = dt.date.fromisoformat(self.HEADER_DATE.search(self.text).group(1))
        newest, who = None, None
        for name in sorted(set(re.findall(r"`?([a-z0-9_]+\.md)`?", self.text))):
            p = REPORTS / name
            if not p.exists():
                continue
            d = dt.date.fromtimestamp(p.stat().st_mtime)
            if newest is None or d > newest:
                newest, who = d, name
        if newest is None:
            self.skipTest("MODEL_CARD 未引用任何 reports/ 文件")
        self.assertGreaterEqual(
            doc_day, newest,
            f"\n文档日期 {doc_day} 早于它引用的报告 {who}（{newest}）。"
            f"\n报告重跑过而文档没跟上 —— 要么复核后更新表头日期，"
            f"要么说明为何本次重跑不影响结论。")

    def test_version_field_covers_every_layer_that_has_one(self):
        """有版本号的层必须都出现在表头 —— 少一层就是「往低了说自己」。"""
        m = re.search(r"^\|\s*版本\s*\|(.+)\|", self.text, re.M)
        self.assertIsNotNone(m, "表头缺「版本」")
        ver = m.group(1)
        # §12 变更记录里出现过的层，表头都该有
        layers = set(re.findall(r"^\|\s*(?:Agent|Serving|ML 核心)\s*\*?\*?([\w.-]+)",
                                self.text, re.M))
        for tag in ("v4-citable-context", "v1-online-scoring"):
            if tag in self.text:
                self.assertIn(tag, ver, f"§12 记了 {tag}，表头版本却没有它")
