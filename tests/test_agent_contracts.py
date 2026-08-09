"""硬点 ⑦⑨：Agent 层的几条契约 —— 全部是**曾经被烧过**的地方。

本文件不测「Agent 聪明不聪明」，只测**代码承诺的东西有没有兑现**。
选题标准很窄：**每一条都对应一次真实事故**。

  1. `_parse_gold` 跨块顺延      —— 漏填条目被悄悄安上邻条的标注（静默替代）
  2. `enforce_evidence_floor`    —— 管道改写模型输出时**必须留痕**（绝不静默）
  3. `validate_report`           —— 引用不存在的 fact_id 必须被逮住（剥夺实验里模型编过 GRAPH_000）
  4. `classify_grounding`        —— 三分类的边界：编造 vs 只是没挂上引用

不测 LLM 调用，**零 API 花费**。
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.agent.schema import validate_report
from src.agent.tools import Fact, FactRegistry


class _Reg:
    """最小 registry 替身：只需 all_facts() / known_ids()。"""

    def __init__(self, facts):
        self._f = list(facts)

    def all_facts(self):
        return self._f

    def known_ids(self):
        return {f.fact_id for f in self._f}


def mkfact(fid, label_based=False, value=1.0):
    return Fact(fact_id=fid, type="t", entity="e", value=value,
                window=(0, 0), label_based=label_based)


class TestGoldParserDoesNotBleedAcrossBlocks(unittest.TestCase):
    """事故复现：`.*?` + DOTALL 让漏填条目一路吃到下一条的答案上。

    后果不是报错，是**读数悄悄变了**（arm3 一度读成 8/53 而非 9/54，约 2pp）。
    **漏填就该缺席**，由调用方报缺——这就是「读不到就报缺」的原点。
    """

    def _parse(self, text):
        from src.eval.agent_eval import _parse_gold
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "gold.md"
            p.write_text(text, encoding="utf-8")
            return _parse_gold(p)

    def test_unfilled_entry_is_absent_not_inherited(self):
        got = self._parse(
            "## 111#0\nreasoning_valid:\noverclaim:\n\n"      # 漏填
            "## 222#1\nreasoning_valid: Y\noverclaim: N\n")   # 已填
        self.assertNotIn((111, 0), got, "漏填条目被顺延安上了邻条的标注")
        self.assertEqual(got.get((222, 1)), (True, False))

    def test_partially_filled_entry_is_also_absent(self):
        """只填了一半也算缺席——半条记录不能当整条用。"""
        got = self._parse("## 111#0\nreasoning_valid: Y\noverclaim:\n"
                          "## 222#1\nreasoning_valid: N\noverclaim: Y\n")
        self.assertNotIn((111, 0), got)
        self.assertEqual(got.get((222, 1)), (False, True))

    def test_accepts_both_yn_and_truefalse(self):
        got = self._parse("## 1#0\nreasoning_valid: true\noverclaim: false\n"
                          "## 2#0\nreasoning_valid: N\noverclaim: Y\n")
        self.assertEqual(got[(1, 0)], (True, False))
        self.assertEqual(got[(2, 0)], (False, True))

    def test_missing_file_yields_empty_not_crash(self):
        from src.eval.agent_eval import _parse_gold
        self.assertEqual(_parse_gold(Path("/nonexistent/gold.md")), {})


class TestEvidenceFloorNeverActsSilently(unittest.TestCase):
    """管道可以改写模型输出，但**每一次改写都必须落一条 override**。

    一个悄悄改写模型输出的管道，会是本项目「静默替代」的第四次。
    """

    def setUp(self):
        from src.agent.pipeline import enforce_evidence_floor
        self.enforce = enforce_evidence_floor

    def test_r1_forces_insufficient_when_no_label_based_fact(self):
        rep = {"evidence_insufficient": False, "key_findings": []}
        out, ov = self.enforce(rep, _Reg([mkfact("TXN_000", label_based=False)]))
        self.assertTrue(out["evidence_insufficient"])
        self.assertEqual(len(ov), 1)
        self.assertIn("R1", ov[0])

    def test_r1_stays_quiet_when_a_label_based_fact_exists(self):
        rep = {"evidence_insufficient": False, "key_findings": []}
        out, ov = self.enforce(rep, _Reg([mkfact("GRAPH_000", label_based=True)]))
        self.assertFalse(out["evidence_insufficient"])
        self.assertEqual(ov, [])

    def test_r2_caps_uncited_finding_at_tentative(self):
        rep = {"evidence_insufficient": True,
               "key_findings": [{"finding": "x", "assertion_strength": "confirmed",
                                 "evidence_ids": []}]}
        out, ov = self.enforce(rep, _Reg([mkfact("TXN_000")]))
        self.assertEqual(out["key_findings"][0]["assertion_strength"], "tentative")
        self.assertTrue(any("R2" in o for o in ov))

    def test_r2_treats_unknown_fact_id_as_no_citation(self):
        """引了一个**不存在**的 fact_id，等于没引——这正是模型编 GRAPH_000 的形态。"""
        rep = {"evidence_insufficient": True,
               "key_findings": [{"finding": "x", "assertion_strength": "supported",
                                 "evidence_ids": ["GRAPH_999"]}]}
        out, ov = self.enforce(rep, _Reg([mkfact("TXN_000")]))
        self.assertEqual(out["key_findings"][0]["assertion_strength"], "tentative")
        self.assertTrue(any("R2" in o for o in ov))

    def test_r2_leaves_a_properly_cited_confirmed_finding_alone(self):
        """「本笔金额 $28.04」引 TXN_000 标 confirmed 是**合法**的，不得连坐降档。

        曾有一版建议「无标签型证据就全篇封顶」，实现时被否决——
        代码判不了的，就不该让代码判。
        """
        rep = {"evidence_insufficient": True,
               "key_findings": [{"finding": "本笔金额 $28.04",
                                 "assertion_strength": "confirmed",
                                 "evidence_ids": ["TXN_000"]}]}
        out, ov = self.enforce(rep, _Reg([mkfact("TXN_000")]))
        self.assertEqual(out["key_findings"][0]["assertion_strength"], "confirmed")
        self.assertEqual(ov, [])

    def test_none_report_is_survivable(self):
        out, ov = self.enforce(None, _Reg([]))
        self.assertIsNone(out)
        self.assertEqual(ov, [])


class TestSchemaValidator(unittest.TestCase):
    def _base(self):
        from src.agent.schema import EXAMPLE_REPORT
        return json.loads(json.dumps(EXAMPLE_REPORT))

    def test_example_report_is_valid(self):
        from src.agent.schema import EXAMPLE_KNOWN_IDS
        self.assertEqual(validate_report(self._base(), EXAMPLE_KNOWN_IDS), [])

    def test_unknown_evidence_id_is_rejected(self):
        """剥夺实验里模型编出 GRAPH_000/RULE_000 —— 校验器必须拦住这一类。"""
        from src.agent.schema import EXAMPLE_KNOWN_IDS
        rep = self._base()
        rep["key_findings"][0]["evidence_ids"] = ["NOPE_123"]
        self.assertTrue(validate_report(rep, EXAMPLE_KNOWN_IDS))

    def test_illegal_disposition_is_rejected(self):
        rep = self._base()
        rep["disposition"] = "freeze_account"
        self.assertTrue(validate_report(rep))

    def test_illegal_assertion_strength_is_rejected(self):
        rep = self._base()
        rep["key_findings"][0]["assertion_strength"] = "certain"
        self.assertTrue(validate_report(rep))

    def test_missing_required_field_is_rejected(self):
        for field in ("disposition", "risk_level", "key_findings"):
            with self.subTest(field=field):
                rep = self._base()
                rep.pop(field, None)
                self.assertTrue(validate_report(rep))


class TestGroundingClassifier(unittest.TestCase):
    """三分类的分界：**没挂引用**（卫生问题）与**编造**（幻觉）必须分开记账。

    把 citation-gap 记成幻觉会高估幻觉率；反过来则会掩盖真编造。
    """

    def _classify(self, finding, evidence_ids, facts, p=0.5):
        from src.eval.agent_eval import classify_grounding
        return classify_grounding(
            {"p": p, "facts": facts},
            {"finding": finding, "evidence_ids": evidence_ids})[0]

    def test_no_numbers_is_its_own_bucket(self):
        """不含数字的 finding **判不了**，必须单列，不能默默算作接地。"""
        self.assertEqual(
            self._classify("该实体画像可疑", [], []), "no_numbers")

    def test_number_from_cited_fact_is_fully_cited(self):
        facts = [{"fact_id": "G0", "type": "prior_fraud_rate",
                  "entity": "card1=1", "value": 0.0559}]
        self.assertEqual(
            self._classify("成熟窗欺诈率 5.59%", ["G0"], facts), "fully_cited")

    def test_number_only_in_pool_is_citation_gap_not_hallucination(self):
        """值是真的、只是没挂上引用 → 记**引用完整率**，不记幻觉率。"""
        facts = [{"fact_id": "G0", "type": "prior_fraud_rate",
                  "entity": "card1=1", "value": 0.0559},
                 {"fact_id": "G1", "type": "fanout_device",
                  "entity": "card1=1", "value": 375}]
        self.assertEqual(
            self._classify("设备扇出 375", ["G0"], facts), "citation_gap")

    def test_number_absent_from_whole_pool_is_true_ungrounded(self):
        facts = [{"fact_id": "G0", "type": "prior_fraud_rate",
                  "entity": "card1=1", "value": 0.0559}]
        self.assertEqual(
            self._classify("设备扇出 8123", ["G0"], facts), "true_ungrounded")


if __name__ == "__main__":
    unittest.main(verbosity=2)
