"""硬点 ②：两层防泄漏（结构型 / 标签型）。

**这是全项目最贵的一条**。泄漏不会报错、不会崩、线下指标只会变**好看**——
所以它必须由测试守住，不能靠人记得。两层的区别是：

  结构型（label_based=False）：只要求发生在 t **之前**        → window[1] ≤ as_of
  标签型（label_based=True）：还要多留 21 天标签成熟期        → window[1] ≤ as_of − 21d

为什么标签型要多留：**拒付要 21 天才回来**。用 t 时刻还没回来的标签去预测 t，
等于拿未来预测过去——线下虚高、线上崩。

本文件既测 `audit_time_boundary` 这个审计器本身（**审计器错了比没有审计更糟**），
也拿**已发布的演示数据包**当真实样本回归——那 7 笔案例是要给评审者看的，
它们的时间纪律必须在每次改动后仍然成立。
"""

import json
import unittest
from pathlib import Path

from src.agent.tools import EMBARGO_SECS, Fact, FactRegistry, audit_time_boundary

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "reports" / "demo" / "demo_data.json"
DAY = 86400


def mkfact(fid, end, label_based, start=0):
    return Fact(fact_id=fid, type="t", entity="e", value=1,
                window=(start, end), label_based=label_based)


class TestEmbargoConstant(unittest.TestCase):
    def test_embargo_is_exactly_21_days(self):
        """21 天这个数贯穿全项目（特征、规则库、案例库、演示页说明行）。改它要连坐。"""
        self.assertEqual(EMBARGO_SECS, 21 * DAY)


class TestAuditor(unittest.TestCase):
    """先证审计器本身是对的，再拿它去审别人。"""

    AS_OF = 100 * DAY

    def test_structural_may_touch_as_of_but_not_beyond(self):
        self.assertEqual(audit_time_boundary([mkfact("S", self.AS_OF, False)], self.AS_OF), [])
        self.assertEqual(len(audit_time_boundary(
            [mkfact("S", self.AS_OF + 1, False)], self.AS_OF)), 1)

    def test_label_based_must_stop_21_days_early(self):
        limit = self.AS_OF - EMBARGO_SECS
        self.assertEqual(audit_time_boundary([mkfact("L", limit, True)], self.AS_OF), [])
        self.assertEqual(len(audit_time_boundary(
            [mkfact("L", limit + 1, True)], self.AS_OF)), 1)

    def test_label_based_is_strictly_stricter_than_structural(self):
        """同一个窗口，结构型合法、标签型违规——两层不是同一把尺子。

        若哪天有人把两者合并成一条规则，这个测试会炸。
        """
        end = self.AS_OF - EMBARGO_SECS + DAY      # 距 as_of 20 天
        self.assertEqual(audit_time_boundary([mkfact("S", end, False)], self.AS_OF), [])
        self.assertEqual(len(audit_time_boundary([mkfact("L", end, True)], self.AS_OF)), 1)

    def test_reports_every_violation_not_just_the_first(self):
        """漏报比误报更危险：必须把违规**全部**列出，不能命中一条就返回。"""
        bad = [mkfact(f"L{i}", self.AS_OF, True) for i in range(4)]
        self.assertEqual(len(audit_time_boundary(bad, self.AS_OF)), 4)

    def test_violation_message_names_the_fact_and_the_kind(self):
        """审计信息要能直接定位——只说「有违规」等于没说。"""
        msg = audit_time_boundary([mkfact("L_X", self.AS_OF, True)], self.AS_OF)[0]
        self.assertIn("L_X", msg)
        self.assertIn("标签型", msg)


class TestPolicyAndNullFactsSatisfyTheContract(unittest.TestCase):
    """两类特殊事实必须**天然过审计**，而不是靠给它们开豁免。

    开豁免 = 在审计器里加 if，那条 if 迟早会挡住真正的泄漏。
    """

    def test_policy_facts_are_structural_at_time_zero(self):
        from src.agent.tools import policy_facts
        reg = FactRegistry()
        facts = policy_facts(reg, {"c_fp": 25.0, "c_review": 5.0})
        self.assertTrue(facts, "policy_facts 没产出任何事实")
        for f in facts:
            self.assertFalse(f.label_based)
            self.assertEqual(f.window, (0, 0))
        self.assertEqual(audit_time_boundary(facts, as_of=1), [])

    def test_null_fact_keeps_the_declared_window(self):
        """缺席事实必须**照实报窗口**。

        剥夺实验里模型编出 `GRAPH_000`/`RULE_000`，根因正是某路径没按契约给出缺席事实
        （静默失约）——模型不是撒谎，是合同没兑现。
        """
        from src.agent.tools import null_fact
        reg = FactRegistry()
        as_of = 100 * DAY
        f = null_fact(reg, "GRAPH", "card1=1", "无记录",
                      (0, as_of - EMBARGO_SECS), label_based=True)
        self.assertIn(f.fact_id, reg.known_ids())
        self.assertEqual(audit_time_boundary([f], as_of), [])


class TestPublishedDemoDataIsClean(unittest.TestCase):
    """已发布的演示数据包：7 笔案例的全部事实都要过时间纪律。

    这是**回归**测试，不是单元测试——它守的是「给评审者看的那份东西没被改坏」。
    """

    @classmethod
    def setUpClass(cls):
        if not DEMO.exists():
            raise unittest.SkipTest("demo_data.json 不存在")
        cls.doc = json.loads(DEMO.read_text(encoding="utf-8"))

    def _as_of(self, case):
        """交易时点 = 结构型事实窗口右端的最大值（结构型可取到 t 本身）。"""
        ends = [f["window"][1] for f in case["facts"] if not f["label_based"]]
        return max(ends) if ends else None

    def test_every_case_passes_the_time_audit(self):
        checked = 0
        for case in self.doc["cases"]:
            if not case["facts"]:
                continue                      # 闸门放行那笔没进 Agent，无事实
            as_of = self._as_of(case)
            facts = [mkfact(f["fact_id"], f["window"][1], f["label_based"],
                            f["window"][0]) for f in case["facts"]]
            with self.subTest(case=case["key"]):
                self.assertEqual(audit_time_boundary(facts, as_of), [])
            checked += 1
        self.assertGreaterEqual(checked, 5, "抽到的案例太少，回归没有意义")

    def test_label_based_facts_are_at_least_21_days_stale(self):
        """演示页那行固定说明「**至少**早 21 天」必须在数据上为真。

        措辞是「至少」而非「恰好」：本项目最小间隔恰为 21.0 天，
        但另有早至 44 天的（CASE_003）——写死「早 21 天」会被页面上现成的反例证伪。
        """
        seen_exactly_21 = False
        for case in self.doc["cases"]:
            if not case["facts"]:
                continue
            as_of = self._as_of(case)
            for f in case["facts"]:
                if not f["label_based"]:
                    continue
                gap = (as_of - f["window"][1]) / DAY
                with self.subTest(case=case["key"], fact=f["fact_id"]):
                    self.assertGreaterEqual(gap, 21 - 1e-9)
                if abs(gap - 21) < 1e-6:
                    seen_exactly_21 = True
        self.assertTrue(seen_exactly_21,
                        "没有任何事实恰好卡在 21 天——说明 embargo 可能没在生效")

    def test_no_case_reports_a_time_audit_violation(self):
        """管道自己记录的违规清单也必须为空（与上面互为交叉验证）。"""
        for case in self.doc["cases"]:
            with self.subTest(case=case["key"]):
                self.assertEqual(case["acceptance"]["time_audit_violations"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestVelocityTimeDiscipline(unittest.TestCase):
    """Velocity 是**结构型但需要 RANGE 帧**的特征 —— 两层纪律的一个反例。

    它撞破了「结构型→ROWS、标签型→RANGE」这个曾被当成规律的对应：
    帧型（位置 vs 取值）与 embargo（标签是否成熟）是两件**正交**的事。
    """

    @staticmethod
    def _count(dt, codes, window):
        import numpy as np
        from src.features.velocity_features import velocity_counts
        return velocity_counts(np.asarray(dt), np.asarray(codes), window)

    def test_counts_only_strictly_earlier_transactions(self):
        """右端开区间：不含本笔，**也不含同一时刻的并列交易**——「同时」不是「之前」。"""
        got = self._count([0, 100, 100, 200], [1, 1, 1, 1], 1000)
        self.assertEqual(list(got), [0, 1, 1, 3])

    def test_window_excludes_transactions_older_than_the_window(self):
        """落在窗口之外的旧交易不该被数进来。"""
        got = self._count([0, 5000, 5100], [1, 1, 1], 3600)   # 1 小时窗
        self.assertEqual(list(got), [0, 0, 1])

    def test_entities_do_not_leak_into_each_other(self):
        got = self._count([10, 20, 30], [1, 2, 1], 3600)
        self.assertEqual(list(got), [0, 0, 1])

    def test_result_order_matches_input_order(self):
        """内部按 (组, 时间) 排序计算，必须还原成输入顺序——错位不会报错，只会静默给错值。"""
        got = self._count([300, 100, 200], [1, 1, 1], 3600)
        self.assertEqual(list(got), [2, 0, 1])

    def test_velocity_needs_no_embargo(self):
        """velocity 只数笔数、不读标签，因此**不需要 21 天 embargo**。

        若哪天有人「顺手统一」给它加上 embargo，1 小时窗会永远返回 0——
        这个测试会先炸掉。
        """
        got = self._count([0, 1800], [1, 1], 3600)
        self.assertEqual(list(got), [0, 1],
                         "1 小时窗内的前一笔没被数到——是不是误加了 embargo？")
