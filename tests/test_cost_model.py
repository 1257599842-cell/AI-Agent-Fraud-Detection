"""防守点 ③④：代价敏感阈值与期望成本 argmin。

**为什么这些值得焊住**：整条链的结论都建立在「阈值不是拍 0.5、是解出来的」之上。
成本公式一旦被改动（哪怕只是笔误），所有历史数字——应然档、层1/层2、成本归因、
round 3/4——同时失去可复现性，而**不会有任何报错**。

测的是**真值不是快照**：期望成本按 AGENT_DESIGN 的公式手算出来对拍，
不是把当前输出录下来当基准。快照测试只能证明「没变」，证不了「本来就对」。
"""

import unittest

import numpy as np

from src.agent.disposition import ACTIONS, BASE, argmin_action, expected_costs, gang_score
from src.model.stepup import ACTIONS5, STEPUP, argmin5, costs5

A_MED = 76.02


class TestFourTierFormulas(unittest.TestCase):
    """四档公式逐项手算对拍（`disposition.py` 已冻结，任何改动都应在此炸掉）。"""

    def test_approve_is_expected_fraud_loss(self):
        e = expected_costs([0.3], [100.0], [0.0], A_MED, BASE)[0]
        self.assertAlmostEqual(e[ACTIONS.index("approve")], 0.3 * 100.0, places=10)

    def test_decline_is_independent_of_amount(self):
        """`E_decline = (1−p)·c_FP` **不含金额项**——拒绝 $10,000 与拒绝 $1 同价。

        这是一个建模选择（拒绝的代价只算误伤好客户），**不是笔误**。
        它直接导致「p=1 时拒绝免费」，进而推翻了「微额确定欺诈仍最优放行」那个表述。
        焊住它，免得日后有人"顺手修好"它而不知道后果。
        """
        i = ACTIONS.index("decline")
        for amt in (1.0, 100.0, 10_000.0):
            e = expected_costs([0.3], [amt], [0.0], A_MED, BASE)[0]
            self.assertAlmostEqual(e[i], (1 - 0.3) * BASE["c_fp"], places=10)

    def test_decline_is_free_at_certainty(self):
        """p=1 → 无误伤 → 拒绝成本为 0，且任何金额下 argmin 都是 decline。"""
        for amt in (0.01, 0.7143, 20.0, 5_000.0):
            e = expected_costs([1.0], [amt], [0.0], A_MED, BASE)[0]
            self.assertAlmostEqual(e[ACTIONS.index("decline")], 0.0, places=12)
            self.assertEqual(argmin_action([1.0], [amt], [0.0], A_MED, BASE)[0], "decline")

    def test_hold_and_escalate_match_hand_computation(self):
        p, a, g = 0.4, 250.0, 0.6
        e = expected_costs([p], [a], [g], A_MED, BASE)[0]
        want_hold = (BASE["c_review"] + p * BASE["m_h"] * a
                     + (1 - p) * BASE["f_h"] * BASE["c_fp"])
        want_esc = (BASE["c_report"] + p * BASE["m_e"] * a
                    + (1 - p) * BASE["f_e"] * BASE["c_fp"]
                    - p * g * BASE["k_future"] * A_MED)
        self.assertAlmostEqual(e[ACTIONS.index("hold")], want_hold, places=10)
        self.assertAlmostEqual(e[ACTIONS.index("escalate")], want_esc, places=10)

    def test_network_term_only_applies_to_escalate(self):
        """网络项**只**减 escalate；gang 变化不得动其余三档（防双重计数）。"""
        lo = expected_costs([0.4], [250.0], [0.0], A_MED, BASE)[0]
        hi = expected_costs([0.4], [250.0], [1.0], A_MED, BASE)[0]
        for act in ("approve", "hold", "decline"):
            self.assertAlmostEqual(lo[ACTIONS.index(act)], hi[ACTIONS.index(act)], places=12)
        self.assertLess(hi[ACTIONS.index("escalate")], lo[ACTIONS.index("escalate")])

    def test_argmin_agrees_with_elementwise_minimum(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(1e-4, 0.999, 500)
        a = 10 ** rng.uniform(0, 3.5, 500)
        g = rng.choice([0.0, 0.5, 1.0], 500)
        got = argmin_action(p, a, g, A_MED, BASE)
        want = np.asarray(ACTIONS)[expected_costs(p, a, g, A_MED, BASE).argmin(axis=1)]
        self.assertTrue((got == want).all())


class TestGangScore(unittest.TestCase):
    def test_small_sample_gate_blocks_one_of_one(self):
        """cnt < 2 一律 0：防 1/1=100% 的噪声率被当成团伙铁证。"""
        self.assertEqual(float(gang_score(50.0, 1.0, 1)), 0.0)
        self.assertGreater(float(gang_score(50.0, 1.0, 2)), 0.0)

    def test_saturates_at_one_and_nan_is_zero(self):
        self.assertAlmostEqual(float(gang_score(1e6, 1.0, 10)), 1.0, places=12)
        self.assertEqual(float(gang_score(np.nan, np.nan, np.nan)), 0.0)


class TestStepUpClosedForm(unittest.TestCase):
    """第五档的闭式解 —— Q-A 推导出来的那条，必须与数值 argmin 对得上。"""

    @staticmethod
    def _threshold(p):
        """放行优于加验证：a < c_f / [p·r_b − (1−p)·r_ab·r_m]。

        分母里的 `(1−p)·r_ab·r_m` 是**好客户因摩擦放弃**的代价。
        漏掉它（即用近似式 c_f/(p·r_b)）会低估临界金额，p 越小误差越大。
        """
        d = (p * STEPUP["r_block"]
             - (1 - p) * STEPUP["r_abandon"] * STEPUP["margin_rate"])
        return STEPUP["c_friction"] / d if d > 0 else float("inf")

    def _numeric_floor(self, p):
        lo, hi = 1e-9, 1e7
        if argmin5(np.array([p]), np.array([lo]), np.array([0.0]),
                   A_MED, BASE, STEPUP)[0] != "approve":
            return None
        for _ in range(120):
            m = (lo + hi) / 2
            nxt = argmin5(np.array([p]), np.array([m]), np.array([0.0]),
                          A_MED, BASE, STEPUP)[0]
            lo, hi = (m, hi) if nxt == "approve" else (lo, m)
        return (lo + hi) / 2

    def test_closed_form_matches_numeric_where_stepup_binds(self):
        for p in (0.05, 0.10, 0.30, 0.50, 0.70):
            with self.subTest(p=p):
                self.assertAlmostEqual(self._threshold(p), self._numeric_floor(p), places=6)

    def test_p030_floor_is_the_documented_2_44(self):
        """README / reports 引用的 $2.44 —— 变了就说明公式或参数被动过。"""
        self.assertAlmostEqual(self._threshold(0.30), 2.4420, places=3)

    def test_approve_never_optimal_at_certainty(self):
        """p=1 时下限不存在（拒绝免费）。**「微额确定欺诈仍放行」已作废，锁死。**"""
        self.assertIsNone(self._numeric_floor(1.0))


class TestFiveTierDoesNotDisturbFour(unittest.TestCase):
    """五档「并列呈现、不覆盖」：去掉 stepup 后必须逐笔退化回四档。"""

    def test_removing_stepup_recovers_four_tier(self):
        rng = np.random.default_rng(7)
        p = rng.uniform(1e-4, 0.999, 800)
        a = 10 ** rng.uniform(0, 3.5, 800)
        g = rng.choice([0.0, 0.5, 1.0], 800)
        c5 = costs5(p, a, g, A_MED, BASE, STEPUP)
        keep = [i for i, k in enumerate(ACTIONS5) if k != "stepup"]
        without = np.asarray([ACTIONS5[i] for i in keep])[c5[:, keep].argmin(axis=1)]
        self.assertTrue((without == argmin_action(p, a, g, A_MED, BASE)).all())

    def test_four_tier_costs_are_untouched_by_five_tier_module(self):
        rng = np.random.default_rng(11)
        p, a, g = rng.uniform(0, 1, 200), 10 ** rng.uniform(0, 3, 200), rng.uniform(0, 1, 200)
        c4 = expected_costs(p, a, g, A_MED, BASE)
        c5 = costs5(p, a, g, A_MED, BASE, STEPUP)
        for act in ACTIONS:
            np.testing.assert_allclose(c5[:, ACTIONS5.index(act)], c4[:, ACTIONS.index(act)],
                                       rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
