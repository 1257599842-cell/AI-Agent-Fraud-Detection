"""针对同分KS、分箱分母、时间隔离与评分刻度的回归检查。"""
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from src.model.bank_metrics import ks, decile_table, validate_scores
from src.model.scorecard import (apply_bins, bin_numeric, bin_categorical, fit_scorecard,
                                to_woe_frame, build_card, scale, expected_loss)


class BankMetricsTests(unittest.TestCase):
    def test_tied_scores_have_no_discrimination(self):
        for y in ([1, 0], [0, 1]):
            self.assertEqual(ks(y, [.5, .5])[0], 0)

    def test_threshold_matches_roc_and_is_order_invariant(self):
        rng = np.random.default_rng(47)
        y = rng.integers(0, 2, 300)
        p = rng.integers(0, 12, 300) / 12
        fpr, tpr, _ = roc_curve(y, p)
        target = float(max(tpr - fpr))
        for _ in range(5):
            ix = rng.permutation(len(y))
            value, threshold, count = ks(y[ix], p[ix])
            self.assertAlmostEqual(value, target)
            self.assertEqual(count, int((p >= threshold).sum()))

    def test_reverse_ranking_uses_origin(self):
        self.assertEqual(ks([1, 0], [0, 1]), (0, float('inf'), 0))

    def test_invalid_inputs_fail(self):
        for y, p in [([], []), ([1], [.5]), ([0, 2], [.1, .2]),
                     ([0, 1], [.1, np.nan]), ([0, 1], [.1]), ([[0, 1]], [[.1, .2]])]:
            with self.subTest(y=y), self.assertRaises(ValueError):
                ks(y, p)

    def test_decile_conserves_counts(self):
        y = np.tile([1, 0, 0], 37)
        out = decile_table(y, np.linspace(0, 1, len(y)))
        self.assertEqual(len(out), 10)
        self.assertEqual(out['样本'].sum(), len(y))
        self.assertEqual(out['坏'].sum(), y.sum())
        self.assertEqual(out.iloc[-1]['累计坏占比'], 1)
        self.assertEqual(out.iloc[-1]['累计好占比'], 1)
        with self.assertRaises(ValueError):
            decile_table([0, 1], [.1, .2], 10)

    def test_cache_rejects_different_protocol_and_duplicates(self):
        d = pd.DataFrame({'TransactionID': range(6), 'day': [0, 131, 132, 145, 146, 181],
                          'split': ['fit']*2+['val']*2+['test']*2,
                          'isFraud': [0, 1]*3, 'p_tab': [.1, .9]*3, 'p_graph': [.2, .8]*3})
        validate_scores(d)
        for col, value in [('split', 'test'), ('TransactionID', 1), ('day', 182), ('p_tab', 1.1)]:
            bad = d.copy(); bad.loc[0, col] = value
            with self.subTest(col=col), self.assertRaises(ValueError):
                validate_scores(bad)


class ScorecardTests(unittest.TestCase):
    def test_numeric_minimum_uses_all_training_rows(self):
        x = pd.Series([np.nan]*800 + list(range(200)))
        y = np.tile([0, 1], 500)
        cuts, mono = bin_numeric(x, y)
        idx = apply_bins(x, {'kind': 'num', 'cuts': cuts})
        self.assertTrue(mono)
        self.assertTrue(all(int((idx == b).sum()) >= 50 for b in set(idx) if b != -1))
        self.assertEqual(int((idx == -1).sum()), 800)

    def test_categorical_small_other_is_merged(self):
        x = pd.Series(['a']*940 + ['b']*50 + ['c']*10)
        keep = bin_categorical(x, np.zeros(1000))
        idx = apply_bins(x, {'kind': 'cat', 'keep': keep})
        self.assertTrue(all(int((idx == b).sum()) >= 50 for b in set(idx)))

    def test_future_labels_do_not_change_woe(self):
        df = pd.DataFrame({'day': [0]*100+[150]*10, 'x': list(range(100))+[999]*10,
                           'isFraud': [0]*50+[1]*50+[0]*10})
        first = fit_scorecard(df, ['x'])
        df.loc[df.day >= 146, 'isFraud'] = 1
        self.assertEqual(first, fit_scorecard(df, ['x']))

    def test_missing_and_unseen_bins_are_explicit_neutral_fallback(self):
        specs = {'x': {'kind': 'cat', 'keep': ['a']}}
        woes = {'x': {0: {'woe': .5, 'n': 100, 'bad_rate': .1}}}
        lr = SimpleNamespace(coef_=np.array([[.8]]), intercept_=np.array([-2.]))
        df = pd.DataFrame({'x': ['a', 'unseen', None]})
        W = to_woe_frame(df, np.ones(3, dtype=bool), specs, woes)
        np.testing.assert_array_equal(W.x, [.5, 0, 0])
        card, base = build_card(specs, woes, ['x'], lr, 1)
        self.assertEqual(set(card['分箱']), {'a', '其他', '缺失'})
        self.assertTrue((card.loc[card['样本'] == 0, '分值'] == 0).all())
        factor, offset = scale(None, None, 1)
        self.assertAlmostEqual(base + card.loc[card['分箱'] == 'a', '分值'].iloc[0],
                               offset-factor*(-2+.8*.5))

    def test_score_scale_and_pdo(self):
        factor, offset = scale(None, None, 1)
        self.assertAlmostEqual(offset+factor*np.log(50), 600)
        self.assertAlmostEqual(offset+factor*np.log(100), 620)

    def test_loss_uses_label_and_amount_with_strict_threshold(self):
        loss, count = expected_loss([0, 1, 1], [.9, .1, .5], [25, 100, 25])
        self.assertEqual(count, 1)
        self.assertEqual(loss, 150)


if __name__ == '__main__':
    unittest.main()
