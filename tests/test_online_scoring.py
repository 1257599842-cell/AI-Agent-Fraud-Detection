"""在线打分：`/score` 必须**真的算**，不是查表。

原状：`/score` 接受 `transaction_id`，去离线表里查一个算好的 `p`。
「这个 API 是个查表」这句话当时成立——而且比「不是线上系统」严重，
因为后者已被诚实声明，前者没有。

本文件焊住修复后的三条契约：
  1. FeatureStore 的三种时间语义各自正确（位置 / 取值+标签成熟 / 短窗取值）；
  2. NULL 语义与离线对齐（`obs_cnt=0 → NULL`，不是 0）；
  3. 只有**一个**打分模型落盘 —— 两个模型 = 两条真理。

不需要数据集即可运行的部分用合成事件；需要数据的部分自动跳过。
"""

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _store():
    import pandas as pd
    from src.serving.feature_store import FeatureStore
    st = FeatureStore()
    # 三笔历史：同卡同设备，时间递增
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "TransactionDT": [0, 1000, 100_000],
        "isFraud": [1, 0, 0],
        "card1": ["C1", "C1", "C1"],
        "addr1": ["A1", "A1", None],
        "P_emaildomain": ["e@x", "e@x", "e@x"],
        "DeviceInfo": ["D1", "D2", "D1"],
    })
    st.append_frame(df)
    return st


class TestThreeTimeSemantics(unittest.TestCase):
    """位置 / 取值+标签成熟 / 短窗取值 —— 三种语义混用就会静默算错。"""

    def setUp(self):
        self.st = _store()

    def tearDown(self):
        self.st.close()

    def test_structural_uses_position_with_id_tiebreak(self):
        """`dt` 并列时靠 ID 打破 —— 复刻离线 `lexsort` 的稳定次序。"""
        txn = {"TransactionDT": 1000, "card1": "C1", "addr1": "A1",
               "P_emaildomain": "e@x", "DeviceInfo": "D2"}
        # tiebreak_id=2：id<2 的并列行（id=1，dt=0<1000）与更早的都算
        f = self.st.get_features(txn, tiebreak_id=2)
        self.assertEqual(f["card1_prior_cnt"], 1)
        # tiebreak_id=0：并列行 id 更大 → 不算
        f0 = self.st.get_features(txn, tiebreak_id=0)
        self.assertEqual(f0["card1_prior_cnt"], 1)   # dt=0 严格更早，与 tiebreak 无关

    def test_label_features_require_matured_labels(self):
        """标签型必须等 `label_time <= t`；未成熟的标签**不可见**。"""
        early = {"TransactionDT": 2000, "card1": "C1", "addr1": "A1",
                 "P_emaildomain": "e@x", "DeviceInfo": "D1"}
        f = self.st.get_features(early, tiebreak_id=99)
        # 历史里有 1 笔欺诈，但它的 label_time = 0 + 21 天 > 2000 → 还没成熟
        self.assertEqual(f["card1_prior_fraud_cnt"], 0)
        self.assertIsNone(f["card1_prior_fraud_rate"],
                          "obs_cnt=0 时必须是 NULL 而不是 0 —— 与离线对齐")

    def test_label_visible_after_embargo(self):
        """标签**逐笔**成熟，不是一起成熟 —— 分母只含已成熟的那些。"""
        from src.serving.feature_store import EMBARGO_SECS
        # t 刚过第 1 笔（dt=0）的成熟点：只有它可见 → 1 笔成熟、1 笔欺诈
        t1 = {"TransactionDT": EMBARGO_SECS + 1, "card1": "C1", "addr1": "A1",
              "P_emaildomain": "e@x", "DeviceInfo": "D1"}
        f1 = self.st.get_features(t1, tiebreak_id=99)
        self.assertEqual(f1["card1_prior_fraud_cnt"], 1)
        self.assertAlmostEqual(f1["card1_prior_fraud_rate"], 1.0)
        # t 再过第 2 笔（dt=1000，非欺诈）的成熟点：2 笔成熟、1 笔欺诈 → 0.5
        t2 = {"TransactionDT": EMBARGO_SECS + 1001, "card1": "C1", "addr1": "A1",
              "P_emaildomain": "e@x", "DeviceInfo": "D1"}
        f2 = self.st.get_features(t2, tiebreak_id=99)
        self.assertEqual(f2["card1_prior_fraud_cnt"], 1)
        self.assertAlmostEqual(f2["card1_prior_fraud_rate"], 0.5,
                               msg="分母应随标签逐笔成熟而增长")

    def test_velocity_is_value_based_without_tiebreak(self):
        """短窗速度是纯取值语义：`[t−W, t)`，不含本笔、不含并列。"""
        txn = {"TransactionDT": 1000, "card1": "C1", "addr1": "A1",
               "P_emaildomain": "e@x", "DeviceInfo": "D1"}
        f = self.st.get_features(txn, tiebreak_id=99)
        self.assertEqual(f["card1_velocity_1h"], 1)      # 只有 dt=0 落在 1 小时内
        self.assertEqual(f["card1_velocity_24h"], 1)

    def test_null_key_yields_empty_group_not_crash(self):
        """键缺失 → 该实体自成一组（与离线 codes_group 一致），不该抛异常。"""
        txn = {"TransactionDT": 5000, "card1": "C1", "addr1": None,
               "P_emaildomain": "e@x", "DeviceInfo": "D1"}
        f = self.st.get_features(txn, tiebreak_id=99)
        self.assertEqual(f["card1_addr1_prior_cnt"], 0)
        self.assertIsNone(f["card1_addr1_prior_fraud_rate"])


class TestSelectiveBiasLivesInTheSchema(unittest.TestCase):
    """`label_time` 可为 NULL —— 未复核就放行的交易**永远拿不到标签**。

    把它建成 nullable 之后，`prior_fraud_rate` 的分母天然只含被复核过的样本：
    项目里反复讨论的选择性偏差，在存储层是一个**可以指着看的字段**。
    """

    def test_unlabelled_events_never_enter_the_denominator(self):
        import pandas as pd
        from src.serving.feature_store import EMBARGO_SECS, FeatureStore
        st = FeatureStore()
        df = pd.DataFrame({
            "TransactionID": [1, 2],
            "TransactionDT": [0, 0],
            "isFraud": [1, 1],
            "card1": ["C1", "C1"], "addr1": ["A1", "A1"],
            "P_emaildomain": ["e", "e"], "DeviceInfo": ["D", "D"],
        })
        st.append_frame(df, labeled_mask=[True, False])     # 第 2 笔未被复核
        txn = {"TransactionDT": EMBARGO_SECS + 1, "card1": "C1", "addr1": "A1",
               "P_emaildomain": "e", "DeviceInfo": "D"}
        f = st.get_features(txn, tiebreak_id=99)
        self.assertEqual(f["card1_prior_fraud_cnt"], 1, "未复核的那笔不该进分子")
        self.assertAlmostEqual(f["card1_prior_fraud_rate"], 1.0,
                               msg="分母只应含被复核过的样本 —— 这就是偏差的来源")
        st.close()


class TestExactlyOneScoringModel(unittest.TestCase):
    """**两个模型 = 两条真理。**

    曾经落盘过纯表基线模型，而在线算的 15 列图特征它从没见过 —— 白算。
    """

    def test_only_the_canonical_model_is_persisted(self):
        mdir = ROOT / "models"
        if not mdir.exists():
            self.skipTest("模型未落盘")
        models = sorted(p.name for p in mdir.glob("*.txt"))
        self.assertEqual(models, ["scoring_model.txt"],
                         f"落盘了不止一个模型：{models}")

    def test_persisted_model_covers_the_online_features(self):
        """在线算出来的历史特征，必须**真的在模型的输入列里**。"""
        from src.serving.feature_store import FEATURE_COLUMNS
        fc = ROOT / "models" / "feature_columns.json"
        if not fc.exists():
            self.skipTest("模型未落盘")
        cols = set(json.loads(fc.read_text(encoding="utf-8")))
        graph = [c for c in FEATURE_COLUMNS if "velocity" not in c]
        missing = [c for c in graph if c not in cols]
        self.assertEqual(missing, [],
                         f"这些在线特征不在模型输入里，等于白算：{missing}")

    def test_categorical_levels_are_persisted(self):
        """类别层级不落盘 → 服务端单行预测直接报错（训练/线上不一致的第二种形态）。"""
        f = ROOT / "models" / "categorical_levels.json"
        if not (ROOT / "models" / "scoring_model.txt").exists():
            self.skipTest("模型未落盘")
        self.assertTrue(f.exists(), "类别层级未落盘")
        self.assertGreater(len(json.loads(f.read_text(encoding="utf-8"))), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
