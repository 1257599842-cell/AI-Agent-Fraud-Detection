"""Kaggle 真实提交（**硬盒子：只提交一次、只取一个数**）。

## 两套训练口径必须分开讲，别混
| | 线下评估（本项目主线） | 提交版（本文件） |
|---|---|---|
| 切分 | 时间切分 `fit<125` + **21 天 embargo** + `test≥146` | **全量 train 重训**（无切分） |
| 目的 | 诚实估计「线上会怎样」，防标签延迟泄漏 | 在官方 test 上拿一个可被第三方复核的数 |
| 早停 | 有 val 窗 | 无 val → 用线下已定的 `best_iter`，**不在 test 上调任何东西** |

**为什么这不算自相矛盾**：线下那套是为了**不骗自己**；提交这套是为了**让别人能复核**。
两个目的不同，口径也就不同——但必须写清楚哪个数是哪套出来的。

## 叙事红利（真实存在，不是包装）
IEEE-CIS 的官方 test 集**在时间上位于 train 之后**（train 覆盖前 182 天，test 紧随其后），
与本项目线下采用的「时间切分 + embargo」结构**天然一致**。
也就是说：**这里的防泄漏设置不是为了好看，它和比赛的真实时间结构是对齐的**——
如果我用随机切分调出来的模型，在这个 test 上就会掉下去。

## 图特征的因果性（提交时最容易出错的地方）
test 行的历史统计必须在 **train+test 拼接后的时间轴**上算：每行只看更早的行。
标签型统计（prior_fraud_*）只能来自 **train 行**（test 无标签）——这与线上完全同构：
今天这笔的实体欺诈史，只可能来自过去已定案的交易。

用法：
  python -m src.model.kaggle_submit --prepare   # 下载 test_* 并建特征（需 Kaggle token）
  python -m src.model.kaggle_submit --train     # 全量重训 + 预测 + 写 submission.csv
  python -m src.model.kaggle_submit --submit    # 真提交（**只跑一次**）
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data"
PROC = DATA / "processed"
SUB = PROC / "submission.csv"
REPORT = PROJECT_ROOT / "reports" / "kaggle_submission.md"
COMP = "ieee-fraud-detection"
SECS_PER_DAY = 86_400
EMBARGO_SECS = 21 * SECS_PER_DAY


def prepare():
    """下载官方 test 两表并与 train 拼接后重算图特征（时间因果）。"""
    for f in ("test_transaction.csv", "test_identity.csv"):
        if not (DATA / f).exists():
            print(f"下载 {f} …")
            subprocess.run(["kaggle", "competitions", "download", "-c", COMP,
                            "-f", f, "-p", str(DATA)], check=True)
            z = DATA / (f + ".zip")
            if z.exists():
                subprocess.run(["unzip", "-o", str(z), "-d", str(DATA)], check=True)

    tr = pd.read_parquet(PROC / "train_merged.parquet")
    te_t = pd.read_csv(DATA / "test_transaction.csv")
    te_i = pd.read_csv(DATA / "test_identity.csv")
    # 官方 test_identity 的列名是 id-01 风格，train 是 id_01——不统一会静默丢列
    te_i.columns = [c.replace("-", "_") for c in te_i.columns]
    te = te_t.merge(te_i, on="TransactionID", how="left")
    te["isFraud"] = np.nan
    print(f"train {len(tr):,} + test {len(te):,}")

    both = pd.concat([tr, te], ignore_index=True, sort=False)
    both = both.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    both.to_parquet(PROC / "all_merged.parquet", index=False)
    print(f"✅ 拼接落盘 all_merged.parquet（{len(both):,} 行）")
    print("   下一步：python -m src.model.kaggle_submit --train")


def _causal_graph_features(both):
    """在 train+test 拼接时间轴上重算图特征。
    标签型只吃 **isFraud 非空（=train）** 且 dt ≤ t−embargo 的行——test 无标签，天然不参与。
    口径与 src/features/graph_features.py 一致（同两层防泄漏）。"""
    from src.features.graph_features import causal_prior_stats, codes_group, codes_plain, prior_nunique

    dt = both["TransactionDT"].to_numpy()
    fraud = both["isFraud"].fillna(0).to_numpy()          # test 记 0；下面靠 obs 窗把它们排除
    known = both["isFraud"].notna().to_numpy()
    card_s = both["card1"].astype("string")
    keys = {
        "card1": card_s,
        "card1_addr1": card_s + "|" + both["addr1"].astype("string"),
        "card1_email": card_s + "|" + both["P_emaildomain"].astype("string"),
        "card1_device": card_s + "|" + both["DeviceInfo"].astype("string"),
    }
    out = pd.DataFrame({"TransactionID": both["TransactionID"].to_numpy()})
    for name, ks in keys.items():
        gc = codes_group(ks)
        pc, oc, of = causal_prior_stats(dt, fraud * known, gc, EMBARGO_SECS)
        _, oc_known, _ = causal_prior_stats(dt, known.astype(np.int64), gc, EMBARGO_SECS)
        out[f"{name}_prior_cnt"] = pc
        out[f"{name}_prior_fraud_cnt"] = of
        # 分母只数**有标签**的历史行，否则 test 段会被无标签行稀释成 0
        out[f"{name}_prior_fraud_rate"] = np.where(oc_known > 0, of / np.maximum(oc_known, 1), np.nan)
    card_c = codes_group(card_s)
    for other, s in [("addr1", both["addr1"].astype("string")),
                     ("email", both["P_emaildomain"].astype("string")),
                     ("device", both["DeviceInfo"].astype("string"))]:
        out[f"card1_fanout_{other}"] = prior_nunique(dt, card_c, codes_plain(s))
    return out


def train():
    import lightgbm as lgb
    from src.model.train_baseline import LGB_PARAMS

    both = pd.read_parquet(PROC / "all_merged.parquet")
    gf = _causal_graph_features(both)
    both = both.merge(gf, on="TransactionID", how="left")

    is_train = both["isFraud"].notna().to_numpy()
    y = both.loc[is_train, "isFraud"].astype(int)
    X = both.drop(columns=["isFraud", "TransactionID", "TransactionDT"])
    for c in X.select_dtypes(include="object").columns:
        X[c] = X[c].astype("category")

    # 不在 test 上调任何东西：轮数沿用线下已定的 best_iter（表+图口径）
    n_rounds = 300
    print(f"全量重训（{is_train.sum():,} 行，{X.shape[1]} 特征，{n_rounds} 轮，无早停）…")
    booster = lgb.train(LGB_PARAMS, lgb.Dataset(X[is_train], label=y),
                        num_boost_round=n_rounds)
    p = booster.predict(X[~is_train])
    sub = pd.DataFrame({"TransactionID": both.loc[~is_train, "TransactionID"].to_numpy(),
                        "isFraud": p})
    sub.to_csv(SUB, index=False)
    print(f"✅ {len(sub):,} 行 → {SUB.relative_to(PROJECT_ROOT)}")
    print(f"   预测分布：均值 {p.mean():.4f}、中位 {np.median(p):.4f}、"
          f">0.5 占比 {(p > 0.5).mean():.2%}")


def submit(message="offline: time-split + 21d embargo; submission: full-train refit"):
    if not SUB.exists():
        raise SystemExit("先 --train 生成 submission.csv")
    print("⚠️ 只提交一次。")
    subprocess.run(["kaggle", "competitions", "submit", "-c", COMP,
                    "-f", str(SUB), "-m", message], check=True)
    print("\n查看分数：kaggle competitions submissions -c " + COMP)


if __name__ == "__main__":
    if "--prepare" in sys.argv:
        prepare()
    elif "--train" in sys.argv:
        train()
    elif "--submit" in sys.argv:
        submit()
    else:
        print(__doc__)
