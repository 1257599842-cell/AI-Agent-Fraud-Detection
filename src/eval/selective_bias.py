"""选择性偏差演示（硬点⑤）—— 选择性标签 → AUC 衰减 → 随机抽检救回。

真实机制：生产里只有**被拦下复核**的交易拿得到标签；**放行**的交易若不主动查，
就被默认当好人（label 0）喂回训练。于是模型漏掉的欺诈被标成 legit → 越练越窄。

演示设计（时间流式自训练）：
  bootstrap：day<60 真标签，训初始模型。
  批次：[60,90) [90,120) [120,150)，逐批：模型打分 → 拦 top FLAG_FRAC（拿真标签），
        放行的 approved 按 mode 处理后并入训练池，重训。
  test：day>=150 真标签，固定不动，衡量循环把模型带偏多少。
  mode:
    naive    ：approved 一律当 legit(0) —— 选择性偏差全开。
    recovery ：随机抽 AUDIT_FRAC 的 approved 拿真标签（探索预算），其余当 0 —— 无偏锚定。

用法：python -m src.eval.selective_bias
产出：reports/figures/09_selective_bias.png + reports/selective_bias.md
"""

from src.report_io import write_report
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.model.train_baseline import prepare

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG = PROJECT_ROOT / "reports" / "figures" / "09_selective_bias.png"
OUT_MD = PROJECT_ROOT / "reports" / "selective_bias.md"

BOOT_END = 60
BATCHES = [(60, 90), (90, 120), (120, 150)]
TEST_LO = 150
FLAG_FRAC = 0.02       # 拦截（复核）比例
AUDIT_FRAC = 0.02      # 随机放行抽检比例（探索预算）
SEED = 42

# 演示用轻量配置（固定轮数、无早停，为了快；这步讲机制不追极致 AUC）
DEMO_PARAMS = {
    "objective": "binary", "num_leaves": 64, "learning_rate": 0.05,
    "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1,
    "min_child_samples": 100, "verbose": -1, "seed": SEED, "n_jobs": -1,
}
N_ROUNDS = 300


def train(Xp, yp):
    d = lgb.Dataset(Xp, label=yp)
    return lgb.train(DEMO_PARAMS, d, num_boost_round=N_ROUNDS)


def run_loop(X, y, day, mode, Xte, yte):
    rng = np.random.default_rng(SEED)
    boot = day < BOOT_END
    Xp = X[boot].copy()
    yp = y[boot].to_numpy().copy()
    model = train(Xp, yp)
    traj = [{"round": 0, "roc": roc_auc_score(yte, model.predict(Xte)),
             "pr": average_precision_score(yte, model.predict(Xte)),
             "poison": 0, "n_pool": len(yp)}]

    for r, (lo, hi) in enumerate(BATCHES, 1):
        m = (day >= lo) & (day < hi)
        Xb, yb_true = X[m], y[m].to_numpy()
        p = model.predict(Xb)
        thr = np.quantile(p, 1 - FLAG_FRAC)
        flagged = p >= thr
        yb_obs = np.zeros(len(yb_true), dtype=int)     # 默认放行=legit(0)
        yb_obs[flagged] = yb_true[flagged]             # 被拦的拿真标签
        if mode == "recovery":
            approved = ~flagged
            audit = approved & (rng.random(len(yb_true)) < AUDIT_FRAC)
            yb_obs[audit] = yb_true[audit]             # 抽检的拿真标签
        poison = int(((yb_obs == 0) & (yb_true == 1)).sum())   # 被误标成 legit 的真欺诈数
        Xp = _concat(Xp, Xb)
        yp = np.concatenate([yp, yb_obs])
        model = train(Xp, yp)
        traj.append({"round": r, "roc": roc_auc_score(yte, model.predict(Xte)),
                     "pr": average_precision_score(yte, model.predict(Xte)),
                     "poison": poison, "n_pool": len(yp)})
    return traj


def _concat(Xa, Xb):
    import pandas as pd
    return pd.concat([Xa, Xb], axis=0)


def main() -> None:
    print("读取数据 …")
    X, y, day = prepare()
    test = day >= TEST_LO
    Xte, yte = X[test], y[test].to_numpy()
    print(f"  test day>=150：{test.sum():,} 笔，欺诈 {int(yte.sum()):,}")

    print("\n跑 naive（放行全当 legit）…")
    naive = run_loop(X, y, day, "naive", Xte, yte)
    print("跑 recovery（+2% 随机放行抽检）…")
    recov = run_loop(X, y, day, "recovery", Xte, yte)

    for name, tr in [("naive", naive), ("recovery", recov)]:
        print(f"\n  {name}: " + "  ".join(f"r{d['round']}ROC={d['roc']:.4f}" for d in tr)
              + f"  (末轮误标欺诈 {tr[-1]['poison']})")

    _plot(naive, recov)
    _write_md(naive, recov, int(yte.sum()))
    print(f"\n✅ 图 09 + {OUT_MD.relative_to(PROJECT_ROOT)}")


def _plot(naive, recov):
    xs = [d["round"] for d in naive]
    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, [d["roc"] for d in naive], marker="o", color="#C44E52", label="naive (approved→legit)")
    plt.plot(xs, [d["roc"] for d in recov], marker="s", color="#4C72B0", label="recovery (+2% random audit)")
    plt.xlabel("Retraining round (streaming batches)")
    plt.ylabel("Test ROC-AUC (fixed true-label test)")
    plt.title("Selective-label bias: naive decay vs random-audit recovery")
    plt.xticks(xs)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG, dpi=120)
    plt.close()


def _write_md(naive, recov, n_fraud_test):
    n0, nL = naive[0]["roc"], naive[-1]["roc"]
    r0, rL = recov[0]["roc"], recov[-1]["roc"]
    L = [
        "# 选择性偏差演示（硬点⑤）—— 选择性标签 → 衰减 → 抽检救回\n",
        "机制：只有被拦复核的拿真标签；放行的默认当 legit 喂回训练 → 漏掉的欺诈被标成好人 → 模型越练越窄。",
        f"bootstrap day<60 → 批次 [60,90)[90,120)[120,150) → 固定真标签 test day>=150（欺诈 {n_fraud_test:,}）。",
        f"拦截 top {FLAG_FRAC:.0%}；recovery 额外随机抽检 {AUDIT_FRAC:.0%} 放行样本。\n",
        "## 每轮 test ROC-AUC\n",
        "| 轮 | naive ROC | naive 末轮误标欺诈 | recovery ROC |",
        "|----|-----------|-------------------|--------------|",
    ]
    for dn, dr in zip(naive, recov):
        L.append(f"| {dn['round']} | {dn['roc']:.4f} | {dn['poison']} | {dr['roc']:.4f} |")
    L += [
        "",
        "## 结论（按实际数字）",
        f"- naive 衰减：ROC {n0:.4f} → {nL:.4f}（Δ {nL-n0:+.4f}）——放行欺诈被当好人喂回，模型逐轮变窄。",
        f"- recovery：ROC {r0:.4f} → {rL:.4f}；末轮相对 naive 救回 {rL-nL:+.4f}。",
        "- 机制量化：naive 末轮把约 (见表) 笔真欺诈误标成 legit；随机抽检用少量无偏样本打破闭环。",
        "- 接 ⑩：这就是回流偏差的来源，探索预算（随机放行抽检）是无偏锚定；delta 记 naive→recovery。",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    write_report(OUT_MD, "\n".join(L))


if __name__ == "__main__":
    main()
