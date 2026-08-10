"""在线回放对账：把交易逐笔当作新交易喂进 FeatureStore，与离线 parquet 逐行比。

## 这一步在验什么、不在验什么

**在验**：在线点查与离线批算**是不是同一个东西**——训练/线上一致性，
也就是把「特征在服务里会不会算错」从担心变成断言。

**不在验**：时间模型对不对。
本数据集的 `label_time = event_time + 21d` 是**常数平移**，
双时间戳日志与离线 `RANGE ... 21d PRECEDING` 帧**同构、不含新信息**。
对账通过只说明两条代码路径算的是同一件事，**不构成时间模型正确的证据**。

## 契约：有状态、按序、先查后写

    for txn in 按 (dt, id) 升序:
        feats = store.get_features(txn)   # 先查 —— 此时本笔尚未入库
        assert feats == 离线[txn]
        store.append(txn)                 # 后写 —— 成为下一笔的历史

**不能先把整段灌进去再逐笔点查**：那样取值型过滤（`event_time < t`）会碰巧给出
相同答案，对账通过了却什么都没验到——**位置语义的 bug 正好被绕过去**。

## 负对照：预注册期望值，不是「断言失败」

去掉 ID tiebreak，位置语义退化成取值语义。
**期望差异行数由离线数据事先算出**（见 `--preregister`），跑完必须**恰好命中**。

> 「断言它失败」信息量太低——退化成单时间戳是没人会写的 bug。
> 预注册一个**确切的数**，同时验证了对账装置与我对系统的理解：
> 猜错了说明理解有问题，猜对了说明这套装置真的在量那件事。

用法：
  python -m src.eval.online_replay --preregister      # 先算出并冻结期望值
  python -m src.eval.online_replay --run              # 回放 + 对账 + 负对照 + 延迟
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.report_io import write_report

ROOT = Path(__file__).resolve().parents[2]
MERGED = ROOT / "data" / "processed" / "train_merged.parquet"
GRAPH = ROOT / "data" / "processed" / "graph_features.parquet"
VEL = ROOT / "data" / "processed" / "velocity_features.parquet"
PREREG = ROOT / "reports" / "online_replay_preregistration.json"
REPORT = ROOT / "reports" / "online_replay.md"

COLS = ["TransactionID", "TransactionDT", "isFraud",
        "card1", "addr1", "P_emaildomain", "DeviceInfo"]
N_REPLAY = 3000          # 回放笔数：全扫方案下 10 万笔要跑数小时，取连续切片即可验一致性


def _load():
    df = pd.read_parquet(MERGED, columns=COLS)
    df = df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    off = (pd.read_parquet(GRAPH).merge(pd.read_parquet(VEL), on="TransactionID")
           .set_index("TransactionID"))
    return df, off


def _tie_rows(df, cols):
    """同组内 dt 并列、且排在前面有同 dt 行的那些行 —— 位置语义与取值语义在此不等。"""
    dt = df["TransactionDT"].to_numpy()
    k = df[cols[0]].astype("string")
    for c in cols[1:]:
        k = k + "|" + df[c].astype("string")
    na = df[cols].isna().any(axis=1).to_numpy()
    g = pd.factorize(k, use_na_sentinel=False)[0].astype(np.int64)
    g[na] = g.max() + 1 + np.arange(na.sum())
    order = np.lexsort((dt, g))
    gs, ds = g[order], dt[order]
    pos = np.arange(len(ds))
    newg = np.r_[True, gs[1:] != gs[:-1]]
    seg = np.maximum.accumulate(np.where(newg, pos, 0))
    pos_prior = pos - seg
    val_prior = np.empty(len(ds), np.int64)
    b = np.flatnonzero(np.diff(gs)) + 1
    for s, e in zip(np.r_[0, b], np.r_[b, len(ds)]):
        val_prior[s:e] = np.searchsorted(ds[s:e], ds[s:e], side="left")
    tid = df["TransactionID"].to_numpy()[order]
    return set(tid[pos_prior != val_prior].tolist())


def preregister():
    """**跑在线之前**从离线数据算出负对照的期望值，冻结成文件。"""
    from src.serving.feature_store import ENTITY_KEYS
    df, _ = _load()
    slice_ids = set(_pick_slice(df)["TransactionID"].tolist())
    rec = {"n_replay": N_REPLAY, "note": "去掉 ID tiebreak 后位置语义退化为取值语义，"
                                         "期望出现差异的行数（全量 / 回放切片内）"}
    colmap = {"card1": ["card1"], "card1_addr1": ["card1", "addr1"],
              "card1_email": ["card1", "P_emaildomain"],
              "card1_device": ["card1", "DeviceInfo"]}
    for kname in ENTITY_KEYS:
        ties = _tie_rows(df, colmap[kname])
        rec[kname] = {"full_dataset": len(ties),
                      "in_replay_slice": len(ties & slice_ids)}
        print(f"  {kname:<14} 全量 {len(ties):>4} 行　回放切片内 {len(ties & slice_ids):>4} 行")
    PREREG.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n✅ 已冻结 → {PREREG.relative_to(ROOT)}　**跑完不许改**")


def _pick_slice(df):
    """选回放切片：**必须包含并列行**，否则负对照永远通过、等于没验。

    并列行只有 166/590,540（0.028%），随机取 3000 行按期望只撞到 0.85 行——
    第一版按中位数取窗，结果切片内 0 行。**改成滑窗找并列最密的一段。**
    """
    ties = _tie_rows(df, ["card1"])
    flag = df["TransactionID"].isin(ties).to_numpy().astype(np.int32)
    if flag.sum() == 0:
        return df.iloc[len(df) // 2: len(df) // 2 + N_REPLAY]
    cum = np.r_[0, np.cumsum(flag)]
    win = cum[N_REPLAY:] - cum[:-N_REPLAY]         # 每个起点的窗口内并列行数
    start = int(np.argmax(win))
    return df.iloc[start: start + N_REPLAY]


def _cmp(online, offline_row, cols):
    """逐列比对。**NULL 必须单独判**——`nan == nan` 为 False，用 `==` 会静默放过。"""
    bad = []
    for c in cols:
        a, b = online.get(c), offline_row[c]
        a_null = a is None or (isinstance(a, float) and np.isnan(a))
        b_null = b is None or (isinstance(b, float) and np.isnan(b))
        if a_null != b_null:
            bad.append((c, a, b, "NULL 位置不对齐"))
        elif not a_null and not np.isclose(float(a), float(b), rtol=1e-9, atol=0):
            bad.append((c, a, b, "取值不等"))
    return bad


def run():
    from src.serving.feature_store import FEATURE_COLUMNS, FeatureStore
    if not PREREG.exists():
        sys.exit("负对照期望值未冻结，先跑 --preregister")
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    df, off = _load()
    sl = _pick_slice(df)
    hist = df.iloc[:sl.index[0]]
    print(f"历史 {len(hist):,} 笔预灌；回放 {len(sl):,} 笔（有状态、先查后写）")

    store = FeatureStore()
    store.append_frame(hist)
    mismatch, null_mismatch, lat, hist_len = [], 0, [], []
    for _, txn in sl.iterrows():
        t0 = time.perf_counter()
        f = store.get_features(txn, tiebreak_id=int(txn["TransactionID"]))
        lat.append((time.perf_counter() - t0) * 1000)
        hist_len.append(store.entity_history_len(txn))
        bad = _cmp(f, off.loc[int(txn["TransactionID"])], FEATURE_COLUMNS)
        if bad:
            mismatch.append((int(txn["TransactionID"]), bad))
            null_mismatch += sum(1 for b in bad if b[3] == "NULL 位置不对齐")
        store.append_frame(txn.to_frame().T)          # 后写：成为下一笔的历史
    print(f"对账：{len(sl) - len(mismatch)}/{len(sl)} 逐列一致"
          + (f"，{len(mismatch)} 笔不符" if mismatch else " ✅"))

    neg = _negative_control(df, sl, off, pre)
    store.close()
    _write(sl, mismatch, null_mismatch, lat, hist_len, neg, pre, len(hist))
    print(f"\n✅ → {REPORT.relative_to(ROOT)}")


def _negative_control(df, sl, off, pre):
    """去掉 ID tiebreak 重跑，差异行数必须**恰好等于预注册值**。"""
    from src.serving.feature_store import FeatureStore
    print("\n负对照：去掉 ID tiebreak（位置语义 → 取值语义）…")
    store = FeatureStore()
    store.append_frame(df.iloc[:sl.index[0]])
    diff = {k: 0 for k in ("card1", "card1_addr1", "card1_email", "card1_device")}
    for _, txn in sl.iterrows():
        f = store.get_features(txn, tiebreak_id=int(txn["TransactionID"]),
                               use_tiebreak=False)
        o = off.loc[int(txn["TransactionID"])]
        for k in diff:
            if f[f"{k}_prior_cnt"] != o[f"{k}_prior_cnt"]:
                diff[k] += 1
        store.append_frame(txn.to_frame().T)
    store.close()
    for k, v in diff.items():
        exp = pre[k]["in_replay_slice"]
        print(f"  {k:<14} 实测 {v:>3} 行　预注册 {exp:>3} 行　{'✅' if v == exp else '❌'}")
    return diff


def _write(sl, mismatch, null_mismatch, lat, hist_len, neg, pre, n_hist):
    lat = np.array(lat)
    hl = np.array(hist_len)
    L = ["# 在线回放对账：点查特征 vs 离线批算\n",
         "> **它修的窟窿**：`/score` 此前接受 `transaction_id`、去离线表里查一个算好的 `p`——",
         "> **那不是打分，是查表**。本轮让服务能对没见过的交易真正算特征、真正打分。\n",
         "## 契约\n",
         "有状态、按 `(dt, id)` 升序、**先查后写**：每笔先取特征（此时本笔尚未入库），",
         "对账后再写入、成为下一笔的历史。\n",
         "> **不能先整段灌入再逐笔点查**：那样取值型过滤会碰巧给出相同答案，",
         "> 对账通过却什么都没验到——**位置语义的 bug 正好被绕过去**。\n",
         f"历史预灌 **{n_hist:,}** 笔；回放 **{len(sl):,}** 笔；比对 **27** 列。\n",
         "## 对账结果\n",
         f"- 逐列一致：**{len(sl) - len(mismatch)} / {len(sl)}**"
         + ("　✅" if not mismatch else f"　❌ {len(mismatch)} 笔不符"),
         f"- 其中 NULL 位置不对齐：**{null_mismatch}** 处",
         "",
         "> `prior_fraud_rate` 在 `obs_cnt = 0` 时是 **NULL 而非 0**，",
         "> 离线该列 NaN 占比 18%–90%（`card1_device` 高达九成）。",
         "> **比对必须单独判 NULL**——`nan == nan` 为 False，用 `==` 会把不一致静默放过。",
         "> 训练/线上不一致在实践中死于 NaN 语义的次数，远多于死于时间语义。\n"]
    if mismatch:
        L += ["| 交易号 | 列 | 在线 | 离线 | 原因 |", "|---|---|---|---|---|"]
        for tid, bad in mismatch[:10]:
            for c, a, b, why in bad[:3]:
                L.append(f"| {tid} | `{c}` | {a} | {b} | {why} |")

    L += ["", "## 负对照：预注册期望值\n",
          "去掉 ID tiebreak，位置语义退化成取值语义。**期望差异行数在跑之前已从离线数据算出并冻结**",
          f"（`{PREREG.name}`）。\n",
          "| 实体键 | 全量数据集 | 回放切片内（预注册） | 实测 | |",
          "|---|---|---|---|---|"]
    allok = True
    for k, v in neg.items():
        exp = pre[k]["in_replay_slice"]
        ok = v == exp
        allok &= ok
        L.append(f"| `{k}` | {pre[k]['full_dataset']} | **{exp}** | **{v}** | "
                 f"{'✅' if ok else '❌'} |")
    L += ["",
          ("**全部命中预注册值。**" if allok else "**未命中，照实记。**")
          + "　这同时验证了两件事：对账装置真的在量并列语义，"
            "以及我对离线定义的理解是对的——**猜错了说明理解有问题，猜对了说明装置有效**。\n",
          "> 为什么不用「断言它失败」当负对照：退化成单时间戳是没人会写的 bug，",
          "> 断言它失败信息量太低。**预注册一个确切的数，强一个量级。**\n",
          "### 一个真结论：`TransactionID` 在线没有对应物\n",
          "离线 `prior_cnt` 靠 `TransactionID` 打破 `dt` 并列（`lexsort` 的稳定次序）。",
          "**真实系统里同一秒到达的两笔谁在前是到达顺序，不是 ID。**",
          "本模块传入 `tiebreak_id` 只是为了能与离线逐行对账——",
          "它是**离线特征定义里的一个人造依赖，在线无法忠实复现**。这是结论，不是瑕疵。\n",
          "## 延迟（**只测量，不承诺**）\n",
          "测量条件：单机 · DuckDB in-memory · **并发 = 1** · 无预聚合（方案 A，每次点查全扫）·",
          f"事件表 {n_hist:,} 行起步 · 每笔 27 列共 ~15 次查询。\n",
          f"- p50 **{np.percentile(lat, 50):.1f} ms**　p95 **{np.percentile(lat, 95):.1f} ms**"
          f"　p99 **{np.percentile(lat, 99):.1f} ms**　最大 {lat.max():.1f} ms\n",
          "### 延迟 vs 实体历史长度\n",
          "> **只报一个 p95 没有信息量**：`card1` 的历史长度是重尾的，p95 由重实体决定。",
          "> 真正的结论是「延迟随实体历史长度怎么涨」。\n",
          "| 实体历史长度 | 笔数 | 延迟 p50 | 延迟 p95 |", "|---|---|---|---|"]
    edges = [0, 10, 50, 200, 1000, 10 ** 9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (hl >= lo) & (hl < hi)
        if m.sum() == 0:
            continue
        name = f"{lo}–{hi}" if hi < 10 ** 9 else f"≥ {lo}"
        L.append(f"| {name} | {int(m.sum()):,} | {np.percentile(lat[m], 50):.1f} ms | "
                 f"{np.percentile(lat[m], 95):.1f} ms |")
    if len(np.unique(hl)) > 5:
        r = float(np.corrcoef(hl, lat)[0, 1])
        L += ["", f"- 历史长度与延迟的相关系数 **{r:+.2f}**。"]
    L += ["",
          "> **超预算才分层。** 「全扫太慢」在测量之前只是一句未经验证的断言，",
          "> 不该用它驱动架构决策。上面这组数是分层与否的依据；",
          "> 若要分层，改动前后必须记成 delta（X → Y），不能只报改动后的值。\n",
          "## 本轮明确划出去的\n",
          "- **`/investigate` 仍是查表。** 它的工具喂着 r1/r3/r4 那批**归档 eval 运行**，",
          "  改工具行为会让归档失去可比性、需付费重跑。**现状照实记，不含糊过去。**",
          "- 预聚合 / 热存储 / 水平扩展 —— 单机 DuckDB 是**正确性演示**，不是规模演示。\n"]
    write_report(REPORT, "\n".join(L))


if __name__ == "__main__":
    if "--preregister" in sys.argv:
        preregister()
    elif "--run" in sys.argv:
        run()
    else:
        print(__doc__)
