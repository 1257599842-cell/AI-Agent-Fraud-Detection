"""团伙结构可视化（②④ 两张皮的展品）—— card1 fan-out 到多设备的 ego 图。

从图特征找出的高分团伙 card1，画"一张卡扇出到多个设备、多笔小额欺诈"的星形图：
中心=card1，外圈=它触达的各设备，边=该(卡,设备)上的交易，欺诈设备标红、大小∝交易数。
这是面试能拍在桌上的展品，比 AUC 更有说服力。

用法：python -m src.features.gang_viz
产出：reports/figures/11_gang_egographs.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARQUET = PROJECT_ROOT / "data" / "processed" / "train_merged.parquet"
FIG = PROJECT_ROOT / "reports" / "figures" / "11_gang_egographs.png"

CARDS = [16095, 1129, 4606]   # 高设备 fan-out + 高欺诈的团伙候选


def draw_ego(ax, sub, card):
    """星形 ego 图：card1 中心，各设备一个节点。"""
    dev = sub.dropna(subset=["DeviceInfo"]).groupby("DeviceInfo").agg(
        n=("isFraud", "size"), fr=("isFraud", "sum"), amt=("TransactionAmt", "median"))
    k = len(dev)
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
    pos = {d: (np.cos(a), np.sin(a)) for d, a in zip(dev.index, ang)}

    for d, (x, y) in pos.items():
        r = dev.loc[d]
        is_fraud = r.fr > 0
        ax.plot([0, x], [0, y], color="#C44E52" if is_fraud else "#B8B8B8",
                lw=0.8 + 0.5 * r.n, alpha=0.7, zorder=1)
        ax.scatter([x], [y], s=60 + 40 * r.n, color="#C44E52" if is_fraud else "#7F9FC0",
                   edgecolor="white", zorder=2)
    ax.scatter([0], [0], s=420, color="#2E2E2E", marker="s", zorder=3)
    ax.text(0, 0, f"card1\n{card}", color="white", ha="center", va="center", fontsize=7, zorder=4)

    n_txn, n_fraud = len(sub), int(sub["isFraud"].sum())
    ax.set_title(f"card1={card}: {n_txn} tx, {k} devices, {n_fraud} fraud\n"
                 f"median ${sub['TransactionAmt'].median():.0f}", fontsize=9)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.axis("off")


def main():
    df = pd.read_parquet(PARQUET, columns=[
        "TransactionID", "TransactionDT", "TransactionAmt", "isFraud", "card1", "DeviceInfo"])
    fig, axes = plt.subplots(1, len(CARDS), figsize=(15, 5.2))
    for ax, card in zip(axes, CARDS):
        draw_ego(ax, df[df["card1"] == card], card)
    # 图例
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#2E2E2E", markersize=11, label="card1 (shared card)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#C44E52", markersize=10, label="device with fraud"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#7F9FC0", markersize=10, label="device (legit only)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("Gang structure: one card fanned out across many devices, small amounts", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=120)
    plt.close()
    print(f"✅ {FIG.relative_to(PROJECT_ROOT)}")
    for card in CARDS:
        sub = df[df["card1"] == card]
        nd = sub["DeviceInfo"].dropna().nunique()
        print(f"  card1={card}: {len(sub)} tx, {nd} devices, {int(sub.isFraud.sum())} fraud, "
              f"median ${sub.TransactionAmt.median():.0f}")


if __name__ == "__main__":
    main()
