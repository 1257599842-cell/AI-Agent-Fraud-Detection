"""`reports/` 的写入闸门：**机器区由代码重写，人写区原样保留。**

## 为什么需要它
裁定：`reports/` 是构建产物、只能由代码写。但真去重新生成时发现，
五份报告里被删掉的**不是排版，是判读**——

    ## 泄漏审计（surprising 结果必做）        ← graph_vs_tabular
    ## 彩蛋洞察（第二份漂移证据，接 ⑩）        ← embargo_decomposition
    ## 结论（诚实版：ROC 稳、无告警，PR 更敏感）  ← drift_monitor
    ## 3. 这一步真正学到的（比原叙事更值钱）    ← calibration
    ## 结论（按实际数字，两边都说到，不硬凑）    ← calib_window_size

生成器在这些位置写的是**预注册式条件句**（「若…则…；若…则…」），
人后来把**实际发生的那一支**填进去。删掉它们等于删掉分析。

## 真问题不是「谁写的」，是「看不出来是谁写的」
所以不搬家、不另立目录，**在文件内划一条显式边界**：

    <机器区：每次生成都重写>
    <!-- HUMAN:BEGIN -->
    <人写判读：生成器不碰>
    <!-- HUMAN:END -->

约定**人写区只有一块、且在文件末尾**——判读本来就是读完数据之后的事，
这条约定让实现足够简单，简单到不会自己出错。

## 配套约束（在 report_manifest 里执法）
- 哈希**只算机器区**：人写区改动不算「手编构建产物」。
- **人写区出现的每个数字，必须能在同一文件的机器区找到。**
  防的是「机器区重算了、人写区还挂着旧数」——那种漂移没人会发现。
"""

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ── 冻结件登记 ──────────────────────────────────────────────────────────────
# 判据一句话：**该文件是否声称了自己写于某个时点之前 / 是否承载「当时不知道结果」。**
# 理由不写「别动它」，写「动了会毁掉什么」——前者是命令，后者才让人自己判断。
#
# 为什么要落进代码：`round4_preregistration.md` 差点被 `--verify-rerun` 覆盖，
# **是 chmod 444 拦住的，不是设计拦住的**。权限位会在 clone / 打包 / 复制时丢失。
FROZEN = {
    "reports/round4_preregistration.md": (
        "2026-07-31",
        "预注册件：指标在跑之前写死。重写它 = 事后按结果改指标，"
        "整个「预注册」的证明力当场归零。"),
    "reports/eval_runs/r1/anchor_blind.md": (
        "2026-07-27",
        "盲标表：标注者当时未看 judge 结果。重生成会带入现在的认知，**毁掉盲态**——"
        "而盲态正是 judge-人工一致率这个数的全部效力来源。"),
    "reports/eval_runs/r1/anchor_v2.md": (
        "2026-07-31",
        "round3 缩表盲标：条目所属臂被刻意隐藏。重生成即等于先看了分组，"
        "手会对某些条目更紧、对另一些更松，把要测的东西污染掉。"),
    "reports/eval_runs/r1/anchor_v2_manifest.json": (
        "2026-07-30",
        "上表的分臂对照：**标注前打开它就废了那张表**。与 anchor_v2.md 是一对，"
        "必须同冻，否则冻了表却漏了答案。"),
    "reports/eval_runs/r1/anchor_enrich.md": (
        "2026-07-28",
        "富集盲标表：用于算 judge-flag 精确率。重生成会让「独立盲标」不再独立。"),
    "reports/kaggle_submission.md": (
        "2026-08-01",
        "记录一次**已发生的外部提交**（提交号 55152570 / 2026-08-01）。"
        "重跑 `--submit` 会向 Kaggle 真的再交一次，而旧那次的榜单快照无法重建——"
        "它是事件记录，不是可重算的产物。"),
    "reports/eval_runs/round2_rubric_draft.md": (
        "2026-07-29",
        "评分前的 rubric 草案：它证明「评分标准先于评分确定」。"
        "事后重写 rubric 再宣布分数，是最典型的移动球门。"),
}


class FrozenArtifactError(RuntimeError):
    """试图重写一个承载时间声明的文件。"""


def guard_frozen(path):
    """冻结件一律拒写。**权限位靠不住，判断要在代码里。**"""
    try:
        rel = Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return
    if rel in FROZEN:
        when, why = FROZEN[rel]
        raise FrozenArtifactError(
            f"{rel} 是冻结件（冻结于 {when}），拒绝写入。\n理由：{why}")


def frozen_digests():
    """冻结件当前哈希 —— 供清单与测试核对「它确实没被动过」。"""
    out = {}
    for rel, (when, why) in sorted(FROZEN.items()):
        p = ROOT / rel
        if p.exists():
            out[rel] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                        "frozen_at": when, "tier": "frozen", "why": why}
    return out

BEGIN = "<!-- HUMAN:BEGIN -->"
END = "<!-- HUMAN:END -->"
_BLOCK = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)


def split_report(text):
    """拆成 (机器区, 人写区)。无标记时人写区为 `None`（**不是空串**）。

    区分 `None` 与 `""` 是要紧的：前者「本来就没有人写区」，
    后者「有一块空的人写区」——两者在校验时该被区别对待。
    """
    m = _BLOCK.search(text)
    if not m:
        return text, None
    return text[:m.start()], m.group(0)


def human_body(block):
    """取人写区内容（去掉标记本身）。"""
    if not block:
        return ""
    return block[len(BEGIN):-len(END)]


def write_report(path, machine_text):
    """写报告：**重写机器区，原样保留已有的人写区。**

    没有人写区的文件行为与直接 `write_text` 完全一致——
    这样 26 个生成器可以无差别改用它，不必逐个判断。
    """
    guard_frozen(path)                      # 冻结件在这里被挡住，不靠 chmod
    path = Path(path)
    keep = None
    if path.exists():
        _, keep = split_report(path.read_text(encoding="utf-8"))
    body = machine_text if machine_text.endswith("\n") else machine_text + "\n"
    out = body if keep is None else body + "\n" + keep + "\n"
    path.write_text(out, encoding="utf-8")
    return path
