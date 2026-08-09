"""重新生成 `reports/` 前的交叉检查：枚举将改动的每一个值，并回查对外引用。

## 为什么需要它
`reports/` 是构建产物、只能由代码写。但历史上有人手编过，于是「重新生成」会改动一批值。
**在改动之前必须回答：这些值有没有印在对外文档上。**

## 为什么不能只按数值 grep
本项目已三次栽在**同值不同义**上：
  - `96.5%`   闸门放行率 ✗ / bootstrap 概率 P(生产拓扑<全放行) ✓
  - `$74.33`  p=0.300 的闭式解 ✗ / 页面 p=0.302 上是 $73.88 ✓
  - `−0.024`  baseline precision@0.5% delta ✗ / 选择性偏差 ROC 衰减 ✓（**正印在对外文档上**）

所以本工具对每个将改动的值，同时报出：
  ① 它在四份对外文档里出现在哪些行；
  ② 它在 `reports/` 里有**几个不同来源**同样产出这个值。
**只要 ② > 1，就必须人工判读① 引的到底是哪一个来源**——工具不替人做这个判断。

用法：python -m src.eval.report_diff_audit
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ["README.md", "MODEL_CARD.md"]   # 答案库与背诵卡不随仓库发布
# 抓「有意义的数值」：带小数点/百分号的，或带千分位的，或 3 位以上整数。
# **千分位首组可以只有 1–3 位**：初版写 `\d{3,}(?:,\d{3})*`，
# 于是 `1,774` 从 `774` 才开始匹配，报出一个根本不存在的「774 消失了」。
# 这是本工具的第四个分词缺陷，与前三个（按行 diff / 子串匹配 / 匹配到注释）同族。
NUM = re.compile(r"(?<![\w.])-?−?\d{1,3}(?:,\d{3})+|-?−?\d+\.\d+%?|\d{3,}")


def _norm(s):
    """统一负号与千分位，让 `−0.024` 与 `-0.024` 视为同一个值。"""
    return s.replace("−", "-").replace(",", "").rstrip("%")


def changed_files():
    out = subprocess.run(["git", "status", "--porcelain", "reports/"],
                         cwd=ROOT, capture_output=True, text=True).stdout
    return [l[3:].strip() for l in out.splitlines() if l[3:].strip().endswith(".md")]


def diff_values(path):
    """返回 (消失的值, 新增的值)，已按 `_norm` 归一。

    **按整文件比对，不按行 diff。** 初版按行算，把「这一行被删了、但同一个值在
    文件别处仍在」误报成消失——`calib_window_size.md` 的 `0.039` 只是从结论段
    移走、数据表里原样健在，却被报成「对外引用 8 处的值将消失」。
    8 个假警足以让人失去对这份清单的信任，比没有清单更糟。
    """
    old = subprocess.run(["git", "show", f"HEAD:{path}"],
                         cwd=ROOT, capture_output=True, text=True).stdout
    new = (ROOT / path).read_text(encoding="utf-8")
    o = {_norm(x) for x in NUM.findall(old)}
    n = {_norm(x) for x in NUM.findall(new)}
    return o - n, n - o


def cite_lines(val):
    """该值在四份对外文档里出现的位置。

    **按 token 精确比对，不用子串包含。** 初版用 `val in line`，于是 `299` 命中
    「gang_score 最小 0.299」、`0.02` 命中 `0.0212`——一个值报出 16 处假引用。
    审计工具自己发假警，比没有审计更糟：它会训练人忽略告警。
    """
    hits = []
    for doc in DOCS:
        p = ROOT / doc
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if val in {_norm(x) for x in NUM.findall(line)}:
                hits.append((doc, i, line.strip()[:110]))
    return hits


def sources_in_reports(val):
    """`reports/` 里有几个**不同文件**同样产出这个值 —— 用来发现「同值不同义」。"""
    return sorted({p.name for p in (ROOT / "reports").glob("*.md")
                   if val in {_norm(x) for x in NUM.findall(p.read_text(encoding="utf-8"))}})


def main():
    files = changed_files()
    if not files:
        print("✅ reports/ 无改动，无需交叉检查。")
        return
    print(f"待检查 {len(files)} 份报告\n" + "=" * 78)
    risky = 0
    for f in files:
        gone, came = diff_values(f)
        if not gone:
            print(f"\n【{f}】仅排版/文字变化，**无数值消失**")
            continue
        print(f"\n【{f}】将消失 {len(gone)} 个值、新增 {len(came)} 个")
        for v in sorted(gone):
            cites = cite_lines(v)
            srcs = sources_in_reports(v)
            if not cites:
                print(f"  · {v:<12} 对外引用 0 处　（reports 内 {len(srcs)} 处来源）")
                continue
            risky += 1
            amb = "  ⚠️ **同值多源，必须人工判读引的是哪一个**" if len(srcs) > 1 else ""
            print(f"  ❗ {v:<12} **对外引用 {len(cites)} 处**"
                  f"　reports 内来源：{', '.join(srcs)}{amb}")
            for doc, i, txt in cites:
                print(f"       {doc}:{i}  {txt}")
    print("\n" + "=" * 78)
    if risky:
        print(f"❗ {risky} 个将消失的值在对外文档里出现过 —— **逐个判读来源后才能提交**。")
        sys.exit(1)
    print("✅ 所有将消失的值均无对外引用 —— 可以全量重新生成并提交。")


if __name__ == "__main__":
    main()
