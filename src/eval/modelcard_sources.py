"""给 MODEL_CARD 的每条 `[实测]` 提议出处文件，并**核验**该行的数字确实在那份报告里。

## 为什么要做
第四次「同值不同义」暴露的根因：**回源靠 grep 值，就永远会被同值骗过。**
`+0.039` 既是图特征增益（0.0387 的舍入），也是校准近窗 28 天的 top2% gap——
重新生成后按值搜索会落到**看起来对得上的错文件**上。

修法是把「回源 = 按值搜索」换成「回源 = 跟指针走」：
MODEL_CARD 每个 `[实测]` 后面写清出处文件名，README / INTERVIEW 不挂路径、回源到 MODEL_CARD。

## 本工具做什么
它**不替人指派出处**——它对每一行列出「哪些报告能覆盖这一行的全部数字」，
覆盖唯一时给出建议，覆盖多份时如实报出候选（这正是同值不同义的现场）。
最后由人拍板写进文档。

用法：
  python -m src.eval.modelcard_sources            # 提议 + 候选
  python -m src.eval.modelcard_sources --verify   # 只核验已写的出处对不对
"""

import pathlib
import re
import sys
from pathlib import Path

from src.report_io import split_report

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "MODEL_CARD.md"
REPORTS = ROOT / "reports"
NUM = re.compile(r"(?<![\w.-])[-−+]?\d[\d,]*(?:\.\d+)?%?")


def _vals(text):
    out = set()
    for tok in NUM.findall(text):
        t = tok.replace("−", "-").replace(",", "").lstrip("+").rstrip("%")
        try:
            v = float(t)
        except ValueError:
            continue
        if abs(v) > 1e-12:
            out.add(v)
    return out


def _covers(pool, need):
    """报告能覆盖这一行的多少个数（含常见转写：舍入 / 百分数两形态）。"""
    hit = 0
    for v in need:
        ok = v in pool
        if not ok:
            ok = any(round(m, nd) == v for m in pool for nd in range(0, 5))
        if not ok:
            ok = any(round(m * 100, nd) == v for m in pool for nd in (0, 1, 2, 3))
        hit += ok
    return hit


def report_pools():
    pools = {}
    for p in sorted(REPORTS.glob("*.md")):
        machine, _ = split_report(p.read_text(encoding="utf-8"))
        pools[p.name] = _vals(machine)
    return pools


def main():
    pools = report_pools()
    lines = CARD.read_text(encoding="utf-8").splitlines()
    verify = "--verify" in sys.argv
    n_ok = n_amb = n_none = n_bad = 0
    for i, line in enumerate(lines, 1):
        if "[实测]" not in line:
            continue
        need = _vals(line)
        if not need:
            continue
        # 出处不限于 .md：`demo_data.json` 之类也是合法来源，
        # 只要它在 reports/ 下、且能读出数字。初版只认 .md，
        # 把项目负责人已正确标注的一行判成「覆盖不全」——**校验器比被校验的更严是假警**。
        named = re.findall(r"`?(?:reports/)?([a-z0-9_/]+\.(?:md|json))`?", line)
        cands = [(name, _covers(pool, need)) for name, pool in pools.items()]
        best = max(c for _, c in cands)
        winners = [n for n, c in cands if c == best and c > 0]
        if verify:
            if not named:
                continue
            def _pool_of(name):
                if name in pools:
                    return pools[name]
                for q in REPORTS.rglob(pathlib.Path(name).name):
                    return _vals(q.read_text(encoding="utf-8"))
                return None
            got = {n: _pool_of(n) for n in named}
            miss = [n for n, v in got.items() if v is None]
            union = set().union(*[v for v in got.values() if v]) if got else set()
            # **多个来源可以合起来覆盖一行**——一行引两个文件是合法的
            short = [] if _covers(union, need) == len(need) else [n for n in named]
            if miss or short:
                n_bad += 1
                print(f"❌ L{i}: 出处 {named} —— "
                      + (f"文件不存在 {miss}；" if miss else "")
                      + (f"覆盖不全 {short}（本行 {len(need)} 个数）" if short else ""))
            continue
        tag = f"L{i:>4}"
        if named:
            n_ok += 1
            continue
        if best == 0:
            n_none += 1
            print(f"{tag} ⚠️ 无报告能覆盖本行数字：{sorted(need)}")
        elif len(winners) == 1 and best == len(need):
            n_ok += 1
            print(f"{tag} ✅ 建议出处 `{winners[0]}`（覆盖 {best}/{len(need)}）")
        else:
            n_amb += 1
            print(f"{tag} ⚠️ 候选 {best}/{len(need)}：{winners[:4]}"
                  f"{' …' if len(winners) > 4 else ''}　← 同值多源，需人工判读")
            print(f"        本行数字：{sorted(need)}")
    if verify:
        print(f"\n{'✅ 已写出处全部核验通过' if not n_bad else f'❌ {n_bad} 行出处有问题'}")
        sys.exit(1 if n_bad else 0)
    print(f"\n唯一建议 {n_ok} · 多源待判 {n_amb} · 无覆盖 {n_none}")


if __name__ == "__main__":
    main()
