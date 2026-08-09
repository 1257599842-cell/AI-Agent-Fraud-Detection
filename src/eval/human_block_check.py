"""人写区数字对账：**人写区出现的每个数，必须能在同一文件的机器区找到。**

## 防的是什么
划出人写区之后，机器区每次重新生成都会刷新，**人写区不会**。
于是可能出现：机器区已经重算成 0.0234，人写区还挂着上一版的 0.0198——
**没人会发现**，因为两处都「看起来是对的」，只是不再一致。

这是第 3 条硬规矩（每个数字都有可回的源）往文件内部再走一层：
源不只要存在，还要**就在同一份文件里、且是当前值**。

## 转写规则必须具名
人写散文里 `1,307` 会写成 `1.3k`、`0.0387` 会写成 `+0.039`——这些是合法转写，不是漂移。
但**「差不多就算过」等于没有校验**。所以每条允许的转写都有名字、都会在报告里打出来，
读的人能判断这次放行是否合理。设计沿用 `agent_eval._derived_reason`。

用法：python -m src.eval.human_block_check
"""

import re
import sys
from pathlib import Path

from src.report_io import human_body, split_report

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
# 取数三条排除，全是踩过的坑：
#  · `top-3` 的连字符不是负号 → 负号前不能紧跟字母/数字
#  · `3. 这一步…` 是列表序号 → 行首「数字.」后跟空格的，跳过
#  · 结尾的句点不属于数字（`+0.023。`）
NUM = re.compile(r"(?<![\w.-])[-−+]?\d[\d,]*(?:\.\d+)?")
_ORDINAL = re.compile(r"^\s*\d+\.\s")


def _f(tok):
    t = tok.replace("−", "-").replace(",", "").lstrip("+")
    try:
        return float(t)
    except ValueError:
        return None


def _tokens(text):
    """逐行取数，跳过 markdown 列表序号行首的「N.」。"""
    for line in text.splitlines():
        body = re.sub(r"^#+\s+", "", line)          # 先剥 markdown 标题标记
        body = _ORDINAL.sub("", body) if _ORDINAL.match(body) else body
        yield from NUM.findall(body)


def machine_numbers(machine_text):
    out = set()
    for tok in _tokens(machine_text):
        v = _f(tok)
        if v is not None:
            out.add(v)
    return out


def explain(value, pool):
    """能否由机器区某个值转写而来？返回规则名，或 None。"""
    if value in pool:
        return "原样"
    for m in pool:
        if abs(m) > 1e-12:
            # 舍入：人写散文常用更少的小数位（0.0387 → 0.039）
            for nd in range(0, 5):
                if round(m, nd) == value:
                    return f"舍入（{nd} 位小数）"
            # 量级缩写：1,307 → 1.3（k）。**要求源值 ≥ 该量级**——否则
            # round(20000/1e6, 2) == 0.02 会让任意五位数「解释」掉 0.02，
            # 那是假放行，比假警更危险。
            for div, unit in ((1e3, "k"), (1e4, "万"), (1e6, "M")):
                # **至少 1 位小数**：0 位时 round(3,556,658/1e6, 0) == 4，
                # 交易号能把任意个位数「解释」掉——又一次假放行。
                if abs(m) >= div:
                    for nd in (1, 2):
                        if round(m / div, nd) == value:
                            return f"量级缩写（{unit}，{nd} 位）"
            # 百分数两形态：0.0559 ↔ 5.59。**至少保留 1 位小数**——
            # 0 位时 round(m*100,0)==3 对 m∈[0.025,0.035) 全成立，太松。
            for nd in (1, 2, 3):
                if round(m * 100, nd) == value:
                    return f"百分数形态（{nd} 位）"
            # 地板截断：0.4555 → 0.455（散文里常直接砍位，不进位）
            for nd in (1, 2, 3, 4):
                f = int(abs(m) * 10 ** nd) / 10 ** nd * (1 if m >= 0 else -1)
                if abs(f - value) < 1e-12:
                    return f"地板截断（{nd} 位）"
            # 补数：0.94 ↔ 0.06
            if abs((1 - m) - value) < 1e-9:
                return "补数（1−x）"
    return None


def check_file(path):
    text = path.read_text(encoding="utf-8")
    machine, block = split_report(text)
    if block is None:
        return None
    pool = machine_numbers(machine)
    rows, bad = [], []
    # **跨文件引用**：要引别份报告里的数，必须在同一行写明来源 `xxx.md`。
    # 这就是「回源 = 跟指针走」在文件内部的落地——不写指针就不放行。
    for line in human_body(block).splitlines():
        extra = set()
        for ref in re.findall(r"`?([a-z0-9_]+\.md)`?", line):
            q = REPORTS / ref
            if q.exists():
                extra |= machine_numbers(split_report(q.read_text(encoding="utf-8"))[0])
        body = re.sub(r"^#+\s+", "", line)
        body = _ORDINAL.sub("", body) if _ORDINAL.match(body) else body
        for tok in NUM.findall(body):
            v = _f(tok)
            if v is None or abs(v) < 1e-12:
                continue
            rule = explain(v, pool)
            if not rule and extra:
                r2 = explain(v, extra)
                rule = f"跨文件引用（{r2}）" if r2 else None
            (rows if rule else bad).append((tok, rule))
    return rows, bad


def main():
    files = sorted(REPORTS.glob("*.md"))
    checked, all_bad = 0, 0
    print(f"{'报告':<34}{'人写数字':>8}{'需转写':>8}{'对不上':>8}")
    for p in files:
        r = check_file(p)
        if r is None:
            continue
        rows, bad = r
        checked += 1
        all_bad += len(bad)
        derived = sum(1 for _, rule in rows if rule != "原样")
        flag = "" if not bad else "  ❌"
        print(f"{p.name:<34}{len(rows) + len(bad):>8}{derived:>8}{len(bad):>8}{flag}")
        for tok, rule in rows:
            if rule != "原样":
                print(f"      · {tok:<10} ← {rule}")
        for tok, _ in bad:
            print(f"      ❌ {tok:<10} 机器区里找不到出处")
    print(f"\n{checked} 份含人写区。", end=" ")
    if all_bad:
        print(f"❌ {all_bad} 个数字在机器区无出处 —— "
              "要么机器区重算后人写区没跟上，要么这个数根本没来源。")
        sys.exit(1)
    print("✅ 人写区所有数字都能回到本文件机器区。")


if __name__ == "__main__":
    main()
