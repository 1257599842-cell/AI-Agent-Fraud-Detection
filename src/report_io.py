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

import re
from pathlib import Path

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
    path = Path(path)
    keep = None
    if path.exists():
        _, keep = split_report(path.read_text(encoding="utf-8"))
    body = machine_text if machine_text.endswith("\n") else machine_text + "\n"
    out = body if keep is None else body + "\n" + keep + "\n"
    path.write_text(out, encoding="utf-8")
    return path
