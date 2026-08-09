"""命令行入口的共用守卫：**缺参即报错，绝不顶默认值。**

## 来历（真实事故）
`round3_metrics.py` 曾写作 `run(args or ["r1"])`。批量重跑时**裸跑**该模块，
于是它顶上默认的单轮口径、重新生成了 `agent_round3_metrics.md`，
**把两轮对比（McNemar 配对、delta、「改了没用」的结论）整段覆盖掉，退出码仍是 0。**

这属于本项目的「**静默替代**」属类：读不到 → 顶一个上去 → 不报错。
同类前科：金标解析器跨块顺延、M-H 跳过空格、兜底顶替缺测、滑块吸附。

## 规矩
参数是**口径**的一部分，不是可省的便利项。省掉它就等于让调用方在不知情的情况下
换了口径。所以：**没给就退出，并把正确用法打出来。**
"""

import sys


def require_tags(argv, least=1, usage=""):
    """从 argv 取位置参数（run tag）；不足 `least` 个就报错退出。

    `--xxx` 形式的开关不计入。返回 tag 列表。
    """
    tags = [a for a in argv if not a.startswith("-")]
    if len(tags) < least:
        prog = f"python -m {_caller_module()}"
        sys.exit(
            f"缺少参数：需要至少 {least} 个 run tag，实际收到 {len(tags)} 个。\n"
            f"**本模块不设默认值**——参数是口径的一部分，顶默认会静默改写产物。\n"
            f"用法：{usage or prog + ' <tag> [tag...]'}"
        )
    return tags


def _caller_module():
    """取调用方的模块名，让错误信息里的用法能直接复制运行。"""
    import inspect
    for fr in inspect.stack()[1:]:
        name = fr.frame.f_globals.get("__name__", "")
        if name and not name.endswith("_cli"):
            return "src.eval." + (fr.frame.f_globals.get("__file__", "")
                                  .rsplit("/", 1)[-1].removesuffix(".py"))
    return "<module>"
