"""公开路径与内容检查。只检查git跟踪文件，不修改index。

这是测试而非提交钩子，必须主动执行。内容扫描是词表启发式，既不能覆盖所有
私密信息，也不应自称完整防泄漏系统。检测器源码需要包含被检测词表，因此
只豁免自身的内容扫描，路径检查仍执行；豁免必须保持用途明确。
"""

import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 已知私有路径：本地保留、不随仓库发布。与 .gitignore 同源，但在这里**再断言一次**——
# .gitignore 可能被误删一行，而误删不会有任何提示。
PRIVATE_PATHS = ("STUDY/", "INTERVIEW.md", "CLAUDE.md", "AGENTS.md",
                 "PROGRESS.md", "REHEARSAL.md", "会话全程记录",
                 "旗舰项目自我学习与回顾PDF", "早期选岗位资料")

# 内部用语：**通用词**，不写内部原句（见模块 docstring 第二条约束）。
INTERNAL_VOCAB = ("秋招", "简历", "面试", "talking point", "答案库", "背诵")

# 检测器本身是必要豁免，其他文件不豁免。
# 将来若不得不加，豁免必须连同「它为什么还需要存在」一起写，并定期复核是否过期。
VOCAB_EXEMPT: set[str] = {"tests/test_no_private_leak.py"}  # 词表声明本身必然命中


def tracked_files():
    """`git ls-files` —— 判据是「**是否被跟踪**」，不是「是否存在于磁盘」。

    工作区里躺着私有文件是正常的（它们本来就该在本地）；
    出事的是它们**进了 index**。所以扫描对象必须是 tracked 集合。
    """
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                         capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise unittest.SkipTest("不在 git 仓库内")
    return [f for f in out.stdout.split("\0") if f]


class TestNoPrivateFileIsTracked(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = tracked_files()
        if not cls.files:
            raise unittest.SkipTest("没有被跟踪的文件")

    def test_no_private_path_is_tracked(self):
        """第一道：已知私有路径一个都不许在 index 里。"""
        for prefix in PRIVATE_PATHS:
            hits = [f for f in self.files if f.startswith(prefix) or prefix in f]
            with self.subTest(private=prefix):
                self.assertEqual(
                    hits, [],
                    f"\n私有路径 {prefix} 被跟踪了：{hits}"
                    f"\n它应当只存在于本地。清除：git rm --cached <文件>"
                    f"\n若确属公开内容，请从 PRIVATE_PATHS 移除并说明理由。")

    def test_no_tracked_file_leaks_internal_vocabulary(self):
        """第二道：内容级，**不依赖文件名**——抓的是还没被预判到的下一个。

        这一层才是补上「黑名单落后一次事故」的那一层。
        """
        pat = re.compile("|".join(re.escape(w) for w in INTERNAL_VOCAB))
        leaks = []
        for rel in self.files:
            if rel in VOCAB_EXEMPT:
                continue
            p = ROOT / rel
            if not p.is_file():
                continue                        # 已删除但仍在 index：由上一条测试兜
            try:
                text = p.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue                        # 二进制/图片不参与内容扫描
            for m in pat.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                leaks.append(f"{rel}:{line} 命中「{m.group(0)}」")
        self.assertEqual(
            leaks, [],
            "\n公开文件里出现内部用语：\n  " + "\n  ".join(leaks[:20])
            + f"\n（共 {len(leaks)} 处）"
            "\n对外文档应以项目与方法为主语，不以求职过程为主语。")

    def test_every_exemption_still_earns_its_place(self):
        """**豁免自己也要被执法** —— 不再命中的豁免就该删，不是留着。

        本项目已烧过一次：初版给 `.gitignore` 留的豁免，在写下的那一刻就已经过期
        （`.gitignore` 的注释早已改成中性表述）。当前仅检测器本身豁免；本测试保证它仍包含被检测词——
        留着它是为了「将来有人加豁免时，过期会被自动抓到」，不是为了现在断言什么。
        """
        pat = re.compile("|".join(re.escape(w) for w in INTERNAL_VOCAB))
        for rel in sorted(VOCAB_EXEMPT):
            p = ROOT / rel
            if not p.is_file():
                continue
            with self.subTest(exempt=rel):
                self.assertRegex(
                    p.read_text(encoding="utf-8"), pat,
                    f"{rel} 已不再命中词表 —— 豁免过期，请从 VOCAB_EXEMPT 删除")


if __name__ == "__main__":
    unittest.main(verbosity=2)
