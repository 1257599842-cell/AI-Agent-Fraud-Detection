"""`reports/` 的出处清单：**每份报告的哈希 + 生成它的命令**。

## 它解决什么
裁定：`reports/` 是构建产物，**只能由代码写，手工编辑就此废止**。
但「不许手编」如果只写在文档里，就是又一条要靠人记的纪律——
本项目已经证明**写下来不等于形成反射**。

所以把它变成会自动拦住的东西：
  · 每份报告记一条 `sha256` + 生成命令 + 开销档；
  · `tests/test_report_provenance.py` 每次跑测试都核对哈希；
  · **手编 → 哈希对不上 → 测试红**。要改内容只能改生成器再重跑。

## 两个开销档
`cheap`  —— 秒级、确定性，可在 `--verify-rerun` 里真正重跑对拍「逐字节相同」。
`heavy`  —— 需要训练（20s–130s），不进快速测试集；只记哈希与重生成命令。
`frozen` —— 预注册件，文件只读、**永不重跑**（重跑即毁掉预注册的意义）。
`api`    —— **重跑要花钱**（走 LLM API）。出处照记，但永不自动重跑——
           把「可重现」和「可免费重现」分开说，是诚实的一部分。
> 分档不是偷懒：把 10 分钟的训练塞进单元测试，结果是**没人再跑测试**。

用法：
  python -m src.eval.report_manifest              # 核对（CI / 手动）
  python -m src.eval.report_manifest --update     # 重生成后刷新清单
  python -m src.eval.report_manifest --verify-rerun  # 只对 cheap 档真重跑对拍
"""

import hashlib
import json
import os
import pathlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
MANIFEST = REPORTS / "_manifest.json"

# 报告 → (生成命令, 开销档)。**命令要能直接复制运行**，参数即口径。
GENERATORS = {
    "baseline_metrics.md": (["src.model.train_baseline"], "heavy"),
    "calibration.md": (["src.model.calibration"], "heavy"),
    "calib_window_size.md": (["src.model.calib_window_size"], "heavy"),
    "cost_sensitive.md": (["src.model.cost_sensitive"], "heavy"),
    "embargo_decomposition.md": (["src.model.embargo_control"], "heavy"),
    "graph_vs_tabular.md": (["src.model.graph_vs_tabular"], "heavy"),
    "graph_leak_audit.md": (["src.model.graph_leak_audit"], "heavy"),
    "graph_feature_ablation.md": (["src.model.graph_feature_ablation"], "heavy"),
    "imbalance_ablation.md": (["src.model.imbalance_ablation"], "heavy"),
    "drift_monitor.md": (["src.eval.drift_monitor"], "heavy"),
    "selective_bias.md": (["src.eval.selective_bias"], "heavy"),
    "agent_knowledge.md": (["src.agent.knowledge"], "heavy"),
    "eda_summary.md": (["notebooks.eda"], "heavy"),
    "graph_eda.md": (["src.features.graph_eda"], "cheap"),
    "sql_vs_pandas_reconciliation.md": (["src.features.build_duckdb"], "cheap"),
    "disposition.md": (["src.agent.disposition"], "cheap"),
    "agent_round4_metrics.md": (["src.eval.round4_metrics", "--score", "r3", "r4"], "cheap"),
    # round4_preregistration.md **不在此表**：冻结件的唯一真源是 report_io.FROZEN，
    # 登记两处必然有一天对不上（本轮就撞了一次）。
    "agent_v4_paired.md": (["src.eval.v4_paired"], "cheap"),
    "agent_flip_experiment.md": (["src.eval.flip_experiment", "--score"], "cheap"),
    # ↓ 需真实图特征的 60 天版本：靠环境变量切换，命令里必须带上，否则重跑的是 21 天
    "graph_vs_tabular_e60.md": (["GRAPH_FILE=data/processed/graph_features_e60.parquet",
                                 "src.model.graph_vs_tabular"], "heavy"),
    # ↓ 由归档离线重算，**零 API 花费且确定性**（已实测逐字节相同）→ cheap
    "agent_pipeline.md": (["src.agent.pipeline", "--report-only"], "cheap"),
    "agent_abstention.md": (["src.eval.abstention_test", "--score"], "cheap"),
    "agent_grounding.md": (["src.eval.agent_eval", "--grounding", "r1"], "cheap"),
    # 缺陷分类由 --score 顺带产出（无独立子命令）
    "agent_defect_taxonomy.md": (["src.eval.agent_eval", "--relabel-score", "r1"], "cheap"),
    "rules_vs_model.md": (["src.model.rules_vs_model"], "cheap"),
    "small_amount_floor.md": (["src.model.small_amount_floor"], "cheap"),
    "stepup.md": (["src.model.stepup"], "cheap"),
    "agent_gang_validity.md": (["src.eval.gang_validity"], "cheap"),
    "agent_gang_validity_replication.md": (["src.eval.gang_validity", "--compare"], "cheap"),
    # ↓ 参数即口径：裸跑会顶单轮默认、覆盖两轮对比（已改为缺参报错，此处记明正确用法）
    "agent_round3_metrics.md": (["src.eval.round3_metrics", "r1", "r3"], "cheap"),
    "agent_evidence_vs_decision.md": (["src.eval.evidence_vs_decision", "r1", "r3"], "cheap"),
    "agent_cost_attribution.md": (["src.eval.cost_attribution", "r1"], "cheap"),
    "agent_disposition_sensitivity.md": (["src.eval.disposition_sensitivity", "r1"], "cheap"),
    "agent_selfcheck.md": (["src.eval.self_awareness", "r1"], "cheap"),
}


# 归档锚：LLM 原始返回。整目录聚合成一个哈希，逐文件记会把清单撑爆。
ARCHIVES = {
    "reports/samples": "管道演习的原始返回（agent_pipeline.md 的锚）",
    "reports/eval_runs/r1": "round 1 全量调查原始返回",
    "reports/eval_runs/r3": "round 3 全量调查原始返回",
    "reports/eval_runs/r4": "round 4 全量调查原始返回",
    "reports/eval_runs/r4v4": "round 4 v4-prompt 配对原始返回",
    "reports/eval_runs/abstain": "剥夺实验原始返回",
    "reports/eval_runs/flip": "喂分翻转实验原始返回",
}


def tree_sha(d):
    """目录聚合哈希：按文件名排序，把「相对路径 + 内容哈希」串起来再哈希。

    这样任何一个归档件被改动 / 增删都会翻，而清单只多一行。
    """
    h = hashlib.sha256()
    files = sorted(p for p in d.rglob("*") if p.is_file())
    for p in files:
        h.update(str(p.relative_to(d)).encode("utf-8"))
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest(), len(files)


def sha(path):
    """**只哈希机器区。** 人写区归人，改它不算「手编构建产物」。

    若整文件参与哈希，判读文字每改一个字都会让测试变红，
    结果必然是有人把这条检查关掉——**过严的检查等于没有检查**。
    """
    from src.report_io import split_report
    machine, _ = split_report(path.read_text(encoding="utf-8"))
    return hashlib.sha256(machine.encode("utf-8")).hexdigest()


def has_human(path):
    from src.report_io import split_report
    return split_report(path.read_text(encoding="utf-8"))[1] is not None


def cmd_of(entry):
    """命令串。带 `=` 的首项视为环境变量前缀（如 GRAPH_FILE=...）。"""
    env = [e for e in entry if "=" in e and not e.startswith("-")]
    rest = [e for e in entry if e not in env]
    return (" ".join(env) + " " if env else "") + "python -m " + " ".join(rest)


def validate_generators():
    """**清单元数据自身也要被执法。**

    来历：我登记的生成器名有 1 处是错的（`eda_summary.md` 写成 `load_data`，实为
    `notebooks.eda`），另有 3 处登记的文件根本不存在——**执法工具差点带着
    未经核实的断言上线**。一个自己没被核实过的检查器，比没有检查器更坏：
    它会让人以为已经查过了。

    校验两件事，都不靠跑（跑一遍要十几分钟）：
      1. 生成器模块**存在且可导入**；
      2. 模块源码里**确实出现它所声称的那个输出文件名**（路径断言）。
    无法断言的一律标 `[未验证]`，**不许留空**——留空看起来像通过。
    """
    import importlib.util
    problems, unverified = [], []
    for name, (argv, tier) in sorted(GENERATORS.items()):
        mod = next((a for a in argv if not a.startswith("-") and "=" not in a), None)
        if mod is None:
            problems.append((name, "命令里没有模块名"))
            continue
        spec = importlib.util.find_spec(mod)
        if spec is None or not spec.origin:
            problems.append((name, f"模块不存在或不可导入：{mod}"))
            continue
        src = pathlib.Path(spec.origin).read_text(encoding="utf-8")
        stem = name.removesuffix(".md")
        # 直接出现全名，或以 f-string 拼接（如 graph_vs_tabular{_gsuffix}.md）
        if f'"{name}"' in src or f"'{name}'" in src:
            continue
        if any(f'"{p}' in src or f'f"{p}' in src for p in (stem.split("_e60")[0],)):
            unverified.append((name, mod, "文件名由 f-string 拼接，仅前缀匹配"))
            continue
        unverified.append((name, mod, "源码里找不到该输出文件名"))
    return problems, unverified


def build():
    out = {}
    for name, (argv, tier) in sorted(GENERATORS.items()):
        p = REPORTS / name
        if not p.exists():
            print(f"⚠️ 清单登记了但文件不存在：{name}")
            continue
        out[name] = {"sha256_machine_only": sha(p), "tier": tier,
                     "command": cmd_of(argv), "has_human_block": has_human(p)}
    _, unver = validate_generators()
    _unver_names = {n for n, _, _ in unver}
    for name in out:
        if name in _unver_names:
            out[name]["generator_verified"] = "[未验证]"
        elif not name.endswith("/"):
            out[name]["generator_verified"] = True
    out.update(__import__("src.report_io", fromlist=["x"]).frozen_digests())
    for rel, why in sorted(ARCHIVES.items()):
        d = ROOT / rel
        if not d.exists():
            print(f"⚠️ 归档目录不存在：{rel}")
            continue
        digest, n = tree_sha(d)
        out[rel + "/"] = {"sha256_tree": digest, "n_files": n, "tier": "archive",
                          "command": "（不可复现：LLM 原始返回。它是锚，不是产物）",
                          "why": why}
    return out


def verify():
    if not MANIFEST.exists():
        sys.exit("清单不存在，先跑 --update")
    want = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bad = []
    for name, rec in sorted(want.items()):
        # 冻结件的「出处」写的是理由而非命令，取值要按档分派——
        # 上一版直接读 rec["command"] 当场 KeyError。
        cmd = rec.get("command") or rec.get("why", "（冻结件，不可重跑）")
        if rec["tier"] == "frozen":
            p = ROOT / name
            if not p.exists():
                bad.append((name, "冻结件缺失", cmd))
            elif hashlib.sha256(p.read_bytes()).hexdigest() != rec["sha256"]:
                bad.append((name, "冻结件被改动 —— 其时间声明已失效", cmd))
            continue
        if rec["tier"] == "archive":
            d = ROOT / name.rstrip("/")
            if not d.exists():
                bad.append((name, "归档目录缺失", cmd))
            elif tree_sha(d)[0] != rec["sha256_tree"]:
                bad.append((name, "归档内容变了 —— 原始返回**不该**被改动", cmd))
            continue
        p = REPORTS / name
        if not p.exists():
            bad.append((name, "文件缺失", cmd))
        elif sha(p) != rec["sha256_machine_only"]:
            bad.append((name, "哈希不符（被手编过，或生成器改了没重跑）", cmd))
    print(f"核对 {len(want)} 份报告 —— {'✅ 全部一致' if not bad else f'❌ {len(bad)} 份不符'}")
    for name, why, c in bad:
        print(f"  {name}\n     {why}\n     重生成：{c}")
    sys.exit(1 if bad else 0)


def verify_rerun():
    """只对 cheap 档：真重跑一次，断言逐字节相同。

    **会临时改写 reports/**，跑完自动还原成重跑前的内容；
    因为它有副作用，所以不进快速测试集，由本命令显式触发。
    """
    ok, fail = [], []
    for name, (argv, tier) in sorted(GENERATORS.items()):
        if tier != "cheap":
            continue
        if not (p := REPORTS / name).exists() or not (p.stat().st_mode & 0o200):
            continue          # 只读 = 冻结件，绝不触碰
        p = REPORTS / name
        if not p.exists():
            continue
        before, mtime_before = p.read_bytes(), p.stat().st_mtime_ns
        env = dict(os.environ)
        for e in argv:
            if "=" in e and not e.startswith("-"):
                k, v = e.split("=", 1)
                env[k] = str(ROOT / v)
        r = subprocess.run([sys.executable, "-m", *[a for a in argv if "=" not in a]],
                           cwd=ROOT, capture_output=True, text=True, timeout=900, env=env)
        after = p.read_bytes() if p.exists() else b""
        touched = p.exists() and p.stat().st_mtime_ns != mtime_before
        if after != before:
            p.write_bytes(before)                   # 只在真被改动时才还原，不做无谓写入
        if r.returncode != 0:
            fail.append((name, f"生成器退出码 {r.returncode}"))
        elif not touched:
            # **假放行修复**：`agent_eval grounding r1`（漏了 --）退出码 0、
            # 只打印用法、**什么都不写**，而内容比对当然说「相同」。
            # 「内容没变」与「根本没被写」必须分开判——后者是登记的命令是错的。
            fail.append((name, "生成器退出码 0 但**没有写这个文件** —— 登记的命令多半是错的"))
        elif after != before:
            fail.append((name, "重新生成后内容不同 —— 非确定性或已被手编"))
        else:
            ok.append(name)
    print(f"cheap 档重跑对拍：{len(ok)} 份逐字节相同"
          + (f"、{len(fail)} 份不符" if fail else " ✅"))
    for name, why in fail:
        print(f"  ❌ {name}：{why}")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    _bad, _unver = validate_generators()
    if _bad:
        print("❌ 清单元数据有误（生成器登记不可信）：")
        for n, w in _bad:
            print(f"   {n}: {w}")
        sys.exit(1)
    if _unver:
        print(f"⚠️ {len(_unver)} 条生成器登记 **[未验证]**（源码里没直接出现该文件名）：")
        for n, m, w in _unver:
            print(f"   {n} ← {m}　{w}")
    if "--update" in sys.argv:
        MANIFEST.write_text(json.dumps(build(), ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print(f"✅ 清单已刷新：{len(json.loads(MANIFEST.read_text()))} 份 → "
              f"{MANIFEST.relative_to(ROOT)}")
    elif "--verify-rerun" in sys.argv:
        verify_rerun()
    else:
        verify()
