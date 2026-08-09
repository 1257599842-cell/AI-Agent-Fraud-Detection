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
    "agent_round4_metrics.md": (["src.eval.round4_metrics"], "cheap"),
    # **预注册件，文件权限 444 故意只读**：跑完不许回头改指标。
    # 它绝不进任何自动重跑——重跑它就等于毁掉预注册本身的意义。
    "round4_preregistration.md": (["src.eval.round4_metrics"], "frozen"),
    "agent_v4_paired.md": (["src.eval.v4_paired"], "cheap"),
    "agent_flip_experiment.md": (["src.eval.flip_experiment"], "cheap"),
    "kaggle_submission.md": (["src.model.kaggle_submit"], "cheap"),
    # ↓ 需真实图特征的 60 天版本：靠环境变量切换，命令里必须带上，否则重跑的是 21 天
    "graph_vs_tabular_e60.md": (["GRAPH_FILE=data/processed/graph_features_e60.parquet",
                                 "src.model.graph_vs_tabular"], "heavy"),
    # ↓ **api 档：重跑要花钱**。出处照记，但不列入任何自动重跑。
    "agent_pipeline.md": (["src.agent.pipeline"], "api"),
    "agent_abstention.md": (["src.eval.abstention_test", "--score"], "api"),
    "agent_grounding.md": (["src.eval.agent_eval", "grounding", "r1"], "api"),
    "agent_defect_taxonomy.md": (["src.eval.agent_eval", "taxonomy", "r1"], "api"),
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


def build():
    out = {}
    for name, (argv, tier) in sorted(GENERATORS.items()):
        p = REPORTS / name
        if not p.exists():
            print(f"⚠️ 清单登记了但文件不存在：{name}")
            continue
        out[name] = {"sha256_machine_only": sha(p), "tier": tier,
                     "command": cmd_of(argv), "has_human_block": has_human(p)}
    return out


def verify():
    if not MANIFEST.exists():
        sys.exit("清单不存在，先跑 --update")
    want = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bad = []
    for name, rec in sorted(want.items()):
        p = REPORTS / name
        if not p.exists():
            bad.append((name, "文件缺失", rec["command"]))
        elif sha(p) != rec["sha256_machine_only"]:
            bad.append((name, "哈希不符（被手编过，或生成器改了没重跑）", rec["command"]))
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
        before = p.read_bytes()
        r = subprocess.run([sys.executable, "-m", *argv], cwd=ROOT,
                           capture_output=True, text=True, timeout=900)
        after = p.read_bytes() if p.exists() else b""
        if after != before:
            p.write_bytes(before)                   # 只在真被改动时才还原，不做无谓写入
        if r.returncode != 0:
            fail.append((name, f"生成器退出码 {r.returncode}"))
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
    if "--update" in sys.argv:
        MANIFEST.write_text(json.dumps(build(), ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print(f"✅ 清单已刷新：{len(json.loads(MANIFEST.read_text()))} 份 → "
              f"{MANIFEST.relative_to(ROOT)}")
    elif "--verify-rerun" in sys.argv:
        verify_rerun()
    else:
        verify()
