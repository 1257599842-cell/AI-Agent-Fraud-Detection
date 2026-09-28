"""从主实验两臂缓存复算 KS、十等分表与各窗指标；不训练模型。

运行：python -m src.model.bank_metrics
"""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.report_io import write_report

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / 'data' / 'processed' / 'gvt_scores.parquet'
REPORT = ROOT / 'reports' / 'bank_metrics.md'
N_BINS = 10


def _validated(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or not len(y):
        raise ValueError('标签和分数须为非空、等长一维数组')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all():
        raise ValueError('标签须为0/1，分数须有限')
    if len(np.unique(y)) != 2:
        raise ValueError('KS要求同时存在好样本与坏样本')
    return y.astype(bool), p


def ks(y, p):
    """返回单侧 max(TPR-FPR)、分数阈值、阈值及以上的样本数。

    同分必须整体纳入；逐行累计会凭同分样本的排列产生虚假 KS。
    包含全不选的 ROC 原点，返回阈值 inf、样本数0时 KS 为0。
    """
    y, p = _validated(y, p)
    order = np.argsort(-p, kind='stable')
    ys, ps = y[order], p[order]
    ends = np.r_[np.flatnonzero(ps[:-1] != ps[1:]), len(ps) - 1]
    delta = np.cumsum(ys)[ends] / ys.sum() - np.cumsum(~ys)[ends] / (~ys).sum()
    best = int(np.argmax(np.r_[0.0, delta]))
    if best == 0:
        return 0.0, float('inf'), 0
    i = int(ends[best - 1])
    return float(delta[best - 1]), float(ps[i]), i + 1


def decile_table(y, p, n_bins=N_BINS):
    """按分数稳定降序等人数分组；同分可跨组，组边界仅作描述。"""
    y, p = _validated(y, p)
    if not isinstance(n_bins, int) or not 1 <= n_bins <= len(y):
        raise ValueError('分组数须为1到样本数之间的整数')
    order = np.argsort(-p, kind='stable')
    ys, ps = y[order], p[order]
    edges = [int(round(len(y) * k / n_bins)) for k in range(n_bins + 1)]
    rows, cb, cg = [], 0, 0
    for k, (s, e) in enumerate(zip(edges[:-1], edges[1:])):
        bad = int(ys[s:e].sum())
        good = e - s - bad
        cb += bad
        cg += good
        rows.append({'组': k + 1, '分数区间': f'{ps[e-1]:.5f} – {ps[s]:.5f}',
                     '样本': e - s, '坏': bad, '好': good, '组内坏率': bad / (e - s),
                     '累计坏占比': cb / ys.sum(), '累计好占比': cg / (~ys).sum(),
                     'KS': cb / ys.sum() - cg / (~ys).sum()})
    return pd.DataFrame(rows)


def validate_scores(d):
    required = {'TransactionID', 'day', 'split', 'isFraud', 'p_tab', 'p_graph'}
    if not required <= set(d):
        raise ValueError(f'缓存缺列：{sorted(required - set(d))}')
    if d.TransactionID.isna().any() or d.TransactionID.duplicated().any():
        raise ValueError('缓存交易ID须非空且唯一')
    if d.day.isna().any() or not d.day.between(0, 181).all() or (d.day % 1 != 0).any():
        raise ValueError('缓存不符合主实验day [0,182)')
    expected = np.where(d.day < 132, 'fit', np.where(d.day < 146, 'val', 'test'))
    if not np.array_equal(expected, d.split.to_numpy()):
        raise ValueError('缓存切分不符合主实验fit<132 / val[132,146) / test[146,182)')
    for split in ('fit', 'val', 'test'):
        part = d[d.split == split]
        for col in ('p_tab', 'p_graph'):
            _validated(part.isFraud, part[col])
            if not part[col].between(0, 1).all():
                raise ValueError('缓存概率须在[0,1]')


def main():
    if not SCORES.exists():
        raise SystemExit(f'缺分数文件 {SCORES.name}，请先生成主实验缓存')
    d = pd.read_parquet(SCORES)
    validate_scores(d)
    rows = {}
    for split in ('fit', 'val', 'test'):
        part = d[d.split == split]
        rows[split] = {col: {'ks': ks(part.isFraud, part[col])[0],
                            'pr': average_precision_score(part.isFraud, part[col]),
                            'roc': roc_auc_score(part.isFraud, part[col])}
                       for col in ('p_tab', 'p_graph')}
    te = d[d.split == 'test']
    a, b = rows['test']['p_tab'], rows['test']['p_graph']
    L = ['# 银行风控口径：KS、十等分表与时间窗对照\n',
         '**来源**：`data/processed/gvt_scores.parquet`，本次直接读取两臂缓存，不重训。',
         f'缓存 SHA-256：`{hashlib.sha256(SCORES.read_bytes()).hexdigest()}`。',
         '缓存由主实验生成器落盘；哈希标识本次输入，不单独证明其历史生成过程。',
         'fit [0,132)、早停 val [132,146)、test [146,182)；训练/早停无统一标签成熟隔离。',
         '图标签历史的21天延迟与训练隔离是两回事。所有下表指标均从缓存复算。\n',
         f'测试 {len(te):,} 笔，欺诈 {int(te.isFraud.sum()):,} 笔，欺诈率 {te.isFraud.mean():.2%}。\n',
         '## 1. 同源测试窗 delta\n',
         '| 模型 | KS | PR-AUC（AP） | ROC-AUC |', '|---|---|---|---|',
         f"| 纯表 | {a['ks']:.4f} | {a['pr']:.4f} | {a['roc']:.4f} |",
         f"| 表+图 | {b['ks']:.4f} | {b['pr']:.4f} | {b['roc']:.4f} |",
         f"| delta | {b['ks']-a['ks']:+.4f} | {b['pr']-a['pr']:+.4f} | {b['roc']-a['roc']:+.4f} |", '',
         f"KS 由 **{a['ks']:.4f}** 升至 **{b['ks']:.4f}**。delta使用未舍入值计算。",
         f"表+图的精确KS切点为 p ≥ {ks(te.isFraud, te.p_graph)[1]:.8f}；不是业务处置阈值。\n",
         '## 2. 十等分表\n',
         '按分数稳定降序等人数分组，同分以缓存行序排序，可能跨组。',
         '逐组KS仅在十个组末计算，是描述性统计；精确KS在每个不同分数阈值处计算，同分整体纳入。',
         '因此十等分表最大值不必等于精确KS，也不可把跨同分的组边界当成可执行阈值。\n']
    for col, name in [('p_tab', '纯表'), ('p_graph', '表+图')]:
        L += [f'### {name}\n', '| 组 | 分数区间 | 样本 | 坏 | 好 | 组内坏率 | 累计坏占比 | 累计好占比 | 逐组KS |',
              '|---|---|---|---|---|---|---|---|---|']
        for _, r in decile_table(te.isFraud, te[col]).iterrows():
            L.append(f"| {r['组']} | {r['分数区间']} | {r['样本']:,} | {r['坏']:,} | {r['好']:,} | "
                     f"{r['组内坏率']:.2%} | {r['累计坏占比']:.2%} | {r['累计好占比']:.2%} | {r['KS']:.4f} |")
        L.append('')
    L += ['## 3. 各窗KS：样本内乐观与时间变化\n',
          '| 窗口 | 样本 | 纯表KS | 表+图KS |', '|---|---|---|---|']
    for split, name in [('fit', '训练 [0,132)，样本内'), ('val', '验证 [132,146)，用于早停'), ('test', '测试 [146,182)')]:
        L.append(f"| {name} | {int((d.split==split).sum()):,} | {rows[split]['p_tab']['ks']:.4f} | {rows[split]['p_graph']['ks']:.4f} |")
    L += [f"| 训练减测试 | — | {rows['fit']['p_tab']['ks']-a['ks']:+.4f} | {rows['fit']['p_graph']['ks']-b['ks']:+.4f} |", '',
          '训练窗是样本内预测，验证窗参与早停；两者都不是独立时间外测试。',
          '训练减测试混合了样本内乐观、时间分布变化等因素，不能单独测量漂移或证明稳定。',
          '旧窗隔离对照也同时改变训练样本、验证窗与早停，不能把这个KS差拆成纯新鲜度效应。\n',
          '## 4. 结果解释与限制\n',
          '> 本数据上的交易反欺诈KS较高，但跨任务的目标、样本与标签机制不同，不能直接比较。',
          '> 高KS本身不能排除泄漏。图标签延迟从21天改为60天时，PR-AUC增益仍为正',
          '> （+0.0387→+0.0234），只支持这项延迟敏感性结果，不能证明缩水完全来自新鲜度，',
          '> 也不能替未公开生成过程的匿名C/V特征背书。参见 `graph_leak_audit.md` 与 `../EXPERIMENT_PROTOCOL.md`。\n',
          '本报告测试窗此前已用于研究，属于回顾性复算，不能包装成新的盲测。',
          'PSI与滚动AUC已有独立早期模型实验，见 `drift_monitor.md`；不能与这里的主模型窗口混用。',
          'KS阈值定义参照 [scikit-learn ROC文档](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.roc_curve.html)。']
    write_report(REPORT, '\n'.join(L))
    print(f"KS {a['ks']:.6f} → {b['ks']:.6f}; AP {a['pr']:.6f} → {b['pr']:.6f}; {REPORT}")


if __name__ == '__main__':
    main()
