"""五动作下 approve 的金额上界：逐一比较所有仿射成本，而非只比 stepup/decline。

边界依赖 p、gang_score 和全部成本假设。金额为零时可能并列；本文讨论正金额。
运行 python -m src.model.small_amount_floor 重生成报告。
"""
from pathlib import Path
import numpy as np
import pandas as pd
from src.agent.disposition import BASE
from src.model.stepup import ACTIONS5, STEPUP, costs5, argmin5
from src.report_io import write_report

ROOT = Path(__file__).resolve().parents[2]
GT = ROOT / 'data/processed/agent_disposition_gt.parquet'
REPORT = ROOT / 'reports/small_amount_floor.md'
A_MED = 76.02


def action_boundaries(p, g=0.0, a_med=A_MED, prm=None, stepup=None):
    """返回 approve 相对每个干预动作、从金额 0 起保持最优的区间上界。

    E_j - E_approve = b_j - d_j*a。b<0 时该动作在 0 已更便宜；
    b>=0,d<=0 时无有限上界；否则上界 b/d。并列按 ACTIONS5 顺序选 approve。
    """
    if not np.isfinite(p) or not 0 <= p <= 1 or not np.isfinite(g) or not 0 <= g <= 1:
        raise ValueError('p 和 gang_score 必须为 [0,1] 内有限数')
    b = BASE if prm is None else prm
    s = STEPUP if stepup is None else stepup
    terms = {
        'stepup': (s['c_friction'], p*s['r_block']-(1-p)*s['r_abandon']*s['margin_rate']),
        'hold': (b['c_review']+(1-p)*b['f_h']*b['c_fp'], p*(1-b['m_h'])),
        'decline': ((1-p)*b['c_fp'], p),
        'escalate': (b['c_report']+(1-p)*b['f_e']*b['c_fp']-p*g*b['k_future']*a_med,
                     p*(1-b['m_e'])),
    }
    return {k: 0.0 if intercept < 0 else intercept/d if d > 0 else np.inf
            for k, (intercept, d) in terms.items()}


def approve_floor(p, g=0.0, a_med=A_MED, prm=None, stepup=None):
    """正金额放行区间的上界；0=没有从零开始的正金额区间，inf=无有限上界。"""
    return min(action_boundaries(p, g, a_med, prm, stepup).values())


def floor_numeric(p, g=0.0):
    """独立地调用五档成本，倍增找括区再二分；仅用于基准参数核验。"""
    def approve(a):
        return argmin5(np.array([p]), np.array([a]), np.array([g]), A_MED, BASE, STEPUP)[0] == 'approve'
    if not approve(0.0) or not approve(1e-12):
        return 0.0
    lo, hi = 0.0, 1.0
    while approve(hi):
        lo, hi = hi, hi*2
        if hi > 1e16:
            return np.inf
    for _ in range(100):
        mid = (lo+hi)/2
        if approve(mid): lo = mid
        else: hi = mid
    return (lo+hi)/2


def _money(x):
    return '∞' if np.isinf(x) else f'${x:.4f}'


def main():
    lines = ['# 小额放行边界：完整五动作核验', '',
        '本报告修正旧版遗漏 hold 的公式。所有成本参数均为假设；结论是模型内的成本排序，不能当作实际拦截效果。', '',
        '## 1. 完整闭式解', '',
        '设 `E_j − E_approve = b_j − d_j·a`，逐一比较四种干预：', '',
        '| 动作 | b_j | d_j |', '|---|---|---|',
        '| stepup | c_friction | p·r_block − (1−p)·r_abandon·margin_rate |',
        '| hold | c_review + (1−p)·f_h·c_fp | p·(1−m_h) |',
        '| decline | (1−p)·c_fp | p |',
        '| escalate | c_report + (1−p)·f_e·c_fp − p·g·k_future·A_med | p·(1−m_e) |', '',
        '`b_j<0` 表示该动作在金额零处已胜出，放行区间上界为零；否则 `d_j>0` 时上界为 `b_j/d_j`，`d_j≤0` 时为无穷。',
        '**完整上界是四项的最小值。** 仅取 stepup 与 decline，要求另两项边界均不更小。',
        '边界处可能并列；实现按动作顺序优先 approve。p=1、g=0 时 decline 成本为零，对任何正金额都优于 approve；金额为零时二者并列。', '',
        f'基准 BASE={BASE}；STEPUP={STEPUP}；A_med={A_MED}。', '',
        '## 2. gang=0：闭式与独立数值二分', '',
        '| p | stepup | hold | decline | escalate | 完整上界 | 数值二分 | 约束动作 |',
        '|---|---|---|---|---|---|---|---|---|']
    for p in [0, .01, .05, .1, .3, .5, .7, .9, .99, .999, 1]:
        bounds=action_boundaries(p); boundary=min(bounds.values()); numeric=floor_numeric(p)
        if not np.isclose(boundary, numeric, rtol=1e-9, atol=1e-8):
            raise AssertionError((p,boundary,numeric))
        binding='无有限约束' if np.isinf(boundary) else min(bounds,key=bounds.get)
        lines.append(f'| {p:g} | '+ ' | '.join(_money(x) for x in bounds.values())+
                     f' | **{_money(boundary)}** | {_money(numeric)} | {binding} |')
    lines += ['', '**全部核验通过。** p=0.01 时旧两项公式为 $2475，完整边界为 $610.5556，由 hold 约束；p=0.30 时为 $2.4420，由 stepup 约束。',
              '“低 p 必由 stepup 约束”与“定性形状不随参数变化”均不成立。', '',
              '## 3. 网络项也可能改变边界', '', '| p | gang | 完整上界 | 数值二分 |','|---|---|---|---|']
    for p,g in [(.01,1),(.1,1),(.3,.5),(.3,1),(.99,1)]:
        v,n=approve_floor(p,g),floor_numeric(p,g)
        if not np.isclose(v,n,rtol=1e-9,atol=1e-8): raise AssertionError((p,g,v,n))
        lines.append(f'| {p} | {g} | {_money(v)} | {_money(n)} |')
    gt=pd.read_parquet(GT)
    p,a,g,y=(gt[c].to_numpy() for c in ['p','TransactionAmt','gang_score','isFraud'])
    five=argmin5(p,a,g,A_MED,BASE,STEPUP)
    zero=argmin5(p,a,np.zeros(len(gt)),A_MED,BASE,STEPUP)
    lines += ['', '## 4. 主实验历史分数上的离线金额分带', '',
        '这些分数来自主实验既有评估窗，已被多轮分析使用，不是新增未使用测试集。', '',
        '| 金额段 | 笔数 | 欺诈笔数 | 五档 approve | 令 gang=0 后 approve | 网络项取消后转为 approve |',
        '|---|---|---|---|---|---|']
    for name,m in [('< $1',a<1),('< $2.44',a<2.4420),('$2.44–$10',(a>=2.4420)&(a<10)),('全体',np.ones(len(a),bool))]:
        n=int(m.sum())
        lines.append(f'| {name} | {n:,} | {int(y[m].sum())} | {(five[m]=="approve").mean():.1%} | '
                     f'{(zero[m]=="approve").mean():.1%} | {int((m&(five!="approve")&(zero=="approve")).sum())} |')
    lines += ['', '## 5. 适用范围', '',
        '- 微额不意味着必然放行：概率高时 decline 会胜出，网络项也可使 escalate 胜出。',
        '- 训练期试卡簇的标签分布不能外推为测试期分布；测试期低于 $1 的 109 笔中没有欺诈。',
        '- 旧版称“非 approve 全部来自网络项”也不准确：取消网络项后仍有其他干预，见表。',
        '- 金额边界随全部成本假设变化；$2.44 仅是 p=0.30、gang=0、基准参数的具体例子。',
        '- 五动作仅在离线分析与沙盘中比较；当前服务及 Agent 历史评估仍为四动作。']
    write_report(REPORT,'\n'.join(lines))
    print(f'全部闭式/数值核验通过 → {REPORT}')

if __name__ == '__main__': main()
