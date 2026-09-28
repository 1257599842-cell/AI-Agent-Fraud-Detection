"""标签可得性回顾性审计；独立产物，不覆盖主实验模型、分数或 Kaggle 提交。

协议所有模型选择在冻结时刻 day146 前完成；标签可得时间是假设 event+21d。
[146,182) 已在项目中使用过，因此这是 retrospective audit，不是新的盲测。
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from src.model.graph_vs_tabular import load
from src.model.train_baseline import LGB_PARAMS
from src.model.cost_sensitive import T_GRID,cost_at,metrics_at
from src.report_io import write_report

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/label_availability_audit.md'
RESULT=ROOT/'reports/label_availability_audit.json'
PROTOCOL={
 'kind':'retrospective_label_availability_audit',
 'freeze_day':146,'label_delay_days':21,
 'tree_fit':[0,90],'early_stopping':[90,104],
 'calibrator_fit':[104,115],'selection':[115,125],
 'evaluation':[146,182],
 'selection_criterion':'lowest Brier on selection; ties prefer raw, then platt, then isotonic',
 'threshold_criterion':'minimum labeled cost on selection; c_fp=25; fixed T_GRID',
 'limitations':['evaluation period was previously used; no claim of untouched holdout',
                'synthetic constant label delay; anonymous feature availability unknown',
                'costs assumed; no observed treatment outcomes'],
}


def masks_for(day):
    masks={k:(day>=lo)&(day<hi) for k,(lo,hi) in
           ((k,PROTOCOL[k]) for k in ['tree_fit','early_stopping','calibrator_fit','selection','evaluation'])}
    used=np.zeros(len(day),dtype=int)
    for k,m in masks.items():
        if not m.any(): raise ValueError(f'{k} 没有样本')
        used+=m
        if k!='evaluation' and np.any(day[m]+PROTOCOL['label_delay_days']>=PROTOCOL['freeze_day']):
            raise AssertionError(f'{k} 含冻结时刻尚不可得的标签')
    if (used>1).any(): raise AssertionError('阶段窗口重叠')
    return masks


def run():
    X,y,day,gcols=load();m=masks_for(day)
    # 类别字典只来自树训练窗；其他窗口新类别变为缺失。
    for c in X.select_dtypes(include='category').columns:
        X[c]=pd.Categorical(X[c],categories=X.loc[m['tree_fit'],c].dropna().unique())
    y=y.to_numpy(); results={}
    for name,cols in [('tabular',[c for c in X if c not in gcols]),('tabular_graph',list(X))]:
        print('训练',name,flush=True)
        bst=lgb.train({**LGB_PARAMS,'num_threads':4},lgb.Dataset(X.loc[m['tree_fit'],cols],label=y[m['tree_fit']]),
            num_boost_round=2000,
            valid_sets=[lgb.Dataset(X.loc[m['early_stopping'],cols],label=y[m['early_stopping']])],
            callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(0)])
        ps={k:bst.predict(X.loc[v,cols]) for k,v in m.items() if k in ['calibrator_fit','selection','evaluation']}
        margins={k:bst.predict(X.loc[m[k],cols],raw_score=True).reshape(-1,1) for k in ps}
        iso=IsotonicRegression(out_of_bounds='clip').fit(ps['calibrator_fit'],y[m['calibrator_fit']])
        platt=LogisticRegression().fit(margins['calibrator_fit'],y[m['calibrator_fit']])
        candidates={
            'raw':ps,
            'platt':{k:platt.predict_proba(v)[:,1] for k,v in margins.items()},
            'isotonic':{k:iso.predict(v) for k,v in ps.items()},
        }
        briers={k:float(brier_score_loss(y[m['selection']],v['selection'])) for k,v in candidates.items()}
        chosen=min(briers,key=briers.get)
        pp=candidates[chosen]['selection']; amt=X.loc[m['selection'],'TransactionAmt'].to_numpy()
        threshold=float(T_GRID[np.argmin([cost_at(t,pp,y[m['selection']],amt,25) for t in T_GRID])])
        ptest=candidates[chosen]['evaluation']; ye=y[m['evaluation']]; ae=X.loc[m['evaluation'],'TransactionAmt'].to_numpy()
        ti=25/(ae+25);blocked=ptest>ti
        results[name]={
            'best_iteration':int(bst.best_iteration),'features':len(cols),
            'selection_brier':briers,'chosen_calibration':chosen,'selection_threshold':threshold,
            'test_roc_auc':float(roc_auc_score(ye,ptest)),
            'test_pr_auc':float(average_precision_score(ye,ptest)),
            'test_brier':float(brier_score_loss(ye,ptest)),
            'test_threshold_metrics':metrics_at(threshold,ptest,ye,ae),
            'test_cost_selected_threshold':float(cost_at(threshold,ptest,ye,ae,25)),
            'test_cost_05':float(cost_at(.5,ptest,ye,ae,25)),
            'test_cost_amount_policy':float((25*((ye==0)&blocked)).sum()+ae[(ye==1)&~blocked].sum()),
        }
    rec={'protocol':PROTOCOL,'phase_counts':{k:int(v.sum()) for k,v in m.items()},'results':results,
         'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    RESULT.write_text(json.dumps(rec,ensure_ascii=False,indent=2)+'\n')
    lines=['# 标签可得性：独立回顾性审计','',
      '**本实验不覆盖主实验，也不是新增盲测。** 评估窗已经被既有研究使用；固定 21 天延迟也不是实际拒付时间。','',
      '## 固定协议','', '整个模型包在 day146 冻结。树训练、早停、校准器拟合和方法/阈值选择的标签均在冻结前成熟。',
      '方法选择以 selection 窗 Brier 最小为准；并列依次偏好 raw、Platt、Isotonic。阈值在同一 selection 窗按假设 c_FP=$25 选好，再评估末段。', '',
      '| 阶段 | 左闭右开日区间 | 样本数 |','|---|---|---|']
    for k,n in rec['phase_counts'].items():lines.append(f'| {k} | {PROTOCOL[k]} | {n:,} |')
    lines+=['','类别字典仅来自树训练窗。表+图使用既有图特征的逐笔成熟历史；测试期较早交易标签成熟后可供之后交易特征使用，模型不更新。',
      '匿名 C/V 字段的原始计算时间不可验证；本实验不消除该数据限制。','',
      '## 结果','', '| 臂 | 早停轮数 | 选中校准 | 阈值 | PR-AUC | ROC-AUC | Brier |','|---|---|---|---|---|---|---|']
    for k,r in results.items(): lines.append(f'| {k} | {r["best_iteration"]} | {r["chosen_calibration"]} | {r["selection_threshold"]:.6f} | {r["test_pr_auc"]:.6f} | {r["test_roc_auc"]:.6f} | {r["test_brier"]:.6f} |')
    lines+=['','| 臂 | cost@0.5 | cost@预选阈值 | 金额感知策略成本 |','|---|---|---|---|']
    for k,r in results.items():lines.append(f'| {k} | ${r["test_cost_05"]:,.2f} | ${r["test_cost_selected_threshold"]:,.2f} | ${r["test_cost_amount_policy"]:,.2f} |')
    lines+=['','成本为真标签代入假设成本函数的离线计算，非实际业务节省。',
      '主实验与本审计同时改变了拟合窗、验证窗、校准及类别字典；不得把差异全归因于标签延迟。',
      '## 尚缺的最终验证','',
      '取得未被研究/选参使用的新时段与真实标签到达时间后，先冻结协议与全部参数，再做一次最终评估。现有末段重切不能恢复盲态。',
      '机器可读结果及协议见 `label_availability_audit.json`；可重跑 `python -m src.model.label_availability_audit`。']
    write_report(REPORT,'\n'.join(lines)); print(json.dumps(results,ensure_ascii=False),flush=True)
if __name__=='__main__':run()
