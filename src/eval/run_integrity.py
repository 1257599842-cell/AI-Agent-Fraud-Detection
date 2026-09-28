"""全轮汇总必须有完整的既定样本；缺测不能经 inner join 静默消失。"""
import json
from pathlib import Path


def validate_complete_run(results, eval_set):
    from src.agent.disposition import ACTIONS
    ids = [r.get('txn_id') for r in results]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('评估轮次为空或交易ID重复')
    if eval_set['TransactionID'].duplicated().any():
        raise ValueError('评估集ID重复')
    allowed = [set(eval_set['TransactionID'])]
    allowed += [set(g['TransactionID']) for _,g in eval_set.groupby('split')]
    if set(ids) not in allowed:
        raise ValueError('轮次必须恰好覆盖完整dev、完整holdout或完整评估集；缺测/额外ID不可静默丢弃')
    for r in results:
        rep=r.get('report')
        if r.get('mode') != 'llm' or not isinstance(rep,dict) or rep.get('disposition') not in ACTIONS:
            raise ValueError(f"{r.get('txn_id')} 不是有效LLM处置；降级或无效输出必须作为缺测处理")
    return results


def load_complete_run(directory, eval_set):
    results=[json.loads(p.read_text()) for p in sorted(Path(directory).glob('txn_*.json'))]
    return validate_complete_run(results,eval_set)
