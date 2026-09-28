"""独立审查的可复现反例；不调用外部 LLM、不依赖交易数据。"""
import copy
import json
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pandas as pd


class KnownLabelDenominator(unittest.TestCase):
    def frame(self, labels):
        return pd.DataFrame({'TransactionID':np.arange(len(labels)),
            'TransactionDT':np.arange(len(labels))*30*86400,'isFraud':labels,
            'card1':[1]*len(labels),'addr1':[100.0]*len(labels),
            'P_emaildomain':['x']*len(labels),'DeviceInfo':['d']*len(labels)})

    def test_unknown_mature_history_is_not_negative_label(self):
        from src.model.kaggle_submit import _causal_graph_features
        out=_causal_graph_features(self.frame([1,np.nan,np.nan]))
        for key in ['card1','card1_addr1','card1_email','card1_device']:
            self.assertEqual(out.iloc[2][key+'_prior_cnt'],2)
            self.assertEqual(out.iloc[2][key+'_prior_fraud_rate'],1.0)

    def test_only_unknown_history_has_nan_rate(self):
        from src.model.kaggle_submit import _causal_graph_features
        self.assertTrue(np.isnan(_causal_graph_features(self.frame([np.nan]*3)).iloc[2]['card1_prior_fraud_rate']))

    def test_online_accepts_actual_null_labels_and_integer_keys(self):
        from src.serving.feature_store import FeatureStore
        st=FeatureStore(); self.addCleanup(st.close)
        df=self.frame([1,np.nan,np.nan]); st.append_frame(df)
        row=df.iloc[2].to_dict(); row['addr1']=100
        out=st.get_features(row)
        self.assertEqual(out['card1_addr1_prior_fraud_rate'],1.0)
        self.assertEqual(out['card1_prior_cnt'],2)

    def test_zero_delay_cannot_read_current_or_future_event(self):
        from src.serving.feature_store import FeatureStore
        st=FeatureStore(); self.addCleanup(st.close)
        df=self.frame([1,0,1]); st.append_frame(df,label_delay=0)
        self.assertEqual(st.get_features(df.iloc[1].to_dict())['card1_prior_fraud_rate'],1.0)
        with self.assertRaises(ValueError): st.append_frame(df,label_delay=-1)


class FiveActionBoundaries(unittest.TestCase):
    def test_counterexample_hold_binds(self):
        from src.model.small_amount_floor import approve_floor, action_boundaries
        bounds=action_boundaries(.01)
        self.assertAlmostEqual(min(bounds['stepup'],bounds['decline']),2475.)
        self.assertAlmostEqual(approve_floor(.01),610.5555555556)

    def test_all_actions_over_probability_and_gang_grid(self):
        from src.agent.disposition import BASE
        from src.model.stepup import argmin5,STEPUP
        from src.model.small_amount_floor import approve_floor,A_MED
        for p in [.00001,.001,.01,.05,.1,.3,.9,.99,.999,1]:
            for g in [0,.1,.5,1]:
                boundary=approve_floor(p,g)
                with self.subTest(p=p,g=g):
                    if boundary>0:
                        self.assertEqual(argmin5([p],[boundary*.9999],[g],A_MED,BASE,STEPUP)[0],'approve')
                    self.assertNotEqual(argmin5([p],[max(boundary*1.0001,1e-8)],[g],A_MED,BASE,STEPUP)[0],'approve')
        self.assertTrue(np.isinf(approve_floor(0)))


class AgentBoundaries(unittest.TestCase):
    def test_malformed_nested_values_are_diagnosed_without_crash(self):
        from src.agent.schema import EXAMPLE_REPORT,validate_report
        from src.agent.pipeline import enforce_evidence_floor
        from src.agent.tools import FactRegistry
        for changes in [{'key_findings':None},{'key_findings':[1]},
                        {'key_findings':[{'assertion_strength':[],'evidence_ids':[{}]}]},
                        {'txn_id':True},{'gang_association':{'entities':None}}]:
            rep=copy.deepcopy(EXAMPLE_REPORT);rep.update(changes)
            enforce_evidence_floor(rep,FactRegistry())
            self.assertTrue(validate_report(rep,set()))

    def test_null_result_is_not_mature_label_evidence(self):
        from src.agent.tools import FactRegistry,null_fact
        from src.agent.pipeline import enforce_evidence_floor
        reg=FactRegistry();null_fact(reg,'STAT','e','无样本',(0,0),True)
        out,_=enforce_evidence_floor({'evidence_insufficient':False,'key_findings':[]},reg)
        self.assertTrue(out['evidence_insufficient'])

    def test_single_response_cannot_exceed_tool_budget(self):
        from src.agent import pipeline as pl
        from src.agent.schema import EXAMPLE_REPORT
        from src.agent.tools import ToolResult
        class Backend:
            as_of=10000000
            calls=0
            def __init__(self,*args): pass
            def query_transaction(self):
                Backend.calls+=1
                return ToolResult('query_transaction',[],'')
        report=copy.deepcopy(EXAMPLE_REPORT);report['txn_id']=1
        responses=[NS(stop_reason='tool_use',usage=NS(input_tokens=1,output_tokens=1),
            content=[NS(type='tool_use',id=str(i),name='query_transaction',input={}) for i in range(12)]),
            NS(stop_reason='end_turn',usage=NS(input_tokens=1,output_tokens=1),
               content=[NS(type='text',text=json.dumps(report))])]
        from unittest.mock import Mock
        client=NS(messages=NS(create=Mock(side_effect=responses)))
        res=NS(gt=pd.DataFrame({'p':[.2]},index=[1]))
        with patch.object(pl,'DataBackedTools',Backend): out=pl.investigate(res,1,client)
        self.assertEqual(Backend.calls,8);self.assertEqual(out['tool_calls'],8)
        self.assertEqual(out['budget_rejections'],4)
        self.assertEqual(client.messages.create.call_args.kwargs['tool_choice'],{'type':'none'})

    def test_invalid_draft_is_retained_only_for_diagnostics(self):
        from src.agent import pipeline as pl
        res=NS(gt=pd.DataFrame({'p':[.5],'disposition_gt':['hold']},index=[1]))
        bad={'report':{'summary':'bad'},'schema_violations':['bad id'],'time_audit_violations':[],
             'cost_usd':.2,'tokens':{'input':10},'api_calls':1,'tool_calls':1}
        fallback={'report':{'summary':'safe'},'mode':'degraded'}
        with patch.object(pl,'investigate',return_value=bad),patch.object(pl,'degraded_report',return_value=fallback):
            out=pl.run_one(res,1,object())
        self.assertEqual(out['report']['summary'],'safe')
        self.assertEqual(out['rejected_draft'],bad)
        self.assertEqual(out['cost_usd'],.2)

    def test_gate_does_not_construct_optional_client(self):
        from src.agent import pipeline as pl
        res=NS(gt=pd.DataFrame({'p':[.001],'disposition_gt':['approve']},index=[1]))
        with patch.object(pl,'_make_client',side_effect=AssertionError('不应创建')):
            self.assertEqual(pl.run_one(res,1,None)['mode'],'gated')

    def test_missing_sdk_degrades(self):
        from src.agent import pipeline as pl
        res=NS(gt=pd.DataFrame({'p':[.5],'disposition_gt':['hold']},index=[1]))
        with patch.object(pl,'_make_client',side_effect=RuntimeError('缺 SDK')),patch.object(pl,'degraded_report',return_value={'mode':'degraded'}):
            self.assertEqual(pl.run_one(res,1,None)['mode'],'degraded')


class InputValidation(unittest.TestCase):
    def test_amount_cannot_silently_be_zero(self):
        from pydantic import ValidationError
        from src.serving.app import RawTxnRequest
        for amount in [None,-1,float('nan'),float('inf'),'not-money',True]:
            with self.subTest(amount=amount),self.assertRaises(ValidationError):
                RawTxnRequest(transaction_dt=1,fields={'TransactionAmt':amount})
        self.assertEqual(RawTxnRequest(transaction_dt=1,fields={'TransactionAmt':'12.5'}).fields['TransactionAmt'],12.5)

class AdditionalAuditRegressions(unittest.TestCase):
    def test_missing_fields_rejected(self):
        from pydantic import ValidationError
        from src.serving.app import RawTxnRequest
        with self.assertRaises(ValidationError): RawTxnRequest(transaction_dt=1)

    def test_demo_passes_full_model_input(self):
        from src.serving import app as service
        raw=pd.DataFrame({'TransactionID':[1],'TransactionDT':[20000000],
                          'TransactionAmt':[10.],'V307':[3.75],'isFraud':[1]})
        res=NS(meta=pd.DataFrame({'TransactionDT':[20000000]},index=[1]))
        with patch.object(service,'_res',return_value=res),patch('pandas.read_parquet',return_value=raw),patch.object(service,'score') as score:
            service.demo_score(1)
        fields=score.call_args.args[0].fields
        self.assertEqual(fields['V307'],3.75)
        self.assertNotIn('isFraud',fields)

    def test_sql_date_does_not_round_at_noon(self):
        import duckdb
        from pathlib import Path
        times=[86400,129600,172799,172800,8*86400]
        raw=pd.DataFrame({'TransactionID':range(5),'TransactionDT':times,
                          'TransactionAmt':[1.]*5,'isFraud':[0]*5})
        for col in ['card1','card4','card6','addr1','addr2','P_emaildomain','R_emaildomain','DeviceInfo','DeviceType','ProductCD']:
            raw[col]='x'
        con=duckdb.connect();self.addCleanup(con.close);con.register('raw_txn',raw)
        con.execute((Path(__file__).resolve().parents[1]/'src/features/sql/01_star_schema.sql').read_text())
        self.assertEqual(con.execute('select date_sk from fact_transaction order by transaction_id').fetchnumpy()['date_sk'].tolist(),[0,0,0,1,7])
        self.assertEqual(con.execute('select week_idx from dim_date where day=7').fetchone()[0],1)

    def test_unknown_disposition_cannot_get_free_cost(self):
        from src.agent.disposition import realized_cost, BASE
        with self.assertRaises(ValueError): realized_cost(['bogus'],[1],[100],[0],76.02,BASE)
        self.assertEqual(realized_cost(['approve'],[1],[100],[0],76.02,BASE),100)

    def test_report_snapshot_restores_side_outputs_on_error(self):
        import tempfile
        from pathlib import Path
        from src.eval import report_manifest as m
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'a.md';b=root/'b.md';a.write_text('A');b.write_text('B')
            with patch.object(m,'REPORTS',root),self.assertRaises(RuntimeError):
                with m.preserve_reports():
                    a.write_text('changed');b.unlink();(root/'new.md').write_text('extra')
                    raise RuntimeError('模拟中断')
            self.assertEqual(a.read_text(),'A');self.assertEqual(b.read_text(),'B')
            self.assertFalse((root/'new.md').exists())

    def test_report_split_refuses_trailing_machine_text(self):
        from src.report_io import split_report
        with self.assertRaises(ValueError):
            split_report('machine\n<!-- HUMAN:BEGIN -->\nhuman\n<!-- HUMAN:END -->\nlost machine')

    def test_protocol_failure_retains_known_api_usage(self):
        from src.agent import pipeline as pl
        res=NS(gt=pd.DataFrame({'p':[.5],'disposition_gt':['hold']},index=[1]))
        def fail(*args,usage_trace=None,**kwargs):
            usage_trace.update(tokens={'input':10000,'output':1000,'api_calls':1},tool_calls=2,usage_complete=False)
            raise pl.AgentProtocolError('failure')
        with patch.object(pl,'investigate',side_effect=fail),patch.object(pl,'degraded_report',return_value={'mode':'degraded'}):
            out=pl.run_one(res,1,object())
        self.assertEqual(out['api_calls'],1);self.assertEqual(out['tool_calls'],2)
        self.assertGreater(out['cost_usd'],0);self.assertFalse(out['usage_complete'])

class EvalIntegrity(unittest.TestCase):
    def test_incomplete_duplicate_or_degraded_round_is_rejected(self):
        from src.eval.run_integrity import validate_complete_run
        es=pd.DataFrame({'TransactionID':[1,2,3,4],'split':['dev','dev','holdout','holdout']})
        rows=[{'txn_id':i,'mode':'llm','report':{'disposition':'hold'}} for i in [1,2]]
        self.assertEqual(len(validate_complete_run(rows,es)),2)
        for bad in [rows[:1],rows+rows[:1],[*rows,{'txn_id':99}],
                    [rows[0],{**rows[1],'mode':'degraded'}]]:
            with self.subTest(bad=bad),self.assertRaises(ValueError): validate_complete_run(bad,es)

    def test_absent_report_is_not_structurally_valid(self):
        from src.eval.agent_eval import hard_metrics
        self.assertFalse(hard_metrics({'txn_id':1,'report':None})['structure_ok'])

class PortableArchiveDigest(unittest.TestCase):
    def test_os_metadata_does_not_change_archive_anchor(self):
        import tempfile
        from pathlib import Path
        from src.eval.report_manifest import tree_sha
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'txn_1.json').write_text('{}');before=tree_sha(root)
            (root/'.DS_Store').write_bytes(b'mac finder metadata')
            self.assertEqual(tree_sha(root),before)
            (root/'txn_1.json').write_text('{"changed":true}')
            self.assertNotEqual(tree_sha(root),before)

class WholePipelineBudget(unittest.TestCase):
    def test_rejected_draft_cannot_exceed_budget_via_fallback(self):
        from src.agent import pipeline as pl
        from src.agent.tools import ToolResult
        res=NS(gt=pd.DataFrame({'p':[.5],'gang_score':[0.],'disposition_gt':['hold']},index=[1]))
        calls=[]
        class Backend:
            as_of=20000000
            def __init__(self,*a):pass
            def retrieve_rules_and_cases(self):
                calls.append('rules');return ToolResult('rules',[],'')
            def query_entity_graph(self):
                calls.append('graph');return ToolResult('graph',[],'')
        for spent in [7,8]:
            calls.clear()
            bad={'report':{},'schema_violations':['bad'],'time_audit_violations':[],
                 'tool_calls':spent,'api_calls':2,'cost_usd':.1,'tokens':{'input':10,'output':1}}
            with patch.object(pl,'investigate',return_value=bad),patch.object(pl,'DataBackedTools',Backend):
                out=pl.run_one(res,1,object())
            self.assertEqual(len(calls),8-spent)
            self.assertEqual(out['tool_calls'],8)
            self.assertIsNotNone(out['report'])
            self.assertTrue(out['report']['evidence_insufficient'])
            self.assertFalse(out['schema_violations'])

if __name__=='__main__': unittest.main()
