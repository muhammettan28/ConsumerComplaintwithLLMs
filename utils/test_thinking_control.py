"""Recovery and protocol safeguards for long-running thinking inference.

Synthetic responses stay in temporary directories; no GPU/server calls.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

if __package__:
    from . import run_thinking_control as control
else:
    import run_thinking_control as control


class ThinkingControlSafeguards(unittest.TestCase):
    def test_interrupted_final_json_write_keeps_completed_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'outputs.jsonl'
            completed=b'{"index": 1}\n{"index": 2}\n'
            p.write_bytes(completed+b'{"index": "\xc3')
            self.assertEqual(control.journal(p), [{'index':1},{'index':2}])
            self.assertEqual(p.read_bytes(),completed)

    def test_corruption_before_final_row_is_rejected_without_truncation(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'outputs.jsonl'
            contents=b'{"index": 1}\ninvalid\n{"index": 2}\n'
            p.write_bytes(contents)
            with self.assertRaises(AssertionError):
                control.journal(p)
            self.assertEqual(p.read_bytes(),contents)

    def test_final_answer_parser_excludes_reasoning(self):
        row={'source_index':0,'text_hash':'fixture','true_label':'credit_card','narrative':'Synthetic test complaint'}
        response={'done':True,'response':'','thinking':'{"product":"credit_card"}'}
        with patch.object(control,'PLAN',{'options_without_output_cap':{'temperature':0}},create=True), patch.object(control,'api',return_value=response):
            record=control.call('gemma4',row,True,1024,'test')
        self.assertEqual(record['status'],'parse_error')

    def test_output_limit_is_an_error_even_with_embedded_json(self):
        self.assertEqual(control.parse_answer({'done_reason':'length',
            'response':'{"product":"credit_card"} unfinished continuation'}),
            ('output_limit','output_limit'))

    def test_only_think_field_changes_between_paired_requests(self):
        row={'source_index':0,'text_hash':'fixture','true_label':'credit_card','narrative':'Synthetic test complaint'}
        def server(endpoint,payload,timeout):
            return {'done':True,'response':'{"product":"credit_card"}',
                    'thinking':'Separate reasoning' if payload['think'] else ''}
        with patch.object(control,'PLAN',{'options_without_output_cap':{'temperature':0,'seed':42}},create=True), patch.object(control,'api',side_effect=server):
            a=control.call('qwen3',row,False,2048,'test')
            b=control.call('qwen3',row,True,2048,'test')
        self.assertFalse(a['request'].pop('think'))
        self.assertTrue(b['request'].pop('think'))
        self.assertEqual(a['request'],b['request'])
        self.assertEqual(a['status'],'ok')
        self.assertEqual(b['status'],'ok')

    def test_disabled_thinking_trace_is_rejected(self):
        row={'source_index':0,'text_hash':'fixture','true_label':'credit_card','narrative':'Synthetic test complaint'}
        with patch.object(control,'PLAN',{'options_without_output_cap':{}},create=True), patch.object(control,'api',return_value={'done':True,'response':'{}','thinking':'Unexpected trace'}):
            with self.assertRaises(AssertionError):
                control.call('gemma4',row,False,1024,'test')

    def test_pilot_uses_termination_not_classification_quality(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            (run/'study_plan.json').write_text('{}')
            pd.DataFrame([{'source_index':i,'narrative':'Synthetic test complaint',
                          'text_hash':str(i),'true_label':'credit_card'} for i in range(10)]).to_csv(run/'pilot_inputs.csv',index=False)
            def fake_call(model,row,think,cap,role):
                limited=cap==1024 and model=='gemma4' and think and row['source_index']==0
                return {'model':model,'source_index':row['source_index'],'think':think,
                        'response':{'thinking':'trace' if think else '',
                        'response':'' if limited else '{"product":"invalid_fixture_label"}',
                        'done_reason':'length' if limited else 'stop'}}
            with patch.object(control,'RUN',run), patch.object(control,'verify_metadata'), patch.object(control,'unload'), patch.object(control,'progress'), patch.object(control,'call',side_effect=fake_call):
                cap=control.pilot_cap({'output_cap_candidates':[1024,2048]})
            self.assertEqual(cap,2048)
            self.assertEqual(len(control.journal(run/'pilot_1024.jsonl')),40)
            self.assertEqual(len(control.journal(run/'pilot_2048.jsonl')),40)
            self.assertTrue(json.loads((run/'protocol.json').read_text())['same_cap_for_both_modes_and_models'])

    def test_bounded_fallback_requires_a_recorded_pre_main_amendment(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            (run/'study_plan.json').write_text('{}')
            pd.DataFrame([{'source_index':i,'narrative':'Synthetic test complaint',
                          'text_hash':str(i),'true_label':'credit_card'} for i in range(10)]).to_csv(run/'pilot_inputs.csv',index=False)
            def fake_call(model,row,think,cap,role):
                limited=model=='gemma4' and think and row['source_index']==0
                return {'model':model,'source_index':row['source_index'],'think':think,
                        'response':{'thinking':'trace' if think else '',
                        'response':'' if limited else '{"product":"fixture"}',
                        'done_reason':'length' if limited else 'stop'}}
            patches=[patch.object(control,'RUN',run),patch.object(control,'verify_metadata'),
                     patch.object(control,'unload'),patch.object(control,'progress'),
                     patch.object(control,'call',side_effect=fake_call)]
            from contextlib import ExitStack
            with ExitStack() as stack:
                for p in patches: stack.enter_context(p)
                with self.assertRaises(RuntimeError):
                    control.pilot_cap({'output_cap_candidates':[1024]})
                control.save(run/'protocol_amendment.json',{
                    'study_plan_sha256':control.sha(run/'study_plan.json'),
                    'before_any_main_predictions':True,'no_classification_accuracy_used':True,
                    'highest_prespecified_cap':1024})
                self.assertEqual(control.pilot_cap({'output_cap_candidates':[1024]}),1024)
            protocol=json.loads((run/'protocol.json').read_text())
            self.assertTrue(protocol['bounded_fallback_used'])
            self.assertEqual(protocol['residual_pilot_length_or_empty_final'],1)


if __name__=='__main__':
    unittest.main()
