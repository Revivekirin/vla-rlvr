"""CPU-only suite guards. No simulator, model import or network access."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.libero_plus import suite_common as c, suite_report as r
from experiments.nominal import checkpoint_sync as sync


class SuiteTests(unittest.TestCase):
    def test_download_selection_excludes_other_suites_and_requires_headers(self):
        root_files=(['config.json','dataset_statistics.json','model.safetensors.index.json',
                    'preprocessor_config.json','processor_config.json','tokenizer.json',
                    'tokenizer.model','tokenizer_config.json','special_tokens_map.json',
                    'added_tokens.json','generation_config.json','configuration_prismatic.py',
                    'modeling_prismatic.py','processing_prismatic.py',
                    'action_head--150000_checkpoint.pt','proprio_projector--150000_checkpoint.pt']+
                   [f'model-{i:05d}-of-00004.safetensors' for i in range(1,5)])
        prefix='openvla_oft/ript_adaptors/LIBERO_LONG_lora/'
        ript_files=([f'openvla_oft/scale_header/LIBERO_LONG_scale_header.pth']+
                   [prefix+n for n in ('adapter_config.json','adapter_model.safetensors','openvla_headers.pt')]+
                   ['openvla_oft/scale_header/LIBERO_OBJECT_scale_header.pth'])
        obj=lambda names:SimpleNamespace(siblings=[SimpleNamespace(rfilename=n) for n in names])
        selected=sync.select('long',obj(root_files),obj(ript_files))
        self.assertEqual(len(selected[sync.RELEASE]),4)
        self.assertEqual(len(selected[sync.SOURCES['long']]),20)
        with self.assertRaises(ValueError):sync.select('long',obj(root_files),obj(ript_files[:-2]))

    def test_checkpoint_mapping_is_suite_specific(self):
        for suite,key in [('long','LIBERO_LONG'),('spatial','LIBERO_SPATIAL')]:
            s=c.checkpoint_paths(suite,'sft')
            a=c.checkpoint_paths(suite,'scale')
            b=c.checkpoint_paths(suite,'ript')
            self.assertEqual(s['base'],a['base'])
            self.assertEqual(s['base'],b['base'])
            self.assertNotIn('adapter',a)
            self.assertIn(key,a['scale'])
            self.assertIn(key,b['rl_heads'].replace('_lora/','_lora/'))
            self.assertNotEqual(a['scale'],c.checkpoint_paths('spatial' if suite=='long' else 'long','scale')['scale'])

    def test_local_normalization_and_missing_weights(self):
        for suite,key in [('long','libero_10_no_noops'),('spatial','libero_spatial_no_noops')]:
            info=c.checkpoint_check(suite,'ript')
            self.assertEqual(info['norm_key'],key)
            self.assertEqual(info['architecture'],'OpenVLAForActionPrediction')
            self.assertEqual(len(info['vision_encoders']),2)
            self.assertFalse(info['ready'])
            self.assertTrue(any('adapter_model.safetensors' in x for x in info['missing']))

    def test_nominal_task_order_and_plus_is_not_fabricated(self):
        original=c.ROOT/'vendor/LIBERO'
        for suite in c.SUITES:
            entries=c.task_plan(suite,original,'nominal')
            self.assertEqual(len(entries),10)
            self.assertEqual([e['id'] for e in entries],list(range(1,11)))
            with tempfile.TemporaryDirectory() as t:
                with self.assertRaises(FileNotFoundError):c.task_plan(suite,Path(t),'plus')

    def test_result_paths_do_not_overlap(self):
        root=Path('/tmp/fixture')
        self.assertEqual(c.result_path(root,1,0,'nominal').name,'task-0001-trial-00.json')
        self.assertEqual(c.result_path(root,1,0,'plus').name,'task-0001.json')
        self.assertNotEqual(c.checkpoint_paths('long','sft')['base'],c.checkpoint_paths('spatial','sft')['base'])

    def test_paired_alignment_and_transitions(self):
        base={'status':'COMPLETE','benchmark_revision':'a'}
        left=dict(base,_rows=[dict(task_id=1,task_name='a',shift='Noise',success=True),
                              dict(task_id=2,task_name='b',shift='Noise',success=False)])
        right=dict(base,_rows=[dict(task_id=1,task_name='a',shift='Noise',success=False),
                               dict(task_id=2,task_name='b',shift='Noise',success=True)])
        pair=r.paired(left,right)
        self.assertEqual(pair['overall']['transitions']['lost'],1)
        self.assertEqual(pair['overall']['transitions']['gained'],1)
        right['_rows'][1]['task_name']='different'
        with self.assertRaises(ValueError):r.paired(left,right)
        self.assertEqual(r.paired({'status':'INCOMPLETE'},left)['status'],'INCOMPLETE')

    def test_manifest_resume_identity_is_checked(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            folder=root/'outputs/libero-nominal/long/scale/seed-1'
            folder.mkdir(parents=True)
            (folder/'manifest.json').write_text(json.dumps(dict(suite_label='spatial',dataset='nominal',model='scale')))
            with self.assertRaisesRegex(ValueError,'suite/model mismatch'):
                r.load_run(root,'long','nominal','scale')

if __name__=='__main__':unittest.main()
