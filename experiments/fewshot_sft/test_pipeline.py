import ast
import copy
import importlib.util
import unittest
import tempfile
import json
from pathlib import Path

from .common import OFT, OUT, ROOT, destination, digest, tasks
from .prepare_dataset import episode_record, select, validate_manifest
from .train import parser, plan
from .worker import instrument
from .checkpoint import check


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.names = tasks()
        self.rows = [dict(episode_id=f'{task}:{i}', task=task, steps=i+1, complete=True,
                          sha256=digest([task,i])) for task in self.names for i in range(8)]
        self.selected = select(self.rows, 5, 42, self.names)
        self.manifest = dict(schema=1, dataset_name='libero_goal_no_noops', shots=5, selection_seed=42,
            episodes=self.selected, selection_sha256=digest(self.selected), source={'revision':'fixture'})

    def test_ten_tasks_fifty_complete_episodes(self):
        self.assertEqual(len(self.names),10)
        self.assertEqual(len(self.selected),50)
        validate_manifest(self.manifest)

    def test_one_shot(self):
        self.assertEqual(len(select(self.rows,1,42,self.names)),10)

    def test_deterministic_even_if_source_iteration_changes(self):
        self.assertEqual(self.selected,select(list(reversed(self.rows)),5,42,self.names))

    def test_different_seed(self):
        self.assertNotEqual(self.selected,select(self.rows,5,43,self.names))

    def test_duplicate_source_ids(self):
        with self.assertRaises(ValueError): select(self.rows+[self.rows[0]],5,42,self.names)

    def test_cross_task_leakage(self):
        self.rows.append(dict(self.rows[0],episode_id='other',task='spatial'))
        with self.assertRaises(ValueError): select(self.rows,5,42,self.names)

    def test_missing_task(self):
        with self.assertRaises(ValueError): select(self.rows[8:],5,42,self.names)

    def test_source_integrity(self):
        with self.assertRaises(ValueError): validate_manifest(self.manifest,{'revision':'changed'})

    def test_manifest_integrity(self):
        self.manifest['episodes'][0]['steps'] += 1
        with self.assertRaises(ValueError): validate_manifest(self.manifest)

    def test_complete_episodes_only(self):
        self.manifest['episodes'][0]['complete'] = False
        self.manifest['selection_sha256'] = digest(self.manifest['episodes'])
        with self.assertRaises(ValueError): validate_manifest(self.manifest)

    def test_duplicate_selection(self):
        self.manifest['episodes'][0] = self.manifest['episodes'][1]
        self.manifest['selection_sha256'] = digest(self.manifest['episodes'])
        with self.assertRaises(ValueError): validate_manifest(self.manifest)


class LauncherTests(unittest.TestCase):
    def test_normalization_metadata(self):
        c = plan(parser().parse_args(['--run-id','fixture','--max-steps','1','--save-freq','1']))
        c.update(step=1, selected_transitions=100)
        stats = {'num_trajectories':50,'num_transitions':100}
        for kind, dim in [('action',7),('proprio',8)]:
            stats[kind] = {field:[0.]*dim for field in ('mean','std','min','max','q01','q99')}
        stats['action']['mask'] = [True]*6+[False]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path/'fewshot_checkpoint.json').write_text(json.dumps(c))
            (path/'dataset_statistics.json').write_text(json.dumps({'libero_goal_no_noops':stats}))
            for name in ('action_head','proprio_projector'):
                (path/f'{name}--1_checkpoint.pt').touch()
            check(path, merged=False)
            stats['num_trajectories'] = 500
            (path/'dataset_statistics.json').write_text(json.dumps({'libero_goal_no_noops':stats}))
            with self.assertRaises(ValueError): check(path, merged=False)

    def test_existing_results_protected(self):
        for path in [ROOT/'outputs/goal',ROOT/'checkpoints/goal/oft',OUT/'../goal',OUT]:
            with self.assertRaises(ValueError): destination(path)

    def test_explicit_steps_required(self):
        with self.assertRaises(SystemExit): parser().parse_args(['--run-id','test'])

    def test_bounded_plan(self):
        c = plan(parser().parse_args(['--run-id','pilot','--max-steps','200','--save-freq','50']))
        self.assertEqual(c['max_steps'],200)
        self.assertEqual(c['base_identity'],'openvla/openvla-7b')
        self.assertFalse(c['architecture']['use_diffusion'])

    def test_unsafe_id(self):
        with self.assertRaises(ValueError):
            plan(parser().parse_args(['--run-id','../goal','--max-steps','1','--save-freq','1']))

    def test_official_shim_parses(self):
        source = (OFT/'vla-scripts/finetune.py').read_text()
        ast.parse(instrument(source))
        with self.assertRaises(ValueError): instrument(source.replace('normalized_loss.backward()','changed()'))

    def test_optimizer_boundary_semantics(self):
        # Verify the patched upstream expressions against accumulation and final-save cases.
        tree = ast.parse(instrument((OFT/'vla-scripts/finetune.py').read_text()))
        index = next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
                     and any(isinstance(t,ast.Name) and t.id=='gradient_step_idx' for t in n.targets))
        from types import SimpleNamespace
        for accumulation in (1,2,4):
            cfg = SimpleNamespace(grad_accumulation_steps=accumulation)
            observed = [eval(compile(ast.Expression(index),'<test>','eval'),dict(batch_idx=b,cfg=cfg))
                        for b in range(3*accumulation)]
            self.assertEqual([observed[b] for b in range(len(observed)) if (b+1)%accumulation==0],[1,2,3])


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'numpy absent: real array shape/dtype tests unavailable')
class EpisodeTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.e = {'episode_metadata':{'file_path':b'fixture'}, 'steps':[
            dict(is_first=i==0,is_last=i==2,is_terminal=i==2,
                 language_instruction=tasks()[0].replace('_',' ').encode(),
                 action=np.zeros(7,dtype=np.float32),
                 observation=dict(state=np.zeros(8,dtype=np.float32),
                                  image=np.zeros((8,8,3),dtype=np.uint8),
                                  wrist_image=np.zeros((8,8,3),dtype=np.uint8))) for i in range(3)]}

    def test_valid_rlds_shape_dtype_and_temporal_hash(self):
        row = episode_record(self.e,'fixture')
        self.assertEqual(row['steps'],3)
        self.e['steps'][1]['action'][0] = 1
        self.assertNotEqual(row['sha256'],episode_record(self.e,'fixture')['sha256'])

    def test_reject_wrong_action_shape(self):
        self.e['steps'][0]['action'] = self.e['steps'][0]['action'][:6]
        with self.assertRaises(ValueError): episode_record(self.e,'fixture')

    def test_reject_wrong_action_dtype(self):
        self.e['steps'][0]['action'] = self.e['steps'][0]['action'].astype('float64')
        with self.assertRaises(ValueError): episode_record(self.e,'fixture')

    def test_reject_truncated_episode(self):
        self.e['steps'].pop()
        with self.assertRaises(ValueError): episode_record(self.e,'fixture')

    def test_reject_mixed_task(self):
        self.e['steps'][1]['language_instruction'] = tasks()[1].replace('_',' ').encode()
        with self.assertRaises(ValueError): episode_record(self.e,'fixture')


if __name__ == '__main__':
    unittest.main()
