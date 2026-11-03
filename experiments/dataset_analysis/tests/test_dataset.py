from pathlib import Path
import copy
import json
import tempfile
import unittest
import numpy as np
from experiments.shot_scaling.core import REVISION,SUITES,task_names,digest,write,sha
from experiments.dataset_analysis.archive import create,verify,files,record_loss,verify_guards
from experiments.dataset_analysis.audit import record,task_identity,no_leakage,convention
from experiments.dataset_analysis.analysis import analyze,train_inventory,path_statistics


def example(suite,task,i):
    lang=task
    import re
    lang=re.sub(r'^.*?_SCENE\d+_','',lang).replace('_',' ')
    steps=[]
    for j in range(3):
        steps.append(dict(action=np.array([i*.001,0,0,0,0,0,-1 if j==0 else 1],dtype=np.float32),
            observation=dict(state=np.array([j*.01,i*.001,1,0,0,0,.03,-.03],dtype=np.float32),
                joint_state=np.zeros(7,dtype=np.float32),image=np.zeros((2,2,3),dtype=np.uint8),wrist_image=np.ones((2,2,3),dtype=np.uint8)),
            language_instruction=lang.encode(),is_first=j==0,is_last=j==2,is_terminal=j==2,
            reward=float(j==2),discount=1.))
    return dict(steps=steps,episode_metadata={'file_path':f'/fixture/{task}/{i}'.encode()})


def fixture(suite='goal',physical=False):
    rows=[record(example(suite,t,i),f'{suite}/{t}/{i}',suite,'train') for t in task_names(suite) for i in range(20)]
    return dict(schema=1,status='AUDITED',synthetic=False,suite=suite,dataset_name=SUITES[suite]+'_no_noops',
        source=dict(repo='openvla/modified_libero_rlds',revision=REVISION,files={'fixture':dict(bytes=1,sha256='a'*64)}),
        episodes=rows,episodes_sha256=digest(rows),declared_splits={'train':len(rows)},integrity={},
        physical_convention={'status':'REVIEWED' if physical else 'UNAVAILABLE','fixture_only':True})


class DatasetTests(unittest.TestCase):
    def test_loss_record_preserves_guards_and_does_not_claim_archive(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);src=root/'missing';guard=root/'checkpoint';guard.mkdir()
            weights=guard/'weights';weights.write_bytes(b'original')
            out=record_loss(src,root/'loss','User deleted previous results',[guard])
            m=verify(out);verify_guards(m)
            self.assertEqual(m['status'],'HISTORICAL_RESULTS_LOST')
            self.assertFalse(m['originals_retained']);self.assertFalse((out/'artifacts').exists())
            src.mkdir();verify_guards(m)
            (src/'recovered.json').write_text('{}')
            with self.assertRaises(ValueError):verify_guards(m)
            with self.assertRaises(ValueError):record_loss(src,root/'other','deleted',[guard])
            (src/'recovered.json').unlink()
            weights.write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_guards(m)
            out.chmod(0o755);(out/'archive_manifest.json').chmod(0o644)

    def test_task_identity_all_suites_and_long_scene_prefix(self):
        for suite in SUITES:
            for t in task_names(suite):
                self.assertEqual(task_identity(t.replace('_',' ').encode(),suite),t)
                self.assertEqual(record(example(suite,t,1),'id',suite,'train')['task'],t)
        with self.assertRaises(ValueError):task_identity(b'unknown task','goal')

    def test_state_action_alignment_and_terminals_not_success(self):
        t=task_names('goal')[0];r=record(example('goal',t,1),'id','goal','train')
        self.assertEqual(len(r['observation_state']),len(r['actions']))
        self.assertTrue(r['terminal']);self.assertIsNone(r['success'])
        self.assertEqual(r['observation_state'][-1][0],np.float32(.02))

    def test_invalid_or_missing_fields(self):
        t=task_names('goal')[0];e=example('goal',t,1)
        e['steps'][1]['observation']['state'][0]=np.nan
        with self.assertRaises(ValueError):record(e,'id','goal','train')
        e=example('goal',t,1);del e['steps'][1]['observation']['state']
        with self.assertRaises(ValueError):record(e,'id','goal','train')
        e=example('goal',t,1)
        for s in e['steps']:del s['observation']['state']
        self.assertIsNone(record(e,'id','goal','train')['observation_state'])

    def test_split_payload_leakage(self):
        row=record(example('goal',task_names('goal')[0],1),'a','goal','train')
        with self.assertRaises(ValueError):no_leakage([row,dict(row,episode_id='b',split='test')])
        with self.assertRaises(ValueError):no_leakage([row,dict(row,episode_id='b')])

    def test_archive_verified_readonly_originals_retained(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);src=root/'source';src.mkdir();write(src/'manifest.json',{'id':'historical'})
            protected=root/'checkpoint';protected.mkdir();(protected/'weights').write_bytes(b'unchanged')
            out=create(src,root/'archive',[protected]);self.assertEqual(verify(out)['files'],files(src))
            self.assertTrue((src/'manifest.json').exists());self.assertEqual((out/'artifacts/manifest.json').stat().st_mode&0o222,0)
            with self.assertRaises(ValueError):create(src,out,[protected])
            # Restore test-fixture permissions for TemporaryDirectory cleanup only.
            for p in out.rglob('*'):p.chmod(0o755 if p.is_dir() else 0o644)
            out.chmod(0o755)

    def test_convention_not_inferred_from_state_shape(self):
        self.assertEqual(convention(None,'x')['status'],'UNAVAILABLE')
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'unreviewed.json';write(p,dict(status='REVIEWED',dataset_revision=REVISION,source_files_sha256='x'))
            with self.assertRaises(ValueError):convention(p,'x')

    def test_geometry_measured_positions_only(self):
        p=path_statistics([[0,0,0],[3,0,0],[3,4,0]])
        self.assertEqual(p['path_length'],7.);self.assertEqual(p['net_displacement'],5.)
        for s in SUITES:
            d=analyze(fixture(s),seeds=[42,43])
            self.assertTrue(all(e['xyz'] is None and e['physical'] is None for e in d['episodes']))
            self.assertEqual(d['physical_pairs'],[])
            for shots in d['selections'].values():
                prev=set()
                for k in ('1','2','5','10','20','Full'):
                    selected={r['episode_id'] for r in shots[k]['episodes']};self.assertTrue(prev<=selected);prev=selected

    def test_historical_identity_and_physical_metrics(self):
        from experiments.fewshot_sft.prepare_dataset import select
        a=fixture('goal',physical=True);inventory=train_inventory(a)
        chosen=select(inventory['episodes'],5,42,task_names('goal'))
        historical=dict(schema=1,dataset_name=inventory['dataset_name'],shots=5,selection_seed=42,
            episodes=chosen,selection_sha256=digest(chosen),source=dict(revision=REVISION,files=a['source']['files']))
        d=analyze(a,[42],historical)
        self.assertEqual(len(d['historical_ids']),50)
        self.assertAlmostEqual(d['episodes'][0]['physical']['path_length'],.02)
        bad=copy.deepcopy(historical);bad['episodes'][0]['sha256']='f'*64;bad['selection_sha256']=digest(bad['episodes'])
        with self.assertRaises(ValueError):analyze(a,[42],bad)
        with self.assertRaises(ValueError):analyze(dict(a,status='SMOKE_ONLY'))

    def test_four_figure_saved_artifact_reproduction(self):
        from experiments.dataset_analysis.plots import plot
        bundle=dict(suites={s:analyze(fixture(s,physical=True),[42]) for s in SUITES},display_seed=42)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);write(p/'analysis.json',bundle);out=plot(p/'analysis.json',p/'figures')
            self.assertEqual(len(list(out.glob('*.png'))),4);self.assertEqual(len(list(out.glob('*.pdf'))),4)
            self.assertEqual(json.loads((out/'figure_manifest.json').read_text())['source_sha256'],sha(p/'analysis.json'))
            self.assertFalse((out/'supplementary').exists())
            with self.assertRaises(FileExistsError):plot(p/'analysis.json',out)


if __name__=='__main__':unittest.main()
