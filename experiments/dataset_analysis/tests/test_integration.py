import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments.dataset_analysis.pose_logging import PoseLogger
from experiments.dataset_analysis.integration import validate_labels, evidence, LABELS
from experiments.dataset_analysis.audit import convention
from experiments.shot_scaling.core import ROOT, read, sha


class Env:
    def __init__(self):self.t=0;self.actions=[]
    def obs(self):return {'robot0_eef_pos':np.array([self.t,2.,3.]),'robot0_eef_quat':np.array([0.,0.,0.,1.])}
    def reset(self):self.t=0;return self.obs()
    def set_init_state(self,x):self.t=int(x[0]);return self.obs()
    def step(self,a):self.actions.append(a);self.t+=1;return self.obs(),float(self.t),self.t==3,{'t':self.t}


class IntegrationTests(unittest.TestCase):
    def test_logging_preserves_action_identity_results_and_time_alignment(self):
        with tempfile.TemporaryDirectory() as t:
            env=Env();ref=Env();wrapped=PoseLogger(env,Path(t)/'poses',{'family':'released_checkpoint'})
            wrapped.reset();ref.reset();wrapped.set_init_state([0]);ref.set_init_state([0])
            actions=[np.arange(7,dtype=float) for _ in range(3)]
            for i,a in enumerate(actions):
                before=a.copy();got=wrapped.step(a);expected=ref.step(a)
                np.testing.assert_array_equal(got[0]['robot0_eef_pos'],expected[0]['robot0_eef_pos'])
                self.assertEqual(got[1:],expected[1:]);self.assertIs(env.actions[-1],a)
                np.testing.assert_array_equal(a,before)
                wrapped.phase='policy'
            wrapped.finished=True;wrapped.save()
            with np.load(Path(t)/'poses/poses.npz',allow_pickle=False) as z:
                np.testing.assert_equal(z['pre_pose'][:,0],[0,1,2]);np.testing.assert_equal(z['post_pose'][:,0],[1,2,3])
                np.testing.assert_equal(z['executed_action'],actions);self.assertEqual(z['phase'][0],'settling')
            with self.assertRaises(ValueError):wrapped.save()

    def test_missing_pose_is_flagged_not_reconstructed(self):
        with tempfile.TemporaryDirectory() as t:
            env=Env();env.obs=lambda:{}
            wrapped=PoseLogger(env,Path(t)/'poses',{});wrapped.reset();wrapped.step([0]*7);wrapped.save()
            with np.load(Path(t)/'poses/poses.npz',allow_pickle=False) as z:
                self.assertFalse(z['pre_valid'][0]);self.assertTrue(np.isnan(z['pre_pose']).all())

    def test_labels_do_not_mix_released_shot_or_nominal_ood(self):
        row=dict(model='scale',family='released_checkpoint',shots=None,condition='nominal',suite='goal',successes=1,n=2)
        validate_labels([row]);self.assertNotEqual(LABELS['scale'],LABELS['sft'])
        for bad in (dict(row,shots=5),dict(row,family='new_shot_scaling'),dict(row,condition='nominal+ood'),dict(row,model='d0')):
            with self.assertRaises(ValueError):validate_labels([bad])

    def test_real_source_evidence_is_bound_but_does_not_bypass_gate(self):
        if not (ROOT/'outputs/dataset_analysis/audits/goal.json').exists():
            self.skipTest('Real audit artifacts not available in this checkout')
        audits={s:read(ROOT/f'outputs/dataset_analysis/audits/{s}.json') for s in ('goal','spatial','long')}
        ev=evidence(audits)
        self.assertEqual(ev['status'],'BLOCKED');self.assertFalse(ev['physical_plots_allowed'])
        for source in ev['sources']:
            self.assertEqual(sha(ROOT/source['path']),source['sha256'])
            self.assertGreaterEqual(len((ROOT/source['path']).read_text().splitlines()),source['end_line'])
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'evidence.json';p.write_text(json.dumps(ev))
            with self.assertRaises(ValueError):convention(p,audits['goal']['source']['source_files_sha256'])

    def test_restore_rejects_blocked_evidence_without_output(self):
        from experiments.dataset_analysis.restore import restore
        if not (ROOT/'outputs/dataset_analysis/audits/goal.json').exists():
            self.skipTest('Real audit artifacts not available in this checkout')
        audits=[ROOT/f'outputs/dataset_analysis/audits/{s}.json' for s in ('goal','spatial','long')]
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'blocked.json';p.write_text('{"status":"BLOCKED"}')
            output=Path(t)/'restore'
            with self.assertRaises(ValueError):
                restore(audits,[p,p,p],ROOT/'outputs/dataset_analysis/libero_expert_trajectories/analysis.json',output)
            self.assertFalse(output.exists())

    def test_saved_result_counts_are_not_mixed_across_conditions(self):
        from experiments.dataset_analysis.integration import dashboard_data
        b=ROOT/'outputs/dataset_analysis/libero_expert_trajectories/analysis.json'
        if not b.exists():self.skipTest('Real result artifacts unavailable')
        d=dashboard_data(read(b),read(ROOT/'outputs/rlvr_analysis/research_summary_all.json'))
        expected={'nominal':{'sft':(486,500),'scale':(488,500),'ript':(488,500)},
                  'ood':{'sft':(1675,2591),'scale':(1667,2591),'ript':(1620,2591)}}
        for condition,models in expected.items():
            for model,counts in models.items():
                rows=[r for r in d['released'] if r['condition']==condition and r['model']==model]
                self.assertEqual((sum(r['successes'] for r in rows),sum(r['n'] for r in rows)),counts)

    def test_pinned_replay_captures_state_before_action(self):
        source=(ROOT/'vendor/openvla-oft/experiments/robot/libero/regenerate_libero_dataset.py').read_text()
        self.assertLess(source.index('ee_states.append('),source.index('env.step(action.tolist())'))
        # Actual reference converter expressions, executed only on a synthetic index fixture.
        import ast
        tree=ast.parse((ROOT/'experiments/dataset_analysis/evidence_sources/builder_LIBERO_Goal_LIBERO_Goal_dataset_builder.py').read_text())
        expressions={}
        for node in ast.walk(tree):
            if isinstance(node,ast.Dict):
                for k,v in zip(node.keys,node.values):
                    if isinstance(k,ast.Constant) and k.value in ('state','action') and isinstance(v,ast.Call) and isinstance(v.func,ast.Attribute) and v.func.attr=='asarray':expressions[k.value]=v
        inputs=dict(np=np,states=np.arange(18).reshape(3,6),gripper_states=np.arange(6).reshape(3,2),actions=np.arange(21).reshape(3,7))
        for i in range(3):
            scope=dict(inputs,i=i)
            state=eval(compile(ast.Expression(expressions['state']),'<upstream expression>','eval'),scope)
            action=eval(compile(ast.Expression(expressions['action']),'<upstream expression>','eval'),scope)
            np.testing.assert_array_equal(state[:6],inputs['states'][i]);np.testing.assert_array_equal(action,inputs['actions'][i])
            self.assertEqual(state.dtype,np.float32)


if __name__=='__main__':unittest.main()
