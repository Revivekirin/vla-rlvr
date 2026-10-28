"""CPU synthetic tests; never import torch, simulator, or checkpoint weights."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import p3b_metrics as met
import p3_representation_analysis as b
import research_report as report


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.x=np.random.default_rng(7).normal(size=(12,6))
    def test_identical_orthogonal_and_translation(self):
        q,_=np.linalg.qr(np.random.default_rng(8).normal(size=(6,6)))
        self.assertAlmostEqual(met.cka(self.x,self.x)['value'],1)
        self.assertAlmostEqual(met.cka(self.x,self.x@q+3)['value'],1)
    def test_symmetry_range(self):
        y=np.random.default_rng(9).normal(size=(12,8))
        value=met.cka(self.x,y)['value']
        self.assertAlmostEqual(value,met.cka(y,self.x)['value'])
        self.assertTrue(0<=value<=1)
    def test_constant_and_insufficient(self):
        self.assertIsNone(met.cka(np.ones((4,3)),np.ones((4,2)))['value'])
        self.assertIsNone(met.cka(self.x[:2],self.x[:2])['value'])
        self.assertIsNone(met.effective_rank(np.ones((3,4)))['effective_rank'])
        with self.assertRaises(ValueError): met.cka(self.x,self.x[:5])
        with self.assertRaises(ValueError): met.cka(self.x,np.full((12,6),np.nan))
    def test_known_effective_rank(self):
        rank1=met.effective_rank(np.array([[-1,0],[0,0],[1,0]]))
        self.assertEqual(rank1['numerical_rank'],1)
        self.assertAlmostEqual(rank1['effective_rank'],1)
        self.assertAlmostEqual(rank1['normalized_effective_rank'],.5)
        rank2=met.effective_rank(np.array([[-1,0],[1,0],[0,-1],[0,1]]))
        self.assertAlmostEqual(rank2['effective_rank'],2)
        self.assertEqual(rank2['N'],4)
        self.assertEqual(rank2['D'],2)
    def test_alignment_and_bootstrap_units(self):
        met.align(['a','b'],['a','b'])
        for y in (['b','a'],['a'],['a','a']):
            with self.assertRaises(ValueError): met.align(['a','b'],y)
        self.assertIsNone(met.bootstrap(self.x,self.x,['task']*12,10)['interval'])
        result=met.bootstrap(self.x,self.x,np.repeat(['a','b','c','d'],3),20)
        np.testing.assert_allclose(result['interval'],[1,1])
        self.assertEqual(result['clusters'],4)
    def test_subsets_and_repeated_frames(self):
        result=met.sensitivity(self.x,self.x,np.repeat(['a','b','c','d'],3))
        self.assertEqual(result['one_per_rollout']['N'],4)
        self.assertAlmostEqual(result['uniform_frame_duplication_cka']['value'],1)


class BankTests(unittest.TestCase):
    def test_stage_sampling_no_slot_pseudoreplication(self):
        idx,counts=b.sampling_indices(38,10)
        self.assertEqual(len(idx),10)
        self.assertEqual(len({i for i,_ in idx}),10)
        self.assertEqual([counts[s]['collected'] for s in ('early','middle','late')],[4,3,3])
        idx,counts=b.sampling_indices(2,10)
        self.assertEqual(len(idx),2)
        self.assertEqual(sum(c['shortfall'] for c in counts.values()),8)
        self.assertEqual(len(b.sampling_indices(1,1)[0]),1)
    def test_hash_reload_and_corruption(self):
        with tempfile.TemporaryDirectory() as t:
            run=Path(t)
            obs=dict(full_image=np.zeros((224,224,3),np.uint8),wrist_image=np.ones((224,224,3),np.uint8),state=np.arange(8,dtype=float))
            row=dict(id='r:0',language='test instruction',input_sha256=b.input_sha(obs,'test instruction'))
            dest=run/'bank/r'
            b.atomic_bundle(run,dest,{k:v[None] for k,v in obs.items()},dict(observations=[row]))
            loaded,_=b.validate_observations(dest)
            for k in obs: np.testing.assert_array_equal(loaded[k][0],obs[k])
            with self.assertRaises(ValueError): b.atomic_bundle(run,dest,{}, {})
            with (dest/'arrays.npz').open('ab') as f: f.write(b'bad')
            with self.assertRaisesRegex(ValueError,'Corrupt'): b.validate_observations(dest)
    def test_missing_physical_task_is_not_invented(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            case=dict(dataset='plus',task_name='test_view_1_15_100_0_0_initstate_0_noise_1')
            for folder,filename in [('bddl_files','test.bddl'),('init_files','test.pruned_init')]:
                p=root/'libero/libero'/folder/'libero_goal'/filename
                p.parent.mkdir(parents=True)
                p.write_text('fixture')
            paths=b.task_files(case,root,root)
            self.assertEqual(paths[0].name,'test.bddl')
            case['task_name']='missing_light_1'
            with self.assertRaises(ValueError): b.task_files(case,root,root)

    def test_balanced_selection_is_reproducible(self):
        cases,_=b.selection(b.ROOT,b.ROOT/'vendor/LIBERO','balanced')
        again,_=b.selection(b.ROOT,b.ROOT/'vendor/LIBERO','balanced')
        self.assertEqual(cases,again)
        self.assertEqual(len(cases),34)
        for shift in b.STRATA[1:]:
            subset=[c for c in cases if c['shift']==shift]
            self.assertEqual(len(subset),8)
            for label in ('lost','gained','both_success','both_fail'):
                self.assertEqual(sum(c['historical_case']==label for c in subset),2)

    def test_frozen_bank_to_analysis_synthetic_roundtrip(self):
        with tempfile.TemporaryDirectory() as t:
            run=Path(t)
            cases=[dict(dataset=dataset,task_id=1,task_name=name,base_task=name,shift=shift,
                        historical_case='nominal' if dataset=='nominal' else 'lost',ambiguous=False)
                   for dataset,name,shift in [('nominal','task_a','Nominal'),('plus','task_b','Camera Viewpoints')]]
            key,norm=b.a1.norm_spec(b.ROOT/'checkpoints/goal/oft')
            manifest=dict(cases=cases,mode='smoke',per_rollout=1,horizon=16,layers=[0],seed=1,
                          normalization=norm,normalization_key=key)
            b.save_json(run/'manifest.json',manifest)
            bundles={}
            for case in cases:
                for source in b.SOURCES:
                    rid=b.rollout_id(case,source)
                    obs=dict(full_image=np.zeros((224,224,3),np.uint8),wrist_image=np.ones((224,224,3),np.uint8),state=np.zeros(8))
                    row=dict(case,id=rid+':0',rollout_id=rid,source=source,step=0,stage='middle',success=False,
                             language='task',input_sha256=b.input_sha(obs,'task'))
                    mean=np.zeros((1,8,7),np.float32)
                    a=norm['action']
                    raw=np.where(a['mask'],.5*(mean+1)*(np.array(a['q99'])-a['q01']+1e-8)+a['q01'],mean)
                    executed=raw.copy(); executed[...,6]=-np.sign(2*raw[...,6]-1)
                    arrays=dict(**{k:v[None] for k,v in obs.items()},normalized_mean=mean,unnormalized=raw,executed=executed)
                    info=dict(observations=[row],execution={'fixture':'CPU synthetic'},steps=16,success=False,truncated=True,
                              composition={'middle':dict(target=1,collected=1,shortfall=0)})
                    b.atomic_bundle(run,run/'bank'/rid,arrays,info)
                    bundles[rid]=b.sha(run/'bank'/rid/'info.json')
            b.save_json(run/'bank.json',dict(bundles=bundles,manifest_sha256=b.sha(run/'manifest.json')))
            for rid in bundles:
                arrays,info=b.bundle(run/'bank'/rid)
                features={k:np.ones((1,3)) for k in ('vision_projection','layer_0','action_head_input')}
                desc={k:dict(module=k,pooling='synthetic fixture') for k in features}
                for model in b.SOURCES:
                    output=dict(features,**{k:arrays[k] for k in ('normalized_mean','unnormalized','executed')})
                    b.atomic_bundle(run,run/'features'/model/rid,output,dict(observations=info['observations'],features=[desc],
                                    execution={'fixture':'CPU synthetic'},bank_sha256=b.sha(run/'bank.json')))
            self.assertEqual(b.validate_run(run,True)['observations'],4)
            result=b.analyze(run,10)
            self.assertEqual(result['observations'],4)
            self.assertTrue(all(v['features']['layer_0']['cka']['value'] is None for v in result['groups'].values()))
            self.assertEqual(result['matched_N_control']['N'],1)
            data=report.build(b.ROOT)
            data['p3_b']=[dict(run='synthetic',analysis=result,manifest=manifest)]
            self.assertIn('extraction verification',report.markdown(data).lower())
            altered=copy.deepcopy(manifest); altered['seed']=99
            (run/'manifest.json').write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError,'manifest changed'): b.frozen_bank(run)


class Tensor:
    def __init__(self,x): self.x=np.asarray(x); self.shape=self.x.shape; self.ndim=self.x.ndim; self.dtype='torch.bfloat16'
    def detach(self): return self
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.x


class HookTests(unittest.TestCase):
    def make(self):
        module=SimpleNamespace(register_forward_hook=lambda fn:SimpleNamespace(remove=lambda:None))
        base=SimpleNamespace(projector=module,language_model=SimpleNamespace(model=SimpleNamespace(layers=[module])),
                             vision_backbone=SimpleNamespace(get_num_patches=lambda:2))
        return b.Features(SimpleNamespace(model=SimpleNamespace(get_base_model=lambda:base),
                                         action_head=SimpleNamespace(predict_action=lambda x:x)),[0])
    def test_exact_cached_action_slice(self):
        hooks=self.make()
        hooks.record('vision_projection',Tensor(np.ones((1,4,3))))
        hooks.record('layer_0',Tensor(np.zeros((1,7,3))))
        suffix=np.ones((1,58,3)); suffix[:,-2:]=1000
        hooks.record('layer_0',Tensor(suffix))
        hooks.record('action_head_input',Tensor(np.ones((1,56,3))))
        values,desc=hooks.finish()
        np.testing.assert_array_equal(values['layer_0'],np.ones(3))
        self.assertIn('Second cached',desc['layer_0']['pooling'])
        hooks.close()
    def test_missing_or_full_sequence_hook_fails(self):
        hooks=self.make()
        with self.assertRaises(ValueError): hooks.finish()
        hooks.record('vision_projection',Tensor(np.ones((1,4,3))))
        hooks.record('layer_0',Tensor(np.zeros((1,65,3))))
        hooks.record('action_head_input',Tensor(np.ones((1,56,3))))
        with self.assertRaisesRegex(ValueError,'prefix/action'): hooks.finish()
        hooks.close()


class ReportTests(unittest.TestCase):
    def test_local_disk_counts_and_optional_unavailable(self):
        data=report.build(b.ROOT)
        rows,_=report.historical(b.ROOT)
        self.assertEqual(data['ood']['overall']['conditions'],len(rows))
        self.assertEqual(sum(data['ood']['overall']['transitions'].values()),len(rows))
        self.assertEqual(data['nominal']['matched_difference_pp'],0)
        text=report.markdown(data)
        self.assertIn('SFT full sequence',text)
        if not (b.ROOT/'outputs/rlvr_analysis/p3/p3_a2_rollouts.csv').exists():
            self.assertEqual(data['p3_a2']['status'],'NOT AVAILABLE')
    def test_a2_schema_and_success_only_steps(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); p=root/'outputs/rlvr_analysis/p3/p3_a2_rollouts.csv'
            p.parent.mkdir(parents=True)
            header=['repeat','model','task_id','historical_case','seed','success','steps','host','evaluator_sha256']
            with p.open('w') as f:
                writer=csv.writer(f); writer.writerow(header)
                writer.writerows([[1,'scale',7,'lost',1,1,20,'test','hash'],[2,'scale',7,'lost',1,0,300,'test','hash'],[1,'ript',7,'lost',1,0,300,'test','hash']])
            result=report.a2_summary(root)
            self.assertEqual(result['total_rollouts'],3)
            self.assertEqual(result['tasks']['7']['models']['scale']['successful_steps']['mean'],20)
            self.assertIsNone(result['tasks']['7']['models']['ript']['successful_steps']['mean'])
    def test_report_overwrite_requires_explicit_refresh(self):
        data=report.build(b.ROOT)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            report.write_report(root,data)
            with self.assertRaises(ValueError): report.write_report(root,data)
            report.write_report(root,data,True)
            self.assertTrue((root/'outputs/rlvr_analysis/research_summary.json').is_file())
    def test_mandatory_missing_results_fail(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError): report.historical(Path(t))
            with self.assertRaises(ValueError): report.nominal(Path(t))


if __name__=='__main__': unittest.main()
