import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from .common import OFT, OUT, digest, sha, tasks
from .evaluation_checks import check_gate, summarize
from .prepare_dataset import select
from .train import parser, plan
from .validation import select_validation, check_validation
from .worker import instrument


class ContinuationTests(unittest.TestCase):
    def test_incomplete_new_checkpoint_is_not_treated_as_legacy(self):
        from .checkpoint import check
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            (path/'fewshot_checkpoint.json').write_text(json.dumps({'cumulative_optimizer_steps':1000}))
            with self.assertRaisesRegex(ValueError, 'Incomplete new checkpoint'):
                check(path,merged=False)

    def test_exposure_and_smoothing_survive_resume_without_inventing_legacy_counts(self):
        from .diagnostics import Diagnostics
        from types import SimpleNamespace
        OUT.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=OUT) as tmp:
            root=Path(tmp); old=root/'old'; old.mkdir(); run=root/'run'; run.mkdir()
            c=dict(output=str(run),resume=str(old),resume_step=200,loss_window=2,validation=None)
            d=Diagnostics(c,None)
            self.assertEqual(d.since,200)
            self.assertEqual(sum(d.exposure.values()),0)
            for step,loss in [(201,2.),(202,4.)]:
                d.exposure.update([tasks()[0]])
                d.pending=[dict(loss=loss,samples=1,tasks=[tasks()[0]],actions={})]
                d.after_step(dict(log_step=step,optimizer=SimpleNamespace(param_groups=[{'lr':.0005}])))
            d.save(old)
            restored=Diagnostics(c,None)
            self.assertEqual(list(restored.window),[2.,4.])
            self.assertEqual(restored.exposure[tasks()[0]],2)
            self.assertEqual(restored.since,200)
            self.assertEqual(restored.ema,d.ema)

    def test_disjoint_selection_excludes_duplicate_content_under_new_id(self):
        rows = [dict(episode_id=f'{t}:{i}', task=t, steps=3, complete=True, sha256=digest([t,i]))
                for t in tasks() for i in range(8)]
        train = {'episodes':select(rows,5,42,tasks())}
        duplicates = [dict(r, episode_id='copy:'+r['episode_id']) for r in train['episodes']]
        chosen = select_validation(rows+duplicates, train)
        self.assertEqual(len(chosen),10)
        self.assertFalse({r['sha256'] for r in chosen} & {r['sha256'] for r in train['episodes']})
        self.assertEqual(chosen, select_validation(list(reversed(rows+duplicates)), train))
        with self.assertRaises(ValueError): select_validation(train['episodes']+duplicates,train)

    def test_checkpoint_schedule_and_12000_rejected(self):
        c = plan(parser().parse_args(['--run-id','test','--max-steps','6000','--save-freq','6000']))
        self.assertEqual(c['checkpoint_steps'],[1000,3000,6000])
        self.assertEqual(c['scheduler_config']['milestones'],[100000])
        with self.assertRaises(ValueError):
            plan(parser().parse_args(['--run-id','test','--max-steps','12000','--save-freq','1000']))

    def test_instrumented_resume_steps_stop_and_save_at_update_boundaries(self):
        tree = ast.parse(instrument((OFT/'vla-scripts/finetune.py').read_text()))
        index = next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and
                     any(isinstance(t,ast.Name) and t.id=='gradient_step_idx' for t in n.targets))
        save_condition = next(n.test for n in ast.walk(tree) if isinstance(n,ast.If) and
                              any(isinstance(x,ast.Name) and x.id=='_should_save' for x in ast.walk(n.test)))
        from types import SimpleNamespace
        for accumulation in (1,2,4):
            cfg = SimpleNamespace(grad_accumulation_steps=accumulation)
            saved = []
            for batch_idx in range((6000-200)*accumulation):
                scope = dict(batch_idx=batch_idx,cfg=cfg)
                index_value = eval(compile(ast.Expression(index),'index','eval'),scope)
                scope.update(log_step=200+index_value, _should_save=lambda step,cfg: step in (1000,3000,6000))
                if eval(compile(ast.Expression(save_condition),'save','eval'),scope):
                    saved.append(scope['log_step'])
            self.assertEqual(saved,[1000,3000,6000])

    def test_gate_requires_complete_all_task_positive_unchanged_results(self):
        OUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=OUT) as tmp:
            root=Path(tmp); result=root/'goal/sft/seed-1'; result.mkdir(parents=True)
            identity=dict(seed=1, kind='fewshot', checkpoint_identity='model', evaluator_sha256='eval', wrapper_sha256='wrapper')
            for t in range(1,11):
                (result/f'task-{t:04d}-trial-00.json').write_text(json.dumps(dict(task_id=t, trial_id=0,
                    success=t==1, steps=300, max_abs_executed_action=1.)))
            summarize(root, list(range(1,11)), 1, identity)
            path=root/'summary.json'
            check_gate(path,'fewshot','model','eval','wrapper')
            with self.assertRaises(ValueError): check_gate(path,'fewshot','other-model','eval','wrapper')
            s=json.loads(path.read_text()); s['successes']=0; path.write_text(json.dumps(s))
            with self.assertRaises(ValueError): check_gate(path,'fewshot','model','eval','wrapper')
            s['successes']=1; path.write_text(json.dumps(s))
            (result/'task-0001-trial-00.json').write_text('{}')
            with self.assertRaises(ValueError): check_gate(path,'fewshot','model','eval','wrapper')

    def test_validation_manifest_rejects_overlap_and_changed_train_statistics(self):
        OUT.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=OUT) as tmp:
            root=Path(tmp); train=root/'train'; val=root/'val'; train.mkdir(); val.mkdir()
            rows=[dict(episode_id=f'{t}:{i}', task=t, steps=3, complete=True, sha256=digest([t,i]))
                  for t in tasks() for i in range(8)]
            selected=select(rows,5,42,tasks())
            m=dict(schema=1,dataset_name='libero_goal_no_noops',shots=5,episodes=selected,
                   source={'revision':'fixture'},selection_sha256=digest(selected))
            (train/'selection_manifest.json').write_text(json.dumps(m)); (train/'normalization.json').write_text('{}')
            chosen=select_validation(rows,m)
            v=dict(episodes=chosen,per_task=1,source=m['source'],selection_sha256=digest(chosen),
                   train_manifest_sha256=sha(train/'selection_manifest.json'))
            (val/'validation_manifest.json').write_text(json.dumps(v))
            a=dict(status='verified_materialization',selection_sha256=v['selection_sha256'],
                   train_normalization_sha256=sha(train/'normalization.json'))
            (val/'validation_audit.json').write_text(json.dumps(a))
            check_validation(val,train,verify_files=False)
            (train/'normalization.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError): check_validation(val,train,verify_files=False)
            v['episodes'][0]=selected[0]; v['selection_sha256']=digest(v['episodes'])
            (val/'validation_manifest.json').write_text(json.dumps(v))
            with self.assertRaises(ValueError): check_validation(val,train,verify_files=False)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch unavailable; run this CPU continuity test on server')
class OptimizerContinuityTests(unittest.TestCase):
    def test_adam_scheduler_weights_match_uninterrupted_next_update(self):
        import torch
        from .resume_checks import restore_optimizer, equal_state
        torch.manual_seed(7)
        model=torch.nn.Linear(3,2)
        optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4)
        scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[201],gamma=.1)
        x=torch.tensor([[.2,.5,-.7]])
        def update(m,o,s):
            o.zero_grad(); m(x).square().mean().backward(); o.step(); s.step()
        for _ in range(200): update(model,optimizer,scheduler)
        weights=copy.deepcopy(model.state_dict())
        saved=copy.deepcopy(dict(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                                 parameter_names=['weight','bias']))
        restored=torch.nn.Linear(3,2); restored.load_state_dict(weights)
        opt2=torch.optim.AdamW(restored.parameters(),lr=5e-4)
        sch2=torch.optim.lr_scheduler.MultiStepLR(opt2,milestones=[201],gamma=.1)
        restore_optimizer(opt2,sch2,saved,200,['weight','bias'])
        update(model,optimizer,scheduler); update(restored,opt2,sch2)
        equal_state(model.state_dict(),restored.state_dict(),'next-step weights')
        equal_state(optimizer.state_dict(),opt2.state_dict(),'next-step optimizer')
        equal_state(scheduler.state_dict(),sch2.state_dict(),'next-step scheduler')
        with self.assertRaises(ValueError): restore_optimizer(opt2,sch2,saved,200,['bias','weight'])
        bad=copy.deepcopy(saved); bad['scheduler']['last_epoch']=199
        with self.assertRaises(ValueError): restore_optimizer(opt2,sch2,bad,200,['weight','bias'])
        legacy=copy.deepcopy(saved); legacy.pop('parameter_names')
        restore_optimizer(opt2,sch2,legacy,200,['weight','bias'])
        with self.assertRaises(ValueError): equal_state(torch.ones(1,dtype=torch.bfloat16),torch.ones(1),'dtype')


if __name__ == '__main__':
    unittest.main()
