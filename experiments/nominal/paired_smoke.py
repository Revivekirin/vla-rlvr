"""Bounded Scale/RIPT action comparison on one identical nominal observation.

Three isolated processes: environment capture, then each policy's cached inference.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.libero_plus import suite_common as common


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=common.SUITES,required=True)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--task-id',type=int,default=1)
    p.add_argument('--run-id',default='smoke-01')
    p.add_argument('--phase',choices=('capture','scale','ript','compare'),default='capture')
    a=p.parse_args()
    import numpy as np
    import re
    common.require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',a.run_id), 'Invalid run ID')
    folder=ROOT/'outputs/libero-paired-smoke'/a.suite/a.run_id/f'seed-{a.seed}'/f'task-{a.task_id:04d}'
    benchmark=ROOT/'vendor/LIBERO'
    if a.phase=='capture':
        for model in ('scale','ript'):
            check=common.preflight(a.suite,model,'nominal',benchmark)
            common.require(check['ready'],json.dumps(check['errors']))
        folder.mkdir(parents=True,exist_ok=False)
        common.configure('nominal',benchmark)
        from libero.libero import benchmark as lib_benchmark
        from experiments.robot.libero import run_libero_eval as oft
        suite=lib_benchmark.get_benchmark_dict()[common.SUITES[a.suite][0]](task_order_index=0)
        common.require(0<a.task_id<=suite.n_tasks,'Unknown task ID')
        task=suite.get_task(a.task_id-1)
        oft.set_seed_everywhere(a.seed)
        env,language=oft.get_libero_env(task,'openvla',resolution=256)
        try:
            env.reset()
            obs=env.set_init_state(suite.get_task_init_states(a.task_id-1)[0])
            for _ in range(10):obs,_,_,_=env.step(oft.get_libero_dummy_action('openvla'))
            prepared,_=oft.prepare_observation(obs,224)
            with (folder/'observation.npz').open('xb') as f:
                np.savez_compressed(f,**prepared)
            (folder/'metadata.json').write_text(json.dumps({'suite':a.suite,'task_id':a.task_id,
                'task_name':task.name,'language':language,'seed':a.seed,
                'observation_sha256':common.sha(folder/'observation.npz'),
                'checkpoints':{m:common.checkpoint_paths(a.suite,m) for m in ('scale','ript')},
                'normalization_key':common.SUITES[a.suite][1],
                'evaluator_sha256':common.sha(__file__)},indent=2))
        finally:env.close()
        for model in ('scale','ript'):
            subprocess.run([sys.executable,'-u',__file__,'--suite',a.suite,'--seed',str(a.seed),
                            '--task-id',str(a.task_id),'--run-id',a.run_id,'--phase',model],check=True)
        subprocess.run([sys.executable,__file__,'--suite',a.suite,'--seed',str(a.seed),
                        '--task-id',str(a.task_id),'--run-id',a.run_id,'--phase','compare'],check=True)
    elif a.phase in ('scale','ript'):
        meta=json.loads((folder/'metadata.json').read_text())
        common.require(common.sha(folder/'observation.npz')==meta['observation_sha256'],'Observation changed')
        common.configure('nominal',benchmark)
        import torch
        with np.load(folder/'observation.npz',allow_pickle=False) as data:
            obs={k:data[k] for k in ('full_image','wrist_image','state')}
        predict=common.predict_fn(a.suite,a.phase,a.seed)
        with torch.inference_mode():actions=np.asarray(predict(obs,meta['language']))
        common.require(actions.shape==(8,7) and np.isfinite(actions).all(),'Invalid action chunk')
        with (folder/f'{a.phase}_actions.npz').open('xb') as f:np.savez_compressed(f,actions=actions)
        print(a.phase,'action shape',actions.shape,flush=True)
    else:
        with np.load(folder/'scale_actions.npz',allow_pickle=False) as f:scale=f['actions']
        with np.load(folder/'ript_actions.npz',allow_pickle=False) as f:ript=f['actions']
        common.require(scale.shape==ript.shape==(8,7),'Action shape mismatch')
        diff=np.abs(scale-ript)
        report={'suite':a.suite,'task_id':a.task_id,'observation_sha256':
            json.loads((folder/'metadata.json').read_text())['observation_sha256'],
            'max_abs':float(diff.max()),'mean_chunk_l2':float(np.linalg.norm(scale-ript,axis=1).mean()),
            'translation_l2':float(np.linalg.norm(scale[:,:3]-ript[:,:3],axis=1).mean()),
            'rotation_l2':float(np.linalg.norm(scale[:,3:6]-ript[:,3:6],axis=1).mean()),
            'gripper_disagreement':float(np.mean(np.sign(scale[:,6])!=np.sign(ript[:,6])))}
        with (folder/'summary.json').open('x') as f:json.dump(report,f,indent=2)
        print(json.dumps(report,indent=2))

if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError) as e:
        print('Paired smoke failed: '+str(e),file=sys.stderr)
        sys.exit(2)
