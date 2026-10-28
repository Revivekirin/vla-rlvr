"""P3-B frozen observations and matched cached-path features. No training.

Default smoke: two conditions, two sources, one observation each (four total).
Balanced mode is explicit: 34 conditions x 2 sources x <=10 observations.
"""
import argparse
import ast
from collections import Counter
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import uuid

sys.dont_write_bytecode = True
from evaluate import ROOT, REVISION, SUITE, checkpoints, task_mapping
import p3_action_analysis as a1
from p3_action_analysis import require, sha, save_json, save_npz, input_sha, check_obs, check_actions
import p3b_metrics as metrics

OUTPUT = ROOT/'outputs/rlvr_analysis/p3b'
STRATA = ('Nominal', 'Camera Viewpoints', 'Sensor Noise', 'Light Conditions')
SOURCES = ('scale', 'ript')
CODE = ('p3_representation_analysis.py', 'p3b_metrics.py', 'p3_action_analysis.py')


def read(path):
    return json.loads(path.read_text())


def revision(path):
    return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'], text=True).strip()


def recognized_reset_patch(reference, actual):
    """Accept only the known fix for LIBERO's reset loop, byte for byte."""
    before=(b'        success = False\n'
            b'        while not success:\n'
            b'            try:\n'
            b'                ret = self.env.reset()\n'
            b'                success = True\n'
            b'            except RandomizationError:\n'
            b'                pass\n'
            b'            finally:\n'
            b'                continue\n')
    after=(b'        while True:\n'
           b'            try:\n'
           b'                ret = self.env.reset()\n'
           b'                break\n'
           b'            except RandomizationError:\n'
           b'                pass\n')
    return reference.count(before)==1 and actual==reference.replace(before,after)


def original_runtime_state(path):
    relative='libero/libero/envs/env_wrapper.py'
    changed=subprocess.check_output(['git','-C',str(path),'diff','--name-only','HEAD','--'],text=True).splitlines()
    source=path/relative
    if not changed:
        return dict(state='clean',path=str(source),sha256=sha(source))
    require(changed==[relative],f'LIBERO tracked changes outside the recognized reset fix: {changed}')
    reference=subprocess.check_output(['git','-C',str(path),'show','HEAD:'+relative])
    require(recognized_reset_patch(reference,source.read_bytes()),
            'LIBERO env_wrapper.py differs from the recognized reset fix')
    return dict(state='recognized_reset_fix',path=str(source),sha256=sha(source))


def runpath(name):
    import re
    require(re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}',name), 'Invalid run ID')
    path=OUTPUT/name
    require(not path.is_symlink(), 'Symlink run forbidden')
    return path


def nominal_names(original):
    folder=original/'libero/libero/benchmark'
    tree=ast.parse((folder/'libero_suite_task_map.py').read_text())
    names=next(ast.literal_eval(n.value)[SUITE] for n in tree.body if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='libero_task_map' for t in n.targets))
    tree=ast.parse((folder/'__init__.py').read_text())
    order=next(ast.literal_eval(n.value)[0] for n in tree.body if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='task_orders' for t in n.targets))
    require(len(names)==10 and sorted(order)==list(range(10)), 'Unexpected nominal mapping')
    return [names[i] for i in order]


def base_task(name, names):
    candidates=[n for n in names if name==n or name.startswith(n+'_')]
    require(bool(candidates), f'Cannot map condition to nominal base task: {name}')
    return max(candidates,key=len)


def selection(root, original, mode):
    from research_report import historical, a2_records
    names=nominal_names(original)
    records, provenance=historical(root)
    repeats,_=a2_records(root)
    ambiguous=set()
    studied={r['task_id'] for r in repeats}
    for tid in studied:
        for model in SOURCES:
            outcomes=[r['success'] for r in repeats if r['task_id']==tid and r['model']==model]
            if outcomes and any(outcomes) and not all(outcomes):
                ambiguous.add(tid)
    chosen=[dict(dataset='nominal',shift='Nominal',task_id=i+1,task_name=n,base_task=n,
                 historical_case='nominal',ambiguous=False) for i,n in enumerate(names)]
    for shift in STRATA[1:]:
        used=Counter()
        for label in ('lost','gained','both_success','both_fail'):
            pool=[dict(r,base_task=base_task(r['task_name'],names)) for r in records if r['shift']==shift and r['historical_case']==label]
            require(len(pool)>=2, f'Insufficient {shift}/{label} conditions')
            for _ in range(2):
                # Diversity first, then unstable/studied A2 conditions, then ID.
                pool.sort(key=lambda r:(used[r['base_task']], r['task_id'] not in ambiguous,
                                        r['task_id'] not in studied, r['task_id']))
                r=pool.pop(0)
                used[r['base_task']]+=1
                chosen.append(dict(r,dataset='plus',ambiguous=r['task_id'] in ambiguous))
    if mode=='smoke':
        chosen=[chosen[0],next(r for r in chosen if r['shift']=='Camera Viewpoints')]
    return chosen,provenance


def sampling_indices(length, maximum):
    import numpy as np
    require(length>0 and maximum>0, 'Empty trajectory/budget')
    bins=np.array_split(np.arange(length),3)
    quotas=[0,1,0] if maximum==1 else [maximum//3+(i<maximum%3) for i in range(3)]
    selected=[]
    counts={}
    for label,group,q in zip(('early','middle','late'),bins,quotas):
        n=min(len(group),q)
        ix=group[np.linspace(0,len(group)-1,n,dtype=int)] if n else []
        selected.extend((int(i),label) for i in ix)
        counts[label]=dict(target=q,collected=n,shortfall=q-n)
    # Very short smoke trajectory may have no middle third; use its only observation.
    if not selected and maximum==1:
        selected=[(0,'early')]
        counts['early']['collected']=1
        counts['middle']['shortfall']=1
    return sorted(selected),counts


def task_files(case, original, plus):
    root=(original if case['dataset']=='nominal' else plus)/'libero/libero'
    name=case['task_name']
    bddl=name.split('_view_')[0] if '_view_' in name else name
    initial=name.split('_view_')[0].split('_light_')[0]
    paths=[root/'bddl_files'/SUITE/(bddl+'.bddl'), root/'init_files'/SUITE/(initial+'.pruned_init')]
    for path in paths:
        require(path.is_file(),f'Missing physical task input: {path}')
    return paths


def preflight(args):
    errors=[]
    result=dict(mode=args.mode,seed=args.seed,errors=errors)
    try:
        cases,provenance=selection(ROOT,args.original,args.mode)
        result.update(cases=cases,historical_provenance=provenance)
        result['task_files']={f"{c['dataset']}:{c['task_id']}": [str(p) for p in task_files(c,args.original,args.plus)] for c in cases}
    except Exception as e:
        errors.append('selection: '+str(e))
    locks=read(ROOT/'stage0/repos.lock.json')
    for name,path in [('LIBERO',args.original),('LIBERO-plus',args.plus)]:
        try:
            require(path.is_dir(),f'Missing checkout {path}')
            require(revision(path)==locks[name]['revision'],f'{name} revision mismatch')
            if name=='LIBERO':
                result['original_runtime']=original_runtime_state(path)
            else:
                require(not subprocess.check_output(['git','-C',str(path),'diff','HEAD','--'],text=True), f'{name} tracked source changed')
            require((path/'libero/libero/assets').is_dir(),f'{name} assets absent')
        except Exception as e:
            errors.append(str(e))
    try:
        result['checkpoints']=a1.checkpoint_files()
        result['normalization_key'],result['normalization']=a1.norm_spec(ROOT/checkpoints('scale')['base'])
        result['action_order']=a1.verify_ordering()
    except Exception as e:
        errors.append(str(e))
    for dep in ('numpy','torch','transformers','peft','tensorflow','mujoco','wand','skimage'):
        if importlib.util.find_spec(dep) is None:
            errors.append('Missing dependency: '+dep)
    if args.plus.is_dir() and 'cases' in result:
        try:
            mapping=task_mapping(args.plus)
            for c in result['cases']:
                if c['dataset']=='plus':
                    e=mapping[c['task_id']-1]
                    require(e['name']==c['task_name'] and e['category']==c['shift'],'Plus mapping mismatch')
        except Exception as e:
            errors.append(str(e))
    result.update(ready=not errors,ready_meaning='CPU checks only; CUDA/EGL runtime not tested',
                  max_observations=4 if args.mode=='smoke' else 680)
    return result


def verify_manifest(run):
    m=read(run/'manifest.json')
    for path,digest in m['file_sha256'].items():
        require(sha(Path(path))==digest,f'Changed provenance: {path}')
    for dataset in ('nominal','plus'):
        require(revision(Path(m[dataset]))==m['revisions'][dataset],'Dataset revision changed')
    return m


def worker_setup(run,m,dataset):
    """No original configuration is modified; attempts remain within run/work."""
    path=run/'work'/uuid.uuid4().hex
    path.mkdir(parents=True)
    config=path/'libero'
    config.mkdir()
    base=Path(m[dataset])/'libero/libero'
    save_json(config/'config.yaml',dict(benchmark_root=str(base), bddl_files=str(base/'bddl_files'),
        init_states=str(base/'init_files'),assets=str(base/'assets'),datasets=str(Path(m[dataset])/'datasets')))
    os.environ.update(LIBERO_CONFIG_PATH=str(config),MUJOCO_GL='egl',TOKENIZERS_PARALLELISM='false',
                      WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_MODULES_CACHE=str(path/'hf-modules'))
    sys.path[:0]=[m[dataset],str(ROOT/'runtime/openvla-oft'),str(ROOT/'runtime/ript-vla')]
    os.chdir(path)
    import torch
    from experiments.robot import openvla_utils
    from experiments.robot.libero import run_libero_eval as oft
    openvla_utils.update_auto_map=lambda _:None
    openvla_utils.check_model_logic_mismatch=lambda _:None
    require(torch.cuda.is_available(),'CUDA unavailable; refusing CPU model load')
    oft.set_seed_everywhere(m['seed'])
    return torch,oft


def policy(model,m):
    from ript.algos.rl_optimizers.openvla_oft_interface import OpenVLA_OFT_Policy
    from ript.env_runner.openvla_oft_libero_runner import get_vla_action_batch
    from experiments.robot.libero import run_libero_eval as oft
    import torch
    import numpy as np
    ck=checkpoints(model)
    p=OpenVLA_OFT_Policy(str(ROOT/ck['base']),str(ROOT/ck['scale']),SUITE,32,0.,
                         str(ROOT/ck['adapter']) if model=='ript' else None, seed=m['seed'],fix_scale_head=True)
    for module in (p.model,p.action_head,p.proprio_projector,p.scale_head):
        module.eval().requires_grad_(False)
    require(p.cfg.unnorm_key==m['normalization_key'] and p.model.norm_stats[p.cfg.unnorm_key]==m['normalization'],'Loaded normalization mismatch')
    require(p.cfg.center_crop and p.cfg.use_proprio and p.cfg.num_images_in_input==2,'Preprocessing mismatch')
    def predict(obs,language):
        check_obs(obs,language)
        before=input_sha(obs,language)
        with torch.inference_mode():
            result=get_vla_action_batch(p.cfg,p.model,p.processor,[{k:v.copy() for k,v in obs.items()}],language,
                       action_head=p.action_head,proprio_projector=p.proprio_projector,scale_head=p.scale_head,
                       use_laplace_sampling=False,scale_factor=1.)
        raw,sampled,logprob,mean,logscale=result
        require(tuple(mean.shape)==(1,8,7) and raw.shape==(1,8,7),'Wrong action batch')
        require(torch.equal(sampled,mean) and all(torch.isfinite(v).all() for v in (mean,logprob,logscale)), 'Nonfinite/stochastic output')
        mean=mean[0].float().cpu().numpy().copy()
        raw=raw[0].copy()
        executed=np.stack([oft.process_action(a.copy(),'openvla') for a in raw])
        check_actions(mean,raw,executed,m['normalization'])
        require(before==input_sha(obs,language),'Input mutated')
        return mean,raw,executed
    import inspect
    implementation={inspect.getfile(obj):sha(Path(inspect.getfile(obj))) for obj in (OpenVLA_OFT_Policy,get_vla_action_batch,type(p.model.get_base_model()))}
    info=dict(host=socket.gethostname(),gpu=torch.cuda.get_device_name(0),cuda_version=torch.version.cuda,
              implementation=implementation,model=model,dtype=str(next(p.action_head.parameters()).dtype),
              versions={name:__import__(name).__version__ for name in ('torch','transformers','peft','numpy')},
              normalization_key=p.cfg.unnorm_key,cached=True,sampling=False,
              attention_implementation=p.model.language_model.config._attn_implementation)
    return p,predict,info


def rollout_id(case,source):
    return f"{case['dataset']}-{case['task_id']:04d}-{source}"


def atomic_bundle(run,destination,arrays,info):
    require(not destination.exists(),f'Artifact exists: {destination}')
    temp=run/'work'/uuid.uuid4().hex
    temp.mkdir(parents=True)
    save_npz(temp/'arrays.npz',**arrays)
    info=dict(info,array_sha256=sha(temp/'arrays.npz'))
    save_json(temp/'info.json',info)
    destination.parent.mkdir(parents=True,exist_ok=True)
    os.rename(temp,destination)


def bundle(path):
    import numpy as np
    info=read(path/'info.json')
    require(sha(path/'arrays.npz')==info['array_sha256'],f'Corrupt bundle: {path}')
    with np.load(path/'arrays.npz',allow_pickle=False) as f:
        arrays={k:f[k] for k in f.files}
    return arrays,info


def validate_observations(path):
    arrays,info=bundle(path)
    n=len(info['observations'])
    require(n>0 and all(len(v)==n for v in arrays.values()),'Observation count mismatch')
    for i,row in enumerate(info['observations']):
        obs={k:arrays[k][i] for k in ('full_image','wrist_image','state')}
        check_obs(obs,row['language'])
        require(input_sha(obs,row['language'])==row['input_sha256'],'Observation hash mismatch')
    return arrays,info


def capture_worker(run,m,dataset,source):
    import numpy as np
    torch,oft=worker_setup(run,m,dataset)
    import libero.libero as libero
    from libero.libero import benchmark
    require(Path(libero.__file__).resolve().is_relative_to(Path(m[dataset])),'Wrong benchmark import')
    suite=benchmark.get_benchmark_dict()[SUITE](task_order_index=0)
    expected=nominal_names(Path(m[dataset])) if dataset=='nominal' else [r['name'] for r in task_mapping(Path(m[dataset]))]
    require([suite.get_task(i).name for i in range(suite.n_tasks)]==expected,'Benchmark order changed')
    _,predict,execution=policy(source,m)
    for case in m['cases']:
        if case['dataset']!=dataset:
            continue
        rid=rollout_id(case,source)
        dest=run/'bank'/rid
        if dest.exists():
            validate_observations(dest)
            continue
        oft.set_seed_everywhere(m['seed'])
        task=suite.get_task(case['task_id']-1)
        require(task.name==case['task_name'],'Selected task mismatch')
        env,language=oft.get_libero_env(task,'openvla',resolution=256)
        candidates=[]
        try:
            env.reset()
            initial=suite.get_task_init_states(case['task_id']-1)[0]
            raw=env.set_init_state(initial)
            for _ in range(10):
                raw,_,_,_=env.step(oft.get_libero_dummy_action('openvla'))
            step,success=0,False
            while step<m['horizon'] and not success:
                obs,_=oft.prepare_observation(raw,224)
                mean,actions,executed=predict(obs,language)
                candidates.append((step,{k:v.copy() for k,v in obs.items()},mean,actions,executed))
                for action in executed:
                    if step>=m['horizon']:
                        break
                    raw,_,done,_=env.step(action.tolist())
                    step+=1
                    if done:
                        success=True
                        break
            selected,counts=sampling_indices(len(candidates),m['per_rollout'])
            observations=[]
            values=[]
            for i,stage in selected:
                t,obs,mean,actions,executed=candidates[i]
                observations.append(dict(id=f'{rid}:{t}',input_sha256=input_sha(obs,language),language=language,
                    rollout_id=rid,task_id=case['task_id'],task_name=case['task_name'],base_task=case['base_task'],
                    shift=case['shift'],dataset=dataset,source=source,step=t,stage=stage,success=success,
                    historical_case=case['historical_case'],ambiguous=case['ambiguous']))
                values.append(dict(**obs,normalized_mean=mean,unnormalized=actions,executed=executed))
            arrays={k:np.stack([v[k] for v in values]) for k in values[0]}
            info=dict(observations=observations,composition=counts,steps=step,success=success,
                      truncated=not success and m['horizon']<300,execution=execution,
                      initial_state_sha256=__import__('hashlib').sha256(np.ascontiguousarray(initial).tobytes()).hexdigest(),
                      dtypes={k:str(v.dtype) for k,v in arrays.items()})
            atomic_bundle(run,dest,arrays,info)
            validate_observations(dest)
            print(f'Captured {rid}: {len(values)} observations; success={success}; steps={step}',flush=True)
        finally:
            env.close()


class Features:
    """Hooks verified against cached runner: two decoder calls, 56 action + 2 tail tokens."""
    def __init__(self,p,layers):
        self.layers=layers
        self.base=p.model.get_base_model()
        require(hasattr(self.base,'projector'),'No vision-to-language projector')
        decoder=self.base.language_model.model.layers
        require(all(0<=i<len(decoder) for i in layers),'Requested decoder layer unavailable')
        self.handles=[]
        self.head=p.action_head
        self.original=self.head.predict_action
        self.handles.append(self.base.projector.register_forward_hook(lambda mod,args,out:self.record('vision_projection',out)))
        for i in layers:
            self.handles.append(decoder[i].register_forward_hook(lambda mod,args,out,i=i:self.record(f'layer_{i}',out[0])))
        def head_call(hidden,*args,**kwargs):
            self.record('action_head_input',hidden)
            return self.original(hidden,*args,**kwargs)
        self.head.predict_action=head_call
        self.reset()
    def reset(self):
        self.values={}
        self.details={}
    def record(self,key,tensor):
        require(tensor.ndim==3 and tensor.shape[0]==1,'Unexpected feature tensor')
        self.values.setdefault(key,[]).append(tensor.detach().float().cpu().numpy().copy())
        self.details.setdefault(key,[]).append(dict(shape=list(tensor.shape),dtype=str(tensor.dtype)))
    def finish(self):
        import numpy as np
        out,desc={},{}
        expected={'vision_projection','action_head_input'}|{f'layer_{i}' for i in self.layers}
        require(set(self.values)==expected,'Missing expected feature hook')
        for key,values in self.values.items():
            if key.startswith('layer_'):
                require(len(values)==2 and values[1].shape[1]==58,'Expected prefix/action decoder calls with 58 suffix tokens')
                selected=values[1][:,:56,:]
                rule='Second cached decoder call; first 56 action tokens, exclude two tail tokens; mean across tokens'
                module='base.language_model.model.layers.'+key.split('_')[1]
            else:
                require(len(values)==1,'Feature hook called unexpected number of times')
                selected=values[0]
                if key=='action_head_input':
                    require(selected.shape[1]==56,'Expected 56 action-head input tokens')
                    rule='Mean of 56 action token inputs before action-head reshape'
                    module='policy.action_head.predict_action argument[0]'
                else:
                    require(selected.shape[1]==self.base.vision_backbone.get_num_patches()*2,'Vision patch count mismatch')
                    rule='Mean of projected patches from both images; excludes proprio and language'
                    module='base.projector'
            feature=selected.mean(axis=1)[0]
            require(np.isfinite(feature).all(),'Nonfinite feature')
            out[key]=feature
            desc[key]=dict(module=module,pooling=rule,calls=self.details[key],output_dimension=len(feature))
        return out,desc
    def close(self):
        self.head.predict_action=self.original
        for h in self.handles:
            h.remove()


def frozen_bank(run):
    bank=read(run/'bank.json')
    require(bank['manifest_sha256']==sha(run/'manifest.json'),'Frozen manifest changed')
    for rid,digest in bank['bundles'].items():
        require(sha(run/'bank'/rid/'info.json')==digest,'Frozen bank changed')
        validate_observations(run/'bank'/rid)
    return bank


def extract_worker(run,m,model):
    import numpy as np
    bank=frozen_bank(run)
    worker_setup(run,m,'nominal')  # No environment is constructed during replay.
    p,predict,execution=policy(model,m)
    hooks=Features(p,m['layers'])
    try:
        for rid in bank['bundles']:
            dest=run/'features'/model/rid
            if dest.exists():
                bundle(dest)
                continue
            arrays,info=validate_observations(run/'bank'/rid)
            outputs=[]
            descriptions=[]
            for i,row in enumerate(info['observations']):
                hooks.reset()
                actions=predict({k:arrays[k][i] for k in ('full_image','wrist_image','state')},row['language'])
                features,description=hooks.finish()
                outputs.append(dict(**features,normalized_mean=actions[0],unnormalized=actions[1],executed=actions[2]))
                descriptions.append(description)
            atomic_bundle(run,dest,{k:np.stack([v[k] for v in outputs]) for k in outputs[0]},
                dict(observations=info['observations'],features=descriptions,execution=execution,
                     bank_sha256=sha(run/'bank.json')))
    finally:
        hooks.close()


def validate_run(run,require_features=False):
    import numpy as np
    m=read(run/'manifest.json')
    bank=frozen_bank(run)
    expected={rollout_id(c,s) for c in m['cases'] for s in SOURCES}
    require(set(bank['bundles'])==expected,'Bank coverage differs from planned rollouts')
    all_ids=[]
    for rid in bank['bundles']:
        obs,info=validate_observations(run/'bank'/rid)
        ids=[r['id'] for r in info['observations']]
        all_ids+=ids
        candidates=[(c,s) for c in m['cases'] for s in SOURCES if rollout_id(c,s)==rid]
        require(len(candidates)==1,'Unexpected rollout identity')
        case,source=candidates[0]
        require(len(ids)<=m['per_rollout'],'Observation budget exceeded')
        for row in info['observations']:
            require(row['rollout_id']==rid and row['source']==source and row['id']==f"{rid}:{row['step']}",'Wrong observation identity')
            require(all(row[k]==case[k] for k in ('dataset','task_id','task_name','base_task','shift')),'Wrong task metadata')
            require(0<=row['step']<m['horizon'] and row['step']%8==0,'Invalid query timestep')
        check_actions(obs['normalized_mean'],obs['unnormalized'],obs['executed'],m['normalization'])
        if require_features:
            models=[]
            for model in SOURCES:
                arrays,fi=bundle(run/'features'/model/rid)
                require(fi['bank_sha256']==sha(run/'bank.json'),'Features reference different bank')
                metrics.align(ids,[r['id'] for r in fi['observations']])
                require(fi['observations']==info['observations'],'Feature input metadata differs')
                keys={'vision_projection','action_head_input'}|{f'layer_{i}' for i in m['layers']}
                require(set(arrays)==keys|{'normalized_mean','unnormalized','executed'},'Unexpected feature keys')
                require(len(fi['features'])==len(ids),'Missing feature descriptions')
                for key in keys:
                    require(arrays[key].ndim==2 and len(arrays[key])==len(ids) and np.isfinite(arrays[key]).all(),'Invalid feature matrix')
                check_actions(arrays['normalized_mean'],arrays['unnormalized'],arrays['executed'],m['normalization'])
                models.append((arrays,fi))
            for key in keys:
                require(models[0][0][key].shape==models[1][0][key].shape,'Feature dimensions differ')
            require(models[0][1]['features']==models[1][1]['features'],'Feature pooling/call shapes/dtypes differ')
    require(len(set(all_ids))==len(all_ids),'Duplicate observation IDs')
    return dict(valid=True,observations=len(all_ids),rollouts=len(expected),features_checked=require_features)


def analyze(run,repeats):
    import numpy as np
    validate_run(run,True)
    m=read(run/'manifest.json')
    bank=frozen_bank(run)
    records=[]
    executions=[]
    feature_definitions=None
    data={model:[] for model in SOURCES}
    composition=[]
    for rid in bank['bundles']:
        _,info=validate_observations(run/'bank'/rid)
        records+=info['observations']
        executions.append(info['execution'])
        composition.append(dict(rollout_id=rid,stages=info['composition'],steps=info['steps'],success=info['success'],truncated=info['truncated']))
        for model in SOURCES:
            values,feature_info=bundle(run/'features'/model/rid)
            data[model].append(values)
            executions.append(feature_info['execution'])
            if feature_definitions is None:
                feature_definitions=feature_info['features'][0]
    arrays={model:{k:np.concatenate([v[k] for v in values]) for k in values[0]} for model,values in data.items()}
    feature_keys=[k for k in arrays['scale'] if k not in ('normalized_mean','unnormalized','executed')]
    groups={}
    selectors={}
    for i,r in enumerate(records):
        for key in (r['shift']+'|'+r['source'],r['shift']+'|ALL_SOURCES',r['shift']+'|'+r['source']+'|TASK:'+r['base_task']):
            selectors.setdefault(key,[]).append(i)
    for key,idx in selectors.items():
        rs=[records[i] for i in idx]
        clusters=[r['base_task'] for r in rs]
        rollouts=[r['rollout_id'] for r in rs]
        entry=dict(N=len(idx),distinct_tasks=len(set(clusters)),distinct_conditions=len({(r['dataset'],r['task_id']) for r in rs}),
                   distinct_rollouts=len(set(rollouts)),independent_unit='base task, not frames or action slots',features={})
        for feature in feature_keys:
            x,y=[arrays[model][feature][idx] for model in SOURCES]
            entry['features'][feature]=dict(**metrics.paired(x,y),uncertainty=metrics.bootstrap(x,y,clusters,repeats,m['seed']),
                                           sensitivity=metrics.sensitivity(x,y,rollouts,m['seed']))
        entry['actions']={}
        for space in ('normalized_mean','unnormalized','executed'):
            a,b=[arrays[model][space][idx] for model in SOURCES]
            dist=a1.distances(a,b)
            entry['actions'][space]={name:metrics.stats(v.mean(axis=1)) for name,v in dist.items()}
            entry['actions'][space]['chunk_l2']=metrics.stats(np.linalg.norm(dist['action_l2'],axis=1))
        a,b=[arrays[model]['executed'][idx] for model in SOURCES]
        entry['actions']['gripper_disagreement_per_observation']=metrics.stats((a[...,6]!=b[...,6]).mean(axis=1))
        groups[key]=entry
    # Cross-stratum control: same N, same pooling, deterministic subset; not independence adjustment.
    common=min(len(selectors[s+'|'+src]) for s in {r['shift'] for r in records} for src in SOURCES)
    matched={}
    for key,idx in selectors.items():
        if '|TASK:' in key or key.endswith('ALL_SOURCES'):
            continue
        subset=np.random.default_rng(m['seed']).choice(idx,common,replace=False)
        matched[key]={k:metrics.paired(arrays['scale'][k][subset],arrays['ript'][k][subset]) for k in feature_keys}
    result=dict(feature_definitions_example=feature_definitions,
                execution_provenance=[json.loads(v) for v in sorted({json.dumps(e,sort_keys=True) for e in executions})],
                mode=m['mode'],observations=len(records),rollouts=len(bank['bundles']),
                distinct_base_tasks=len({r['base_task'] for r in records}),composition=composition,
                groups=groups,matched_N_control=dict(N=common,groups=matched),bootstrap_repeats=repeats,
                bank_sha256=sha(run/'bank.json'),feature_files={str(p.relative_to(run)):sha(p) for p in sorted((run/'features').glob('*/*/info.json'))},warnings=[
                    'Selected conditions and one initial state per task; not a representative population sample.',
                    'CKA and singular-value entropy do not directly measure task-relevant information or causality.',
                    'Frame duplication does not increase independent sample size; pooled sources share task clusters.',
                    'Effective ranks across different N or pooling are not directly comparable; use matched-N control.',
                    'Small smoke is extraction verification only, not representation evidence.'])
    save_json(run/'analysis.json',result)
    return result


def launch(run,m,operation,model,dataset='nominal'):
    label=operation+'-'+dataset+'-'+model+'-'+uuid.uuid4().hex[:8]
    a1.run_logged([sys.executable,'-u',str(Path(__file__).resolve()),'_worker','--run-id',run.name,
                   '--worker-op',operation,'--model',model,'--dataset',dataset],run,label)


def capture(args,run):
    if (run/'manifest.json').exists():
        m=verify_manifest(run)
        require(m['mode']==args.mode and m['seed']==args.seed and m['layers']==args.layers,'Resume options differ from manifest')
    else:
        require(not run.exists(),'Partial run without manifest; choose another run ID')
        plan=preflight(args)
        require(plan['ready'],json.dumps(plan['errors']))
        files=set(sum(plan['checkpoints'].values(),[]))|{str(Path(__file__).parent/f) for f in CODE}
        files|={str(ROOT/f) for f in a1.IMPL}
        files|=set(plan['historical_provenance'].get('files',{}))
        files|=set(sum(plan['task_files'].values(),[]))
        files|=set(plan['action_order']['evidence'])
        files.add(plan['original_runtime']['path'])
        p=ROOT/'outputs/rlvr_analysis/p3/p3_a2_rollouts.csv'
        if p.exists(): files.add(str(p))
        m=dict(plan,nominal=str(args.original),plus=str(args.plus),mode=args.mode,seed=args.seed,
               horizon=16 if args.mode=='smoke' else 300,per_rollout=1 if args.mode=='smoke' else 10,
               layers=args.layers,created_host=socket.gethostname(),repository_revision=revision(ROOT),
               revisions=dict(nominal=revision(args.original),plus=revision(args.plus)),
               file_sha256={p:sha(Path(p)) for p in sorted(files)},
               protocol=dict(initial_state_index=0,wait_steps=10,env_seed=0,image_resolution=256,policy_resolution=224,
                             center_crop=True,action_chunk=[8,7],sampling=False,reseed_per_rollout=True,
                             note='Nominal per-environment semantics; one initial state, singleton cached inference, not a rerun of all 50 historical trials'))
        run.mkdir(parents=True)
        save_json(run/'manifest.json',m)
    if (run/'bank.json').exists():
        validate_run(run)
        print('Frozen bank already complete; no capture performed')
        return
    for dataset in ('nominal','plus'):
        for model in SOURCES:
            pending=[c for c in m['cases'] if c['dataset']==dataset and not (run/'bank'/rollout_id(c,model)).exists()]
            if pending:
                launch(run,m,'capture',model,dataset)
    bundles={}
    for case in m['cases']:
        for source in SOURCES:
            rid=rollout_id(case,source)
            validate_observations(run/'bank'/rid)
            bundles[rid]=sha(run/'bank'/rid/'info.json')
    save_json(run/'bank.json',dict(bundles=bundles,manifest_sha256=sha(run/'manifest.json')))
    print(validate_run(run))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('dry-run','capture','extract','analyze','validate','report','_worker'))
    p.add_argument('--run-id',required=True)
    p.add_argument('--mode',choices=('smoke','balanced'),default='smoke')
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--layers',type=int,nargs='+',default=[0,15,31])
    p.add_argument('--original',type=Path,default=ROOT/'vendor/LIBERO')
    p.add_argument('--plus',type=Path,default=ROOT/'vendor/LIBERO-plus')
    p.add_argument('--bootstrap',type=int,default=200)
    p.add_argument('--refresh',action='store_true',help='Derived unified reports only')
    p.add_argument('--worker-op',choices=('capture','extract'))
    p.add_argument('--model',choices=SOURCES)
    p.add_argument('--dataset',choices=('nominal','plus'),default='nominal')
    args=p.parse_args()
    args.original,args.plus=args.original.resolve(),args.plus.resolve()
    require(len(set(args.layers))==len(args.layers) and 0<len(args.layers)<=4,'Select 1..4 distinct layers')
    require(0<=args.bootstrap<=2000,'Bootstrap must be 0..2000')
    run=runpath(args.run_id)
    if args.operation=='dry-run':
        result=preflight(args)
        print(json.dumps(result,indent=2))
        return 0 if result['ready'] else 2
    if args.operation=='_worker':
        m=read(run/'manifest.json')
        require(args.model in SOURCES,'Worker model required')
        if args.worker_op=='capture': capture_worker(run,m,args.dataset,args.model)
        elif args.worker_op=='extract': extract_worker(run,m,args.model)
        else: raise ValueError('Worker operation required')
        return 0
    # Exclusive lock outside the run lets fresh capture keep an exclusive mkdir.
    import fcntl
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/(args.run_id+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.operation=='capture':
            capture(args,run)
        elif args.operation=='report':
            from research_report import build,write_report
            print(*write_report(ROOT,build(ROOT,args.run_id),args.refresh))
        else:
            m=verify_manifest(run)
            if args.operation=='extract':
                validate_run(run)
                for model in SOURCES:
                    if any(not (run/'features'/model/rid).exists() for rid in frozen_bank(run)['bundles']):
                        launch(run,m,'extract',model)
                print(validate_run(run,True))
            elif args.operation=='validate':
                print(validate_run(run,require_features=True))
                if (run/'analysis.json').exists():
                    result=read(run/'analysis.json')
                    require(result['bank_sha256']==sha(run/'bank.json'),'Stale analysis')
                    for path,digest in result['feature_files'].items():
                        require(sha(run/path)==digest,'Feature artifacts changed after analysis')
            else:
                require(not (run/'analysis.json').exists(),'Analysis exists; refusing overwrite')
                analyze(run,args.bootstrap)
    return 0


if __name__=='__main__':
    try:
        sys.exit(main())
    except (ValueError,FileNotFoundError,KeyError) as e:
        print('P3-B failed: '+str(e),file=sys.stderr)
        sys.exit(2)
