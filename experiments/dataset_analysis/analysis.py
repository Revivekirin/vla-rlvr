"""Descriptive task-conditioned analysis; no coverage ranking or random percentiles."""
from collections import Counter
import numpy as np
from experiments.shot_scaling.core import require,digest,SUITES
from experiments.shot_scaling.data import select,validate_selection
from .audit import no_leakage


def train_inventory(audit):
    rows=[{k:r[k] for k in ('episode_id','task','steps','complete','sha256')} for r in audit['episodes'] if r['split']=='train']
    return dict(schema=1,suite=audit['suite'],dataset_name=audit['dataset_name'],synthetic=False,
        complete_source=True,source_episode_count=len(rows),episodes=rows,episodes_sha256=digest(rows),
        source=dict(repo=audit['source']['repo'],revision=audit['source']['revision'],split='train',
                    files={k:v['sha256'] for k,v in audit['source']['files'].items()}))


def align_path(x,n=32):
    x=np.asarray(x);return np.stack([np.interp(np.linspace(0,1,n),np.linspace(0,1,len(x)),x[:,j]) for j in range(3)],1)


def path_statistics(x):
    x=np.asarray(x,dtype=float)
    require(x.ndim==2 and x.shape[1]==3 and len(x)>0 and np.isfinite(x).all(),'Invalid measured XYZ sequence')
    return dict(initial_xyz=x[0].tolist(),last_recorded_xyz=x[-1].tolist(),
        path_length=float(np.linalg.norm(np.diff(x,axis=0),axis=1).sum()),
        net_displacement=float(np.linalg.norm(x[-1]-x[0])))


def analyze(audit,seeds=(42,),historical=None):
    require(audit['status']=='AUDITED' and not audit.get('synthetic'),'A complete real source audit is required')
    require(audit['episodes_sha256']==digest(audit['episodes']),'Audit episode checksum mismatch')
    no_leakage(audit['episodes'])
    inventory=train_inventory(audit)
    selections={str(seed):{str(k):select(inventory,audit['suite'],k,seed) for k in (1,2,5,10,20,'Full')} for seed in seeds}
    historical_ids=validate_selection(historical,inventory) if historical is not None else None
    physical=audit['physical_convention']['status']=='REVIEWED'
    es=[]
    for r in audit['episodes']:
        a=np.asarray(r['actions']);s=np.asarray(r['observation_state']) if r['observation_state'] is not None else None
        require(a.shape==(r['steps'],7) and np.isfinite(a).all(),'Invalid action alignment')
        require(s is None or (s.shape==(len(a),8) and np.isfinite(s).all()),'State/action length alignment mismatch')
        valid=physical and s is not None
        closed=a[:,6]>=.5;events=np.flatnonzero(closed[1:]!=closed[:-1])+1
        e=dict(episode_id=r['episode_id'],task=r['task'],split=r['split'],steps=r['steps'],source_payload_sha256=r['sha256'],
            xyz=s[:,:3].tolist() if valid else None,orientation=s[:,3:6].tolist() if valid else None,
            initial_proprio=s[0].tolist() if s is not None else None,
            physical=path_statistics(s[:,:3]) if valid else None,
            physical_status='REVIEWED_CONVENTION' if valid else 'UNAVAILABLE',
            gripper_command=a[:,6].tolist(),gripper_events=[dict(step=int(j),command='high' if closed[j] else 'low') for j in events],
            gripper_event_definition='Crossing raw command threshold 0.5; opening/closing polarity is not independently verified',
            gripper_state=s[:,6:].tolist() if s is not None else None,
            terminal=r['terminal'],success=r['success'],last_reward=r['reward'][-1])
        for label,x in [('translation',a[:,:3]),('rotation',a[:,3:6])]:
            mag=np.linalg.norm(x,axis=1);good=mag>1e-10
            az=np.arctan2(x[good,1],x[good,0]);elev=np.arctan2(x[good,2],np.linalg.norm(x[good,:2],axis=1))
            unit=np.divide(x,mag[:,None],out=np.zeros_like(x),where=good[:,None]);pairs=good[1:]&good[:-1]
            cosine=(unit[1:]*unit[:-1]).sum(1)[pairs]
            e[label]=dict(magnitude=mag.tolist(),magnitude_mean=float(mag.mean()),magnitude_std=float(mag.std()),
                azimuth_counts=np.histogram(az,bins=12,range=(-np.pi,np.pi))[0].tolist(),
                elevation_counts=np.histogram(elev,bins=6,range=(-np.pi/2,np.pi/2))[0].tolist(),
                zero_commands=int((~good).sum()),successive_cosine_mean=float(cosine.mean()) if len(cosine) else None,
                reversals=int((cosine<0).sum()),valid_successive_pairs=len(cosine))
        es.append(e)
    tasks={};pairs=[]
    # Scientific question: within the same task, how much do measured positions differ at comparable progress?
    # Temporal normalization is not DTW or policy diversity; use at most 64 deterministic episodes/task/split.
    for split in audit['declared_splits']:
        tasks[split]={}
        for task in sorted({e['task'] for e in es if e['split']==split}):
            rows=[e for e in es if e['task']==task and e['split']==split]
            paths=[e for e in rows if e['xyz'] is not None]
            sampled=sorted(paths,key=lambda e:digest(['physical-pairs-v1',e['episode_id']]))[:64]
            for j,left in enumerate(sampled):
                for right in sampled[j+1:]:
                    rms=float(np.sqrt(np.mean(np.sum((align_path(left['xyz'])-align_path(right['xyz']))**2,axis=1))))
                    pairs.append(dict(suite=audit['suite'],split=split,task=task,episode_a=left['episode_id'],episode_b=right['episode_id'],
                        aligned_xyz_rms_m=rms,path_length_difference_m=abs(left['physical']['path_length']-right['physical']['path_length']),
                        endpoint_distance_m=float(np.linalg.norm(np.array(left['xyz'][-1])-right['xyz'][-1]))))
            raw=[np.asarray(r['actions']) for r in audit['episodes'] if r['task']==task and r['split']==split]
            allraw=np.concatenate(raw);mean=np.mean([x.mean(0) for x in raw],axis=0)
            tasks[split][task]=dict(episodes=len(rows),transitions=sum(e['steps'] for e in rows),lengths=[e['steps'] for e in rows],
                gripper_event_counts=dict(Counter(len(e['gripper_events']) for e in rows)),
                transition_weighted_action=dict(mean=allraw.mean(0).tolist(),std=allraw.std(0).tolist()),
                episode_weighted_action=dict(mean=mean.tolist(),std=np.sqrt(np.maximum(np.mean([(x*x).mean(0) for x in raw],axis=0)-mean*mean,0)).tolist()),
                physical_path_episodes=len(paths),pairwise_sample_ids=[e['episode_id'] for e in sampled],
                physical_path_status='REVIEWED_CONVENTION' if len(paths)==len(rows) else 'UNAVAILABLE_OR_PARTIAL')
    return dict(schema=1,suite=audit['suite'],status='DESCRIPTIVE_ANALYSIS',synthetic=False,source=audit['source'],
        physical_convention=audit['physical_convention'],declared_splits=audit['declared_splits'],integrity=audit['integrity'],
        episodes=es,tasks=tasks,physical_pairs=pairs,selections=selections,
        historical_selection=historical,historical_ids=sorted(historical_ids) if historical_ids else None,
        historical_note='Historical Goal5 is separate from every newly selected nested condition',
        unavailable=['Object positions are not inferred from images or controller actions','No independent success verification from terminal alone'],
        pairwise_question='Within-task differences in measured XYZ at normalized progress, using 32 samples and at most 64 source-ID-ranked episodes per task/split',
        endpoint_note='Last recorded observation, not an inferred post-final-action environment state')
