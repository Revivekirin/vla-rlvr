"""CPU extraction of all recorded states. Physical semantics require reviewed evidence."""
from collections import Counter
from pathlib import Path
import io
import os
import re
import numpy as np
from experiments.shot_scaling.core import ROOT,REVISION,SUITES,require,read,sha,digest,task_names
from experiments.fewshot_sft.prepare_dataset import payload_hash


def task_identity(language,suite):
    text=language.decode() if isinstance(language,bytes) else str(language)
    normalized='_'.join(text.strip().split()).lower()
    matches=[]
    for task in task_names(suite):
        short=re.sub(r'^.*?_SCENE\d+_','',task)
        if normalized in (task.lower(),short.lower()):matches.append(task)
    require(len(matches)==1,f'Ambiguous/unknown {suite} instruction: {text!r}')
    return matches[0]


def record(episode,eid,suite,split):
    steps=list(episode['steps']);require(bool(steps),'Empty episode')
    require([bool(s['is_first']) for s in steps]==[True]+[False]*(len(steps)-1),'Broken first flags')
    require([bool(s['is_last']) for s in steps]==[False]*(len(steps)-1)+[True],'Broken last flags')
    require(not any(bool(s['is_terminal']) for s in steps[:-1]),'Terminal before last step')
    tasks={task_identity(s['language_instruction'],suite) for s in steps};require(len(tasks)==1,'Task changes within episode')
    actions=[];states=[];joints=[];rewards=[];discounts=[]
    state_presence=[];joint_presence=[];fields=set()
    for step in steps:
        action=np.asarray(step['action']);obs=step['observation'];fields.update(obs)
        require(action.shape==(7,) and action.dtype==np.float32 and np.isfinite(action).all(),'Invalid float32 action[7]')
        actions.append(action.tolist())
        for key,dim,target,presence in [('state',8,states,state_presence),('joint_state',7,joints,joint_presence)]:
            present=key in obs;presence.append(present)
            if present:
                x=np.asarray(obs[key]);require(x.shape==(dim,) and x.dtype==np.float32 and np.isfinite(x).all(),f'Invalid {key}')
                target.append(x.tolist())
        for camera in ('image','wrist_image'):
            require(camera in obs,f'Missing {camera}')
            value=obs[camera]
            if isinstance(value,bytes):
                from PIL import Image
                with Image.open(io.BytesIO(value)) as im: x=np.asarray(im.convert('RGB'))
            else:x=np.asarray(value)
            require(x.ndim==3 and x.shape[-1]==3 and x.dtype==np.uint8,'Invalid camera observation')
        for key,target in [('reward',rewards),('discount',discounts)]:
            value=float(step[key]) if key in step else None
            require(value is None or np.isfinite(value),f'Nonfinite {key}');target.append(value)
    require(len(set(state_presence))==1 and len(set(joint_presence))==1,'Intermittently missing proprio fields')
    task=tasks.pop();h=payload_hash(dict(episode,steps=steps))
    metadata=episode.get('episode_metadata',{});source_file=metadata.get('file_path')
    if isinstance(source_file,np.ndarray):source_file=source_file.item()
    if isinstance(source_file,bytes):source_file=source_file.decode()
    return dict(episode_id=eid,task=task,suite=suite,split=split,steps=len(steps),complete=True,sha256=h,
        actions=actions,observation_state=states or None,joint_state=joints or None,
        observation_fields=sorted(fields),source_file_path=source_file,
        terminal=bool(steps[-1]['is_terminal']),reward=rewards,discount=discounts,
        success=None,success_note='No independently verified success field; terminal/reward are preserved as recorded',
        alignment='state and action at the same serialized RLDS step; pre/post-action timing needs convention evidence')


def no_leakage(rows):
    ids=[r['episode_id'] for r in rows];require(len(ids)==len(set(ids)),'Duplicate source episode IDs')
    seen={}
    for r in rows:
        previous=seen.setdefault(r['sha256'],r['split'])
        require(previous==r['split'],'Identical payload occurs in train and another official split')
    # Also reject duplicated payloads within a split for unambiguous selection identity.
    require(len({r['sha256'] for r in rows})==len(rows),'Duplicate episode payloads')


def convention(path,source_hash):
    if path is None:return dict(status='UNAVAILABLE',reason='Coordinate frame, units, representation and action alignment have not been reviewed')
    c=read(path)
    require(c.get('status')=='REVIEWED' and c.get('dataset_revision')==REVISION and c.get('source_files_sha256')==source_hash,
            'Convention record must bind to the exact audited source')
    require(c.get('state_position_indices')==[0,1,2] and c.get('state_orientation_indices')==[3,4,5], 'Unexpected state mapping')
    require(c.get('position_units')=='m' and c.get('orientation_representation')=='axis_angle_radians', 'Unverified physical units/representation')
    require(c.get('coordinate_frame') and c.get('state_timing') in ('pre_action','post_action') and c.get('reviewer'), 'Incomplete convention review')
    required={'position_mapping','orientation_mapping','coordinate_frame','units','state_action_alignment'}
    require({e['claim'] for e in c.get('evidence',[])}>=required,'Missing convention evidence')
    for e in c['evidence']:
        p=Path(e['path']);require(p.is_file() and sha(p)==e['sha256'],'Convention evidence file changed/missing')
        require(e.get('explanation'),'Each evidence item needs a review explanation')
    return dict(c,verification_note='Reviewer-supplied, source-bound evidence; file hashes verified, semantic review not automated')


def extract(source,suite,revision,convention_path=None,limit=None):
    from experiments.fewshot_sft.common import inventory
    from experiments.fewshot_sft.prepare_dataset import episodes
    source=Path(source).resolve()
    require(suite in SUITES and revision==REVISION,'Wrong suite/revision')
    require((source/'dataset_info.json').is_file() and (source/'features.json').is_file(),'Missing actual TFDS version directory')
    require(SUITES[suite]+'_no_noops' in source.parts,'Wrong source suite directory')
    require(not source.is_relative_to(ROOT/'outputs'),'Use original RLDS, not a selected dataset')
    require(limit is None or limit>0,'Invalid smoke bound')
    files=inventory(source);source_hash=digest(files)
    physical=convention(convention_path,source_hash)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    import tensorflow_datasets as tfds
    builder=tfds.builder_from_directory(str(source))
    aliases={SUITES[suite],SUITES[suite]+'_no_noops'}
    if suite=='long':aliases.add('liber_o10')
    require(builder.info.name in aliases,'Wrong TFDS builder name')
    declared={k:v.num_examples for k,v in builder.info.splits.items()}
    require('train' in declared,'Missing train split')
    rows=[]
    import itertools
    for split,count in declared.items():
        stream=episodes(builder,split)
        subset=[record(e,eid,suite,split) for eid,e in itertools.islice(stream,limit)]
        require(len(subset)==(min(count,limit) if limit else count),'Observed/declaration split count mismatch')
        rows.extend(subset)
    no_leakage(rows)
    if limit is None:
        require({r['task'] for r in rows if r['split']=='train'}==set(task_names(suite)),'Incomplete train task mapping')
    require(inventory(source)==files,'Source changed during extraction')
    return dict(schema=1,status='SMOKE_ONLY' if limit else 'AUDITED',suite=suite,dataset_name=SUITES[suite]+'_no_noops',
        synthetic=False,source=dict(path=str(source),repo='openvla/modified_libero_rlds',revision=revision,files=files,source_files_sha256=source_hash),
        schema_fields=read(source/'features.json'),declared_splits=declared,observed_splits=dict(Counter(r['split'] for r in rows)),
        physical_convention=physical,episodes=rows,episodes_sha256=digest(rows),
        integrity=dict(duplicate_ids=0,duplicate_payloads=0,nonfinite_actions=0,nonfinite_present_states=0,
            missing_state_episodes=sum(r['observation_state'] is None for r in rows),
            missing_joint_state_episodes=sum(r['joint_state'] is None for r in rows)),
        official_test_available='test' in declared,success_verified=False)
