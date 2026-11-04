"""Read-only integration of expert and released-policy artifacts. Never runs models."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from experiments.shot_scaling.core import ROOT, read, write, sha, digest, require
from .plots import plot

LABELS={'sft':'Released Full-SFT (full sequence)', 'scale':'Released Scale-cached', 'ript':'Released RIPT-cached'}
BLOCKERS=[
 'The pinned RLDS snapshot does not record the producer commit of moojink/rlds_dataset_builder. The retrieved converter reference is not an authenticated producer revision.',
 'No matching regenerated HDF5 demonstration is available locally for an identity-matched numerical check. Original unfiltered HDF5 or unrelated simulator initial states cannot substitute.',
 'Recorded robosuite versions differ: LIBERO requirements pin 1.4.0, server freeze records 1.4.1. The dataset-generation environment and SI asset scaling are not authenticated.'
]


def inventory(roots):
    return {str(p.relative_to(ROOT)):sha(p) for root in roots for p in sorted(root.rglob('*'))
            if p.is_file() and p.name!='.DS_Store'}


def source_entry(path, start, end, claim, repo=None):
    p=ROOT/path
    result=dict(path=path,start_line=start,end_line=end,sha256=sha(p),claim=claim)
    if repo:
        pin=read(ROOT/'stage0/repos.lock.json')[repo]['revision']
        relative=str(p.relative_to(ROOT/'vendor'/repo))
        original=subprocess.check_output(['git','-C',str(ROOT/'vendor'/repo),'show',f'{pin}:{relative}'])
        result.update(pinned_revision=pin,matches_pinned_bytes=hashlib.sha256(original).hexdigest()==sha(p))
        require(result['matches_pinned_bytes'],f'Pinned source differs: {path}')
    return result


def evidence(audits):
    sources=[
      source_entry('vendor/openvla-oft/experiments/robot/libero/regenerate_libero_dataset.py',115,199,
        'HDF5 ee_states stores robot0_eef_pos then quat2axisangle(robot0_eef_quat); captures before env.step of the corresponding kept action, after 10 settling steps. Initial simulator states[0] is copied from original, so it is not the same as the recorded settled EEF observation.','openvla-oft'),
      source_entry('vendor/openvla-oft/experiments/robot/libero/run_libero_eval.py',243,265,
        'Evaluation prepare_observation concatenates raw EEF position, axis-angle and gripper; sampled P3 state is before policy normalization.','openvla-oft'),
      source_entry('vendor/LIBERO/scripts/create_dataset.py',180,246,
        'Original LIBERO HDF5 saves ee_states and derived ee_pos/ee_ori. This is not numerical equality evidence for replayed modified no-noop datasets.','LIBERO')]
    base='experiments/dataset_analysis/evidence_sources/'
    for suite in ('Goal','Spatial','10'):
        sources.append(source_entry(base+f'builder_LIBERO_{suite}_LIBERO_{suite}_dataset_builder.py',15,60,
            'Reference converter concatenates HDF5 ee_states[i] and gripper_states[i] and pairs action[i], casts float32. Producer commit linkage unresolved.'))
    sources += [source_entry(base+'robosuite_robosuite_robots_single_arm.py',300,322,
                   'Version 1.4.1 eef_pos reads site_xpos (world position), eef_quat reads end-effector body world quaternion in xyzw order.'),
                source_entry(base+'robosuite_robosuite_utils_transform_utils.py',490,520,
                   'quat2axisangle returns axis multiplied by angle in radians; not Euler angles.')]
    stats={}
    for s,d in audits.items():
        require(d['episodes_sha256']==digest(d['episodes']),f'{s}: episode checksum mismatch')
        states=[np.asarray(e['observation_state']) for e in d['episodes'] if e['observation_state'] is not None]
        require(all(e['observation_state'] is None or np.asarray(e['observation_state']).shape==(e['steps'],8) for e in d['episodes']),'State alignment differs')
        stats[s]=dict(episodes=len(d['episodes']),state_episodes=len(states),transitions=sum(e['steps'] for e in d['episodes']),
            finite=all(np.isfinite(x).all() for x in states),source_files_sha256=d['source']['source_files_sha256'],
            audit_sha256=sha(ROOT/f'outputs/dataset_analysis/audits/{s}.json'))
    return dict(schema=1,status='BLOCKED',physical_plots_allowed=False,
        confirmed_reference_definitions=dict(position='EEF site world position in reference robosuite',
            orientation='EEF body axis-angle rotation vector, radians',
            alignment='pre_action in pinned replay code and same-index reference converter',
            units='meters expected under standard SI assets; dataset-specific generation scale not authenticated'),
        dataset_specific_status='UNVERIFIED: reference-code agreement is not authenticated conversion provenance',
        sources=sources,reference_retrieval=read(ROOT/base/'retrieval.json'),audits=stats,
        numerical_comparison=dict(status='UNAVAILABLE',matched_demonstrations=0,
            reason='No matching regenerated HDF5 locally. No unrelated simulator state was compared.'),
        unresolved_assumptions=BLOCKERS,decision='Do not change REVIEWED gate; retain unavailable physical panels.')


def audit_npz(root):
    result=[]
    for p in sorted(root.rglob('*.npz')):
        row=dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
        with np.load(p,allow_pickle=False) as z:
            row['keys']=z.files
            if 'state' in z:
                state=z['state'];require(state.ndim==2 and state.shape[1]==8 and np.isfinite(state).all(),'Invalid P3 state')
                meta=read(p.with_name('info.json'));require(meta['array_sha256']==row['sha256'],'P3 bundle checksum mismatch')
                steps=[r['step'] for r in meta['observations']]
                require(len(steps)==len(state),'P3 state/index count mismatch')
                row.update(kind='sparse_raw_observation_state',samples=len(state),steps=steps,
                    full_simulator_step_trajectory=False,initial_state_sha256=meta['initial_state_sha256'],
                    observation_ids=[r['id'] for r in meta['observations']],
                    note='state[:3] available at selected policy-query times; not full path length or final endpoint')
            else:row.update(kind='features_or_actions_only',samples=0,full_simulator_step_trajectory=False)
        result.append(row)
    return dict(files=result,counts=dict(Counter(r['kind'] for r in result)),
        full_paths_available=False,policy_pose_samples=sum(r['samples'] for r in result),
        limitation='Sparse measured states exist. Do not join across rollouts or treat planned action chunks as executed physical paths.')


def validate_labels(rows):
    for r in rows:
        require(r['model'] in LABELS,'Unknown released model')
        require(r['family']=='released_checkpoint' and r.get('shots') is None,'Released/shot condition mixing')
        require(r['condition'] in ('nominal','ood'),'Mixed nominal/OOD label')
        require(r['suite'] in ('goal','spatial','long'),'Invalid suite')
        require(0<=r['successes']<=r['n'] and r['n']>0,'Invalid sample count')


def dashboard_data(bundle,summary):
    expert=[]
    for suite,d in bundle['suites'].items():
        for split,ts in d['tasks'].items():
            for task,t in ts.items():
                expert.append(dict(suite=suite,task=task,split=split,n=t['episodes'],transitions=t['transitions'],
                    mean_length=sum(t['lengths'])/len(t['lengths']),physical=t['physical_path_episodes']))
    released=[]
    for model,d in summary['goal']['nominal']['models'].items():
        for task,t in d['per_task'].items():
            released.append(dict(suite='goal',task=task,model=model,label=LABELS[model],family='released_checkpoint',shots=None,
                condition='nominal',shift='Original LIBERO',n=t['episodes'],successes=t['successes'],
                source='experiments/nominal/goal.csv',checkpoint=d['checkpoint'],
                controlled='Matched inference reference for Scale/RIPT; Full-SFT inference differs. Single seed; not a shot-scaling causal test.'))
    paired=ROOT/'experiments/libero_plus/results/abc_paired.csv'
    with paired.open(newline='') as f: raw=list(csv.DictReader(f))
    require(len(raw)==summary['goal']['ood']['overall']['conditions'],'OOD summary coverage differs from CSV')
    for model in ('scale','ript'):
        require(sum(int(r[model]) for r in raw)==summary['goal']['ood']['overall']['successes'][model],'OOD summary outcome mismatch')
    names=list(summary['goal']['nominal']['models']['sft']['per_task'])
    groups={}
    for r in raw:
        matches=[n for n in names if r['task_name']==n or r['task_name'].startswith(n+'_')]
        require(len(matches)==1,'Ambiguous OOD base task mapping')
        for model in LABELS:
            key=(matches[0],r['shift'],model)
            if key not in groups: groups[key]=[0,0]
            require(r[model] in ('0','1'),'Invalid OOD outcome')
            groups[key][0]+=1;groups[key][1]+=int(r[model])
    for (task,shift,model),(n,k) in groups.items():
        released.append(dict(suite='goal',task=task,model=model,label=LABELS[model],family='released_checkpoint',shots=None,
            condition='ood',shift=shift,n=n,successes=k,source=str(paired.relative_to(ROOT)),
            controlled='Condition-paired CSV; Scale/RIPT use cached inference, Full-SFT differs. Original manifest provenance is unavailable from this CSV alone.'))
    validate_labels(released)
    p3=[]
    for run in summary['goal']['p3_b']:
        a=run['analysis']
        for group,g in a['groups'].items():
            features=g.get('features',{})
            p3.append(dict(run=run['run'],group=group,n=g['N'],rollouts=g.get('distinct_rollouts'),tasks=g.get('distinct_tasks'),
                action_l2=g['actions']['executed']['action_l2']['mean'],
                cka={k:v.get('cka',{}).get('value') for k,v in features.items()},
                source='outputs/rlvr_analysis/p3b/'+run['run']+'/analysis.json'))
    return dict(expert=expert,released=released,p3=p3,p3a=summary['goal']['p3_a2'],
        missing=summary['suites'],fewshot_rl='MISSING: no new shot-conditioned D0/D1 RL results are included',
        seeds=bundle['seeds'],historical='Historical Goal5 remains distinct from newly sampled nested5',
        p3b_counts=[dict(run=r['run'],n=r['analysis']['observations'],rollouts=r['analysis']['rollouts']) for r in summary['goal']['p3_b']])


def build(output):
    require(not output.exists(),'Choose a fresh output; originals and earlier reports must be preserved')
    dataset=ROOT/'outputs/dataset_analysis';rlvr=ROOT/'outputs/rlvr_analysis'
    protected=inventory([dataset,rlvr]);audits={s:read(dataset/f'audits/{s}.json') for s in ('goal','spatial','long')}
    bundle_path=dataset/'libero_expert_trajectories/analysis.json';bundle=read(bundle_path)
    summary_path=rlvr/'research_summary_all.json';summary=read(summary_path)
    output.mkdir(parents=True)
    ev=evidence(audits);write(output/'physical_convention_evidence.json',ev)
    npz=audit_npz(rlvr);write(output/'p3_pose_inventory.json',npz)
    data=dashboard_data(bundle,summary)
    write(output/'dashboard_data.json',data)
    plot(bundle_path,output/'figures')
    template=(Path(__file__).parent/'dashboard.html').read_text()
    data['links']={name:os.path.relpath(ROOT/path,output) for name,path in {
        'dataset':'outputs/dataset_analysis/libero_expert_trajectories/report.md',
        'rlvr':'outputs/rlvr_analysis/research_summary_all.md',
        'p3':'outputs/rlvr_analysis/p3b/p3b-balanced-01/analysis.json'}.items()}
    data['npz_summary']={k:v for k,v in npz.items() if k!='files'}
    (output/'index.html').write_text(template.replace('__DATA__',json.dumps(data).replace('<','\\u003c')))
    report='''# Unified LIBERO research report

Status: INCOMPLETE — physical EEF verification is blocked. No training or evaluation was run.

1. Dataset overview: all three saved source audits were loaded and their episode checksums checked. Goal 428, Spatial 432, Long 379 episodes. Raw state is present, but physical interpretation is not yet authenticated.
2. Expert variation: action/gripper/length statistics and unchanged shot manifests remain available. Figures 1–4 were reproduced from the saved analysis in this directory. Figure 2 explicitly documents the coordinate-evidence blocker, with 3D/XY/XZ/YZ panel slots. No XYZ was integrated from actions or enabled by editing status.
3. Released models: Goal nominal Full-SFT 486/500, Scale-cached 488/500, RIPT-cached 488/500. Goal OOD Full-SFT 1675/2591, Scale 1667/2591, RIPT 1620/2591. These are separate protocols; pooled nominal/OOD rates are invalid. Full-SFT is not the matched cached-inference baseline. Long/Spatial evaluations are incomplete in the supplied summary. New shot-conditioned RL results are missing.
4. Goal P3: 80 fixed-seed repeated diagnostic rollouts across 8 outcome-selected conditions. Balanced P3B: 678 selected observations from 68 rollouts; smoke: 4 observations from 4 rollouts. Shared-observation action/representation comparisons are exploratory, not causal evidence of RL improvements. P3 sparse state samples do not support full executed-path metrics.

## Physical-coordinate blocker

'''+ '\n'.join('- '+s for s in BLOCKERS)+'''

See physical_convention_evidence.json for source paths, line ranges, SHA256, exact local pins, reference-only converter revision, and unavailable numerical comparison. No unrelated initial state was compared.

## Use

Open index.html directly; no web server or external dependencies required. Filters are scoped by section so unsupported task-level OOD filters are not invented. Original reports are linked, not overwritten. Files and figures are referenced without copying datasets.

Reproduce: `python -m experiments.dataset_analysis.integration --output outputs/research_dashboard/rebuild-02`

Optional future logging: add `--log-eef` to existing nominal Long/Spatial or Goal/Long/Spatial PLUS evaluators, using a fresh output path. No rollout is launched by this integration command. The logger records every env.step, including settling, with exact actions and pre/post pose; disable or filter settling for policy comparisons. New D0/D1 training/evaluation integration remains separate and must bind shot manifest and checkpoint lineage.
'''
    (output/'report.md').write_text(report)
    require(inventory([dataset,rlvr])==protected,'Original analysis artifacts changed')
    write(output/'validation.json',dict(status='PARTIAL_WITH_EXPLICIT_PHYSICAL_BLOCKER',originals_unchanged=True,
        original_file_hashes=protected,figure_input_sha256=sha(bundle_path),rlvr_input_sha256=sha(summary_path),
        physical_figure2_complete=False,training_or_rollouts_executed=False,
        code_hashes={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}))
    print(output.resolve())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.output.resolve())
