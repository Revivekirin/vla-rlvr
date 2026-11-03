"""Phase 2: within-task temporal action variation from saved expert artifacts only."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

from .core import ROOT, REVISION, digest, read, require, runtime, sha, write
from .data import validate_inventory, validate_selection

SHOTS = (1, 2, 5, 10, 20)
METRICS = ('temporal_action', 'descriptor', 'initial_state', 'gripper_pattern')


def align(values, size=32):
    """Linear interpolation in normalized episode phase; not elapsed physical time."""
    x = np.asarray(values, dtype=float)
    require(x.ndim == 2 and len(x) > 0 and np.isfinite(x).all(), 'Invalid sequence')
    grid = np.linspace(0, 1, size)
    return np.stack([np.interp(grid, np.linspace(0, 1, len(x)), x[:, d]) for d in range(x.shape[1])], 1)


def command_features(x):
    magnitude = np.linalg.norm(x, axis=1)
    nonzero = magnitude > 1e-10
    unit = np.divide(x, magnitude[:, None], out=np.zeros_like(x), where=nonzero[:, None])
    valid = nonzero[1:] & nonzero[:-1]
    cos = np.clip((unit[1:] * unit[:-1]).sum(1)[valid], -1, 1)
    azimuth = np.arctan2(unit[nonzero, 1], unit[nonzero, 0])
    elevation = np.arcsin(np.clip(unit[nonzero, 2], -1, 1))
    hist, _, _ = np.histogram2d(azimuth, elevation, bins=(8, 4), range=((-np.pi, np.pi), (-np.pi/2, np.pi/2)))
    return dict(magnitude_mean=float(magnitude.mean()), magnitude_std=float(magnitude.std()),
        magnitude_max=float(magnitude.max()), zero_count=int((~nonzero).sum()),
        direction_histogram=hist.astype(int).tolist(), direction_distribution=(hist/max(1, hist.sum())).ravel().tolist(),
        successive_cosine_mean=float(cos.mean()) if len(cos) else None,
        reversal_count=int((cos < 0).sum()), valid_successive_pairs=len(cos),
        reversal_fraction=float((cos < 0).mean()) if len(cos) else None)


def episode_features(e, grid):
    actions = np.asarray(e['actions'], dtype=float)
    initial = np.asarray(e['initial_proprio'], dtype=float)
    require(actions.ndim == 2 and actions.shape[1] == 7 and len(actions) > 0 and
            initial.shape == (8,) and np.isfinite(actions).all() and np.isfinite(initial).all(), 'Invalid episode arrays')
    trans, rot = command_features(actions[:, :3]), command_features(actions[:, 3:6])
    # Raw LIBERO -1=open, +1=close; threshold matches the existing clip/invert transform.
    closed = actions[:, 6] >= .5
    event_indices = np.flatnonzero(closed[1:] != closed[:-1]) + 1
    events = [dict(phase=float(i/max(1, len(actions)-1)), command='close' if closed[i] else 'open') for i in event_indices]
    phase_indices = np.floor(np.linspace(0, len(actions)-1, grid)).astype(int)
    vector = [float(np.log1p(len(actions)))]
    names = ['log1p_length']
    for prefix, block in (('translation', trans), ('rotation', rot)):
        for k in ('magnitude_mean', 'magnitude_std', 'successive_cosine_mean', 'reversal_fraction'):
            vector.append(block[k] if block[k] is not None else 0.0)
            names.append(prefix+'_'+k)
        vector.append(block['zero_count']/len(actions)); names.append(prefix+'_zero_fraction')
        vector.extend(block['direction_distribution'])
        names.extend(f'{prefix}_direction_bin_{i}' for i in range(32))
    vector.extend([len(events), float(closed.mean())]); names.extend(['gripper_events', 'gripper_closed_fraction'])
    return dict(episode_id=e['episode_id'], task=e['task'], source='expert', length=len(actions),
        initial_proprio=initial.tolist(), initial_eef_pose_source_values=initial[:6].tolist(),
        final_eef_pose=None, eef_path_length=None, eef_net_displacement=None,
        physical_path_status='UNAVAILABLE: full observation/state sequence was not retained',
        success=None, success_status='UNAVAILABLE: terminal/reward are not independent success evidence',
        translation=trans, rotation=rot, gripper_events=events, gripper_closed_fraction=float(closed.mean()),
        gripper_phase_pattern=closed[phase_indices].astype(int).tolist(),
        descriptor=vector, descriptor_names=names, action_phase_32=align(actions, grid).tolist())


def rms_matrix(x):
    x = np.asarray(x, dtype=float)
    x = x.reshape(len(x), -1)
    # Explicit differences avoid cancellation when testing exact duplicate trajectories.
    return np.sqrt(np.mean((x[:, None, :] - x[None, :, :])**2, axis=-1))


def standardize(x, fit):
    mean, std = x[fit].mean(0), x[fit].std(0)
    # Constant candidate-pool axes have no meaningful scale and are excluded from distances.
    mask = std > 1e-8
    require(mask.any(), 'All metric dimensions constant in candidate pool')
    return (x[:, mask]-mean[mask])/std[mask], dict(mean=mean.tolist(), std=std.tolist(),
                                                used_dimensions=np.flatnonzero(mask).tolist())


def task_matrices(episodes, candidate_ids, grid):
    ids = [e['episode_id'] for e in episodes]
    fit = np.array([i for i, eid in enumerate(ids) if eid in candidate_ids])
    require(len(fit) >= 20, 'Need 20 non-reference candidates per task')
    descriptor, dscale = standardize(np.array([e['descriptor'] for e in episodes]), fit)
    initial, iscale = standardize(np.array([e['initial_proprio'] for e in episodes]), fit)
    aligned = np.array([e['action_phase_32'] for e in episodes])[:, :, :6]
    flat = aligned[fit].reshape(-1, 6)
    mean, std = flat.mean(0), flat.std(0)
    mask = std > 1e-8
    require(mask.any(), 'All action dimensions constant')
    action = (aligned[:, :, mask]-mean[mask])/std[mask]
    patterns = np.array([e['gripper_phase_pattern'] for e in episodes])
    matrices = dict(temporal_action=rms_matrix(action), descriptor=rms_matrix(descriptor),
                    initial_state=rms_matrix(initial), gripper_pattern=np.mean(patterns[:, None] != patterns[None, :], axis=-1))
    return dict(ids=ids, matrices=matrices, normalization=dict(descriptor=dscale, initial_state=iscale,
        temporal_action=dict(mean=mean.tolist(), std=std.tolist(), used_dimensions=np.flatnonzero(mask).tolist()),
        fit='20-shot candidate pool only; no fixed-reference episodes', temporal_grid=grid))


def average_pair_distance(matrix, indices):
    indices = np.asarray(indices, dtype=int)
    if len(indices) < 2:
        return None
    block = matrix[np.ix_(indices, indices)]
    return float(block[np.triu_indices(len(indices), 1)].mean())


def nearest(matrix, selected, reference):
    require(len(selected) and len(reference), 'Empty selected/reference set')
    require(not set(selected) & set(reference), 'Selected/reference leakage')
    distances = matrix[np.ix_(reference, selected)]
    return distances.min(1)


def random_order(ids, seed, task):
    # Analysis-only seeded permutations. No selection manifests or training data are written.
    return sorted(ids, key=lambda eid: (digest(['phase2-reference-excluded-v1', seed, task, eid]), eid))


def band(values, actual):
    x = np.asarray(values, dtype=float)
    require(np.isfinite(x).all(), 'Nonfinite random baseline')
    tied = np.isclose(x, actual, rtol=1e-10, atol=1e-12)
    return dict(mean=float(x.mean()), p05=float(np.quantile(x, .05)), p50=float(np.median(x)),
                p95=float(np.quantile(x, .95)), seed42_percentile=float(100*(((x < actual)&~tied).sum()+.5*tied.sum())/len(x)),
                degenerate=bool(np.all(tied)),
                interpretation='Empirical conditional selection spread, not a confidence interval or policy score')


def audit_metadata(metadata_dir, inventory):
    result = {}
    for suite in ('goal', 'spatial', 'long'):
        feature_path, info_path = [metadata_dir/f'{suite}_{name}.json' for name in ('features', 'dataset_info')]
        features, info = read(feature_path), read(info_path)
        steps = features['featuresDict']['features']['steps']['sequence']['feature']['featuresDict']['features']
        observation = steps['observation']['featuresDict']['features']
        if suite == 'goal':
            for path, name in ((feature_path, 'features.json'), (info_path, 'dataset_info.json')):
                require(sha(path) == inventory['source']['files'][name], 'Pinned Goal schema differs from verified RLDS source')
        result[suite] = dict(status='VERIFIED_SCHEMA', actual_episode_values='VERIFIED_SAVED_ARTIFACTS' if suite == 'goal' else 'UNAVAILABLE',
            schema_sha256=sha(feature_path), metadata_sha256=sha(info_path), builder=info['name'],
            declared_train_episodes=sum(int(n) for split in info['splits'] if split['name']=='train' for n in split['shardLengths']),
            step_fields=list(steps), observation_fields={k: {'description':v.get('description'),
                'shape':v.get('tensor', v.get('image', {})).get('shape'),
                'dtype':v.get('tensor', v.get('image', {})).get('dtype')} for k,v in observation.items()},
            eef_pose='state[:6], 6D pose per pinned schema; xyz plus axis-angle per local OFT regeneration code',
            proprio='8D state and 7D joint_state declared; Goal artifact retains initial state only',
            objects='UNAVAILABLE: no object/relative-position fields in serialized observation schema',
            gripper='state[-2:] 2D physical state; action[-1] command (-1 open / +1 close per OFT transform)',
            success='No success field. reward=1 on final demo step and terminal flags do not independently establish task success',
            identifiers='TFDS source IDs retained; episode_metadata.file_path exists in raw schema but is not retained in trajectory artifact',
            units_frame='UNAVAILABLE: pinned metadata does not specify coordinate frame or units; no metre/radian conversion asserted',
            convention_code_evidence={str(p.relative_to(ROOT)):sha(p) for p in (
                ROOT/'runtime/openvla-oft/experiments/robot/libero/regenerate_libero_dataset.py',
                ROOT/'runtime/openvla-oft/prismatic/vla/datasets/rlds/oxe/transforms.py')},
            physical_path_analysis='UNAVAILABLE: no full state sequence in local artifacts',
            source_url=f'https://huggingface.co/datasets/openvla/modified_libero_rlds/tree/{REVISION}/'+
                       {'goal':'libero_goal_no_noops','spatial':'libero_spatial_no_noops','long':'libero_10_no_noops'}[suite]+'/1.0.0')
    return result


def load_source(run_root):
    runs = []
    for p in sorted(run_root.iterdir()):
        if (p/'manifest.json').is_file():
            m = read(p/'manifest.json')
            if m.get('command') in ('prepare','dataset-analyze','plot'):
                runs.append((p,m))
    analyses = [p for p,m in runs if m['command']=='dataset-analyze' and (p/'inventory.json').is_file()]
    require(len(analyses)==1, 'Expected one explicit verified Goal source analysis; use a dedicated run root')
    base = analyses[0]
    inventory, trajectories, prior = [read(base/f'{n}.json') for n in ('inventory','trajectories','analysis')]
    validate_inventory(inventory, 'goal')
    require(not inventory.get('synthetic') and len(inventory['episodes'])==428, 'Expected real verified Goal 428 episodes')
    require(trajectories['inventory_sha256']==digest(inventory)==prior['inventory_sha256'], 'Source artifact identity mismatch')
    records={e['episode_id']:e for e in inventory['episodes']}
    trs=trajectories['episodes']
    require(len(trs)==len(records) and len({e['episode_id'] for e in trs})==len(trs), 'Trajectory duplicates/count mismatch')
    for e in trs:
        r=records[e['episode_id']]
        require(e['task']==r['task'] and len(e['actions'])==r['steps'] and e['source_payload_sha256']==r['sha256'], 'Episode identity mismatch')
        require(e['trajectory_sha256']==digest([e['actions'],e['initial_proprio']]), 'Trajectory checksum mismatch')
    selections={}; historical=set()
    for p,m in runs:
        require(read(p/'completion.json')['status']=='COMPLETE', 'Incomplete input run')
        if m['command']=='prepare':
            s=read(p/'selection.json')
            require(s['selection_seed']==42, 'Expected selection seed42')
            ids=validate_selection(s,inventory)
            require(s['shots'] not in selections, 'Ambiguous shot manifest')
            selections[s['shots']]=ids
    require(set(selections)==set(SHOTS)|{'Full'}, 'Missing selection manifests')
    previous=set()
    for k in (*SHOTS,'Full'):
        require(previous<=selections[k], 'Subsets are not nested'); previous=selections[k]
    prior_ids=[e['episode_id'] for e in prior['episodes']]
    require(len(prior_ids)==len(set(prior_ids)) and set(prior_ids)==set(records), 'Historical analysis ID mismatch')
    lookup={e['episode_id']:e for e in trs}
    for e in prior['episodes']:
        r=records[e['episode_id']]
        require(e['task']==r['task'] and e['length']==r['steps'] and e['initial_proprio']==lookup[e['episode_id']]['initial_proprio'], 'Historical row identity mismatch')
        if e['selected']: historical.add(e['episode_id'])
    require(len(historical)==50 and all(n==5 for n in Counter(records[k]['task'] for k in historical).values()), 'Historical 5-shot mismatch')
    require(len(historical & selections[5])==3, 'Historical/nested overlap changed')
    fixed=set(records)-selections[20]
    require(len(fixed)==228, 'Fixed reference pool changed')
    # Hash every file in the eight completed runs, including existing figures, to verify preservation.
    hashes={str(p):sha(p) for p in sorted(run_root.rglob('*')) if p.is_file()}
    return inventory,trs,selections,historical,fixed,hashes


def compute(inventory, trs, selections, historical, fixed, grid=32, seeds=64, max_task_episodes=64):
    require(grid==32, 'This version uses a fixed 32-point normalized-phase grid')
    require(16<=seeds<=256, 'Use 16..256 analysis seeds')
    candidates=set(e['episode_id'] for e in trs)-fixed
    require(candidates==selections[20], 'Reference/candidate split mismatch')
    episodes=[episode_features(e,grid) for e in sorted(trs,key=lambda e:e['episode_id'])]
    source_by_id={e['episode_id']:e for e in trs}
    task_results={}; pair_rows=[]; coverage=[]; historical_rows=[]
    for task in sorted({e['task'] for e in episodes}):
        es=[e for e in episodes if e['task']==task]
        require(len(es)<=max_task_episodes<=128, 'Pairwise budget exceeded; explicitly revise bounded strategy')
        block=task_matrices(es,candidates,grid)
        ids=block['ids']; index={eid:i for i,eid in enumerate(ids)}
        ref=[index[eid] for eid in ids if eid in fixed]
        pool=[eid for eid in ids if eid in candidates]
        random_indices=[[index[eid] for eid in random_order(pool,seed,task)] for seed in range(seeds)]
        matrices=block['matrices']
        for x in range(len(ids)):
            for y in range(x+1,len(ids)):
                pair_rows.append(dict(task=task,episode_a=ids[x],episode_b=ids[y],
                    **{k:float(m[x,y]) for k,m in matrices.items()},
                    gripper_event_count_difference=abs(len(es[x]['gripper_events'])-len(es[y]['gripper_events'])),
                    endpoint_distance=None,path_length_difference=None,physical_path_distance=None))
        for shot in SHOTS:
            selected=[index[eid] for eid in ids if eid in selections[shot]]
            for metric,matrix in matrices.items():
                distances=nearest(matrix,selected,ref)
                random_means=[float(nearest(matrix,order[:shot],ref).mean()) for order in random_indices]
                diversity=average_pair_distance(matrix,selected)
                random_diversities=[average_pair_distance(matrix,order[:shot]) for order in random_indices] if shot>1 else []
                coverage.append(dict(task=task,shots=shot,metric=metric,reference_episode_ids=[ids[j] for j in ref],
                    nearest_distances=distances.tolist(),nearest_mean=float(distances.mean()),
                    selected_diversity=diversity,random_coverage_means=random_means,
                    random_coverage=band(random_means,float(distances.mean())),
                    random_diversity=band(random_diversities,diversity) if shot>1 else None,
                    bias_status='DEGENERATE: all 20 candidate episodes selected' if shot==20 else 'CONDITIONAL: reference-excluded 20-episode pool'))
        hselected=[index[eid] for eid in ids if eid in historical]
        nselected=[index[eid] for eid in ids if eid in selections[5]]
        # Secondary historical comparison excludes historical IDs from the fixed pool too.
        href=[j for j in ref if ids[j] not in historical]
        for metric,matrix in matrices.items():
            historical_rows.append(dict(task=task,metric=metric,reference_episode_ids=[ids[j] for j in href],
                historical_distance=float(nearest(matrix,hselected,href).mean()),
                nested5_distance=float(nearest(matrix,nselected,href).mean()),
                shared_selected_ids=sorted(set(ids[j] for j in hselected)&set(ids[j] for j in nselected))))
        raw=[np.asarray(source_by_id[eid]['actions']) for eid in ids]
        all_raw=np.concatenate(raw)
        task_results[task]=dict(episode_ids=ids,episodes=len(es),transitions=len(all_raw),
            normalization=block['normalization'],pairwise_matrices={k:m.tolist() for k,m in matrices.items()},
            pairwise_means={k:average_pair_distance(m,list(range(len(ids)))) for k,m in matrices.items()},
            action_transition_weighted=dict(mean=all_raw.mean(0).tolist(),std=all_raw.std(0).tolist()),
            action_episode_weighted=dict(mean=np.mean([x.mean(0) for x in raw],axis=0).tolist(),
                std=np.sqrt(np.maximum(np.mean([(x*x).mean(0) for x in raw],axis=0)-np.mean([x.mean(0) for x in raw],axis=0)**2,0)).tolist()),
            initial_proprio_std=np.std([e['initial_proprio'] for e in es],axis=0).tolist(),
            length=dict(min=min(e['length'] for e in es),max=max(e['length'] for e in es),mean=float(np.mean([e['length'] for e in es]))),
            gripper_event_counts=dict(Counter(len(e['gripper_events']) for e in es)),
            physical_path_status='UNAVAILABLE')
    stratified=[]
    for shot in SHOTS:
        for metric in METRICS:
            blocks=[r for r in coverage if r['shots']==shot and r['metric']==metric]
            actual=float(np.mean([r['nearest_mean'] for r in blocks]))
            random_means=np.mean([r['random_coverage_means'] for r in blocks],axis=0)
            stratified.append(dict(shots=shot,metric=metric,task_weighted_mean=actual,random_coverage=band(random_means,actual)))
    return dict(schema=2,kind='expert_variation',synthetic=False,suite='goal',episodes=episodes,tasks=task_results,
        coverage=coverage,historical_comparison=historical_rows,stratified_coverage=stratified,
        fixed_reference_ids=sorted(fixed),candidate_ids=sorted(candidates),analysis_seeds=list(range(seeds)),
        pairwise_budget=dict(grid_points=grid,max_episodes_per_task=max_task_episodes,actual_pairs=len(pair_rows),
            algorithm='All within-task pairs; RMS of phase-aligned commands. No cross-task pairs, no DTW, no physical reconstruction.'),
        random_baseline='Conditional random nested permutations of the 20 candidates per task outside the fixed 228 reference set; 20-shot is degenerate.',
        historical_identity='Source episode IDs and payload-linked prior analysis aligned; original historical selection file unavailable locally.'),pair_rows


def write_csv(path, rows):
    require(bool(rows), 'No CSV rows')
    with path.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def report(result):
    lines=['# Phase 2 — Goal expert variation', '',
        '## VERIFIED', '',
        f"Real Goal artifacts: {len(result['episodes'])} episodes, {sum(e['length'] for e in result['episodes'])} transitions, 10 tasks. No synthetic research results, model inference or training.",
        'Pinned Goal metadata matches the original audited source hashes. Spatial/Long are schema-only audits; their episode values were not locally available.',
        'All temporal distances compare action-command sequences within the same task. They are NOT physical EEF path distances.',
        'The prior mean/std descriptor coverage and all eight completed runs remain unchanged.', '',
        'Temporal action RMS uses 32 normalized-phase samples of the first six command axes; each axis is standardized using the reference-excluded candidate pool for its task. Duration is retained separately. Gripper distance is binary command disagreement over 32 phase bins.',
        'Direction statistics use an 8 azimuth × 4 elevation histogram. Zero commands are excluded; successive cosines exclude pairs containing zero vectors. Reversal means cosine < 0.', '',
        '## Fixed-reference coverage and conditional selection bias', '',
        'All 1/2/5/10/20-shot comparisons use the same 228 reference IDs. Random selections draw only from the remaining 200 episodes (20/task), using '+str(len(result['analysis_seeds']))+' deterministic nested permutations. No training manifests are created or replaced.',
        'This restriction prevents reference leakage but conditions the baseline on seed42’s 20-shot pool. It cannot establish whether that 20-shot pool is unusually narrow relative to all 428 episodes. At 20 shots every random draw is identical; no seed-bias claim is possible.',
        'Reference bands are empirical 5th–95th percentiles of conditional random selections, not confidence intervals. Suite rows below are equal-task averages; per-task rows and every nearest-neighbor distance are saved in variation.json.', '',
        '| Shot | Temporal-command coverage ↓ | Random p05–p95 | Seed42 percentile |', '|---|---:|---:|---:|']
    for row in result['stratified_coverage']:
        if row['metric']=='temporal_action':
            b=row['random_coverage'];pct='N/A (degenerate)' if b['degenerate'] else f"{b['seed42_percentile']:.1f}"
            lines.append(f"| {row['shots']} | {row['task_weighted_mean']:.4f} | {b['p05']:.4f}–{b['p95']:.4f} | {pct} |")
    lines+=['', '### Per-task 5-shot coverage and event-pattern variation', '',
        '| Task | Modal gripper event count / episodes | Temporal coverage ↓ | Random percentile | Selected temporal diversity |',
        '|---|---:|---:|---:|---:|']
    modal_total=0
    for task,t in result['tasks'].items():
        events,count=max(t['gripper_event_counts'].items(),key=lambda kv:kv[1]);modal_total+=count
        cov=next(r for r in result['coverage'] if r['task']==task and r['shots']==5 and r['metric']=='temporal_action')
        lines.append(f"| {task} | {events} events / {count} of {t['episodes']} | {cov['nearest_mean']:.4f} | {cov['random_coverage']['seed42_percentile']:.1f} | {cov['selected_diversity']:.4f} |")
    lines+=['', f"Across tasks, {modal_total}/{len(result['episodes'])} episodes share their task's most frequent gripper-event count. Event count is a coarse action pattern, not a full trajectory/strategy label.",
        'Initial-state distance is standardized Euclidean distance in the eight stored proprio components. Its orientation components are not converted into an SO(3) geodesic, so it is a coordinate-space proxy, not a physical pose distance. Constant candidate-pool dimensions are excluded from metrics.',
        'Phase alignment removes duration differences and can miss brief motions; raw lengths, reversal statistics and gripper event times are saved separately. No physical seconds/control frequency are inferred.']
    lines+=['', 'Higher coverage distance means less representation of the held-out command patterns. Smaller distances from larger nested sets are expected mechanically and are not policy improvements.', '',
        'Historical vs nested 5-shot shares only 3 selected source IDs. Its secondary reference set is the fixed 228 pool minus historical selected IDs, identically used for both selections. This does not replace the original reference pool for the main shot comparison.', '',
        '## UNAVAILABLE', '',
        '- Full EEF/proprio state sequence, measured 3D path overlays/distances, endpoint/final pose, path length and net displacement. Original RLDS schema declares state[8], but prior extraction saved only initial_proprio.',
        '- Object/relative positions: not present in the pinned RLDS observation schema. Full simulator state is not retained here.',
        '- Verified physical coordinate frame, position units and timestamp/control interval: metadata does not declare them. Initial-state scatter uses source values without claiming metres or a world frame.',
        '- Independent episode success labels: terminal/reward conventions for demonstrations are not a separate task-success measurement.',
        '- Spatial/Long trajectory-value analysis and full-source random-selection bias under this fixed reference design.',
        '- Original historical selection manifest for independent file-hash revalidation; historical membership is sourced from the preserved, ID-aligned prior analysis.', '',
        '## HYPOTHESIS / interpretation boundary', '',
        'Nonzero within-task action-sequence distances and differences in command directions, event timing and initial proprio support variation in executed command patterns. They do not establish distinct physical routes or distinct task strategies. Nearest-neighbor distances have no externally calibrated threshold defining a new behavior mode.',
        'Accordingly, the current results can distinguish more/less similar command sequences, but cannot yet decide whether shots cover genuinely distinct physical trajectory patterns rather than variants of similar motions. Full measured state extraction is required for that claim.',
        'No claim is made that expert variation predicts SFT/RIPT or RLVR robustness. That requires later matched closed-loop evaluation through the existing evaluator.', '',
        '## Artifacts', '',
        '- `state_audit.json`: pinned schema fields, source links, units and availability.',
        '- `episode_features.csv`: scalar features, initial state and descriptor vectors.',
        '- `task_variation.json`, `shot_coverage.csv`, `pairwise_comparisons.csv`, `variation.json`.',
        '- `figures/index.md`: task-specific PNG/SVG figures; physical plots explicitly unavailable.',
        '- `provenance.json`: exact commands, input hashes and CPU environment; `completion.json`: preservation audit.',
        '- Future expert/SFT/D0/D1 contract: `experiments/shot_scaling/trajectory.schema.json`.', '']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--metadata-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--random-seeds',type=int,default=64)
    a=p.parse_args()
    require(not a.output.exists(),'Output already exists; choose a new version directory')
    inventory,trs,selections,historical,fixed,hashes=load_source(a.run_root)
    audit=audit_metadata(a.metadata_dir,inventory)
    result,pairs=compute(inventory,trs,selections,historical,fixed,seeds=a.random_seeds)
    a.output.mkdir(parents=True,exist_ok=False)
    write(a.output/'state_audit.json',audit)
    write(a.output/'variation.json',result)
    write(a.output/'task_variation.json',result['tasks'])
    scalar_rows=[]
    for e in result['episodes']:
        row={k:e[k] for k in ('episode_id','task','length','eef_path_length','eef_net_displacement','success')}
        row.update({f'initial_proprio_{i}':v for i,v in enumerate(e['initial_proprio'])})
        for family in ('translation','rotation'):
            row.update({family+'_'+k:e[family][k] for k in ('magnitude_mean','magnitude_std','magnitude_max','zero_count','successive_cosine_mean','reversal_count','valid_successive_pairs','reversal_fraction')})
        row.update(gripper_events=len(e['gripper_events']),gripper_closed_fraction=e['gripper_closed_fraction'],
                   descriptor_json=json.dumps(e['descriptor']),initial_eef_pose_json=json.dumps(e['initial_eef_pose_source_values']),final_eef_pose=None)
        scalar_rows.append(row)
    write_csv(a.output/'episode_features.csv',scalar_rows)
    write_csv(a.output/'pairwise_comparisons.csv',pairs)
    coverage_rows=[]
    for r in result['coverage']:
        row={k:r[k] for k in ('task','shots','metric','nearest_mean','selected_diversity','bias_status')}
        row.update(reference_episodes=len(r['reference_episode_ids']),**{f'random_{k}':r['random_coverage'][k] for k in ('mean','p05','p50','p95','seed42_percentile')})
        coverage_rows.append(row)
    write_csv(a.output/'shot_coverage.csv',coverage_rows)
    (a.output/'report.md').write_text(report(result))
    source_files=[Path(__file__),Path(__file__).with_name('variation_plot.py'),Path(__file__).with_name('trajectory.schema.json')]
    write(a.output/'provenance.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable,'-m','experiments.shot_scaling.variation',*sys.argv[1:]],
        input_hashes=hashes,metadata_hashes={str(p):sha(p) for p in a.metadata_dir.glob('*.json')},
        code_hashes={str(p):sha(p) for p in source_files},runtime=runtime()))
    require(all(sha(Path(p))==h for p,h in hashes.items()),'An original input changed during analysis')
    write(a.output/'completion.json',dict(status='COMPLETE',scope='Real Goal command-trajectory analysis; physical path analysis unavailable',
        original_files_verified_unchanged=len(hashes),episodes=len(result['episodes']),within_task_pairs=len(pairs)))
    print(a.output.resolve())


if __name__=='__main__':
    main()
