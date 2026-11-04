"""Stage 1 only: reuse audited numerical sequences, never select shots or run models."""
import argparse
import csv
from pathlib import Path
import numpy as np
from experiments.shot_scaling.core import ROOT, REVISION, SUITES, read, write, require, sha, digest, task_names
from .analysis import analyze
from .audit import convention

EXPECTED = {'goal': (428, 52042), 'spatial': (432, 52970), 'long': (379, 101469)}


def validate(a):
    require(a['source']['revision'] == REVISION, 'Wrong source revision label')
    require(a['source']['repo'] == 'openvla/modified_libero_rlds', 'Wrong source repository')
    require(a['dataset_name'] == SUITES[a['suite']]+'_no_noops', 'Wrong dataset name')
    require(digest(a['source']['files']) == a['source']['source_files_sha256'], 'Source manifest digest mismatch')
    require({e['task'] for e in a['episodes']} == set(task_names(a['suite'])), 'Missing tasks')
    for split, count in a['declared_splits'].items():
        require(sum(e['split'] == split for e in a['episodes']) == count, 'Split count mismatch')
    for e in a['episodes']:
        require(e['split'] in a['declared_splits'], 'Unknown split')
        for field, width in [('observation_state', 8), ('joint_state', 7)]:
            x = e[field]
            require(x is None or (np.shape(x) == (e['steps'], width) and np.isfinite(x).all()), 'State alignment/finite check failed')
    return (len(a['episodes']), sum(e['steps'] for e in a['episodes']))


def build(paths, output, conventions=()):
    require(not output.exists(), 'Output exists; preserve it or complete approved cleanup first')
    require(not any(output.resolve().is_relative_to(ROOT/'outputs'/x) for x in
                    ('fewshot_sft', 'rlvr_analysis', 'research_dashboard')), 'Future-stage output forbidden')
    audits = [read(p) for p in paths]
    require(len(audits) == 3 and {a['suite'] for a in audits} == set(SUITES), 'Three distinct suites required')
    reviewed = {read(p)['source_files_sha256']: p for p in conventions}
    suites = {}; checks = {}; features = []
    for a in audits:
        count = validate(a)
        # Never trust a manually edited REVIEWED flag in a cached audit.
        a['physical_convention'] = convention(reviewed.get(a['source']['source_files_sha256']), a['source']['source_files_sha256'])
        d = analyze(a, seeds=())
        for k in ('selections', 'historical_selection', 'historical_ids', 'historical_note'): d.pop(k)
        suites[a['suite']] = d
        checks[a['suite']] = dict(episodes=count[0], transitions=count[1], expected_match=count == EXPECTED[a['suite']],
                                  raw_source_revalidated=False, revision_authenticated=False)
        for e in d['episodes']:
            features.append(dict(suite=a['suite'], task=e['task'], split=e['split'], episode_id=e['episode_id'],
                steps=e['steps'], gripper_events=len(e['gripper_events']),
                translation_command_mean=e['translation']['magnitude_mean'],
                rotation_command_mean=e['rotation']['magnitude_mean'],
                physical_path_m=e['physical']['path_length'] if e['physical'] else None))
    output.mkdir(parents=True)
    # Preserve every raw state/action sequence, independently of physical interpretation.
    for a in audits: write(output/'trajectory_data'/f"{a['suite']}.json", a)
    bundle = dict(stage=1, suites=suites, source_scope='Saved real-data audits; original TFRecords not redecoded')
    write(output/'analysis.json', bundle)
    with (output/'episode_features.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(features[0])); writer.writeheader(); writer.writerows(features)
    inventory = []
    for suite, d in suites.items():
        for split, tasks in d['tasks'].items():
            for task, t in tasks.items():
                inventory.append(dict(suite=suite, split=split, task=task, episodes=t['episodes'], transitions=t['transitions']))
    with (output/'inventory.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(inventory[0])); writer.writeheader(); writer.writerows(inventory)
    write(output/'manifest.json', dict(stage=1, revision=REVISION, input_sha256={str(p.resolve()):sha(p) for p in paths},
          code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}, selection_generated=False,
          source_scope=bundle['source_scope']))
    write(output/'validation.json', dict(status='PARTIAL', suites=checks, physical_status={s:d['physical_convention']['status'] for s,d in suites.items()},
          blockers=['Fresh original RLDS count/hash/revision authentication required on server',
                    'Physical verification requires authenticated producer linkage and matched HDF5/simulator evidence'],
          tests='CPU checks on saved real numerical data; not a fresh RLDS extraction or numerical HDF5 comparison'))
    plot(output/'analysis.json', output/'figures')
    lines = ['# Stage 1 — PARTIAL', '', bundle['source_scope']+'.', '',
        'Original source counts/revision are not independently revalidated by this cached build. Physical EEF remains blocked without source-bound reviewed evidence.', '',
        '| Suite | Episodes | Transitions |', '|---|---:|---:|']
    lines += [f"| {s} | {c['episodes']} | {c['transitions']} |" for s,c in checks.items()]
    lines += ['', 'All state/action sequences are retained in trajectory_data. Raw action statistics, lengths and threshold-crossing gripper events are descriptive; commands are not integrated into physical positions.', '',
              '[Figure index](figures/index.md). No shot selections or model jobs are generated. Figure 2 is omitted when EEF verification is unavailable. Figure 4 uses raw state components with unknown physical units.', '',
              'See validation.json for blockers and manifest.json for reproducibility inputs.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def plot(source, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    b = read(source); output.mkdir(parents=True, exist_ok=False)
    suites = [s for s in SUITES if s in b['suites']]; ds = [b['suites'][s] for s in suites]
    index = []; files = {}
    def save(fig, name, caption):
        fig.suptitle(caption, fontsize=11); fig.tight_layout(rect=(0,0,1,.94))
        fig.savefig(output/(name+'.png'), dpi=160); plt.close(fig)
        files[name+'.png'] = sha(output/(name+'.png')); index.append(f'![{caption}]({name}.png)\n')
    fig, ax = plt.subplots(1,2,figsize=(11,4))
    ax[0].bar(suites,[len(d['episodes']) for d in ds]); ax[0].set_ylabel('Expert episodes')
    ax[1].boxplot([[e['steps'] for e in d['episodes']] for d in ds],tick_labels=suites)
    ax[1].set_ylabel('Serialized steps per episode')
    save(fig,'figure_1_counts_lengths','Saved original-dataset audit counts and episode lengths')
    valid = [(s,d) for s,d in zip(suites,ds) if any(e['xyz'] is not None for e in d['episodes'])]
    if valid:
        fig = plt.figure(figsize=(5*len(valid),4))
        for i,(s,d) in enumerate(valid,1):
            ax = fig.add_subplot(1,len(valid),i,projection='3d'); task=sorted(d['tasks']['train'])[0]
            for e in sorted(d['episodes'],key=lambda e:e['episode_id']):
                if e['task']==task and e['xyz'] is not None:
                    x=np.array(e['xyz']); ax.plot(*x.T,alpha=.3)
            ax.set(title=s+' / task 0',xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)')
        save(fig,'figure_2_eef','Measured paths: alphabetically first task per suite; all expert episodes')
    fig, axes = plt.subplots(1,len(ds),figsize=(13,4),squeeze=False)
    for ax,s,d in zip(axes[0],suites,ds):
        tasks=sorted(d['tasks']['train'])
        ax.boxplot([[e['translation']['magnitude_mean'] for e in d['episodes'] if e['task']==t and e['split']=='train'] for t in tasks])
        ax.set(title=s,xlabel='Task index (alphabetical)',ylabel='Episode mean raw translation command norm')
    save(fig,'figure_3_action_variation','Within-task action variation; raw controller units, not physical path length')
    fig, axes = plt.subplots(2,len(ds),figsize=(13,7),squeeze=False,sharey="row")
    for col,(s,d) in enumerate(zip(suites,ds)):
        initial=[e['initial_proprio'] for e in d['episodes'] if e['initial_proprio'] is not None]
        if initial:
            axes[0,col].boxplot(np.array(initial),tick_labels=[str(i) for i in range(8)])
        axes[0,col].set(title=s,xlabel='Raw initial state component',ylabel='Stored value (units unverified)')
        axes[1,col].boxplot([ [e['translation']['magnitude_mean'] for e in d['episodes']], [e['rotation']['magnitude_mean'] for e in d['episodes']] ],tick_labels=['translation','rotation'])
        axes[1,col].set(ylabel='Episode mean raw command norm')
    save(fig,'figure_4_initial_action','Across-suite initial-state and expert command variation; physical trajectories unverified')
    write(output/'figure_manifest.json',dict(input_sha256=sha(source),files=files,
          task_index={s:sorted(d['tasks']['train']) for s,d in zip(suites,ds)},
          omitted=[] if valid else ['Figure 2: physical EEF BLOCKED; no placeholder generated']))
    (output/'index.md').write_text('# Stage 1 figures\n\n'+'\n'.join(index))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,nargs=3,required=True)
    p.add_argument('--convention',type=Path,nargs='*',default=[])
    p.add_argument('--output',type=Path,default=ROOT/'outputs/dataset_analysis')
    a=p.parse_args(); build(a.audit,a.output,a.convention)


if __name__=='__main__': main()
