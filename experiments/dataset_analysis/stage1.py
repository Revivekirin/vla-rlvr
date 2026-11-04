"""Stage 1 only: reuse audited numerical sequences, never select shots or run models."""
import argparse
import csv
from pathlib import Path
import numpy as np
from experiments.shot_scaling.core import ROOT, REVISION, SUITES, read, require, digest, task_names
from .analysis import analyze


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


def save(path, value):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def build(paths, output, conventions=(), refresh=False):
    require(not conventions, 'Stage 1 uses documented semantics; no REVIEWED override is needed')
    require(not output.exists() or refresh, 'Output exists; use --refresh to replace derived analysis in place')
    require(not output.is_symlink(), 'Output symlink rejected')
    require(not any(output.resolve().is_relative_to(ROOT/'outputs'/x) for x in
                    ('fewshot_sft', 'rlvr_analysis', 'research_dashboard')), 'Future-stage output forbidden')
    # Load inputs before replacing derived files, including when inputs live in this output.
    audits = [read(p) for p in paths]
    require(len(audits) == 3 and {a['suite'] for a in audits} == set(SUITES), 'Three distinct suites required')
    from .semantics import documented_mapping, sanity
    suites = {}; checks = {}; features = []; variation=[]
    for a in audits:
        count = validate(a)
        numerical=sanity(a['episodes'])
        # Source interpretation is distinct from exact producer / bitwise verification.
        interpreted=dict(a, physical_convention=documented_mapping())
        d = analyze(interpreted, seeds=(), documented=True)
        for k in ('selections', 'historical_selection', 'historical_ids', 'historical_note'): d.pop(k)
        suites[a['suite']] = d
        checks[a['suite']] = dict(episodes=count[0], transitions=count[1], expected_match=count == EXPECTED[a['suite']],
                                  saved_sequence_checks='PASS', numerical_sanity=numerical)
        for e in d['episodes']:
            features.append(dict(suite=a['suite'], task=e['task'], split=e['split'], episode_id=e['episode_id'],
                steps=e['steps'], gripper_events=len(e['gripper_events']),
                translation_command_mean=e['translation']['magnitude_mean'],
                rotation_command_mean=e['rotation']['magnitude_mean'],
                physical_path_m=e['physical']['path_length'], net_displacement_m=e['physical']['net_displacement']))
        for split,tasks in d['tasks'].items():
            for task,t in tasks.items():
                es=[e for e in d['episodes'] if e['split']==split and e['task']==task]
                lengths=np.array([e['physical']['path_length'] for e in es])
                initial=np.array([e['physical']['initial_xyz'] for e in es])
                rms=[r['aligned_xyz_rms_m'] for r in d['physical_pairs'] if r['split']==split and r['task']==task]
                stats=dict(path_length_mean_m=float(lengths.mean()), path_length_std_m=float(lengths.std()),
                    initial_xyz_std_m=initial.std(0).tolist(), pair_count=len(rms),
                    pairwise_rms_mean_m=float(np.mean(rms)) if rms else None,
                    pairwise_rms_median_m=float(np.median(rms)) if rms else None)
                t['trajectory_variation']=stats
                variation.append(dict(suite=a['suite'],split=split,task=task,episodes=len(es),**stats))
    status='PASS' if all(c['expected_match'] and c['numerical_sanity']['status']=='PASS' for c in checks.values()) else 'PARTIAL'
    output.mkdir(parents=True,exist_ok=refresh)
    for a,p in zip(audits,paths):
        destination=output/'trajectory_data'/f"{a['suite']}.json"
        if destination.resolve()!=p.resolve(): save(destination,a)
    bundle = dict(stage=1, suites=suites, source_scope='Full extracted RLDS expert sequences; this build does not decode TFRecords again')
    save(output/'analysis.json', bundle)
    def csv_file(name,rows):
        with (output/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    csv_file('episode_features.csv',features)
    csv_file('task_variation.csv',variation)
    inventory=[dict(suite=s,split=split,task=task,episodes=t['episodes'],transitions=t['transitions'])
               for s,d in suites.items() for split,tasks in d['tasks'].items() for task,t in tasks.items()]
    csv_file('inventory.csv',inventory)
    save(output/'manifest.json', dict(stage=1, revision=REVISION, input_paths=[str(p.resolve()) for p in paths],
          selection_generated=False, source_scope=bundle['source_scope'], coordinate_semantics=documented_mapping(),
          pairwise_method='All pairs within each task/split; 32 linearly interpolated normalized-index samples, world-frame XYZ RMS; no centering, DTW or action integration'))
    plot(output/'analysis.json', output/'figures', refresh=refresh)
    save(output/'validation.json', dict(status=status, completion_scope='Descriptive Full expert trajectory analysis',
        suites=checks, physical_status='SOURCE_DOCUMENTED',bitwise_verified=False,
        exact_producer_provenance='UNRESOLVED_NONBLOCKING', matched_hdf5_comparison='NOT_REQUIRED_FOR_THIS_SCOPE',
        raw_source_redecoded_in_this_build=False,
        blockers=[] if status=='PASS' else ['Inspect numerical sanity flags or unexpected dataset counts'],
        tests='Finite values, shapes/alignment, task and episode identities, expected counts, range and temporal screening on extracted real data'))
    lines = [f'# Stage 1 — {status}', '', bundle['source_scope']+'.', '',
        'EEF XYZ, axis-angle orientation, gripper state and pre-action timing follow documented source code. Exact producer provenance and bitwise equality are not verified; these are not completion gates for descriptive analysis.', '',
        '| Suite | Episodes | Transitions | XYZ min (m) | XYZ max (m) |', '|---|---:|---:|---|---|']
    lines += [f"| {s} | {c['episodes']} | {c['transitions']} | {c['numerical_sanity']['xyz_min']} | {c['numerical_sanity']['xyz_max']} |" for s,c in checks.items()]
    lines += ['', '[Four consolidated figures](figures/index.md). Figure 2 shows all episodes of the alphabetically first task in each suite; all tasks are analyzed in task_variation.csv and Figure 3.', '',
        'Task variation uses all within-task episode pairs, resampled to 32 normalized-index points. Distances include initial-position differences, and do not measure velocity or policy quality. Path length sums adjacent recorded XYZ displacements and ends at the last pre-action observation.', '',
        'Figure 4 includes initial XYZ distributions for every episode. Action commands are never integrated into position. No shot selections, HDF5 regeneration, model evaluation or training are performed.', '',
        'See manifest.json for source interpretation and limitations; validation.json for numerical screening.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def plot(source, output, refresh=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    b = read(source); output.mkdir(parents=True, exist_ok=refresh)
    suites = list(SUITES); ds = [b['suites'][s] for s in suites]
    index = []; files = {}; representatives={}
    def savefig(fig,name,caption):
        fig.suptitle(caption,fontsize=13)
        fig.tight_layout(rect=(0,0,1,.96))
        fig.savefig(output/(name+'.png'),dpi=150);plt.close(fig)
        files[name+'.png']=dict(caption=caption)
        index.append(f'![{caption}]({name}.png)\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].bar(suites,[len(d['episodes']) for d in ds]);axes[0].set_ylabel('Expert episodes')
    axes[1].boxplot([[e['steps'] for e in d['episodes']] for d in ds],tick_labels=suites)
    axes[1].set_ylabel('Recorded steps per episode')
    savefig(fig,'figure_1_counts_lengths','Full expert datasets: counts and episode lengths')
    fig=plt.figure(figsize=(15,13))
    for col,(suite,d) in enumerate(zip(suites,ds)):
        task=sorted(d['tasks']['train'])[0]
        es=sorted([e for e in d['episodes'] if e['task']==task and e['split']=='train'],key=lambda e:e['episode_id'])
        representatives[suite]=dict(task=task,episode_ids=[e['episode_id'] for e in es],rule='Alphabetically first task; all its train episodes')
        xyz=np.concatenate([np.asarray(e['xyz']) for e in es]);low=xyz.min(0)-.015;high=xyz.max(0)+.015
        a3=fig.add_subplot(3,3,col+1,projection='3d')
        xy=fig.add_subplot(3,3,col+4);xz=fig.add_subplot(3,3,col+7)
        for i,e in enumerate(es):
            x=np.asarray(e['xyz']);color=plt.colormaps['viridis'](i/max(1,len(es)-1))
            a3.plot(*x.T,color=color,alpha=.45,lw=.7)
            for ax,j in ((xy,1),(xz,2)):
                ax.plot(x[:,0],x[:,j],color=color,alpha=.45,lw=.7)
                ax.scatter(x[0,0],x[0,j],c=[color],s=8,marker='o')
                ax.scatter(x[-1,0],x[-1,j],c=[color],s=8,marker='x')
        a3.set(title=f'{suite}: task 1, n={len(es)}',xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',
               xlim=(low[0],high[0]),ylim=(low[1],high[1]),zlim=(low[2],high[2]))
        a3.set_box_aspect(high-low)
        from matplotlib.ticker import MaxNLocator
        for axis in (a3.xaxis,a3.yaxis,a3.zaxis): axis.set_major_locator(MaxNLocator(4))
        for ax,j in ((xy,1),(xz,2)):
            ax.set(xlabel='X (m)',ylabel=('Y' if j==1 else 'Z')+' (m)',xlim=(low[0],high[0]),ylim=(low[j],high[j]))
            ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
    savefig(fig,'figure_2_eef','Recorded EEF paths: 3D / XY / XZ; circle = initial, cross = last recorded state')
    fig,axes=plt.subplots(3,3,figsize=(15,11),sharey='row')
    for col,(suite,d) in enumerate(zip(suites,ds)):
        tasks=sorted(d['tasks']['train'])
        lengths=[[e['physical']['path_length'] for e in d['episodes'] if e['task']==t and e['split']=='train'] for t in tasks]
        rms=[[r['aligned_xyz_rms_m'] for r in d['physical_pairs'] if r['task']==t and r['split']=='train'] for t in tasks]
        action=[[e['translation']['magnitude_mean'] for e in d['episodes'] if e['task']==t and e['split']=='train'] for t in tasks]
        for row,(values,label) in enumerate(((lengths,'Recorded path length (m)'),(rms,'Pairwise path RMS (m)'),(action,'Mean translation command norm'))):
            axes[row,col].boxplot(values,showfliers=False)
            axes[row,col].set(xlabel='Task index (1–10, alphabetical)',ylabel=label)
        axes[0,col].set_title(suite)
    savefig(fig,'figure_3_expert_variation','All tasks: physical trajectory and action variation (boxplots hide outlier markers only)')
    fig,axes=plt.subplots(3,3,figsize=(15,11),sharex='row',sharey='row')
    for col,(suite,d) in enumerate(zip(suites,ds)):
        tasks=sorted(d['tasks']['train'])
        initial=[]
        for i,t in enumerate(tasks):
            x=np.array([e['physical']['initial_xyz'] for e in d['episodes'] if e['task']==t and e['split']=='train'])
            initial.extend(x);color=plt.colormaps['tab10'](i)
            axes[0,col].scatter(x[:,0],x[:,1],color=color,s=14,alpha=.55,label=str(i+1))
            axes[1,col].scatter(x[:,0],x[:,2],color=color,s=14,alpha=.55)
        for row,j in ((0,1),(1,2)):
            axes[row,col].set(xlabel='Initial X (m)',ylabel='Initial '+('Y' if j==1 else 'Z')+' (m)')
            axes[row,col].set_aspect('equal',adjustable='box');axes[row,col].grid(alpha=.2)
        axes[0,col].set_title(suite)
        axes[2,col].boxplot(np.array(initial),tick_labels=['X','Y','Z'])
        axes[2,col].set_ylabel('Initial position (m)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,title='Task index within each suite',loc='lower center',ncol=10)
    fig.subplots_adjust(bottom=.1)
    # Keep room for the task legend within the common save layout.
    fig.tight_layout(rect=(0,.07,1,.94))
    fig.suptitle('All expert initial EEF positions: XY / XZ / marginal distributions',fontsize=13)
    name='figure_4_initial_positions.png';fig.savefig(output/name,dpi=150);plt.close(fig)
    files[name]=dict(caption='Initial positions for all tasks and episodes')
    index.append(f'![Initial positions for all tasks and episodes]({name})\n')
    for old in ('figure_3_action_variation.png','figure_4_initial_action.png'):
        if (output/old).is_file(): (output/old).unlink()
    save(output/'figure_manifest.json',dict(input=str(source),files=files,representative_tasks=representatives,
        task_index={s:sorted(d['tasks']['train']) for s,d in zip(suites,ds)},
        coordinate_semantics='SOURCE_DOCUMENTED; exact producer provenance and bitwise equivalence not verified'))
    (output/'index.md').write_text('# Stage 1 figures\n\n'+'\n'.join(index)+'\nTask names and representative episode IDs: figure_manifest.json. Units follow documented source conventions.\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,nargs=3,required=True)
    p.add_argument('--convention',type=Path,nargs='*',default=[])
    p.add_argument('--output',type=Path,default=ROOT/'outputs/dataset_analysis')
    p.add_argument('--refresh',action='store_true',help='Replace derived outputs in place, without backups or source decoding')
    a=p.parse_args(); build(a.audit,a.output,a.convention,a.refresh)


if __name__=='__main__': main()
