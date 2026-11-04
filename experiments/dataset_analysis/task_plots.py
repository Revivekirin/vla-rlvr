"""Explicitly requested task-level supplements from saved Full expert analysis."""
import argparse
import csv
import json
from pathlib import Path
import textwrap
import numpy as np
from experiments.shot_scaling.core import read, require, SUITES
from .analysis import align_path


def variation(paths):
    """Compare world paths and paths translated to their own initial position."""
    aligned=np.stack([align_path(x) for x in paths])
    initial=aligned[:,0,:]
    relative=aligned-initial[:,None,:]
    world_pairs=[]; relative_pairs=[]; initial_pairs=[]
    for i in range(len(paths)):
        for j in range(i):
            world_pairs.append(float(np.sqrt(np.mean(np.sum((aligned[i]-aligned[j])**2,axis=1)))))
            relative_pairs.append(float(np.sqrt(np.mean(np.sum((relative[i]-relative[j])**2,axis=1)))))
            initial_pairs.append(float(np.linalg.norm(initial[i]-initial[j])))
    def spread(x):
        return np.sqrt(np.mean(np.sum((x-x.mean(0))**2,axis=-1),axis=0))
    return dict(initial=initial,world_spread=spread(aligned),relative_spread=spread(relative),
                world_pairs=world_pairs,relative_pairs=relative_pairs,initial_pairs=initial_pairs)


def generate(source, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    bundle=read(source)
    require(bundle.get('stage')==1,'Stage 1 analysis.json required')
    output.mkdir(parents=True,exist_ok=True)
    rows=[]; mapping=[]
    for suite in SUITES:
        d=bundle['suites'][suite]
        require(d['physical_convention']['status']=='SOURCE_DOCUMENTED','Documented XYZ analysis required')
        for number,task in enumerate(sorted(d['tasks']['train']),1):
            episodes=sorted([e for e in d['episodes'] if e['split']=='train' and e['task']==task],key=lambda e:e['episode_id'])
            paths=[np.asarray(e['xyz'],dtype=float) for e in episodes]
            require(len(paths)>1 and all(x.ndim==2 and x.shape[1]==3 and len(x)>0 and np.isfinite(x).all() for x in paths),'Invalid task paths')
            metrics=variation(paths); initial=metrics['initial']; allxyz=np.concatenate(paths)
            lengths=np.array([np.linalg.norm(np.diff(x,axis=0),axis=1).sum() for x in paths])
            row=dict(suite=suite,task_index=number,task=task,episodes=len(paths),pairs=len(metrics['world_pairs']),
                path_length_median_m=float(np.median(lengths)),
                initial_pair_distance_median_m=float(np.median(metrics['initial_pairs'])),
                world_path_pair_rms_median_m=float(np.median(metrics['world_pairs'])),
                start_relative_path_pair_rms_median_m=float(np.median(metrics['relative_pairs'])),
                initial_radial_rms_m=float(np.sqrt(np.mean(np.sum((initial-initial.mean(0))**2,axis=1)))))
            rows.append(row)
            fig=plt.figure(figsize=(15,9.5))
            axes=[fig.add_subplot(2,3,1,projection='3d')]+[fig.add_subplot(2,3,i) for i in range(2,7)]
            colors=plt.colormaps['viridis'](np.linspace(.08,.92,len(paths)))
            low=allxyz.min(0)-.01; high=allxyz.max(0)+.01
            for x,color in zip(paths,colors):
                axes[0].plot(*x.T,color=color,alpha=.55,lw=.8)
                axes[0].scatter(*x[0],color=color,s=13,edgecolor='black',linewidth=.25)
                axes[0].scatter(*x[-1],color=color,s=14,marker='x')
                for ax,j in ((axes[1],1),(axes[2],2)):
                    ax.plot(x[:,0],x[:,j],color=color,alpha=.55,lw=.8)
                    ax.scatter(x[0,0],x[0,j],color=color,s=18,edgecolor='black',linewidth=.3,zorder=3)
                    ax.scatter(x[-1,0],x[-1,j],color=color,s=18,marker='x',zorder=3)
            axes[0].set(title='3D EEF trajectories',xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',
                        xlim=(low[0],high[0]),ylim=(low[1],high[1]),zlim=(low[2],high[2]))
            axes[0].set_box_aspect(high-low)
            for axis in (axes[0].xaxis,axes[0].yaxis,axes[0].zaxis):axis.set_major_locator(MaxNLocator(4))
            for ax,j,title in ((axes[1],1,'XY trajectories'),(axes[2],2,'XZ trajectories')):
                ax.set(title=title,xlabel='X (m)',ylabel=('Y' if j==1 else 'Z')+' (m)',
                       xlim=(low[0],high[0]),ylim=(low[j],high[j]))
                ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
            for ax,j,label in ((axes[3],1,'Y'),(axes[4],2,'Z')):
                ax.scatter(initial[:,0],initial[:,j],c=colors,s=32,alpha=.85,edgecolor='black',linewidth=.25)
                # Task-local zoom, same metric scale on both axes. No change to coordinates.
                center=np.array([initial[:,0].mean(),initial[:,j].mean()])
                radius=max(float(np.max(np.abs(initial[:,[0,j]]-center)))*1.2,.003)
                ax.set(title=f'Initial X{label} positions (zoom)',xlabel='Initial X (m)',ylabel=f'Initial {label} (m)',
                       xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius))
                ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
                for axis in (ax.xaxis,ax.yaxis):
                    axis.set_major_locator(MaxNLocator(4));axis.set_major_formatter(ScalarFormatter(useOffset=False))
            progress=np.linspace(0,1,32)
            axes[5].plot(progress,metrics['world_spread']*100,label='World-frame paths',lw=2)
            axes[5].plot(progress,metrics['relative_spread']*100,label='Initial-position-subtracted paths',lw=2)
            axes[5].set(title='Episode spread along progress',xlabel='Normalized recorded-step progress',ylabel='RMS distance to mean path (cm)',ylim=(0,None))
            axes[5].legend(fontsize=8);axes[5].grid(alpha=.2)
            instruction=task.replace('_',' ')
            fig.suptitle(f'{suite.upper()} / Task {number:02d} / {len(paths)} expert episodes\n'+textwrap.fill(instruction,100),fontsize=13)
            footer=(f"Median path length: {row['path_length_median_m']:.3f} m | Initial pair distance: {100*row['initial_pair_distance_median_m']:.2f} cm | "
                    f"Path pair RMS: world {100*row['world_path_pair_rms_median_m']:.2f} cm; start-relative {100*row['start_relative_path_pair_rms_median_m']:.2f} cm\n"
                    'Same color = same episode across panels; circle = first recorded state; cross = last recorded state.\n'
                    'Source-documented world coordinates; no bitwise provenance claim. Start subtraction removes translation only; it is not causal attribution.')
            fig.text(.5,.025,footer,ha='center',va='bottom',fontsize=9)
            fig.tight_layout(rect=(0,.11,1,.91))
            filename=f'{suite}_task{number:02d}_eef_initial.png'
            fig.savefig(output/filename,dpi=160);plt.close(fig)
            mapping.append(dict(**row,figure=filename,episode_ids=[e['episode_id'] for e in episodes]))
            print(f'Saved {filename}',flush=True)
    with (output/'task_trajectory_initial_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    metadata=dict(source=str(source.resolve()),scope='Explicitly requested supplement; all 30 tasks, all train expert episodes',
                  coordinate_status='SOURCE_DOCUMENTED',bitwise_verified=False,
                  method='All unordered within-task episode pairs, linear interpolation to 32 normalized recorded-step positions. Start-relative paths subtract each episode initial XYZ only; no rotation or DTW.',
                  interpretation='World vs start-relative differences are descriptive, not an additive or causal decomposition; task-specific axes are labeled in meters.',tasks=mapping)
    (output/'task_trajectory_initial_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    lines=['# Task-wise EEF trajectories and initial positions','',
           'All 30 tasks; all train expert episodes. Six panels per task: 3D/XY/XZ trajectories, initial XY/XZ zoom, and world/start-relative progress spread.',
           'Task initial EEF position is not the full environment state. No source decoding, model execution, backup or archive.',
           'The spread curves are RMS distances to the episode mean at each progress point; they are not pairwise RMS distributions. Footer values are medians of all episode pairs.',
           '', '| Suite | Task | N | Path median (m) | Initial pair median (cm) | World RMS median (cm) | Start-relative RMS median (cm) | Figure |',
           '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in mapping:
        lines.append(f"| {r['suite']} | {r['task_index']} | {r['episodes']} | {r['path_length_median_m']:.3f} | {100*r['initial_pair_distance_median_m']:.2f} | {100*r['world_path_pair_rms_median_m']:.2f} | {100*r['start_relative_path_pair_rms_median_m']:.2f} | [{r['task']}]({r['figure']}) |")
    lines+=['','Initial subtraction does not necessarily lower dispersion: paths can converge from different starts. Remaining differences include shape and phase timing. Episode pairs are correlated; no statistical significance or policy-quality claims are made.']
    (output/'task_trajectory_initial_index.md').write_text('\n'.join(lines)+'\n')
    index=output/'index.md'
    link='[All-task trajectory and initial-position supplements](task_trajectory_initial_index.md)'
    if index.exists() and link not in index.read_text():index.write_text(index.read_text()+'\n'+link+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();generate(a.input,a.output)
