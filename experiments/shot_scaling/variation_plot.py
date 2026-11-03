"""Standalone Phase 2 visualizations: reads variation.json only; no extraction or inference."""
import argparse
import os
from pathlib import Path
import tempfile
import numpy as np
from .core import read, require, sha, write


def plot(input_path, output):
    d=read(input_path)
    require(d['kind']=='expert_variation' and not d['synthetic'],'Expected real Phase 2 artifact')
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'shot-scaling-mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics=('temporal_action','descriptor','initial_state','gripper_pattern')
    labels={'temporal_action':'Phase-aligned action RMS','descriptor':'Episode descriptor RMS',
            'initial_state':'Initial proprio RMS','gripper_pattern':'Gripper phase disagreement'}
    index=['# Phase 2 — task-specific saved-artifact figures','',
        '**UNAVAILABLE:** measured 3D EEF overlays and EEF path-coverage plots. Full state sequences were not retained; no commands were integrated.',
        'All trajectory-distance plots below measure action commands, not physical EEF paths.',
        'Random bands are empirical conditional 5th–95th percentiles (not confidence intervals). Candidates exclude the same 228 reference episodes. All random 20-shot selections coincide.',
        'Initial state values are shown in source coordinates/units; their physical frame and units remain unverified.','']
    def save(fig,path,title):
        fig.suptitle(title,fontsize=10,wrap=True)
        fig.tight_layout(rect=(0,0,1,.94))
        for ext in ('png','svg'): fig.savefig(output/f'{path}.{ext}',dpi=140)
        plt.close(fig)
        index.append(f'- [{path}]({path}.png) · [SVG]({path}.svg)')
    for task,info in d['tasks'].items():
        name=task
        display=task.replace('_',' ')
        es=[e for e in d['episodes'] if e['task']==task]
        index+=['',f'## {display}','']
        initial=np.array([e['initial_proprio'] for e in es])
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        sc=axes[0].scatter(initial[:,0],initial[:,1],c=initial[:,2],s=25,cmap='viridis')
        axes[0].set(xlabel='Initial state[0] (source units)',ylabel='Initial state[1] (source units)')
        fig.colorbar(sc,ax=axes[0],label='Initial state[2]')
        axes[1].scatter(initial[:,6],initial[:,7],s=25)
        axes[1].set(xlabel='Initial gripper state[6]',ylabel='Initial gripper state[7]')
        save(fig,name+'_initial_state',display+' | measured initial state; physical units/frame unverified')
        fig,axes=plt.subplots(2,2,figsize=(10,7))
        for row,kind in enumerate(('translation','rotation')):
            counts=np.array([e[kind]['direction_histogram'] for e in es],dtype=float)
            totals=counts.sum((1,2))
            episode=np.divide(counts,totals[:,None,None],out=np.zeros_like(counts),where=totals[:,None,None]>0).mean(0)
            transition=counts.sum(0)/max(1,counts.sum())
            for col,(values,label) in enumerate(((transition,'transition weighted'),(episode,'episode weighted'))):
                im=axes[row,col].imshow(values.T,origin='lower',aspect='auto',extent=(-180,180,-90,90),cmap='magma')
                axes[row,col].set(xlabel='Command azimuth (degrees)',ylabel='Command elevation (degrees)',title=f'{kind}: {label}')
                fig.colorbar(im,ax=axes[row,col],label='Bin probability')
        save(fig,name+'_command_directions',display+' | command directions, zero vectors excluded')
        # Deterministic bounded display: evenly spaced source IDs, independent of outcome/coverage.
        indices=np.unique(np.linspace(0,len(es)-1,min(8,len(es))).astype(int))
        fig,ax=plt.subplots(figsize=(10,5))
        for offset,j in enumerate(indices):
            e=es[j]; values=np.array(e['gripper_phase_pattern'])
            ax.step(np.linspace(0,1,len(values)),offset+values*.65,where='post',label=e['episode_id'])
        ax.set(yticks=np.arange(len(indices)),yticklabels=[f'episode index {j}' for j in indices],
               xlabel='Normalized episode phase (not seconds)',ylabel='0=open; +0.65=close within each row')
        save(fig,name+'_gripper_commands',display+' | bounded raw gripper-command patterns; IDs in figure_manifest.json')
        fig,ax=plt.subplots(figsize=(7,6))
        im=ax.imshow(info['pairwise_matrices']['temporal_action'],cmap='viridis',vmin=0)
        ax.set(xlabel='Episode index (source IDs saved in variation.json)',ylabel='Episode index')
        fig.colorbar(im,ax=ax,label='Within-task phase-aligned action RMS')
        save(fig,name+'_action_distance_heatmap',display+' | commands, not EEF paths')
        fig,axes=plt.subplots(2,2,figsize=(10,7))
        for ax,metric in zip(axes.flat,metrics):
            rows=sorted([r for r in d['coverage'] if r['task']==task and r['metric']==metric],key=lambda r:r['shots'])
            x=[r['shots'] for r in rows]
            ax.fill_between(x,[r['random_coverage']['p05'] for r in rows],[r['random_coverage']['p95'] for r in rows],alpha=.2,label='Random p05–p95')
            ax.plot(x,[r['random_coverage']['p50'] for r in rows],':',label='Random median')
            ax.plot(x,[r['nearest_mean'] for r in rows],'o-',label='seed42')
            ax.set(xlabel='Shots per task',ylabel=labels[metric],xticks=[1,2,5,10,20])
            ax.legend(fontsize=7)
        save(fig,name+'_shot_coverage',display+' | fixed reference IDs; conditional random bands')
        fig,axes=plt.subplots(2,2,figsize=(9,6))
        for ax,metric in zip(axes.flat,metrics):
            r=next(r for r in d['historical_comparison'] if r['task']==task and r['metric']==metric)
            ax.bar(['Historical 5','Nested 5'],[r['historical_distance'],r['nested5_distance']],color=['#b87947','#367f9e'])
            ax.set(ylabel=labels[metric],title=f'Shared reference n={len(r["reference_episode_ids"])}')
        save(fig,name+'_historical_vs_nested5',display+' | aligned by source ID; secondary reference excludes both selections')
        print(f'Plotted {task}',flush=True)
    (output/'index.md').write_text('\n'.join(index)+'\n')
    write(output/'figure_manifest.json',dict(input=str(Path(input_path).resolve()),input_sha256=sha(input_path),
        gripper_display={task:[es[j]['episode_id'] for j in np.unique(np.linspace(0,len(es)-1,min(8,len(es))).astype(int))]
            for task in d['tasks'] for es in [[e for e in d['episodes'] if e['task']==task]]},
        figures=len(list(output.glob('*.png'))),note='Reads saved analysis only; unavailable physical plots are listed in index.'))
    print(output.resolve())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plot(a.input,a.output)
