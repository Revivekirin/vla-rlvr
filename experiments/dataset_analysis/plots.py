"""Exactly four consolidated figures from saved analysis. No TFDS/model imports."""
import argparse
from pathlib import Path
import os
import tempfile
import numpy as np
from experiments.shot_scaling.core import read,write,require,sha,digest


def plot(bundle_path,output):
    bundle=read(bundle_path);data=bundle['suites']
    require(set(data)=={'goal','spatial','long'},'All three real suite analyses required for main figures')
    require(all(not d['synthetic'] and d['episodes'] for d in data.values()),'Real episodes required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'libero-dataset-analysis-mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8})
    suites=['goal','spatial','long'];chosen={};figures=[]
    def save(fig,name,title):
        fig.suptitle(title,fontsize=13);fig.tight_layout(rect=(0,0,1,.96))
        for ext in ('png','pdf'):fig.savefig(output/f'{name}.{ext}',dpi=160)
        plt.close(fig);figures.append(name)
    def unavailable(ax,message='UNAVAILABLE\nMeasured EEF conventions/state not verified'):
        ax.text(.5,.5,message,ha='center',va='center',transform=ax.transAxes,fontsize=8);ax.set_xticks([]);ax.set_yticks([])
    def task_rows(s):
        return sorted(data[s]['tasks']['train'])
    def representative(s):
        task=task_rows(s)[0]  # deterministic task-name order, not result quality
        rows=[e for e in data[s]['episodes'] if e['task']==task and e['split']=='train']
        return task,rows
    fig,axes=plt.subplots(3,3,figsize=(15,11))
    for row,s in enumerate(suites):
        tasks=task_rows(s);stats=data[s]['tasks']['train'];positions=np.arange(len(tasks))
        labels=[str(j+1) for j in positions]
        axes[row,0].bar(positions,[stats[t]['episodes'] for t in tasks]);axes[row,0].set(title=f'{s}: train episodes',xlabel='Task ID (mapping in figure_manifest.json)',ylabel='Episodes',xticks=positions,xticklabels=labels)
        axes[row,1].boxplot([stats[t]['lengths'] for t in tasks],tick_labels=labels,showfliers=False);axes[row,1].set(title=f'{s}: recorded episode lengths',xlabel='Task ID',ylabel='Transitions')
        measured=[e for e in data[s]['episodes'] if e['split']=='train' and e['xyz'] is not None]
        if measured:
            for j,t in enumerate(tasks):
                vals=np.array([e['xyz'][0] for e in measured if e['task']==t]);axes[row,2].scatter(vals[:,0],vals[:,1],s=8,label=str(j+1))
            axes[row,2].set(xlabel='Initial X (m)',ylabel='Initial Y (m)',title=f'{s}: initial positions, stratified by task');axes[row,2].legend(ncol=5,fontsize=6)
        else:unavailable(axes[row,2])
    save(fig,'figure_1_dataset_overview','Figure 1 — Actual expert dataset overview (train split; all split counts in report)')
    fig=plt.figure(figsize=(19,11))
    for row,s in enumerate(suites):
        task,rows=representative(s);paths=[e for e in rows if e['xyz'] is not None]
        display=sorted(paths,key=lambda e:digest(e['episode_id']))[:8];chosen[s]={'task':task,'trajectory_ids':[e['episode_id'] for e in display]}
        ax3=fig.add_subplot(3,4,row*4+1,projection='3d');axxy=fig.add_subplot(3,4,row*4+2);axxz=fig.add_subplot(3,4,row*4+3);axyz=fig.add_subplot(3,4,row*4+4)
        if not display:
            ax3.set_axis_off()
            ax3.text2D(.05,.5,f'{s}: physical paths BLOCKED\nRLDS coordinate evidence incomplete\nSee physical_convention_evidence.json',transform=ax3.transAxes)
            for ax in (axxy,axxz,axyz):unavailable(ax,'BLOCKED: coordinate evidence incomplete\nRaw state exists; XYZ interpretation not approved')
        else:
            whole=np.concatenate([e['xyz'] for e in paths]);low=whole.min(0);high=whole.max(0);pad=np.maximum((high-low)*.05,.001)
            for e in display:
                x=np.array(e['xyz']);ax3.plot(*x.T,lw=.8);axxy.plot(x[:,0],x[:,1],lw=.8);axxz.plot(x[:,0],x[:,2],lw=.8);axyz.plot(x[:,1],x[:,2],lw=.8)
            ax3.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',xlim=(low[0]-pad[0],high[0]+pad[0]),ylim=(low[1]-pad[1],high[1]+pad[1]),zlim=(low[2]-pad[2],high[2]+pad[2]))
            for ax,d in ((axxy,1),(axxz,2)):
                ax.set(xlim=(low[0]-pad[0],high[0]+pad[0]),ylim=(low[d]-pad[d],high[d]+pad[d]),xlabel='X (m)',ylabel=('Y' if d==1 else 'Z')+' (m)');ax.set_aspect('equal',adjustable='box')
            axyz.set(xlim=(low[1]-pad[1],high[1]+pad[1]),ylim=(low[2]-pad[2],high[2]+pad[2]),xlabel='Y (m)',ylabel='Z (m)');axyz.set_aspect('equal',adjustable='box')
        ax3.set_title(s+' — '+task.replace('_',' ')[:55],fontsize=7);axxy.set_title('XY projection');axxz.set_title('XZ projection');axyz.set_title('YZ projection')
    save(fig,'figure_2_eef_trajectories','Figure 2 — Measured EEF paths only; no integrated action commands')
    fig,axes=plt.subplots(3,5,figsize=(21,11))
    for row,s in enumerate(suites):
        tasks=task_rows(s);es=[e for e in data[s]['episodes'] if e['split']=='train'];xs=np.arange(len(tasks));labels=[str(j+1) for j in xs]
        physical=[[e['physical']['path_length'] for e in es if e['task']==t and e['physical']] for t in tasks]
        if all(physical):
            axes[row,0].boxplot(physical,tick_labels=labels,showfliers=False);axes[row,0].set(ylabel='Recorded path length (m)')
            pairvals=[[p['aligned_xyz_rms_m'] for p in data[s]['physical_pairs'] if p['task']==t and p['split']=='train'] for t in tasks]
            axes[row,1].boxplot(pairvals,tick_labels=labels,showfliers=False);axes[row,1].set(ylabel='Within-task phase-aligned XYZ RMS (m)')
        else:unavailable(axes[row,0]);unavailable(axes[row,1])
        hist=np.array([np.sum([e['translation']['azimuth_counts'] for e in es if e['task']==t],axis=0) for t in tasks],dtype=float)
        hist=np.divide(hist,hist.sum(1)[:,None],out=np.zeros_like(hist),where=hist.sum(1)[:,None]>0)
        im=axes[row,2].imshow(hist,aspect='auto',extent=(-180,180,len(tasks)+.5,.5));fig.colorbar(im,ax=axes[row,2],label='Command direction fraction');axes[row,2].set(xlabel='Translation command azimuth (deg)',ylabel='Task ID')
        for label,color,offset in [('translation','#367f9e',-.15),('rotation','#b87947',.15)]:
            means=[np.mean([e[label]['magnitude_mean'] for e in es if e['task']==t]) for t in tasks]
            axes[row,3].bar(xs+offset,means,width=.3,label=label,color=color)
        axes[row,3].set(xticks=xs,xticklabels=labels,ylabel='Episode-mean command norm (raw units)');axes[row,3].legend(fontsize=6)
        axes[row,4].boxplot([[len(e['gripper_events']) for e in es if e['task']==t] for t in tasks],tick_labels=labels);axes[row,4].set(ylabel='Gripper command threshold crossings')
        for col,title in enumerate(('Physical length','Physical path difference','Action direction','Action magnitude','Gripper pattern')):
            axes[row,col].set_title(s+': '+title)
    save(fig,'figure_3_expert_variation','Figure 3 — Task-conditioned physical state versus controller command statistics')
    fig,axes=plt.subplots(3,6,figsize=(21,11));selection_ids={}
    for row,s in enumerate(suites):
        task,rows=representative(s);lookup={e['episode_id']:e for e in rows};paths=[e for e in rows if e['xyz'] is not None]
        seed=str(bundle['display_seed']);sels=data[s]['selections'][seed];selection_ids[s]={}
        whole=np.concatenate([e['xyz'] for e in paths]) if paths else None
        for col,shot in enumerate(('1','2','5','10','20','historical')):
            ax=axes[row,col]
            if shot=='historical':
                ids=[k for k in (data[s]['historical_ids'] or []) if k in lookup]
                comparison=[r['episode_id'] for r in sels['5']['episodes'] if r['task']==task]
            else:ids=[r['episode_id'] for r in sels[shot]['episodes'] if r['task']==task];comparison=[]
            selection_ids[s][shot]=dict(task=task,selected_ids=ids,comparison_nested5_ids=comparison)
            if whole is not None and ids:
                for e in paths:
                    x=np.asarray(e['xyz']);ax.plot(x[:,0],x[:,1],color='.8',lw=.5,zorder=0)
                for selected,color,label in ((ids,'#367f9e','historical' if shot=='historical' else 'selected'),(comparison,'#c87335','nested5')):
                    for j,k in enumerate(selected):
                        x=np.asarray(lookup[k]['xyz']);ax.plot(x[:,0],x[:,1],color=color,lw=1,label=label if j==0 else None)
                low=whole.min(0);high=whole.max(0);pad=np.maximum((high-low)*.05,.001)
                ax.set(xlim=(low[0]-pad[0],high[0]+pad[0]),ylim=(low[1]-pad[1],high[1]+pad[1]),xlabel='X (m)',ylabel='Y (m)');ax.set_aspect('equal',adjustable='box');ax.legend(fontsize=6)
            else:unavailable(ax,f'Physical paths UNAVAILABLE\n{len(ids)} selected IDs saved' if ids else 'Historical selection UNAVAILABLE')
            ax.set_title(f'{s}: {shot}-shot; seed {seed}' if shot!='historical' else f'{s}: historical vs nested5',fontsize=8)
    save(fig,'figure_4_shot_selection','Figure 4 — Full population (gray) and random nested selections; no quality ranking')
    write(output/'figure_manifest.json',dict(source=str(Path(bundle_path).resolve()),source_sha256=sha(bundle_path),figures=figures,
        task_mapping={s:{str(j+1):t for j,t in enumerate(task_rows(s))} for s in suites},representatives=chosen,selection_ids=selection_ids,
        policy='Representative task is first sorted task name; up to eight source-ID-ranked paths. No performance/coverage optimization. Supplementary figures not generated.'))
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(plot(a.input,a.output))
