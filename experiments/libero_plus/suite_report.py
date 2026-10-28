"""Consolidated Goal, Long and Spatial result report from on-disk evidence."""
from collections import Counter
import json
from pathlib import Path

from .suite_common import ROOT, SUITES, MODELS, result_path, require, sha

PAPER = {'goal': {'oft':97.9,'ript':99.0}, 'long': {'oft':92.9,'ript':93.8},
         'spatial': {'oft':97.6,'ript':98.6}}


def load_run(root,suite,dataset,model):
    folder=root/'outputs'/('libero-nominal' if dataset=='nominal' else 'libero-plus')/suite/model/'seed-1'
    mp=folder/'manifest.json'
    if not mp.is_file():return {'status':'INCOMPLETE','reason':'No manifest','path':str(folder)}
    manifest=json.loads(mp.read_text())
    require(manifest['suite_label']==suite and manifest['dataset']==dataset and manifest['model']==model,
            'Report manifest suite/model mismatch')
    require(manifest['benchmark_revision'] and manifest['evaluator_sha256'], 'Missing provenance')
    entries=manifest['task_mapping']
    require(len(entries)==(10 if dataset=='nominal' else SUITES[suite][3]), 'Task count mismatch')
    trials=manifest['trials_per_task']
    require(trials==(50 if dataset=='nominal' else 1), 'Trial protocol mismatch')
    rows=[]
    for e in entries:
        for t in range(trials):
            p=result_path(folder,e['id'],t,dataset)
            if not p.is_file():continue
            row=json.loads(p.read_text())
            require(all(row[k]==v for k,v in {'suite':manifest['suite'],'dataset':dataset,
                'model':model,'task_id':e['id'],'task_name':e['name'],'trial_id':t,
                'seed':manifest['seed'],'benchmark_revision':manifest['benchmark_revision'],
                'evaluator_sha256':manifest['evaluator_sha256']}.items()),
                f'Invalid result identity: {p}')
            require(type(row['success']) is bool and type(row['failure']) is bool and
                    row['failure'] != row['success'] and 0<row['steps']<=manifest['horizon'],
                    f'Invalid outcome: {p}')
            rows.append(row)
    per_task={e['name']:{'task_id':e['id'],'episodes':sum(r['task_id']==e['id'] for r in rows),
                 'successes':sum(r['success'] for r in rows if r['task_id']==e['id'])} for e in entries}
    for value in per_task.values():
        value['rate']=value['successes']/value['episodes'] if value['episodes'] else None
    expected=len(entries)*trials
    categories={category:{'episodes':sum(r['shift']==category for r in rows),
                          'successes':sum(r['success'] for r in rows if r['shift']==category)}
                for category in sorted({e['category'] for e in entries})} if dataset=='plus' else {}
    for value in categories.values():
        value['rate']=value['successes']/value['episodes'] if value['episodes'] else None
    return {'status':'COMPLETE' if len(rows)==expected else 'INCOMPLETE',
            'episodes':len(rows),'expected_episodes':expected,
            'successes':sum(r['success'] for r in rows),
            'rate':sum(r['success'] for r in rows)/len(rows) if rows else None,
            'per_task':per_task,'categories':categories,
            'manifest_path':str(mp),'manifest_sha256':sha(mp),
            'benchmark_revision':manifest['benchmark_revision'],
            'evaluator_sha256':manifest['evaluator_sha256'],'checkpoint':manifest['checkpoint'],
            '_rows':rows}


def paired(scale,ript):
    if scale['status']!='COMPLETE' or ript['status']!='COMPLETE':
        return {'status':'INCOMPLETE','reason':'Matched Scale and RIPT coverage required'}
    a={r['task_id']:r for r in scale['_rows']}
    b={r['task_id']:r for r in ript['_rows']}
    require(len(a)==len(scale['_rows']) and len(b)==len(ript['_rows']) and set(a)==set(b),
            'Paired condition IDs differ')
    require(all(a[i]['task_name']==b[i]['task_name'] and a[i]['shift']==b[i]['shift'] for i in a),
            'Cross-suite or category paired mismatch')
    require(scale['benchmark_revision']==ript['benchmark_revision'], 'Paired benchmark revisions differ')
    def block(ids):
        ids=list(ids)
        counts=Counter('both_success' if a[i]['success'] and b[i]['success'] else
                       'lost' if a[i]['success'] else 'gained' if b[i]['success'] else 'both_fail' for i in ids)
        n=len(ids)
        return {'conditions':n,'scale_successes':sum(a[i]['success'] for i in ids),
                'ript_successes':sum(b[i]['success'] for i in ids),
                'scale_rate':sum(a[i]['success'] for i in ids)/n if n else None,
                'ript_rate':sum(b[i]['success'] for i in ids)/n if n else None,
                'difference_pp':100*(sum(b[i]['success'] for i in ids)-sum(a[i]['success'] for i in ids))/n if n else None,
                'transitions':{k:counts[k] for k in ('both_success','lost','gained','both_fail')}}
    return {'status':'COMPLETE','overall':block(a),
            'categories':{shift:block(i for i in a if a[i]['shift']==shift)
                          for shift in sorted({r['shift'] for r in a.values()})},
            'benchmark_revision':scale['benchmark_revision']}


def build(root=ROOT):
    from . import research_report
    goal=research_report.build(root)
    result={'paper_nominal_percent':PAPER,'goal':goal,'suites':{}}
    for suite in SUITES:
        nominal={m:load_run(root,suite,'nominal',m) for m in MODELS}
        ood={m:load_run(root,suite,'plus',m) for m in MODELS}
        pair=paired(ood['scale'],ood['ript'])
        nominal_diff=(100*(nominal['ript']['rate']-nominal['scale']['rate'])
                      if nominal['ript']['status']=='COMPLETE' and nominal['scale']['status']=='COMPLETE'
                      else None)
        for block in (nominal,ood):
            for run in block.values():run.pop('_rows',None)
        result['suites'][suite]={'nominal':nominal,'nominal_scale_to_ript_pp':nominal_diff,
                                  'ood':ood,'ood_paired':pair}
    return result


def markdown(data):
    lines=['# Goal, Long and Spatial evaluation', '',
           'All local rates below come from result files. Goal P3-A2/P3-B analyses apply only to Goal.', '',
           '## Paper-reported nominal rates (not local measurements)', '',
           '| Suite | OFT SFT | RIPT |','|---|---:|---:|']
    for suite,r in data['paper_nominal_percent'].items():
        lines.append(f"| {suite} | {r['oft']:.1f}% | {r['ript']:.1f}% |")
    lines+=['','## Locally reproduced nominal','', '| Suite | SFT-full | Scale-cached | RIPT-cached | Matched Scale→RIPT |',
            '|---|---|---|---|---|']
    g=data['goal']['nominal']['models']
    lines.append('| goal | '+ ' | '.join(f"{g[m]['successes']}/{g[m]['trials']} ({g[m]['rate']:.2%})" for m in MODELS)+
                 f" | {data['goal']['nominal']['matched_difference_pp']:+.3f} pp |")
    for suite,s in data['suites'].items():
        def fmt(r):
            return f"{r['successes']}/{r['expected_episodes']} ({r['rate']:.2%})" if r['status']=='COMPLETE' else f"INCOMPLETE {r.get('episodes',0)}/{r.get('expected_episodes',500)}"
        delta=s['nominal_scale_to_ript_pp']
        lines.append('| '+suite+' | '+' | '.join(fmt(s['nominal'][m]) for m in MODELS)+
                     f" | {delta:+.3f} pp |" if delta is not None else
                     '| '+suite+' | '+' | '.join(fmt(s['nominal'][m]) for m in MODELS)+' | INCOMPLETE |')
    lines+=['','## LIBERO-PLUS OOD','']
    goal_ood=data['goal']['ood']['overall']
    lines.append(f"Goal: {goal_ood['successes']['scale']}/{goal_ood['conditions']} Scale, {goal_ood['successes']['ript']}/{goal_ood['conditions']} RIPT; {goal_ood['difference_pp']:+.3f} pp.")
    for suite,s in data['suites'].items():
        pair=s['ood_paired']
        if pair['status']!='COMPLETE':
            lines.append(f"{suite}: INCOMPLETE; paired OOD unavailable. SFT: {s['ood']['sft']['status']}.")
            continue
        r=pair['overall']
        lines.append(f"{suite}: {r['scale_successes']}/{r['conditions']} Scale, {r['ript_successes']}/{r['conditions']} RIPT; {r['difference_pp']:+.3f} pp; transitions {r['transitions']}.")
        lines+=['','| Category | N | SFT-full | Scale | RIPT | Δ pp | Both / lost / gained / fail |',
                '|---|---:|---:|---:|---:|---:|---|']
        for category,v in pair['categories'].items():
            tr=v['transitions']
            sft=s['ood']['sft']['categories'].get(category,{}).get('rate') if s['ood']['sft']['status']=='COMPLETE' else None
            sft_text=f'{sft:.2%}' if sft is not None else 'INCOMPLETE'
            lines.append(f"| {category} | {v['conditions']} | {sft_text} | {v['scale_rate']:.2%} | {v['ript_rate']:.2%} | {v['difference_pp']:+.3f} | "+
                         '/'.join(str(tr[k]) for k in ('both_success','lost','gained','both_fail'))+' |')
    lines+=['','Task-level nominal rates and full provenance are in the companion JSON. SFT-full and cached policies use different inference paths; their difference does not isolate RL training.']
    return '\n'.join(lines)+'\n'


def write(root,data,refresh=False):
    folder=root/'outputs/rlvr_analysis'
    folder.mkdir(parents=True,exist_ok=True)
    targets=[folder/'research_summary_all.md',folder/'research_summary_all.json']
    require(refresh or not any(p.exists() for p in targets), 'Consolidated report exists; pass --refresh')
    import os,tempfile
    for path,content in zip(targets,[markdown(data),json.dumps(data,indent=2,allow_nan=False)+'\n']):
        with tempfile.NamedTemporaryFile('w',dir=folder,delete=False,prefix=path.name+'.') as f:
            tmp=Path(f.name);f.write(content);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    return targets
