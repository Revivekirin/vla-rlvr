"""Read-only research aggregation; only derived summary files may be refreshed."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
from evaluate import ROOT, REVISION
from report import aggregate, comparable
from p3_action_analysis import require, sha
from p3b_metrics import stats


def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def boolean(value):
    if str(value).lower() in ('1', 'true'):
        return True
    if str(value).lower() in ('0', 'false'):
        return False
    raise ValueError(f'Invalid outcome: {value!r}')


def historical(root):
    """Prefer original per-condition JSON. Explicitly label the local paired-CSV fallback."""
    dirs = [root / f'outputs/libero-plus/{m}/seed-1' for m in ('scale', 'ript')]
    if any((d / 'manifest.json').exists() for d in dirs):
        require(all((d / 'manifest.json').exists() for d in dirs), 'Missing one OOD manifest')
        s, r = [aggregate(d) for d in dirs]
        require(comparable(s, r), 'OOD manifests/coverage do not match')
        require(len(s[1]) == len(s[0]['tasks']) == 2591, 'Incomplete OOD population')
        records = [dict(task_id=i, task_name=a['task_name'], shift=a['shift'], scale=a['success'],
                        ript=r[1][i]['success']) for i, a in sorted(s[1].items())]
        provenance = {'source': 'original JSON', 'manifests': [s[0], r[0]],
                      'files': {str(d / 'manifest.json'): sha(d / 'manifest.json') for d in dirs}}
    else:
        path = root / 'experiments/libero_plus/results/abc_paired.csv'
        require(path.is_file(), 'Mandatory OOD results missing (JSON and paired CSV)')
        records = [dict(task_id=int(v['task_id']), task_name=v['task_name'], shift=v['shift'],
                        scale=boolean(v['scale']), ript=boolean(v['ript'])) for v in rows(path)]
        provenance = {'source': 'derived paired CSV; original manifests unavailable', 'files': {str(path): sha(path)},
                      'warning': 'Seed, dataset revision, host and evaluator hashes cannot be independently established from this CSV.'}
    require({r['task_id'] for r in records} == set(range(1, 2592)) and len(records) == 2591, 'Invalid OOD IDs')
    require(len({r['task_name'] for r in records}) == 2591, 'Duplicate OOD task names')
    for r in records:
        r['historical_case'] = ('both_success' if r['scale'] and r['ript'] else 'lost' if r['scale'] else 'gained' if r['ript'] else 'both_fail')
    return records, provenance


def nominal(root):
    path = root / 'experiments/nominal/goal.csv'
    require(path.is_file(), f'Mandatory nominal CSV missing: {path}')
    result = {}
    for r in rows(path):
        require(r['model'] in ('sft', 'scale', 'ript') and r['model'] not in result, 'Invalid nominal model')
        n, k = int(r['total_episodes']), int(r['successful_episodes'])
        tasks = json.loads(r['per_task_success_rates'])
        require(n > 0 and 0 <= k <= n and sum(t['episodes'] for t in tasks.values()) == n
                and sum(t['successes'] for t in tasks.values()) == k, 'Nominal counts disagree')
        require(abs(float(r['success_rate']) - k/n) < 1e-9, 'Nominal rate disagrees')
        result[r['model']] = dict(trials=n, successes=k, rate=k/n, checkpoint=json.loads(r['checkpoint']),
                                  seed=r['evaluation_seed'], per_task=tasks, evidence=r['source_evidence_path'],
                                  evidence_available=(root / r['source_evidence_path']).is_file(),
                                  inference='SFT full sequence; unmatched baseline' if r['model'] == 'sft' else 'RIPT cached')
    require(set(result) == {'sft', 'scale', 'ript'}, 'Missing nominal model')
    return dict(models=result, matched_difference_pp=100*(result['ript']['rate']-result['scale']['rate']),
                files={str(path): sha(path)})


def a2_records(root):
    path = root / 'outputs/rlvr_analysis/p3/p3_a2_rollouts.csv'
    if not path.exists():
        return [], {'status': 'NOT AVAILABLE', 'reason': str(path)}
    records, seen = [], set()
    for r in rows(path):
        require(set(('repeat', 'model', 'task_id', 'historical_case', 'seed', 'success', 'steps', 'host', 'evaluator_sha256')).issubset(r), 'Unknown P3-A2 CSV schema')
        r = dict(r, task_id=int(r['task_id']), repeat=int(r['repeat']), seed=int(r['seed']), steps=int(r['steps']), success=boolean(r['success']))
        key = (r['task_id'], r['model'], r['repeat'])
        require(key not in seen and r['model'] in ('scale', 'ript') and 1 <= r['steps'] <= 300, 'Invalid A2 row')
        seen.add(key)
        records.append(r)
    require(bool(records), 'Empty P3-A2 CSV')
    return records, {'status': 'AVAILABLE', 'files': {str(path): sha(path)}}


def a2_summary(root):
    records, provenance = a2_records(root)
    if not records:
        return provenance
    tasks = {}
    for tid in sorted({r['task_id'] for r in records}):
        selected = [r for r in records if r['task_id'] == tid]
        require(len({r['historical_case'] for r in selected}) == 1, 'Conflicting A2 historical labels')
        tasks[str(tid)] = dict(historical_case=selected[0]['historical_case'], models={})
        for model in ('scale', 'ript'):
            subset = [r for r in selected if r['model'] == model]
            require(bool(subset), 'Missing A2 model/task cell')
            tasks[str(tid)]['models'][model] = dict(rollouts=len(subset), successes=sum(r['success'] for r in subset),
                success_rate=sum(r['success'] for r in subset)/len(subset),
                successful_steps=stats([r['steps'] for r in subset if r['success']]))
    return dict(provenance, tasks=tasks, total_rollouts=len(records),
                run_count=len({(r['repeat'], r['model']) for r in records}),
                repeats=sorted({r['repeat'] for r in records}), seeds=sorted({r['seed'] for r in records}),
                hosts=sorted({r['host'] for r in records}), evaluator_hashes=sorted({r['evaluator_sha256'] for r in records}),
                models={m: dict(rollouts=sum(r['model']==m for r in records), successes=sum(r['success'] for r in records if r['model']==m)) for m in ('scale','ript')},
                warning='Outcome-selected conditions; fixed-seed repeats are not independent seeds or the population OOD effect. CSV contains outcomes, not images.')


def population(records):
    n = len(records)
    k = {m: sum(r[m] for r in records) for m in ('scale', 'ript')}
    return dict(conditions=n, successes=k, rates={m: v/n for m,v in k.items()},
                difference_pp=100*(k['ript']-k['scale'])/n, transitions=dict(Counter(r['historical_case'] for r in records)))


def build(root, p3b_run=None):
    records, provenance = historical(root)
    b = []
    paths = [root / 'outputs/rlvr_analysis/p3b' / p3b_run] if p3b_run else sorted((root / 'outputs/rlvr_analysis/p3b').glob('*'))
    for path in paths:
        if (path / 'analysis.json').is_file():
            # A representation report is only accepted after full artifact validation.
            from p3_representation_analysis import validate_run
            validate_run(path, require_features=True)
            analyzed=json.loads((path/'analysis.json').read_text())
            require(analyzed['bank_sha256']==sha(path/'bank.json'),'Stale P3-B analysis')
            for rel,digest in analyzed['feature_files'].items():
                require(sha(path/rel)==digest,'Stale P3-B feature digest')
            b.append(dict(run=path.name, analysis=json.loads((path/'analysis.json').read_text()),
                          manifest=json.loads((path/'manifest.json').read_text()),
                          analysis_sha256=sha(path/'analysis.json')))
    return dict(nominal=nominal(root), ood=dict(overall=population(records), categories={s:population([r for r in records if r['shift']==s]) for s in sorted({r['shift'] for r in records})}, provenance=provenance),
                p3_a2=a2_summary(root), p3_b=b if b else {'status':'NOT AVAILABLE', 'reason':'No completed, validated representation analysis'},
                interpretation=['Exploratory diagnostics, not causal evidence.', 'CKA/effective rank do not measure task-relevant information directly.',
                    'SFT-full is not the matched RL baseline.', 'No RLVR-specific improvement without a matched training comparison.',
                    'Eight action slots are not eight independent observations.'],
                repository_locks=json.loads((root/'stage0/repos.lock.json').read_text()))


def markdown(data):
    lines = ['# Unified VLA research summary', '', '## Original LIBERO-Goal', '',
             '| Model / path | Successes / trials | Rate |', '|---|---:|---:|']
    for m,r in data['nominal']['models'].items():
        lines.append(f"| {m}: {r['inference']} | {r['successes']}/{r['trials']} | {r['rate']:.2%} |")
    lines += ['', f"Matched Scale → RIPT: {data['nominal']['matched_difference_pp']:+.3f} pp.", '', '## LIBERO-PLUS Goal', '',
              '| Stratum | N | Scale | RIPT | Difference (pp) | Both / lost / gained / fail |', '|---|---:|---:|---:|---:|---|']
    for name,r in {'Overall':data['ood']['overall'], **data['ood']['categories']}.items():
        counts = '/'.join(str(r['transitions'].get(k,0)) for k in ('both_success','lost','gained','both_fail'))
        lines.append(f"| {name} | {r['conditions']} | {r['rates']['scale']:.2%} | {r['rates']['ript']:.2%} | {r['difference_pp']:+.3f} | {counts} |")
    lines += ['', '## P3-A2 repeated rollouts', '']
    a = data['p3_a2']
    if a['status'] == 'NOT AVAILABLE':
        lines.append('NOT AVAILABLE: '+a['reason'])
    else:
        lines += [a['warning'], '', f"{a['total_rollouts']} rollouts; {a['run_count']} evaluation runs; seeds {a['seeds']}.", '',
                  '| Task | Historical label | Scale successes/n; successful steps mean | RIPT successes/n; successful steps mean |', '|---|---|---|---|']
        for tid,r in a['tasks'].items():
            values = [f"{v['successes']}/{v['rollouts']}; {v['successful_steps']['mean']}" for v in r['models'].values()]
            lines.append(f"| {tid} | {r['historical_case']} | {' | '.join(values)} |")
    lines += ['', '## P3-B representations', '']
    if isinstance(data['p3_b'], dict):
        lines.append('NOT AVAILABLE: '+data['p3_b']['reason'])
    else:
        for b in data['p3_b']:
            analysis=b['analysis']
            lines += [f"### {b['run']}", '',
                      f"Mode: {analysis['mode']}; {analysis['observations']} observations, {analysis['rollouts']} rollouts, {analysis['distinct_base_tasks']} base-task clusters.",
                      'Task-level details, sensitivity checks and matched-N controls are in research_summary.json.', '',
                      '| Stratum/source | N / tasks / rollouts | Feature | CKA | Scale erank | RIPT erank | Cluster interval |',
                      '|---|---|---|---:|---:|---:|---|']
            for group,entry in analysis['groups'].items():
                if '|TASK:' in group:
                    continue
                for name,feature in entry['features'].items():
                    values=[feature['cka']['value'],feature['scale_rank']['effective_rank'],feature['ript_rank']['effective_rank']]
                    text=['NOT AVAILABLE' if v is None else f'{v:.5f}' for v in values]
                    interval=feature['uncertainty']['interval']
                    uncertainty=str(interval) if interval is not None else feature['uncertainty']['reason']
                    lines.append(f"| {group.replace('|',' / ')} | {entry['N']} / {entry['distinct_tasks']} / {entry['distinct_rollouts']} | {name} | {' | '.join(text)} | {uncertainty} |")
            lines += ['', '| Stratum/source | Executed chunk L2 mean | Gripper disagreement mean |', '|---|---:|---:|']
            for group,entry in analysis['groups'].items():
                if '|TASK:' not in group:
                    lines.append(f"| {group.replace('|',' / ')} | {entry['actions']['executed']['chunk_l2']['mean']:.5g} | {entry['actions']['gripper_disagreement_per_observation']['mean']:.5g} |")
            target=sum(sum(v['target'] for v in c['stages'].values()) for c in analysis['composition'])
            lines += ['', 'Feature definitions (one example; variable prefix shapes are recorded per observation):',
                      '```json', json.dumps(analysis['feature_definitions_example'],indent=2), '```',
                      'Runtime provenance:', '```json', json.dumps(analysis['execution_provenance'],indent=2), '```',
                      f"Collected {analysis['observations']} of at most {target}; stage/rollout shortfalls are recorded in JSON.",
                      *['- '+w for w in analysis['warnings']], '']
    lines += ['', '## Provenance and limits', '', *['- '+v for v in data['interpretation']], '',
              'Checkpoint identifiers, seeds, protocol, hosts (when recorded), revisions and evaluator hashes:', '', '```json',
              json.dumps(dict(nominal={k:{f:v for f,v in r.items() if f!='per_task'} for k,r in data['nominal']['models'].items()}, ood={k:([{f:v for f,v in m.items() if f!='tasks'} for m in value] if k=='manifests' else value) for k,value in data['ood']['provenance'].items()}, a2={k:v for k,v in a.items() if k!='tasks'},
                              repository_locks=data['repository_locks'], p3b=[{k:v for k,v in b['manifest'].items() if k in ('protocol','checkpoints','revisions','repository_revision','created_host','seed','layers','mode')} for b in data['p3_b']] if isinstance(data['p3_b'],list) else 'NOT AVAILABLE'), indent=2), '```', '']
    return '\n'.join(lines)


def write_report(root, data, refresh=False):
    folder = root / 'outputs/rlvr_analysis'
    paths = [folder/'research_summary.md', folder/'research_summary.json']
    require(refresh or not any(p.exists() for p in paths), 'Derived reports exist; use --refresh')
    folder.mkdir(parents=True, exist_ok=True)
    contents = [markdown(data), json.dumps(data, indent=2, allow_nan=False)+'\n']
    for path, text in zip(paths, contents):
        fd, temporary = tempfile.mkstemp(prefix=path.name+'.', dir=folder)
        try:
            with os.fdopen(fd,'w') as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return paths


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--all', action='store_true', required=True)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--refresh', action='store_true')
    p.add_argument('--p3b-run')
    p.add_argument('--suites', action='store_true',
                   help='Also produce a separate Goal/Long/Spatial consolidated report')
    a=p.parse_args()
    data=build(a.root, a.p3b_run)
    paths=write_report(a.root,data,a.refresh)
    print(markdown(data))
    print('Saved:', *paths)
    if a.suites:
        if str(a.root) not in sys.path:
            sys.path.insert(0,str(a.root))
        from experiments.libero_plus import suite_report
        consolidated=suite_report.build(a.root)
        print('Saved consolidated:', *suite_report.write(a.root,consolidated,a.refresh))


if __name__=='__main__':
    main()
