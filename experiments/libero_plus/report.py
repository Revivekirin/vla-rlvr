"""Consolidate existing Goal logs and count LIBERO-Plus episode records."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re

from evaluate import ROOT, MODELS, REVISION, checkpoints


def parse_nominal(text, model):
    tasks = defaultdict(list)
    current = None
    for line in text.splitlines():
        match = re.search(r'Task: ([a-z][a-z ]+?)\s*$', line) if model == 'sft' else re.search(r'running (\w+)', line)
        if match:
            current = match[1].strip().replace(' ', '_')
        match = re.search(r'Success: (True|False)', line) if model == 'sft' else re.search(r'\bsuccess \[(True|False)\]', line)
        if match:
            if current is None:
                raise ValueError('Episode without task identifier')
            tasks[current].append(match[1] == 'True')
    if len(tasks) != 10 or any(len(v) != 50 for v in tasks.values()):
        raise ValueError(f'{model}: expected 10 completed tasks of 50 episodes')
    successes = sum(sum(v) for v in tasks.values())
    if model == 'sft':
        assert re.search(r'Total episodes: 500\b', text)
        assert re.search(rf'Total successes: {successes}\b', text)
    else:
        assert "'rollout_count': 500" in text
        assert f"'overall_success_rate': {successes / 500}" in text
    return tasks


def nominal(root):
    rows = []
    for model in MODELS:
        source = f'stage0/evidence/eval-goal-{model}.log'
        path = root / source
        row = dict(model=model, checkpoint=json.dumps(checkpoints(model), sort_keys=True),
                   evaluation_seed='', total_episodes='', successful_episodes='', success_rate='',
                   per_task_success_rates='', source_evidence_path=source, seed_source='')
        if path.exists():
            text = path.read_text()
            tasks = parse_nominal(text, model)
            successes = sum(map(sum, tasks.values()))
            row.update(total_episodes=500, successful_episodes=successes, success_rate=successes / 500,
                       per_task_success_rates=json.dumps({k: {'successes': sum(v), 'episodes': len(v),
                               'success_rate': sum(v) / len(v)} for k, v in tasks.items()}, sort_keys=True))
            seed = re.search(r'Rank 0 setting seed to (\d+)', text)
            row.update(evaluation_seed=int(seed[1]) if seed else 1,
                       seed_source='log' if seed else 'stage0/scripts/run.py --seed 1; not printed in SFT log')
        rows.append(row)
    return rows


def aggregate(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['revision'] == REVISION and manifest['suite'] == 'libero_goal'
    assert manifest['model'] in MODELS
    entries = {e['id']: e for e in manifest['tasks']}
    rows = {}
    for path in sorted(directory.glob('task-*.json')):
        row = json.loads(path.read_text())
        task_id = row['task_id']
        if task_id in rows or task_id not in entries:
            raise ValueError(f'Duplicate or unknown task: {path}')
        entry = entries[task_id]
        for field in ('model', 'seed', 'revision', 'suite'):
            if row[field] != manifest[field]:
                raise ValueError(f'{path}: mismatched {field}')
        if row['task_name'] != entry['name'] or row['shift'] != entry['category']:
            raise ValueError(f'{path}: task mapping mismatch')
        if type(row['success']) is not bool or type(row['failure']) is not bool or row['failure'] == row['success']:
            raise ValueError(f'{path}: invalid outcome')
        rows[task_id] = row
    return manifest, rows


def comparable(left, right):
    # Do not silently compare distinct seeds, task subsets or evaluation conditions.
    lm, lr = left
    rm, rr = right
    ignored = {'model', 'checkpoints'}
    return ({k: v for k, v in lm.items() if k not in ignored} ==
            {k: v for k, v in rm.items() if k not in ignored} and
            set(lr) == set(rr) and bool(lr))


def rate(rows):
    return sum(r['success'] for r in rows) / len(rows) if rows else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--results', type=Path, default=ROOT / 'outputs/libero-plus')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--report', type=Path, default=ROOT / 'experiments/libero_plus/comparison.md')
    args = parser.parse_args()
    nominal_rows = nominal(args.root)
    folder = args.root / 'experiments/nominal'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'goal.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(nominal_rows[0]))
        writer.writeheader()
        writer.writerows(nominal_rows)
    lines = ['# Goal nominal / LIBERO-Plus comparison', '',
             'Goal checkpoints only; episode-weighted rates. Missing OOD data remain unavailable.',
             'Drop = nominal SR − OOD SR (percentage points); shown only for complete OOD runs.', '',
             '| Model | Nominal successes / episodes | Nominal SR | OOD successes / completed (expected) | OOD SR | Drop (pp) |',
             '|---|---:|---:|---:|---:|---:|']
    loaded = {}
    for row in nominal_rows:
        model = row['model']
        path = args.results / model / f'seed-{args.seed}'
        nominal_rate = row['success_rate']
        nominal_count = f"{row['successful_episodes']}/{row['total_episodes']}" if nominal_rate != '' else 'unavailable'
        nominal_text = f'{nominal_rate:.1%}' if nominal_rate != '' else 'unavailable'
        ood_count = ood_text = drop = 'unavailable'
        if (path / 'manifest.json').exists():
            manifest, results = loaded[model] = aggregate(path)
            assert manifest['model'] == model and manifest['seed'] == args.seed
            values = list(results.values())
            ood = rate(values)
            ood_count = f"{sum(r['success'] for r in values)}/{len(values)} ({len(manifest['tasks'])})"
            if ood is not None:
                ood_text = f'{ood:.2%}'
                if len(values) == len(manifest['tasks']) and nominal_rate != '':
                    drop = f'{100 * (nominal_rate - ood):+.2f}'
        lines.append(f'| {model} | {nominal_count} | {nominal_text} | {ood_count} | {ood_text} | {drop} |')
    lines += ['', '| Model | Shift | Successes / completed (expected) | SR |', '|---|---|---:|---:|']
    for model, (manifest, results) in loaded.items():
        for shift in sorted({e['category'] for e in manifest['tasks']}):
            values = [r for r in results.values() if r['shift'] == shift]
            expected = sum(e['category'] == shift for e in manifest['tasks'])
            sr = f'{rate(values):.2%}' if values else 'unavailable'
            lines.append(f"| {model} | {shift} | {sum(r['success'] for r in values)}/{len(values)} ({expected}) | {sr} |")
    if not loaded:
        lines.append('| — | All seven shifts | unavailable | unavailable |')
    lines += ['', '| Comparison (second − first) | Nominal difference (pp) | OOD difference (pp) |', '|---|---:|---:|']
    rates = {r['model']: r['success_rate'] for r in nominal_rows}
    for first, second in [('sft', 'scale'), ('scale', 'ript'), ('sft', 'ript')]:
        delta = f'{100 * (rates[second] - rates[first]):+.2f}' if all(rates[k] != '' for k in (first, second)) else 'unavailable'
        ood_delta = 'unavailable / unmatched coverage'
        if first in loaded and second in loaded and comparable(loaded[first], loaded[second]):
            ood_delta = f'{100 * (rate(list(loaded[second][1].values())) - rate(list(loaded[first][1].values()))):+.2f}'
        lines.append(f'| {first} → {second} | {delta} | {ood_delta} |')
    lines += ['', 'Nominal source: `experiments/nominal/goal.csv` (per-task counts and evidence paths).',
              'No nominal/OOD task pairing is assumed. Nominal and OOD use different task populations.',
              'Nominal SFT uses OFT single-sample inference; Scale/RIPT use RIPT cached batch inference.',
              'OOD shares one environment loop but retains those model inference paths; numerical equivalence is unverified.',
              'The scale head is loaded but does not change mean actions when sampling is disabled.', '']
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text('\n'.join(lines))
    print(args.report)


if __name__ == '__main__':
    main()
