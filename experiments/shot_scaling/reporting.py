"""Read saved results only. No inference and no independent rollout implementation."""
from collections import defaultdict
import math
import statistics
from .core import SUITES, compatible, digest, require


def aggregate(runs):
    """Validated episode bundles; expected keys come from the bounded evaluator manifest."""
    groups, seen, summaries = defaultdict(list), set(), []
    require(bool(runs), 'No evaluation runs supplied')
    synthetic = {bool(r.get('synthetic', False)) for r in runs}
    require(len(synthetic) == 1, 'Synthetic and experimental results cannot be combined')
    for run in runs:
        m, rows = run['manifest'], run['episodes']
        require(m['suite'] in SUITES, 'Unknown suite')
        require(m['condition_sha256'] == digest(m['condition']), 'Training condition digest mismatch')
        require(len(m['checkpoint_sha256']) == 64 and all(c in '0123456789abcdef'
                for c in m['checkpoint_sha256']), 'Invalid checkpoint digest')
        key = (m['suite'], str(m['shots']), m['selection_seed'], m['train_seed'], m['eval_seed'],
               m['checkpoint_sha256'], m['inference'], m['dataset'], m['protocol_sha256'])
        require(key not in seen, 'Duplicate run; refusing pseudoreplication')
        seen.add(key)
        expected = [tuple(x) for x in m['expected_episode_keys']]
        actual = [(r['task_id'], r['condition_id'], r['trial_id']) for r in rows]
        require(expected and len(expected) == len(set(expected)), 'Empty/duplicate protocol episode IDs')
        require(len(actual) == len(set(actual)) and set(actual) <= set(expected), 'Duplicate/unexpected evaluated episode')
        require(all(type(r['success']) is bool and r['seed'] == m['eval_seed'] for r in rows), 'Invalid outcome/seed')
        require(m['dataset'] in ('nominal', 'plus') and m['inference'] in
                ('sft_full', 'd0_cached', 'd1_cached'), 'Unknown evaluation mode')
        require(m['protocol_sha256'] == digest(m['protocol']), 'Protocol digest mismatch')
        require(m['protocol']['expected_episode_keys'] == m['expected_episode_keys'], 'Protocol coverage mismatch')
        status = 'COMPLETE' if set(actual) == set(expected) else 'INCOMPLETE'
        rate = sum(r['success'] for r in rows)/len(rows) if rows else None
        summaries.append(dict(identity=list(key), status=status, episodes=len(rows), expected=len(expected), rate=rate))
        # Keep training seeds distinct from evaluation seeds. Checkpoint step/protocol must match.
        group = (m['suite'], str(m['shots']), m['selection_seed'], m['inference'], m['dataset'],
                 m['protocol_sha256'], m['optimizer_step'], m['condition_sha256'])
        if status == 'COMPLETE':
            groups[group].append((m['train_seed'], m['eval_seed'], rate))
    result = []
    for key, values in groups.items():
        train = defaultdict(list)
        pairs = [(t, e) for t, e, _ in values]
        require(len(pairs) == len(set(pairs)), 'Multiple checkpoints for the same seed/step')
        for t, e, r in values:
            train[t].append(r)
        rates = [statistics.mean(v) for v in train.values()]
        result.append(dict(group=list(key), training_seeds=len(rates), evaluation_runs=len(values),
            rate=statistics.mean(rates), training_seed_sem=statistics.stdev(rates)/math.sqrt(len(rates)) if len(rates)>1 else None))
    return dict(schema=1, kind='evaluation_aggregate', synthetic=synthetic.pop(), runs=summaries, groups=result,
                note='Incomplete runs excluded from seed means. SEM is across training-seed means, not episodes.')


def paired(d0, d1):
    compatible(d0['checkpoint'], d1['checkpoint'])
    a, b = d0['manifest'], d1['manifest']
    for key in ('suite', 'shots', 'selection_seed', 'train_seed', 'eval_seed', 'dataset',
                'protocol_sha256', 'expected_episode_keys', 'condition_sha256'):
        require(a[key] == b[key], f'Unmatched evaluation: {key}')
    require(a['inference'] == 'd0_cached' and b['inference'] == 'd1_cached', 'Full-vs-cached is not an RL effect')
    for run in (d0, d1):
        require(run['manifest']['checkpoint_sha256'] == digest(run['checkpoint']), 'Checkpoint identity mismatch')
        require(aggregate([run])['runs'][0]['status'] == 'COMPLETE', 'Pair requires complete bounded protocol')
    key = lambda r: (r['task_id'], r['condition_id'], r['trial_id'])
    left = {key(r): r for r in d0['episodes']}
    delta = defaultdict(list)
    for r in d1['episodes']:
        old = left[key(r)]
        require(old['shift'] == r['shift'], 'Shift mismatch')
        delta[r['shift']].append(int(r['success'])-int(old['success']))
    return {shift: dict(episodes=len(v), difference_pp=100*statistics.mean(v)) for shift, v in delta.items()}
