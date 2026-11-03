"""Bounded nominal evaluation summaries and explicit expansion gates."""
import math
from pathlib import Path

from .common import read, require, sha, write


def summarize(out, task_ids, trials, identity):
    out = Path(out)
    root = out/'goal/sft'/f'seed-{identity["seed"]}'
    files = sorted(root.glob('task-*-trial-*.json'))
    rows = [read(p) for p in files]
    expected = {(t,i) for t in task_ids for i in range(trials)}
    require(len(rows) == len(expected) and {(r['task_id'],r['trial_id']) for r in rows} == expected,
            'Evaluation incomplete or duplicate trial identities')
    require(all(type(r['success']) is bool and 0 < r['steps'] <= 300 and
                math.isfinite(r.get('max_abs_executed_action', float('nan'))) for r in rows),
            'Invalid evaluation result/action statistics')
    summary = dict(identity, status='complete', episodes=len(rows), expected_episodes=len(expected),
        successes=sum(r['success'] for r in rows), trials_per_task=trials,
        per_task={str(t):dict(episodes=sum(r['task_id']==t for r in rows),
                             successes=sum(r['success'] for r in rows if r['task_id']==t)) for t in task_ids},
        result_files={str(p.relative_to(out)):sha(p) for p in files},
        diagnostic_interpretation='Nominal diagnostic/model selection; overlapping initial states in expanded evaluation')
    write(out/'summary.json', summary)
    return summary


def check_gate(path, kind, checkpoint_identity, evaluator_sha, wrapper_sha):
    path = Path(path)
    s = read(path)
    require(s['status'] == 'complete' and s['episodes'] == s['expected_episodes'], 'Gate evaluation incomplete')
    require(s['kind'] == kind and s['evaluator_sha256'] == evaluator_sha and s['wrapper_sha256'] == wrapper_sha,
            'Gate evaluator/model kind differs')
    if checkpoint_identity is not None:
        require(s['checkpoint_identity'] == checkpoint_identity, 'Gate checkpoint differs')
    require(s['successes'] > 0, 'Zero-success diagnostic: inspect loss/actions/loading before expanding')
    require(set(s['per_task']) == {str(i) for i in range(1,11)} and
            all(v['episodes'] > 0 for v in s['per_task'].values()), 'Gate must cover every Goal task')
    require(all((path.parent/p).is_file() and sha(path.parent/p) == h for p,h in s['result_files'].items()),
            'Gate result files changed')
    return dict(path=str(path.resolve()), sha256=sha(path), successes=s['successes'], episodes=s['episodes'])
