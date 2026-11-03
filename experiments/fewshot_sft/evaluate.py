"""Isolated Goal SFT entrypoint reusing the validated original nominal rollout loop."""
import argparse
import json
from pathlib import Path

from .common import NAME, OUT, ROOT, destination, require, sha, write
from .checkpoint import check


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--phase', choices=['smoke', 'pilot', 'final'], default='pilot')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--selection-note', required=True, help='Record checkpoint choice and any previous nominal tuning')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    m = check(a.checkpoint)
    out = destination(OUT/f'eval/goal_{m["shots"]}shot_seed{m["selection_seed"]}'/a.checkpoint.name/a.phase)
    require(not out.exists(), 'Evaluation phase output already exists; no overwrites')
    trials = {'smoke': 1, 'pilot': 20, 'final': 50}[a.phase]
    print(json.dumps(dict(output=str(out), tasks=1 if a.phase=='smoke' else 10,
                         trials_per_task=trials, checkpoint_step=m['step'], phase=a.phase)))
    if not a.execute or a.dry_run:
        return
    from experiments.libero_plus import suite_common as nominal
    # Process-local extension. No shared evaluator source or full-data path changes.
    nominal.SUITES = dict(nominal.SUITES, goal=('libero_goal', NAME, 300, 0))
    nominal.checkpoint_paths = lambda suite, model: {'base': str(a.checkpoint.resolve())}
    nominal.checkpoint_check = lambda suite, model: dict(ready=True, missing=[], metadata=check(a.checkpoint))
    benchmark = ROOT/'vendor/LIBERO'
    preflight = nominal.preflight('goal', 'sft', 'nominal', benchmark)
    require(preflight['ready'], str(preflight['errors']))
    write(out/'selection.json', dict(phase=a.phase, note=a.selection_note, checkpoint=str(a.checkpoint.resolve()),
        checkpoint_step=m['step'], checkpoint_metadata_sha256=sha(a.checkpoint/'fewshot_checkpoint.json'),
        evaluator_sha256=sha(Path(nominal.__file__)), wrapper_sha256=sha(__file__),
        trials_per_task=trials, seed=a.seed, nominal_model_selection_recorded=True))
    nominal.configure('nominal', benchmark)
    args = argparse.Namespace(suite='goal', model='sft', seed=a.seed, benchmark=benchmark, output=out,
        task_ids=None, max_tasks=1 if a.phase=='smoke' else None, max_trials=trials, check_import=False)
    nominal.run(args, 'nominal')


if __name__ == '__main__':
    main()
