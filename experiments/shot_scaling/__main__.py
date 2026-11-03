"""One CPU-validated CLI; unavailable GPU/data integration stages fail explicitly."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from .core import ROOT, config, read, require, run_dir, runtime, sha, write

BLOCKERS = {
    'sft-train': 'New nested manifests are not materialized RLDS. Existing trainer/checkpoint validation is Goal 1/5-shot only. No validated all-suite integration; no local data, base weights, or CUDA smoke.',
    'sft-eval': 'No validated new SFT export. Existing fewshot evaluator supports Goal full-sequence inference only; run its export verification and bounded smoke first.',
    'scale-train': 'No scale-head NLL fitting entrypoint found in pinned runtime. Released scale headers cannot initialize a newly trained SFT condition. Exact-SFT cached feature/loss validation is required.',
    'ript-train': 'Rollout samples Laplace(exp(log_b)*scale_factor) without clipping, but rollout/PPO log probabilities use exp(clamp(log_b)) without scale_factor. Suite scale_factor=5.0. Also missing exact-SFT scale fit, verified LoRA merge and D0 cached snapshot. Execution disabled.',
    'eval': 'Unified arbitrary-checkpoint nominal/PLUS binding and resumable protocol integration are not validated. Existing evaluators use released checkpoint paths; they cannot be labeled new D0/D1.',
}


def inspect(c):
    sources = ['experiments/fewshot_sft/prepare_dataset.py', 'experiments/fewshot_sft/train.py',
               'experiments/fewshot_sft/checkpoint.py', 'experiments/fewshot_sft/evaluate.py',
               'experiments/nominal/evaluate.py', 'experiments/libero_plus/suite_common.py',
               'runtime/openvla-oft/vla-scripts/finetune.py',
               'runtime/ript-vla/ript/env_runner/openvla_oft_libero_runner.py',
               'runtime/ript-vla/ript/algos/rl_optimizers/rl_optimizer_openvla_oft.py',
               'runtime/ript-vla/ript/algos/rl_optimizers/openvla_oft_interface.py']
    return dict(status='CPU_FOUNDATION_ONLY', config=c, runtime=runtime(), blockers=BLOCKERS,
        source_hashes={s: sha(ROOT/s) if (ROOT/s).is_file() else None for s in sources},
        declared_revisions=read(ROOT/'stage0/repos.lock.json'),
        matrix=[dict(suite=s, shots=k, cpu_contracts='READY', dataset_materialization='BLOCKED',
                     sft='BLOCKED', scale='BLOCKED', d0='BLOCKED', ript='BLOCKED', evaluation='BLOCKED')
                for s in ('goal', 'spatial', 'long') for k in (1, 2, 5, 10, 20, 'Full')],
        historical=dict(status='PRESERVED', interpretation='released-checkpoint comparisons except separately identified fewshot runs',
                        goal5_seed42='Existing selection/checkpoint untouched; not equivalent to nested-sha256-v1',
                        evidence=sorted(str(p.relative_to(ROOT)) for p in (ROOT/'outputs').glob('**/selection_manifest.json'))))


def historical_sft_command(c):
    """Print only; use the existing parser to ensure flags are real and bounds valid."""
    from experiments.fewshot_sft.train import parser, plan
    require(c['suite']['name'] == 'goal' and c['shots'] in (1, 5), 'Legacy SFT only supports Goal 1/5')
    s = c['sft']
    flags = ['--suite', 'goal', '--shots', str(c['shots']), '--selection-seed', str(c['selection_seed']),
             '--train-seed', str(c['train_seed']), '--batch-size', str(s['batch_size']),
             '--gradient-accumulation', str(s['accumulation']), '--learning-rate', str(s['learning_rate']),
             '--max-steps', str(s['max_steps']), '--save-freq', str(s['save_freq']),
             '--run-id', 'goal5-historical-bounded', '--dry-run']
    plan(parser().parse_args(flags))
    return shlex.join([sys.executable, '-m', 'experiments.fewshot_sft.train', *flags])


def commands():
    return '''# Goal raw-RLDS extraction smoke (read-only; no complete inventory emitted).
python -m experiments.shot_scaling dataset-analyze --source data/data/modified_libero_rlds/libero_goal_no_noops/1.0.0 --source-revision a7c9ae18499b6eea8a32f78a9302327b752b1b5f --selection outputs/fewshot_sft/datasets/goal_5shot_seed42/selection_manifest.json --max-episodes 2 suite=goal shots=5 selection_seed=42
# After smoke passes, remove --max-episodes 2 to extract the full source and analyze historical Goal5.
# Alternatively, use existing extracted JSON artifacts:
python -m experiments.shot_scaling dataset-analyze --inventory "$GOAL_INVENTORY" --trajectories "$GOAL_TRAJECTORIES" --selection "$HISTORICAL_GOAL5_MANIFEST" suite=goal
for shot in 1 2 5 10 20 Full; do
  python -m experiments.shot_scaling prepare --inventory "$GOAL_INVENTORY" suite=goal shots=$shot selection_seed=42
done
# Readiness only: these return exit 2 until integration blockers are resolved.
python -m experiments.shot_scaling sft-train --check suite=goal shots=5 sft.max_steps=1
python -m experiments.shot_scaling sft-eval --check suite=goal evaluation.max_tasks=1 evaluation.max_trials=1
python -m experiments.shot_scaling dataset-analyze --inventory "$SPATIAL_INVENTORY" --trajectories "$SPATIAL_TRAJECTORIES" suite=spatial
python -m experiments.shot_scaling dataset-analyze --inventory "$LONG_INVENTORY" --trajectories "$LONG_TRAJECTORIES" suite=long
python -m experiments.shot_scaling scale-train --check
python -m experiments.shot_scaling ript-train --check
python -m experiments.shot_scaling aggregate --historical
python -m experiments.shot_scaling aggregate --input "$EVALUATION_BUNDLES"
python -m experiments.shot_scaling plot --input "$SAVED_ANALYSIS"
'''


def input_file(value):
    if not value.strip():
        raise argparse.ArgumentTypeError('Empty file path: set the referenced environment variable to an existing JSON file')
    path = Path(value).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f'Expected an existing file, not a directory or missing path: {path}')
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['inspect', 'prepare', 'dataset-analyze', 'sft-train', 'sft-eval',
                                      'scale-train', 'ript-train', 'eval', 'aggregate', 'plot'])
    p.add_argument('--inventory', type=input_file)
    p.add_argument('--trajectories', type=input_file)
    p.add_argument('--selection', type=input_file)
    p.add_argument('--source', help='Original Goal TFDS version directory; read-only CPU extraction')
    p.add_argument('--source-revision', help='Explicit pinned HF dataset commit for --source')
    p.add_argument('--max-episodes', type=int, help='Bounded extraction smoke only; does not emit analysis/inventory')
    p.add_argument('--input', type=input_file)
    p.add_argument('--resume', type=input_file, help='Verify an existing selection file exactly; never overwrite')
    p.add_argument('--historical', action='store_true', help='Reuse existing suite report, preserving released provenance')
    p.add_argument('--check', action='store_true', help='Readiness only; blocked stages exit 2')
    p.add_argument('--commands', action='store_true')
    p.add_argument('overrides', nargs='*', help='Hydra overrides, e.g. suite=long shots=Full')
    a = p.parse_intermixed_args(argv)
    c = config(a.overrides)
    require(a.source is None or a.command == 'dataset-analyze', '--source belongs to dataset-analyze')
    require(a.source is not None or (a.source_revision is None and a.max_episodes is None),
            '--source-revision/--max-episodes require --source')
    if a.commands:
        require(a.command == 'inspect', '--commands belongs to inspect')
        print(commands())
        return 0
    if a.command in BLOCKERS:
        result = dict(status='BLOCKED', command=a.command, reason=BLOCKERS[a.command], config=c)
        if a.command == 'sft-train' and c['suite']['name'] == 'goal' and c['shots'] in (1, 5):
            result['historical_dry_run_only'] = historical_sft_command(c)
        print(json.dumps(result, indent=2))
        return 2
    require(not a.check, '--check only applies to training/evaluation readiness stages')
    if a.command == 'inspect':
        result = inspect(c)
        out = run_dir(c, a.command, [])
        write(out/'audit.json', result)
    elif a.command == 'prepare':
        from .data import select
        require(a.inventory is not None, 'BLOCKED: --inventory required; raw RLDS extraction/materialization is not integrated')
        result = select(read(a.inventory), c['suite']['name'], c['shots'], c['selection_seed'])
        if a.resume:
            require(read(a.resume) == result, 'Resume content/config mismatch; existing evidence preserved')
            print(f'Verified existing manifest only: {a.resume}')
            return 0
        out = run_dir(c, a.command, [a.inventory])
        write(out/'selection.json', result)
    elif a.command == 'dataset-analyze':
        from .analysis import analyze, save
        selection = read(a.selection) if a.selection else None
        if selection:
            require(selection['shots'] == c['shots'] and selection['selection_seed'] == c['selection_seed'],
                    'Configured selection mismatch')
        if a.source is not None:
            from .extract import extract
            require(a.source.strip(), 'Empty --source path')
            require(a.inventory is None and a.trajectories is None, 'Use --source or artifact inputs, not both')
            require(c['suite']['name'] == 'goal', 'Raw extraction currently supports Goal only')
            inventory, trajectories = extract(a.source, a.source_revision, selection, a.max_episodes)
            out = run_dir(c, a.command, [Path(a.source)/'dataset_info.json']+([a.selection] if a.selection else []))
            if a.max_episodes is not None:
                write(out/'extraction_smoke.json', dict(inventory, source=str(Path(a.source).resolve()),
                                                     revision=a.source_revision))
                write(out/'completion.json', dict(status='COMPLETE', scope='Bounded extraction smoke only'))
                print(out)
                return 0
            write(out/'inventory.json', inventory)
            write(out/'trajectories.json', trajectories)
        else:
            require(a.inventory is not None and a.trajectories is not None,
                    'Supply --source with --source-revision, or --inventory and --trajectories')
            inventory, trajectories = read(a.inventory), read(a.trajectories)
            require(inventory['suite'] == c['suite']['name'], 'Configured suite mismatch')
            out = run_dir(c, a.command, [a.inventory, a.trajectories]+([a.selection] if a.selection else []))
        result = analyze(inventory, trajectories, selection)
        save(out, result)
    elif a.command == 'aggregate':
        if a.historical:
            require(a.input is None, 'Historical and new results cannot be mixed')
            # Legacy report modules use script-relative imports. Isolate those imports;
            # call the existing read-only builder, never its report-writing main().
            code = ('import sys,json; sys.path.insert(0, sys.argv[1]); '
                    'from experiments.libero_plus.suite_report import build; '
                    'print(json.dumps(build(), allow_nan=False))')
            built = subprocess.run([sys.executable, '-c', code, str(ROOT/'experiments/libero_plus')],
                                   cwd=ROOT, text=True, capture_output=True)
            require(built.returncode == 0, 'Historical report validation failed: '+built.stderr)
            result = dict(schema=1, kind='historical_aggregate',
                comparison='released-checkpoint comparison; NOT directly trained D0/D1', results=json.loads(built.stdout))
            inputs = [ROOT/'outputs/rlvr_analysis/research_summary.json']
        else:
            from .reporting import aggregate
            require(a.input is not None, '--input episode-bundle JSON required')
            result, inputs = aggregate(read(a.input)), [a.input]
        out = run_dir(c, a.command, inputs)
        write(out/'aggregate.json', result)
    else:
        from .plotting import plot
        require(a.input is not None, '--input saved artifact required')
        out = run_dir(c, a.command, [a.input])
        plot(read(a.input), out/'figures')
    write(out/'completion.json', dict(status='COMPLETE', scope='CPU artifact operation only'))
    print(out)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print(f'BLOCKED: {exc}', file=sys.stderr)
        sys.exit(2)
