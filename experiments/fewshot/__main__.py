"""Stage 2 only: plan, bounded smoke, materialize, finish."""
import argparse
from pathlib import Path

from experiments.shot_scaling.core import SUITES, digest, read, require, write
from .stage2 import OUT, SHOTS, plan, plot_task, selection_path, checked_baseline
from .materialize import materialize


def finish(seeds):
    audits = checked_baseline()
    repro = read(OUT / 'analysis' / 'figure_reproducibility.json')
    require(all((OUT / 'analysis' / name).is_file() for name in repro['figures']),
            'Consolidated figures missing')
    require(repro['source_audit_sha256'] == {suite: digest(audit['episodes']) for suite, audit in audits.items()},
            'Figure source data changed')
    require((OUT / 'analysis' / 'smoke_validation.json').is_file(), 'Missing bounded smoke result')
    smoke = read(OUT / 'analysis' / 'smoke_validation.json')
    require(smoke['status'] == 'PASS', 'Bounded smoke did not pass')
    records = []
    for suite in SUITES:
        for seed in seeds:
            for shots in SHOTS:
                path = selection_path(suite, seed, shots)
                s = read(path / 'selection_manifest.json')
                if seed == min(seeds):
                    require(repro['selection_sha256'][f'{suite}/seed{seed}/{shots}'] == s['selection_sha256'],
                            'Figure selection changed')
                v = read(path / 'validation.json')
                require(v['status'] == 'PASS' and v['selection_sha256'] == s['selection_sha256']
                        and v['episode_count'] == len(s['episodes'])
                        and v['transition_count'] == s['selected_transitions']
                        and v['dataset_payload_sha256'] == digest(sorted(r['sha256'] for r in s['episodes'])),
                        f'Incomplete Stage 2 condition: {path}')
                records.append(dict(suite=suite, seed=seed, shots=shots,
                                    episodes=v['episode_count'], transitions=v['transition_count'],
                                    dataset_root=v['rlds_root'], dataset_name=v['dataset_name'],
                                    materialization=v['materialization'], normalization_sha256=v['normalization_sha256'],
                                    dataset_payload_sha256=v['dataset_payload_sha256']))
    inventory = dict(status='PASS', conditions=records, smoke=smoke,
                     historical_goal5=read(OUT/'analysis'/'selection_analysis.json')['historical_goal5'])
    p = OUT / 'analysis' / 'dataset_inventory.json'
    if p.exists():
        require(read(p) == inventory, 'Existing dataset inventory differs')
    else:
        write(p, inventory)
    report = ['# Stage 2 — PASS', '',
              f'{len(records)} nested random suite/seed/shot conditions validated from pinned full expert episodes.',
              'Selections use whole episodes and per-task deterministic SHA order; no success or path-coverage optimization.',
              'Path diversity is descriptive only. Full reuses the original loadable RLDS without copying it.',
              '', 'The original actions, images, state, language, boundaries and terminal flags were checked by full payload hashes after TFDS loading.',
              'Normalization is recomputed for each selected condition after the official LIBERO gripper transform; no historical statistics are reused.',
              'The official OFT loader reads each condition with its verified statistics and yields an 8x7 action chunk with images and proprio.',
              'Its window=1/future=7 transform excludes the last seven chunk starts. RLDS retains those terminal actions; the loader does not use padded terminal training frames.',
              '', '## Outputs', '',
              '- Selection manifests and RLDS: `outputs/fewshot/datasets/{suite}/seed{seed}/{shots}shot/` (or `Full/`).',
              '- Dataset inventory, validation summary and four figures: `outputs/fewshot/analysis/`.',
              '', '## Limitations', '',
              '- Physical XYZ semantics follow documented source mapping; exact producer commit/bitwise HDF5 identity are not claimed.',
              '- Training has not started. Subset OFT statistics caches are pinned to the measured data. Full uses the original RLDS and requires Stage 3 to inject its verified normalization.json.',
              '- Historical Goal 5-shot: '+inventory['historical_goal5']+'.',
              '', '## Stage 3 starter', '',
              f'Example command for Goal seed {min(seeds)} / 5-shot; execute only in Stage 3 after choosing the training schedule:',
              '', '```bash',
              'cd /nfs4/jhkim/repos/vla-rlvr/runtime/openvla-oft',
              'WANDB_MODE=disabled torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \\',
              '  --vla_path openvla/openvla-7b \\',
              f'  --data_root_dir /nfs4/jhkim/repos/vla-rlvr/outputs/fewshot/datasets/goal/seed{min(seeds)}/5shot/rlds \\',
              '  --dataset_name libero_goal_no_noops \\',
              '  --run_root_dir /nfs4/jhkim/repos/vla-rlvr/outputs/stage3_sft \\',
              '  --use_l1_regression True --use_diffusion False --use_film False \\',
              '  --num_images_in_input 2 --use_proprio True --batch_size 1 \\',
              '  --learning_rate 5e-4 --max_steps 1000 --save_freq 1000 \\',
              f'  --image_aug True --lora_rank 32 --run_id_note goal_seed{min(seeds)}_5shot',
              '```']
    (OUT / 'analysis' / 'report.md').write_text('\n'.join(report)+'\n')
    return inventory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('plan', help='Write nested selection manifests and four descriptive figures')
    a.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    a.add_argument('--historical-goal5', type=Path)
    for command in ('smoke', 'materialize'):
        a = sub.add_parser(command)
        a.add_argument('--suite', choices=SUITES, required=True)
        a.add_argument('--seed', type=int, required=True)
        a.add_argument('--shots', choices=['1','2','5','10','20','Full'], required=True)
    a = sub.add_parser('finish')
    a.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    a = sub.add_parser('plot-task')
    a.add_argument('--suite', choices=SUITES, required=True)
    a.add_argument('--task-index', type=int, required=True)
    a.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.command == 'plan':
        x = plan(args.seeds, args.historical_goal5)
        print(f'Planned {x["conditions"]} conditions; four figures in {OUT / "analysis"}')
    elif args.command in ('smoke', 'materialize'):
        x = materialize(args.suite, args.seed,
                        args.shots if args.shots == 'Full' else int(args.shots),
                        smoke=args.command == 'smoke')
        print(f'PASS {x["suite"]} seed{x["seed"]} {x["shots"]}: {x["episode_count"]} episodes, {x["transition_count"]} transitions')
    elif args.command == 'plot-task':
        print(plot_task(args.suite, args.task_index, args.seed))
    else:
        x = finish(args.seeds)
        print(f'PASS {len(x["conditions"])} conditions: {OUT / "analysis" / "report.md"}')


if __name__ == '__main__':
    main()
