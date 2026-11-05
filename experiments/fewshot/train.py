"""Run one independent Goal SFT condition from the verified Stage 2 RLDS."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from experiments.fewshot_sft.common import (
    BASE, OFT, OUT, ROOT, destination, inventory, read, require, sha, write,
)
from experiments.fewshot_sft.checkpoint import check


MILESTONES = (200, 500, 1000, 3000, 6000, 12000, 24000, 52000)
RUN_BOUNDS = (1000, 3000, 6000, 12000, 24000, 52000)
GLOBAL64_MILESTONES = (1000, 5000, 10000, 25000, 50000)
PILOT_MILESTONES = (1000, 6000, 12000, 24000, 52000)
ARCHITECTURE = dict(
    use_l1_regression=True, use_diffusion=False, use_film=False,
    num_images_in_input=2, use_proprio=True, use_lora=True, lora_rank=32,
    action_chunk=8, action_dim=7, proprio_dim=8, image_aug=True,
)
SHOTS = (1, 2, 5, 10, 20, 'Full')


def checkpoint_schedule(profile, max_steps, requested=None):
    milestones = (GLOBAL64_MILESTONES if profile == 'single_gpu_global64' else
                  PILOT_MILESTONES if profile == 'single_gpu_batch1_pilot' else MILESTONES)
    if requested is None:
        return sorted({step for step in milestones if step <= max_steps} | {max_steps})
    require(bool(requested) and all(type(step) is int and 0 < step <= max_steps for step in requested),
            'Checkpoint steps must be positive optimizer steps within max_steps')
    require(len(requested) == len(set(requested)) and max_steps in requested,
            'Checkpoint steps must be unique and include final max_steps')
    return sorted(requested)


def loss_trend(run, first_step, last_step, milestones):
    rows = [json.loads(line) for line in (run / 'diagnostics.jsonl').read_text().splitlines()]
    require([row['step'] for row in rows] == list(range(first_step + 1, last_step + 1)),
            'Missing or duplicate optimizer-step diagnostics')
    losses = [row['normalized_chunk_l1'] for row in rows]
    require(all(math.isfinite(loss) and loss >= 0 for loss in losses), 'Invalid training loss trend')
    width = min(100, len(losses))
    return dict(first_step=first_step, last_step=last_step,
                first_window_mean=sum(losses[:width]) / width,
                last_window_mean=sum(losses[-width:]) / width,
                checkpoints={str(step): rows[step - first_step - 1]['moving_mean']
                             for step in milestones},
                final_task_frame_draws=rows[-1]['task_exposure'],
                interpretation='Descriptive training loss; nominal rollout success is required to assess function.')


def configuration(suite, seed, shots, max_steps, run_id, resume=None,
                  profile='legacy_batch1', checkpoint_steps=None, train_seed=7):
    require(suite == 'goal' and shots in SHOTS, 'Stage 3 supports Goal 1/2/5/10/20/Full')
    require(profile in ('legacy_batch1', 'single_gpu_batch1_pilot', 'single_gpu_global64'),
            'Unknown training profile')
    require(profile != 'legacy_batch1' or shots == 'Full',
            'Legacy profile preserves the historical Goal Full run only')
    require(type(seed) is int and seed >= 0 and type(train_seed) is int and train_seed >= 0,
            'Invalid selection or training seed')
    valid_bound = (max_steps in RUN_BOUNDS if profile == 'legacy_batch1' else
                   max_steps <= 52000 if profile == 'single_gpu_batch1_pilot' else
                   max_steps <= 50000)
    require(type(max_steps) is int and max_steps > 0 and valid_bound,
            'Invalid optimizer-step bound for this profile')
    checkpoints = checkpoint_schedule(profile, max_steps, checkpoint_steps)
    require(run_id and all(ch.isalnum() or ch in '-_' for ch in run_id), 'Unsafe run ID')
    from .train_smoke import preflight
    dataset, validation, stats = preflight(suite, seed, shots)
    output = destination(OUT / 'runs' / run_id)
    require(not output.exists() and not list(output.parent.glob(output.name + '--*')),
            'Run ID or checkpoint prefix already exists')
    from huggingface_hub import snapshot_download
    import torch
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'A CUDA BF16 GPU is required')
    base = Path(snapshot_download(BASE, local_files_only=True)).resolve()
    require(not (base / 'adapter_config.json').exists(), 'Initialization must be the pristine pretrained base')
    smoke_path = None
    if profile == 'single_gpu_global64':
        shot_label = 'Full' if shots == 'Full' else f'{shots}shot'
        candidates = sorted((ROOT / 'outputs/stage3_sft/smoke/goal' / f'seed{seed}').glob(
            f'{shot_label}_batch1_accum64_attempt*/smoke_result.json'))
        for candidate in candidates:
            evidence = read(candidate)
            if (evidence.get('status') == 'PASS' and evidence.get('optimizer_steps') == 1
                    and evidence.get('microbatches') == 64
                    and evidence.get('effective_batch_size') == 64
                    and evidence.get('suite') == suite and evidence.get('shots') == shots
                    and evidence.get('train_seed') == train_seed
                    and evidence.get('selection_sha256') == validation['selection_sha256']
                    and evidence.get('normalization_sha256') == validation['normalization_sha256']
                    and evidence.get('base_snapshot') == str(base)
                    and evidence.get('official_finetune_sha256') == sha(OFT / 'vla-scripts/finetune.py')
                    and evidence.get('worker_shim_sha256') == sha(ROOT / 'experiments/fewshot_sft/worker.py')
                    and evidence.get('smoke_code_sha256') == sha(ROOT / 'experiments/fewshot/train_smoke.py')):
                smoke_path = candidate
        require(smoke_path is not None,
                'Matching one-update global-64 smoke test is required')
    source_paths = [
        'experiments/fewshot_sft/common.py',
        'runtime/openvla-oft/prismatic/vla/datasets/datasets.py',
        'runtime/openvla-oft/prismatic/vla/datasets/rlds/dataset.py',
        'runtime/openvla-oft/prismatic/vla/datasets/rlds/oxe/transforms.py',
        'runtime/openvla-oft/prismatic/vla/constants.py',
        'runtime/openvla-oft/prismatic/extern/hf/modeling_prismatic.py',
        'experiments/fewshot_sft/worker.py',
        'experiments/fewshot_sft/diagnostics.py',
        'experiments/fewshot_sft/resume_checks.py',
        'experiments/fewshot/train_smoke.py',
        'experiments/fewshot/train.py',
    ]
    effective_batch = 64 if profile == 'single_gpu_global64' else 1
    transitions = validation['transition_count']
    config = dict(
        schema=2, stage='stage3_sft', base_identity=BASE, base_snapshot=str(base),
        base_revision=base.name, base_files=inventory(base),
        dataset=str(dataset), rlds_root=validation['rlds_root'],
        dataset_name=validation['dataset_name'], suite=suite, shots=shots,
        selection_seed=seed, source_revision=validation['source_revision'],
        selection_manifest_sha256=sha(dataset / 'selection_manifest.json'),
        selection_sha256=validation['selection_sha256'],
        normalization_sha256=validation['normalization_sha256'],
        episode_count=validation['episode_count'],
        selected_transitions=validation['transition_count'],
        output=str(output), train_seed=train_seed, profile=profile,
        accumulation_smoke=str(smoke_path) if smoke_path else None,
        batch_size=1, gradient_accumulation=effective_batch,
        effective_batch_size=effective_batch,
        world_size=1, learning_rate=5e-4,
        max_steps=max_steps, save_freq=max_steps,
        checkpoint_steps=checkpoints,
        checkpoint_exposure={str(step): dict(frame_draws=step * effective_batch,
                                            nominal_dataset_passes=step * effective_batch / transitions)
                             for step in checkpoints},
        validation=None, validation_freq=max_steps, loss_window=100,
        diagnostics_format='compact' if profile == 'single_gpu_global64' else 'full',
        resume=str(Path(resume).resolve()) if resume else None,
        verify_resume=False, architecture=ARCHITECTURE, merged=False,
        scheduler_config=dict(kind='MultiStepLR', milestones=[100000], gamma=0.1, warmup_steps=0),
        official_reference=dict(gpus=8, batch_per_gpu=8, gradient_accumulation=1,
                                effective_batch=64, learning_rate=5e-4,
                                maximum_steps=150005, goal_reported_best_step=50000,
                                source=str(OFT / 'LIBERO.md'), source_sha256=sha(OFT / 'LIBERO.md')),
        profile_interpretation=(
            'Single-GPU accumulated global batch 64; same sample count per optimizer step '
            'as the official reference, not exact DDP reproduction.'
            if profile == 'single_gpu_global64' else
            'Fresh batch-one low-cost pilot; not an official global-64 reproduction.'
            if profile == 'single_gpu_batch1_pilot' else
            'Historical low-cost batch-one infrastructure/early-training reference.'),
        source_hashes={name: sha(ROOT / name) for name in source_paths},
        official_finetune_sha256=sha(OFT / 'vla-scripts/finetune.py'),
        dataset_exposure=dict(
            frame_draws=max_steps * effective_batch,
            transitions=transitions,
            estimated_repetitions=max_steps * effective_batch / transitions,
            note='Draws count loader samples; augmentation and shuffling change repeats.'),
    )
    if resume:
        resume = Path(resume).resolve()
        previous = check(resume, merged=False)
        from experiments.fewshot_sft.resume_checks import verify_normalization
        verify_normalization(resume, dataset)
        require(previous.get('stage') == 'stage3_sft' and previous['suite'] == suite
                and previous['shots'] == shots and previous['selection_seed'] == seed
                and previous['profile'] == profile,
                'Resume checkpoint belongs to a different Stage 3 condition')
        for field in ('base_revision', 'base_files', 'selection_sha256', 'normalization_sha256',
                      'architecture', 'train_seed', 'batch_size', 'gradient_accumulation',
                      'learning_rate', 'official_finetune_sha256'):
            require(previous[field] == config[field], f'Resume changed {field}')
        require(previous['step'] < max_steps and previous['step'] in previous['checkpoint_steps'],
                'Resume must start from an earlier declared milestone')
        require((resume / 'optimizer.pt').is_file(), 'Resumable optimizer state is missing')
        config.update(resume_step=previous['step'], resume_files=inventory(resume),
                      checkpoint_steps=[s for s in checkpoints if previous['step'] < s <= max_steps],
                      continuity=dict(start_step=previous['step'], tf_iterator_restored=False,
                                      rng_restored=False, bitwise_uninterrupted_equivalence=False,
                                      note='TF shuffle/prefetch restarts from the train seed.'),
                      dataset_exposure=dict(frame_draws=(max_steps - previous['step']) * effective_batch,
                                            transitions=validation['transition_count'],
                                            estimated_repetitions=(max_steps - previous['step']) * effective_batch
                                                                  / validation['transition_count'],
                                            prior_optimizer_steps=previous['step']))
    else:
        if profile == 'legacy_batch1':
            require(max_steps == 1000, 'Begin legacy Goal Full with the 1000-step run')
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['goal'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--train-seed', type=int, default=7)
    parser.add_argument('--shots', choices=['Full', '20', '10', '5', '2', '1'], required=True)
    parser.add_argument('--max-steps', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--profile', choices=['legacy_batch1', 'single_gpu_batch1_pilot',
                                              'single_gpu_global64'],
                        help='Default: legacy_batch1 for Full, single_gpu_batch1_pilot for numeric shots')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--checkpoint-steps', nargs='+', type=int,
                        help='Explicit optimizer-step checkpoints; must include max-steps')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    shots = args.shots if args.shots == 'Full' else int(args.shots)
    profile = args.profile or ('legacy_batch1' if shots == 'Full' else 'single_gpu_batch1_pilot')
    config = configuration(args.suite, args.seed, shots, args.max_steps,
                           args.run_id, args.resume, profile, args.checkpoint_steps,
                           args.train_seed)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node=1', '-m', 'experiments.fewshot_sft.worker',
               '--config', str(Path(config['output']) / 'training_config.json')]
    print(json.dumps({key: config[key] for key in ('output', 'dataset', 'rlds_root',
          'base_snapshot', 'suite', 'shots', 'selection_seed', 'train_seed',
          'profile', 'effective_batch_size', 'max_steps',
          'checkpoint_steps', 'checkpoint_exposure', 'dataset_exposure',
          'official_reference')}, indent=2))
    print(shlex.join(command))
    if not args.execute:
        return
    run = Path(config['output'])
    run.mkdir(parents=True)
    write(run / 'training_config.json', config)
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               WANDB_MODE='disabled', PYTHONHASHSEED=str(config['train_seed']),
               PYTHONPATH=os.pathsep.join([str(ROOT), str(OFT), os.environ.get('PYTHONPATH', '')]))
    with (run / 'train.log').open('x') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    require(result.returncode == 0, f'Training failed; inspect {run / "train.log"}')
    runtime = read(run / 'runtime.json')
    require(runtime['status'] == 'passed' and runtime['actual_optimizer_steps'] ==
            config['max_steps'] - config.get('resume_step', 0), 'Optimizer-step audit failed')
    trend = loss_trend(run, config.get('resume_step', 0), config['max_steps'],
                       config['checkpoint_steps'])
    write(run / 'loss_trend.json', trend)
    for step in config['checkpoint_steps']:
        check(Path(str(run) + f'--{step}_chkpt'), merged=False)
    print(f'PASS Goal {config["shots"]} to optimizer step {config["max_steps"]}: {run}')


if __name__ == '__main__':
    main()
