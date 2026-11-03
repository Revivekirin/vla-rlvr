"""Build a bounded, offline, single-GPU official OFT training invocation."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .common import BASE, NAME, OFT, OUT, ROOT, destination, digest, inventory, read, require, sha, write
from .prepare_dataset import validate_manifest


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=['goal'], default='goal')
    p.add_argument('--shots', type=int, choices=[1, 5], default=5)
    p.add_argument('--selection-seed', type=int, default=42)
    p.add_argument('--train-seed', type=int, default=7)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--gradient-accumulation', type=int, default=1)
    p.add_argument('--learning-rate', type=float, default=5e-4)
    p.add_argument('--max-steps', type=int, required=True)
    p.add_argument('--save-freq', type=int, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--dataset', type=Path)
    p.add_argument('--resume', type=Path, help='Our unmerged checkpoint; optimizer resumes, TF shuffle restarts')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--smoke-evidence', type=Path, help='Measured successful one-step training run runtime.json')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--execute', action='store_true')
    return p


def plan(a):
    require(a.max_steps > 0 and 0 < a.save_freq <= a.max_steps, 'Invalid explicit step/save bounds')
    require(a.batch_size > 0 and a.gradient_accumulation > 0 and a.learning_rate > 0, 'Invalid optimizer configuration')
    require(a.run_id and all(c.isalnum() or c in '-_' for c in a.run_id), 'Unsafe run ID')
    require(not a.smoke or (a.max_steps == a.save_freq == a.batch_size == a.gradient_accumulation == 1),
            'Smoke must use exactly one batch and optimizer step')
    dataset = destination(a.dataset or OUT/f'datasets/goal_{a.shots}shot_seed{a.selection_seed}')
    run = destination(OUT/'runs'/a.run_id)
    return dict(schema=1, base_identity=BASE, dataset=str(dataset), output=str(run),
                dataset_name=NAME, suite=a.suite, shots=a.shots, selection_seed=a.selection_seed,
                train_seed=a.train_seed, batch_size=a.batch_size, gradient_accumulation=a.gradient_accumulation,
                effective_batch_size=a.batch_size*a.gradient_accumulation, world_size=1,
                learning_rate=a.learning_rate, max_steps=a.max_steps, save_freq=a.save_freq,
                resume=str(a.resume.resolve()) if a.resume else None,
                resume_semantics='optimizer/scheduler/weights restored; deterministic TF data stream restarts',
                smoke=a.smoke, architecture=dict(use_l1_regression=True, use_diffusion=False, use_film=False,
                    num_images_in_input=2, use_proprio=True, use_lora=True, lora_rank=32,
                    action_chunk=8, action_dim=7, proprio_dim=8, image_aug=True),
                merged=False, held_out_validation='not configured; no training episodes reused as validation')


def main():
    a = parser().parse_args()
    c = plan(a)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=1',
               '-m', 'experiments.fewshot_sft.worker', '--config', str(Path(c['output'])/'training_config.json')]
    print(json.dumps(c, indent=2))
    print(shlex.join(command))
    if not a.execute or a.dry_run:
        return
    dataset = Path(c['dataset'])
    m, audit = read(dataset/'selection_manifest.json'), read(dataset/'dataset_audit.json')
    validate_manifest(m)
    require((m['shots'], m['selection_seed']) == (a.shots, a.selection_seed), 'Dataset selection mismatch')
    require(audit['status'] == 'verified_materialization', 'Subset has not passed integrity validation')
    require(inventory(dataset/'rlds') == audit['subset_files'], 'Materialized RLDS content changed')
    require(digest(read(dataset/'normalization.json')) == audit['normalization_sha256'], 'Normalization changed')
    c.update(selection_manifest_sha256=sha(dataset/'selection_manifest.json'),
             selected_transitions=audit['selected_transitions'],
             expected_frame_draws=a.max_steps*c['effective_batch_size'],
             expected_dataset_repetitions=a.max_steps*c['effective_batch_size']/audit['selected_transitions'],
             official_finetune_sha256=sha(OFT/'vla-scripts/finetune.py'))
    c['source_hashes'] = {str(p.relative_to(ROOT)): sha(p) for p in [
        OFT/'prismatic/vla/datasets/datasets.py', OFT/'prismatic/vla/datasets/rlds/dataset.py',
        OFT/'prismatic/vla/datasets/rlds/oxe/transforms.py', OFT/'prismatic/vla/constants.py',
        OFT/'prismatic/extern/hf/modeling_prismatic.py', ROOT/'experiments/fewshot_sft/worker.py']}
    import torch
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'CUDA BF16 GPU required')
    if not a.smoke:
        require(a.smoke_evidence is not None, 'Run the one-step GPU smoke before bounded training')
        smoke = read(a.smoke_evidence)
        require(smoke['status'] == 'passed' and smoke['actual_optimizer_steps'] == 1,
                'Smoke did not pass a measured optimizer step')
        require(smoke['selection_manifest_sha256'] == c['selection_manifest_sha256'], 'Smoke dataset mismatch')
        require(smoke['batch_size'] == a.batch_size, 'Measure smoke with matching batch size (currently 1)')
        require(smoke['gpu'] == torch.cuda.get_device_name(), 'Smoke was measured on a different GPU')
        c['smoke_evidence'] = dict(path=str(a.smoke_evidence), sha256=sha(a.smoke_evidence),
                                   peak_reserved_bytes=smoke['peak_reserved_bytes'])
    from huggingface_hub import snapshot_download
    # Only the pristine pretrained model, from the already populated local HF cache.
    base = Path(snapshot_download(BASE, local_files_only=True)).resolve()
    require(not (base/'adapter_config.json').exists(), 'Base must not be an adapter')
    c['base_snapshot'] = str(base)
    c['base_revision'] = base.name
    c['base_files'] = inventory(base)
    if a.resume:
        old = read(a.resume/'fewshot_checkpoint.json')
        require(old['base_revision'] == c['base_revision'] and
                old['selection_manifest_sha256'] == c['selection_manifest_sha256'], 'Resume provenance mismatch')
        require(not old['merged'] and old['step'] < a.max_steps, 'Resume requires unmerged, earlier checkpoint')
        c['resume_step'] = old['step']
        require((a.resume/'optimizer.pt').is_file(), 'Missing resumable optimizer state')
        for key in ('batch_size', 'gradient_accumulation', 'learning_rate', 'train_seed', 'official_finetune_sha256'):
            require(old[key] == c[key], f'Resume configuration changed: {key}')
        c['expected_frame_draws'] = (a.max_steps-old['step'])*c['effective_batch_size']
        c['expected_dataset_repetitions'] = c['expected_frame_draws']/audit['selected_transitions']
    run = Path(c['output'])
    require(not run.exists(), 'Run ID already exists; choose a new ID even when resuming')
    require(not list(run.parent.glob(run.name+'--*')), 'Checkpoint run prefix already exists')
    run.mkdir(parents=True)
    write(run/'training_config.json', c)
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
               PYTHONPATH=os.pathsep.join([str(ROOT), str(OFT), os.environ.get('PYTHONPATH', '')]),
               PYTHONHASHSEED=str(a.train_seed))
    with (run/'train.log').open('x') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    require(result.returncode == 0, f'Training failed; inspect {run / "train.log"}')


if __name__ == '__main__':
    main()
