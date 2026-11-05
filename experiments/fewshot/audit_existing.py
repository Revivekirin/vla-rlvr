"""Read-only audit of the completed Goal Full 1000-step infrastructure run."""
import argparse
import gc
import json
import math
from pathlib import Path

from experiments.fewshot_sft.common import OFT, OUT, ROOT, read, require, sha


STEPS = (200, 500, 1000)


def parameter_changes(left, right, step_left, step_right):
    import torch
    from safetensors import safe_open

    def compare_tensors(a, b):
        require(a.keys() == b.keys(), 'Checkpoint parameter names differ')
        changed = []
        elements = 0
        for name in a:
            x, y = a[name], b[name]
            require(x.shape == y.shape and x.dtype == y.dtype, f'Parameter shape/dtype changed: {name}')
            elements += x.numel()
            if not torch.equal(x, y):
                changed.append(name)
        return dict(tensors=len(a), elements=elements, changed_tensors=len(changed),
                    sample_changed_names=changed[:8])

    result = dict(from_step=step_left, to_step=step_right)
    for component in ('action_head', 'proprio_projector'):
        old = torch.load(left / f'{component}--{step_left}_checkpoint.pt',
                         map_location='cpu', weights_only=True)
        new = torch.load(right / f'{component}--{step_right}_checkpoint.pt',
                         map_location='cpu', weights_only=True)
        result[component] = compare_tensors(old, new)
        del old, new
        gc.collect()
    old_adapter = left / 'lora_adapter/adapter_model.safetensors'
    new_adapter = right / 'lora_adapter/adapter_model.safetensors'
    with safe_open(str(old_adapter), framework='pt', device='cpu') as a, \
         safe_open(str(new_adapter), framework='pt', device='cpu') as b:
        names = a.keys()
        require(names == b.keys(), 'LoRA parameter names differ')
        changed = []
        elements = 0
        for name in names:
            x, y = a.get_tensor(name), b.get_tensor(name)
            require(x.shape == y.shape and x.dtype == y.dtype, f'LoRA shape/dtype changed: {name}')
            elements += x.numel()
            if not torch.equal(x, y):
                changed.append(name)
        result['lora_adapter'] = dict(tensors=len(names), elements=elements,
                                      changed_tensors=len(changed), sample_changed_names=changed[:8])
    result['all_components_changed'] = all(result[k]['changed_tensors'] > 0 for k in
                                           ('action_head', 'proprio_projector', 'lora_adapter'))
    return result


def audit(run, include_parameters=False):
    run = Path(run).resolve()
    require(run.is_relative_to((OUT / 'runs').resolve()), 'Run must be in outputs/fewshot_sft/runs')
    c = read(run / 'training_config.json')
    runtime = read(run / 'runtime.json')
    require(c['suite'] == 'goal' and c['shots'] == 'Full' and c['max_steps'] == 1000,
            'Not the completed Goal Full 1000-step run')
    require(runtime['status'] == 'passed' and runtime['actual_optimizer_steps'] == 1000,
            'Recorded training did not pass 1000 optimizer updates')
    effective_batch = c['batch_size'] * c['gradient_accumulation'] * c['world_size']
    require(effective_batch == c['effective_batch_size'], 'Recorded effective batch is inconsistent')
    rows = [json.loads(line) for line in (run / 'diagnostics.jsonl').read_text().splitlines()]
    require([row['step'] for row in rows] == list(range(1, 1001)),
            'Missing or duplicate optimizer-step diagnostics')
    losses = [row['normalized_chunk_l1'] for row in rows]
    require(all(math.isfinite(value) and value >= 0 for value in losses), 'Nonfinite training loss')
    learning_rates = [row['lr'] for row in rows]
    require(all(len(values) == 1 and math.isfinite(values[0]) and values[0] > 0
                for values in learning_rates), 'Invalid learning-rate trace')
    checkpoints = []
    for step in STEPS:
        folder = Path(str(run) + f'--{step}_chkpt')
        metadata = read(folder / 'fewshot_checkpoint.json')
        require(metadata['step'] == step and metadata['selection_sha256'] == c['selection_sha256']
                and metadata['normalization_sha256'] == c['normalization_sha256']
                and metadata['base_snapshot'] == c['base_snapshot'],
                f'Checkpoint {step} provenance differs from run')
        proof = read(folder / 'component_verification.json')
        require(proof['status'] == 'passed' and (folder / 'checkpoint_files.json').is_file(),
                f'Checkpoint {step} lacks recorded component/file verification')
        files = [p for p in folder.rglob('*') if p.is_file()]
        checkpoints.append(dict(step=step, path=str(folder), file_count=len(files),
                                bytes=sum(p.stat().st_size for p in files),
                                component_verification=proof['status'],
                                loss_moving_mean=rows[step - 1]['moving_mean'],
                                learning_rates=rows[step - 1]['lr']))
    result = dict(
        run=str(run), status='AUDITED_RECORDED_EVIDENCE',
        note='Read-only review; recorded checkpoint hashes are not recomputed here.',
        dataset=c['dataset'], rlds_root=c.get('rlds_root'),
        selection_seed=c['selection_seed'], training_seed=c['train_seed'],
        base_snapshot=c['base_snapshot'], base_revision=c['base_revision'],
        base_identity=c['base_identity'],
        batch=dict(per_gpu=c['batch_size'], accumulation=c['gradient_accumulation'],
                   world_size=c['world_size'], effective_global=effective_batch),
        optimizer=dict(learning_rate=c['learning_rate'], scheduler=c['scheduler_config'],
                       updates=runtime['actual_optimizer_steps'],
                       first_logged_lr=learning_rates[0][0],
                       last_logged_lr=learning_rates[-1][0],
                       min_logged_lr=min(values[0] for values in learning_rates),
                       max_logged_lr=max(values[0] for values in learning_rates)),
        exposure=c['dataset_exposure'],
        losses=dict(first_100_mean=sum(losses[:100]) / 100,
                    last_100_mean=sum(losses[-100:]) / 100,
                    minimum=min(losses), maximum=max(losses),
                    first=losses[0], last=losses[-1]),
        checkpoints=checkpoints,
        current_official_finetune_matches_recorded_hash=(
            sha(OFT / 'vla-scripts/finetune.py') == c['official_finetune_sha256']),
        official_reference=dict(source=str(OFT / 'LIBERO.md'),
                                source_sha256=sha(OFT / 'LIBERO.md'),
                                declared_revision=read(ROOT / 'stage0/repos.lock.json')['openvla-oft']['revision'],
                                has_git_metadata=(OFT / '.git').exists(),
                                per_gpu_batch=8, gpu_count=8, global_batch=64,
                                learning_rate=5e-4, decay_step=100000,
                                max_steps=150005, goal_referenced_checkpoint_step=50000),
    )
    if include_parameters:
        folders = [Path(item['path']) for item in checkpoints]
        result['parameter_changes'] = [parameter_changes(a, b, s, t)
                                       for a, b, s, t in zip(folders, folders[1:], STEPS, STEPS[1:])]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path,
                        default=OUT / 'runs/goal-full-seed42-1000')
    parser.add_argument('--parameter-deltas', action='store_true',
                        help='Compare saved tensors at 200/500/1000 on CPU; may take minutes')
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.parameter_deltas), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
