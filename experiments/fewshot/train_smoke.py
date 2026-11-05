"""One measured OFT optimizer step on a Stage 2 condition; no checkpoint or evaluation."""
import argparse
import copy
import os
from pathlib import Path
import random
import sys
import time
import types

from experiments.shot_scaling.core import ROOT, SUITES, digest, read, require, write
from .stage2 import selection_path

OFT = ROOT / 'runtime/openvla-oft'
RUNS = ROOT / 'outputs/stage3_sft/smoke'


def preflight(suite, seed, shots):
    require(suite in SUITES and type(seed) is int and seed >= 0
            and shots in (1, 2, 5, 10, 20, 'Full'), 'Invalid Stage 3 smoke condition')
    path = selection_path(suite, seed, shots)
    selection = read(path / 'selection_manifest.json')
    validation = read(path / 'validation.json')
    stats = read(path / 'normalization.json')
    require(validation['status'] == 'PASS' and validation['suite'] == suite
            and validation['seed'] == seed and validation['shots'] == shots,
            'Stage 2 dataset has not passed validation')
    require(validation['dataset_name'] == selection['dataset_name'] == SUITES[suite] + '_no_noops',
            'OFT dataset name mismatch')
    require(validation['selection_sha256'] == selection['selection_sha256']
            and validation['dataset_payload_sha256'] == digest(sorted(r['sha256'] for r in selection['episodes'])),
            'Stage 2 selection or payload identity changed')
    require(digest(stats) == validation['normalization_sha256']
            and stats['num_trajectories'] == validation['episode_count']
            and stats['num_transitions'] == validation['transition_count'],
            'Stage 2 normalization identity/count mismatch')
    root = Path(validation['rlds_root'])
    require((root / validation['dataset_name'] / '1.0.0' / 'dataset_info.json').is_file(),
            'Loadable RLDS dataset is missing')
    return path, validation, stats


def smoke(suite, seed, shots, train_seed=7, attempt=1):
    dataset, validation, stats = preflight(suite, seed, shots)
    require(type(train_seed) is int and train_seed >= 0, 'Invalid training seed')
    require(type(attempt) is int and attempt > 0, 'Attempt must be positive')
    name = 'Full' if shots == 'Full' else f'{shots}shot'
    output = RUNS / suite / f'seed{seed}' / f'{name}_attempt{attempt}'
    require(not output.exists(), f'Smoke output already exists: {output}')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['WANDB_MODE'] = 'disabled'
    sys.path.insert(0, str(OFT))
    import numpy as np
    import tensorflow as tf
    import torch
    tf.config.set_visible_devices([], 'GPU')
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'A CUDA BF16 GPU is required')
    require(int(os.environ.get('WORLD_SIZE', '1')) == 1, 'Smoke supports one GPU')
    torch.cuda.reset_peak_memory_stats()
    random.seed(train_seed); np.random.seed(train_seed); torch.manual_seed(train_seed)
    torch.cuda.manual_seed_all(train_seed)
    tf.random.set_seed(train_seed)
    from huggingface_hub import snapshot_download
    base = Path(snapshot_download('openvla/openvla-7b', local_files_only=True)).resolve()
    require(not (base / 'adapter_config.json').exists(), 'Base snapshot is an adapter')
    from experiments.fewshot_sft.worker import instrument
    source = OFT / 'vla-scripts/finetune.py'
    module = types.ModuleType('fewshot_stage3_smoke_finetune')
    module.__file__ = str(source)
    sys.modules[module.__name__] = module
    exec(compile(instrument(source.read_text()), str(source), 'exec'), module.__dict__)
    from prismatic.vla.datasets.rlds import dataset as rlds
    rlds.get_dataset_statistics = lambda *args, **kwargs: copy.deepcopy(stats)
    # Keep the pristine cached base unchanged and use the local continuous-action model class.
    module.model_is_on_hf_hub = lambda _: False
    module.update_auto_map = lambda _: None
    module.check_model_logic_mismatch = lambda _: None
    module.AutoModelForVision2Seq = types.SimpleNamespace(
        register=lambda *args, **kwargs: None,
        from_pretrained=module.OpenVLAForActionPrediction.from_pretrained)
    state = {'optimizer_steps': 0, 'loss': None}
    def setup(optimizer, scheduler, context):
        optimizer.register_step_post_hook(lambda *_: state.__setitem__('optimizer_steps', state['optimizer_steps'] + 1))
    def finite(loss, params):
        require(bool(torch.isfinite(loss).all()), 'Nonfinite SFT loss')
        grads = [p.grad for p in params if p.grad is not None]
        require(bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads), 'Missing/nonfinite SFT gradients')
        state['loss'] = float(loss.detach().cpu())
    module._setup = setup
    module._finite = finite
    module._capture = lambda *args, **kwargs: None
    module._ready = lambda context: None
    module._after_step = lambda context: None
    module._should_save = lambda step, cfg: False  # Feasibility smoke creates no checkpoint.
    module.log_metrics_to_wandb = lambda *args, **kwargs: None
    cfg = module.FinetuneConfig(
        vla_path=str(base), data_root_dir=Path(validation['rlds_root']),
        dataset_name=validation['dataset_name'], run_root_dir=output.parent,
        run_id_override=output.name, use_l1_regression=True, use_diffusion=False,
        use_film=False, num_images_in_input=2, use_proprio=True,
        use_lora=True, lora_rank=32, lora_dropout=0., image_aug=True,
        batch_size=1, grad_accumulation_steps=1, learning_rate=5e-4,
        max_steps=1, save_freq=1, merge_lora_during_training=False,
        shuffle_buffer_size=1000, wandb_log_freq=1,
        wandb_entity=None, wandb_project='fewshot-stage3-smoke')
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        module.finetune.__wrapped__(cfg)
        require(state['optimizer_steps'] == 1, 'Expected exactly one optimizer step')
        status = 'PASS'
        error = None
    except Exception as exc:
        status = 'FAIL'
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        output.mkdir(parents=True, exist_ok=True)
        write(output / 'smoke_result.json', dict(
            status=status, error=error, suite=suite, seed=seed, shots=shots,
            dataset=str(dataset), selection_sha256=validation['selection_sha256'],
            normalization_sha256=validation['normalization_sha256'],
            base_snapshot=str(base), train_seed=train_seed,
            optimizer_steps=state['optimizer_steps'], loss=state['loss'],
            gpu=torch.cuda.get_device_name(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            elapsed_seconds=time.monotonic() - started,
            checkpoints_created=False))
    return output / 'smoke_result.json'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=SUITES, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--shots', choices=['Full', '20', '10', '5', '2', '1'], required=True)
    p.add_argument('--train-seed', type=int, default=7)
    p.add_argument('--attempt', type=int, default=1)
    a = p.parse_args()
    result = smoke(a.suite, a.seed, a.shots if a.shots == 'Full' else int(a.shots),
                   a.train_seed, a.attempt)
    print(f'PASS: {result}')


if __name__ == '__main__':
    main()
