"""Small instrumentation shim; all forward/loss/optimizer logic remains official OFT."""
import argparse
import json
from pathlib import Path
import sys
import time
import types

from .common import OFT, read, require, sha, write


def instrument(source):
    # Fix upstream zero-based stop/save bookkeeping, including accumulation boundaries.
    replacements = [
        ('gradient_step_idx = batch_idx // cfg.grad_accumulation_steps',
         'gradient_step_idx = (batch_idx + 1) // cfg.grad_accumulation_steps'),
        ('if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:',
         'if (batch_idx + 1) % cfg.grad_accumulation_steps == 0 and (log_step % cfg.save_freq == 0 or log_step == cfg.max_steps):'),
        ('if log_step == cfg.max_steps:',
         'if (batch_idx + 1) % cfg.grad_accumulation_steps == 0 and log_step == cfg.max_steps:'),
        ('normalized_loss.backward()', 'normalized_loss.backward()\n            _finite(loss, trainable_params)'),
        ('# Create Action Tokenizer\n', '_setup(optimizer, scheduler)\n\n    # Create Action Tokenizer\n'),
    ]
    for old, new in replacements:
        require(source.count(old) == 1, f'Official trainer changed; review shim anchor: {old}')
        source = source.replace(old, new)
    return source


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    c = read(p.parse_args().config)
    from .common import ROOT
    require(all(sha(ROOT/p) == expected for p,expected in c['source_hashes'].items()),
            'Audited source changed after launch')
    import numpy as np
    import random
    import torch
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    random.seed(c['train_seed']); np.random.seed(c['train_seed']); torch.manual_seed(c['train_seed'])
    torch.cuda.manual_seed_all(c['train_seed']); tf.random.set_seed(c['train_seed'])
    torch.cuda.reset_peak_memory_stats()
    require(int(__import__('os').environ.get('WORLD_SIZE', '1')) == 1, 'Only audited single-GPU training is supported')
    path = OFT/'vla-scripts/finetune.py'
    require(sha(path) == c['official_finetune_sha256'], 'Trainer changed after launch')
    module = types.ModuleType('fewshot_official_finetune')
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(instrument(path.read_text()), str(path), 'exec'), module.__dict__)
    # Avoid upstream global statistics caches keyed only by DatasetInfo/transform.
    from prismatic.vla.datasets.rlds import dataset as rlds
    statistics = read(Path(c['dataset'])/'normalization.json')
    require(statistics['num_trajectories'] == 10*c['shots'], 'Normalization episode count mismatch')
    import copy
    rlds.get_dataset_statistics = lambda *args, **kwargs: copy.deepcopy(statistics)
    run = Path(c['output'])
    state = {'actual_optimizer_steps': 0}
    started = time.monotonic()

    def finite(loss, params):
        require(bool(torch.isfinite(loss).all()), 'Nonfinite loss')
        grads = [v.grad for v in params if v.grad is not None]
        require(bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads), 'Missing/nonfinite gradients')

    def setup(optimizer, scheduler):
        state.update(optimizer=optimizer, scheduler=scheduler)
        if c['resume']:
            saved = torch.load(Path(c['resume'])/'optimizer.pt', map_location='cpu', weights_only=True)
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
        def count(*_):
            state['actual_optimizer_steps'] += 1
        optimizer.register_step_post_hook(count)

    module._finite, module._setup = finite, setup
    # Register local OFT classes, never modify the cached pretrained directory.
    module.model_is_on_hf_hub = lambda _: False
    module.update_auto_map = lambda _: None
    module.check_model_logic_mismatch = lambda _: None
    # trust_remote_code on the pristine base would select its old discrete-action class.
    # Load the audited local continuous-action class directly, keeping base weights intact.
    module.AutoModelForVision2Seq = types.SimpleNamespace(register=lambda *a, **k: None,
        from_pretrained=module.OpenVLAForActionPrediction.from_pretrained)
    if c['resume']:
        from peft import PeftModel
        module.get_peft_model = lambda model, _: PeftModel.from_pretrained(
            model, str(Path(c['resume'])/'lora_adapter'), is_trainable=True)
        load = module.load_checkpoint
        module.load_checkpoint = lambda kind, path, step, device='cpu': load(kind, c['resume'], step, device)
    save = module.save_training_checkpoint
    def checkpoint(**kw):
        save(**kw)
        folder = Path(str(run)+f'--{kw["log_step"]}_chkpt')
        torch.save({'optimizer': state['optimizer'].state_dict(),
                    'scheduler': state['scheduler'].state_dict()}, folder/'optimizer.pt')
        metadata = dict(c, step=kw['log_step'], actual_optimizer_steps=state['actual_optimizer_steps'], merged=False)
        write(folder/'fewshot_checkpoint.json', metadata)
        # Strictly reload all learned modules into their existing architectures.
        for key in ('action_head', 'proprio_projector'):
            saved = torch.load(folder/f'{key}--{kw["log_step"]}_checkpoint.pt', weights_only=True)
            kw[key].load_state_dict(saved, strict=True)
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file
        current = get_peft_model_state_dict(kw['vla'].module)
        reloaded = load_file(str(folder/'lora_adapter/adapter_model.safetensors'))
        require(current.keys() == reloaded.keys(), 'Saved LoRA keys differ')
        require(all(torch.equal(v.cpu(), reloaded[k]) for k,v in current.items()), 'Saved LoRA tensors differ')
    module.save_training_checkpoint = checkpoint
    def metrics(values, prefix, step, wandb):
        with (run/'metrics.jsonl').open('a') as f:
            f.write(json.dumps({'step': step, 'prefix': prefix, 'metrics': values})+'\n')
    module.log_metrics_to_wandb = metrics
    cfg = module.FinetuneConfig(vla_path=c['base_snapshot'], data_root_dir=Path(c['dataset'])/'rlds',
        dataset_name=c['dataset_name'], run_root_dir=run.parent, run_id_override=run.name,
        use_l1_regression=True, use_diffusion=False, use_film=False, num_images_in_input=2,
        use_proprio=True, use_lora=True, lora_rank=32, lora_dropout=0., image_aug=True,
        batch_size=c['batch_size'], grad_accumulation_steps=c['gradient_accumulation'],
        learning_rate=c['learning_rate'], max_steps=c['max_steps'], save_freq=c['save_freq'],
        resume=bool(c['resume']), resume_step=c.get('resume_step'), merge_lora_during_training=False,
        shuffle_buffer_size=1000, wandb_log_freq=1, wandb_entity=None, wandb_project='fewshot-oft')
    result = dict(status='failed', batch_size=c['batch_size'],
                  selection_manifest_sha256=c['selection_manifest_sha256'])
    try:
        module.finetune.__wrapped__(cfg)
        require(state['actual_optimizer_steps'] == c['max_steps']-c.get('resume_step', 0), 'Optimizer step count mismatch')
        result['status'] = 'passed'
    finally:
        result.update(actual_optimizer_steps=state['actual_optimizer_steps'], elapsed_seconds=time.monotonic()-started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(), gpu=torch.cuda.get_device_name(),
                      inference_and_rollout='not run; export and evaluate separately')
        import importlib.metadata
        result['packages'] = {name: importlib.metadata.version(name) for name in
            ('torch','numpy','tensorflow','tensorflow-datasets','transformers','peft','draccus','accelerate')}
        write(run/'runtime.json', result)


if __name__ == '__main__':
    main()
