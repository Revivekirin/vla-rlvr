"""Small instrumentation shim; all forward/loss/optimizer logic remains official OFT."""
import argparse
import json
from pathlib import Path
import sys
import time
import types

from .common import OFT, inventory, read, require, sha, write


def instrument(source):
    # Fix upstream zero-based stop/save bookkeeping, including accumulation boundaries.
    replacements = [
        ('gradient_step_idx = batch_idx // cfg.grad_accumulation_steps',
         'gradient_step_idx = (batch_idx + 1) // cfg.grad_accumulation_steps'),
        ('if gradient_step_idx > 0 and log_step % cfg.save_freq == 0:',
         'if (batch_idx + 1) % cfg.grad_accumulation_steps == 0 and _should_save(log_step, cfg):'),
        ('if log_step == cfg.max_steps:',
         'if (batch_idx + 1) % cfg.grad_accumulation_steps == 0 and log_step == cfg.max_steps:'),
        ('normalized_loss.backward()', 'normalized_loss.backward()\n            _finite(loss, trainable_params)'),
        ('# Create Action Tokenizer\n', '_setup(optimizer, scheduler, locals())\n\n    # Create Action Tokenizer\n'),
        ('loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)',
         'loss = torch.nn.L1Loss()(ground_truth_actions, predicted_actions)\n            _capture(predicted_actions, ground_truth_actions)'),
        ('progress.update()', '_after_step(locals())\n                progress.update()'),
        ('# Deque to store recent train metrics', '_ready(locals())\n\n    # Deque to store recent train metrics'),
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
    from .diagnostics import Diagnostics, install_metadata
    from .resume_checks import optimizer_names, restore_optimizer, verify_components, verify_normalization
    diagnostics = Diagnostics(c, module.run_forward_pass)
    install_metadata(module)
    module.run_forward_pass = diagnostics.train_forward
    module._capture = diagnostics.capture
    module._should_save = lambda step,cfg: step in c['checkpoint_steps'] or step % cfg.save_freq == 0 or step == cfg.max_steps
    class ResumeVerified(Exception):
        pass
    def ready(context):
        if c.get('verify_resume'):
            raise ResumeVerified()
        if c.get('validation'):
            diagnostics.validate(context, c.get('resume_step',0))
    def after_step(context):
        require(state['scheduler'].last_epoch == c.get('resume_step',0)+state['actual_optimizer_steps'],
                'Scheduler continuity lost after optimizer step')
        diagnostics.after_step(context)
    module._ready, module._after_step = ready, after_step

    def finite(loss, params):
        require(bool(torch.isfinite(loss).all()), 'Nonfinite loss')
        grads = [v.grad for v in params if v.grad is not None]
        require(bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads), 'Missing/nonfinite gradients')

    def setup(optimizer, scheduler, context):
        state.update(optimizer=optimizer, scheduler=scheduler)
        state['parameter_names'] = optimizer_names(context)
        if c['resume']:
            require(inventory(c['resume']) == c['resume_files'], 'Resume checkpoint changed after launch')
            verify_normalization(c['resume'], c['dataset'])
            components = verify_components(c['resume'], context['vla'], context['action_head'], context['proprio_projector'])
            saved = torch.load(Path(c['resume'])/'optimizer.pt', map_location='cpu', weights_only=True)
            proof = restore_optimizer(optimizer, scheduler, saved, c['resume_step'], state['parameter_names'])
            write(run/'resume_verification.json', dict(status='passed', before_any_optimizer_update=True,
                step=c['resume_step'], components=components, **proof, continuity=c['continuity']))
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
        # Preserve the legacy optimizer's original parameter construction/order.
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        original_peft = module.get_peft_model
        def restore_adapter(model, config):
            saved_config = read(Path(c['resume'])/'lora_adapter/adapter_config.json')
            for key in ('r','lora_alpha','lora_dropout'):
                require(saved_config[key] == getattr(config,key), f'LoRA configuration changed: {key}')
            model = original_peft(model, config)
            targets = model.peft_config['default'].target_modules
            require(set(saved_config['target_modules']) == set(targets), 'LoRA target modules changed')
            set_peft_model_state_dict(model, load_file(str(Path(c['resume'])/'lora_adapter/adapter_model.safetensors')))
            return model
        module.get_peft_model = restore_adapter
        load = module.load_checkpoint
        module.load_checkpoint = lambda kind, path, step, device='cpu': load(kind, c['resume'], step, device)
    save = module.save_training_checkpoint
    def checkpoint(**kw):
        save(**kw)
        folder = Path(str(run)+f'--{kw["log_step"]}_chkpt')
        torch.save({'optimizer': state['optimizer'].state_dict(),
                    'scheduler': state['scheduler'].state_dict(),
                    'parameter_names':state['parameter_names']}, folder/'optimizer.pt')
        metadata = dict(c, step=kw['log_step'], cumulative_optimizer_steps=kw['log_step'],
                        actual_optimizer_steps=state['actual_optimizer_steps'], merged=False)
        write(folder/'fewshot_checkpoint.json', metadata)
        # Compare BEFORE loading; reloading first would hide a wrongly saved head.
        components = verify_components(folder, kw['vla'], kw['action_head'], kw['proprio_projector'])
        # Strictly reload all learned modules into their existing architectures.
        for key in ('action_head', 'proprio_projector'):
            saved = torch.load(folder/f'{key}--{kw["log_step"]}_checkpoint.pt', map_location='cpu', weights_only=True)
            kw[key].load_state_dict(saved, strict=True)
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file
        current = get_peft_model_state_dict(kw['vla'].module)
        reloaded = load_file(str(folder/'lora_adapter/adapter_model.safetensors'))
        require(current.keys() == reloaded.keys(), 'Saved LoRA keys differ')
        require(all(torch.equal(v.cpu(), reloaded[k]) for k,v in current.items()), 'Saved LoRA tensors differ')
        verify_normalization(folder, c['dataset'])
        write(folder/'component_verification.json', dict(status='passed', step=kw['log_step'], components=components))
        diagnostics.save(folder)
        write(folder/'checkpoint_files.json', inventory(folder))
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
    except ResumeVerified:
        require(state['actual_optimizer_steps'] == 0, 'Resume audit performed an optimizer update')
        result['status'] = 'resume_verified'
    finally:
        result.update(actual_optimizer_steps=state['actual_optimizer_steps'], elapsed_seconds=time.monotonic()-started,
                      cumulative_optimizer_steps=c.get('resume_step',0)+state['actual_optimizer_steps'],
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(), gpu=torch.cuda.get_device_name(),
                      inference_and_rollout='not run; export and evaluate separately')
        import importlib.metadata
        result['packages'] = {name: importlib.metadata.version(name) for name in
            ('torch','numpy','tensorflow','tensorflow-datasets','transformers','peft','draccus','accelerate')}
        write(run/'runtime.json', result)


if __name__ == '__main__':
    main()
