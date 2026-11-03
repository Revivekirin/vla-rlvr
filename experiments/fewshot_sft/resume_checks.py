"""Fail-closed tensor/state checks, shared by training and export."""
from pathlib import Path

from .common import NAME, inventory, read, require, sha


def equal_state(actual, expected, label):
    import torch
    if isinstance(expected, torch.Tensor):
        require(isinstance(actual, torch.Tensor) and actual.shape == expected.shape and actual.dtype == expected.dtype and
                torch.equal(actual.detach().cpu(), expected.detach().cpu()), f'{label}: tensor mismatch')
    elif isinstance(expected, dict):
        require(actual.keys() == expected.keys(), f'{label}: keys mismatch')
        for k in expected:
            equal_state(actual[k], expected[k], f'{label}/{k}')
    elif isinstance(expected, (list, tuple)):
        require(len(actual) == len(expected), f'{label}: length mismatch')
        for i,(a,b) in enumerate(zip(actual, expected)):
            equal_state(a,b,f'{label}/{i}')
    else:
        require(actual == expected, f'{label}: value mismatch')


def optimizer_names(context):
    names, params = [], []
    # Same order as the pinned official AdamW constructor. No optimizer is rebuilt with a different order.
    for kind in ('vla', 'action_head', 'proprio_projector'):
        for name, param in context[kind].named_parameters():
            if param.requires_grad:
                names.append(f'{kind}.{name}')
                params.append(param)
    actual = [p for g in context['optimizer'].param_groups for p in g['params']]
    require(len(params) == len(actual) and all(a is b for a,b in zip(params,actual)), 'Optimizer parameter order changed')
    return names


def restore_optimizer(optimizer, scheduler, saved, step, names):
    import torch
    if 'parameter_names' in saved:
        require(saved['parameter_names'] == names, 'Resume parameter names/order differ')
    groups = saved['optimizer']['param_groups']
    require(len(groups) == len(optimizer.param_groups), 'Optimizer group count differs')
    for old,new in zip(groups,optimizer.param_groups):
        require(len(old['params']) == len(new['params']), 'Optimizer parameter count differs')
        for pid,param in zip(old['params'],new['params']):
            state = saved['optimizer']['state'].get(pid, {})
            for key in ('exp_avg','exp_avg_sq'):
                if key in state:
                    require(tuple(state[key].shape) == tuple(param.shape), f'Optimizer {key} shape mismatch')
                    require(bool(torch.isfinite(state[key]).all()), f'Nonfinite optimizer {key}')
    require(saved['scheduler']['last_epoch'] == step, 'Scheduler cumulative step differs from checkpoint')
    state_steps = [int(s['step']) for s in saved['optimizer']['state'].values() if 'step' in s]
    require(state_steps and max(state_steps) == step and min(state_steps) > 0, 'Adam state step mismatch')
    optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    equal_state(optimizer.state_dict(), saved['optimizer'], 'restored optimizer')
    equal_state(scheduler.state_dict(), saved['scheduler'], 'restored scheduler')
    require([g['lr'] for g in optimizer.param_groups] == scheduler.get_last_lr(), 'Restored optimizer/scheduler LR mismatch')
    return dict(optimizer_exact=True, scheduler_exact=True, scheduler_step=step,
                lr=scheduler.get_last_lr(), adam_step_min=min(state_steps), adam_step_max=max(state_steps),
                parameter_order='saved names verified' if 'parameter_names' in saved else
                'legacy: original get_peft_model construction/order and moment shapes; old checkpoint has no name list')


def verify_components(path, vla, head, proprio, inference=False):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    path = Path(path)
    m = read(path/'fewshot_checkpoint.json')
    model = vla.module if hasattr(vla, 'module') else vla
    saved = load_file(str(path/'lora_adapter/adapter_model.safetensors'))
    current = get_peft_model_state_dict(model)
    if inference:
        require(current.keys() == saved.keys(), 'LoRA inference keys mismatch')
        saved = {k:v.to(dtype=current[k].dtype) for k,v in saved.items()}
    equal_state(current, saved, 'LoRA')
    files = {'lora': inventory(path/'lora_adapter')}
    for kind, component in [('action_head',head), ('proprio_projector',proprio)]:
        file = path/f'{kind}--{m["step"]}_checkpoint.pt'
        state = torch.load(file, map_location='cpu', weights_only=True)
        state = {k.removeprefix('module.'):v for k,v in state.items()}
        component = component.module if hasattr(component,'module') else component
        current_state = component.state_dict()
        if inference:
            # Official inference casts the FP32 trained proprio projector to BF16.
            require(current_state.keys() == state.keys(), f'{kind}: inference keys mismatch')
            state = {k:v.to(dtype=current_state[k].dtype) for k,v in state.items()}
        equal_state(current_state, state, kind)
        files[kind] = sha(file)
    files['normalization'] = sha(path/'dataset_statistics.json')
    files['comparison'] = 'exact saved tensors after official inference dtype casts' if inference else 'exact saved tensors and dtypes'
    return files


def verify_normalization(checkpoint, dataset):
    expected = read(Path(dataset)/'normalization.json')
    actual = dict(read(Path(checkpoint)/'dataset_statistics.json')[NAME])
    require(actual.pop('mask', None) is None, 'Unexpected top-level normalization mask')
    actual['action'] = dict(actual['action'])
    require(actual['action'].pop('mask') == [True]*6+[False], 'Normalization mask changed')
    require(actual == expected, 'Checkpoint does not use this train subset normalization')
