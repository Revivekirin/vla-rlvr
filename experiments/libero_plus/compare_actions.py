"""Bounded, isolated-process SFT/Scale comparison; never writes checkpoints.

capture: save three prepared observations after reset + 10 dummy steps (no policy).
compare: replay those NPZs in fresh SFT and Scale processes, at most 12 forwards.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from evaluate import ROOT, SUITE, checkpoints, configure, task_mapping

CATEGORIES = ('Language Instructions', 'Light Conditions', 'Sensor Noise')


def preflight(plus):
    base = ROOT / checkpoints('sft')['base']
    index = json.loads((base / 'model.safetensors.index.json').read_text())
    missing = [str(base / p) for p in sorted(set(index['weight_map'].values())) if not (base / p).is_file()]
    scale = ROOT / checkpoints('scale')['scale']
    missing += [str(scale)] if not scale.is_file() else []
    missing += [str(plus)] if not plus.is_dir() else []
    missing += [f'Python dependency: {p}' for p in ('numpy', 'torch', 'transformers', 'peft')
                if importlib.util.find_spec(p) is None]
    return {'ready': not missing, 'missing': missing}


def setup(plus):
    configure(plus)
    import numpy as np
    import torch
    from experiments.robot.libero import run_libero_eval as oft
    from experiments.robot import openvla_utils
    openvla_utils.update_auto_map = lambda _: None
    openvla_utils.check_model_logic_mismatch = lambda _: None
    return np, torch, oft


def capture(args):
    np, _, oft = setup(args.plus)
    from libero.libero import benchmark
    entries = task_mapping(args.plus)
    suite = benchmark.get_benchmark_dict()[SUITE](task_order_index=0)
    args.directory.mkdir(parents=True, exist_ok=False)
    for category in CATEGORIES:
        candidates = [e for e in entries if e['category'] == category]
        # Prefer a lost outcome if the original result JSONs are available.
        def lost(e):
            paths = [ROOT / f"outputs/libero-plus/{m}/seed-1/task-{e['id']:04d}.json" for m in ('sft', 'scale')]
            return all(p.exists() for p in paths) and json.loads(paths[0].read_text())['success'] and not json.loads(paths[1].read_text())['success']
        entry = next((e for e in candidates if lost(e)), candidates[0])
        oft.set_seed_everywhere(1)
        task = suite.get_task(entry['id'] - 1)
        env, language = oft.get_libero_env(task, 'openvla', resolution=256)
        try:
            env.reset()
            raw = env.set_init_state(suite.get_task_init_states(entry['id'] - 1)[0])
            for _ in range(10):
                raw, _, _, _ = env.step(oft.get_libero_dummy_action('openvla'))
            obs, _ = oft.prepare_observation(raw, 224)
            np.savez_compressed(args.directory / f"task-{entry['id']:04d}.npz",
                                full_image=obs['full_image'], wrist_image=obs['wrist_image'],
                                state=obs['state'], instruction=language,
                                metadata=json.dumps(entry))
        finally:
            env.close()


def worker(args):
    np, torch, oft = setup(args.plus)
    oft.set_seed_everywhere(1)
    ckpt = checkpoints(args.model)
    cfg = oft.GenerateConfig(pretrained_checkpoint=str(ROOT / ckpt['base']), task_suite_name=SUITE, seed=1, num_trials_per_task=1)
    if args.model == 'sft':
        model, head, proprio, noisy, processor = oft.initialize_model(cfg)
        scale_head = None
    else:
        # Never import this module in the SFT process: it globally patches attention.
        from ript.algos.rl_optimizers.openvla_oft_interface import OpenVLA_OFT_Policy
        from ript.env_runner import openvla_oft_libero_runner as runner
        policy = OpenVLA_OFT_Policy(str(ROOT / ckpt['base']), str(ROOT / ckpt['scale']), SUITE, 32, 0., None, seed=1, fix_scale_head=True)
        model, head, proprio, processor, cfg = policy.model, policy.action_head, policy.proprio_projector, policy.processor, policy.cfg
        scale_head, noisy = policy.scale_head, None
    for module in (model, head, proprio, scale_head):
        if module is not None:
            module.eval().requires_grad_(False)
    base = model.get_base_model() if args.model == 'scale' else model

    def fingerprint(module):
        digest = hashlib.sha256()
        for name, value in sorted(module.state_dict().items()):
            if 'lora_' in name:
                continue
            name = name.replace('.base_layer.', '.')
            digest.update(f'{name}:{value.dtype}:{tuple(value.shape)}'.encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    lora = {n: {'nonzero': int(torch.count_nonzero(p)), 'finite': bool(torch.isfinite(p).all())}
            for n, p in model.named_parameters() if 'lora_' in n}
    if args.model == 'scale':
        assert any('lora_B' in n for n in lora), 'No LoRA B found'
        assert all(v['finite'] for v in lora.values())
        assert all(v['nonzero'] == 0 for n, v in lora.items() if 'lora_B' in n)
    info = dict(weights={k: fingerprint(v) for k, v in [('base', base), ('action_head', head), ('proprio', proprio)]},
                norm_stats=base.norm_stats, unnorm_key=cfg.unnorm_key, lora=lora,
                attention_implementation=base.language_model.config._attn_implementation,
                versions={k: __import__(k).__version__ for k in ('torch', 'transformers', 'peft')})
    (args.output / f'{args.model}-model.json').write_text(json.dumps(info, indent=2))
    trace = {}

    def array(value):
        if torch.is_tensor(value):
            return value.detach().float().cpu().numpy().copy()
        return np.asarray(value).copy()

    def wrap(obj, name, callback):
        original = getattr(obj, name)
        def wrapped(*a, **kw):
            result = original(*a, **kw)
            callback(a, kw, result)
            return result
        setattr(obj, name, wrapped)

    # Capture the exact tensors entering each path, not reconstructed preprocessing.
    def inputs(a, kw, result):
        for key in ('input_ids', 'attention_mask', 'pixel_values', 'proprio'):
            if key in kw and kw[key] is not None:
                trace['input_' + key] = array(kw[key]).reshape(-1) if key == 'proprio' else array(kw[key])
    if args.model == 'sft':
        wrap(base, 'predict_action', inputs)
    else:
        wrap(runner, 'predict_action_batch', inputs)
    wrap(base, '_build_multimodal_attention', lambda a, k, r: trace.update(multimodal=array(r[0]), multimodal_mask=array(r[1])))
    original_unnorm = base._unnormalize_actions
    def unnorm(actions, *a, **kw):
        trace['normalized_mean'] = array(actions).reshape(8, 7)
        return original_unnorm(actions, *a, **kw)
    base._unnormalize_actions = unnorm
    wrap(head, 'predict_action', lambda a, k, r: trace.update(action_hidden=array(a[0])))
    # First two decoder layers: Scale has prompt/action calls, concatenate for alignment.
    layer_outputs = {0: [], 1: []}
    for i, layer in enumerate(base.language_model.model.layers[:2]):
        layer.register_forward_hook(lambda m, a, r, i=i: layer_outputs[i].append(array(r[0])))

    files = sorted(args.directory.glob('*.npz'))
    assert 1 <= len(files) <= 3, 'Use one to three saved observations'
    for path in files:
        with np.load(path, allow_pickle=False) as saved:
            original = {k: saved[k].copy() for k in ('full_image', 'wrist_image', 'state')}
            language = str(saved['instruction'])
        modes = ['native'] if args.model == 'sft' else ['native', 'full', 'scale_perturbed']
        for mode in modes:
            trace.clear()
            for values in layer_outputs.values():
                values.clear()
            obs = {k: v.copy() for k, v in original.items()}  # SFT mutates state!
            handle = None
            if mode == 'scale_perturbed':
                handle = scale_head.register_forward_hook(lambda m, a, r: torch.zeros_like(r) + 3)
            try:
                with torch.inference_mode():
                    if args.model == 'sft' or mode == 'full':
                        actions = oft.get_action(cfg, model, obs, language, processor=processor, action_head=head, proprio_projector=proprio, noisy_action_projector=noisy)
                    else:
                        actions = runner.get_vla_action_batch(cfg, model, processor, [obs], language, action_head=head, proprio_projector=proprio, scale_head=scale_head, use_laplace_sampling=False, scale_factor=1.)[0][0]
                trace['unnormalized'] = np.asarray(actions).copy()
                trace['executed'] = np.stack([oft.process_action(a.copy(), 'openvla') for a in actions])
                for i, values in layer_outputs.items():
                    trace[f'layer_{i}'] = np.concatenate(values, axis=1)
                np.savez_compressed(args.output / f'{path.stem}-{args.model}-{mode}.npz', **trace)
            finally:
                if handle is not None:
                    handle.remove()


def compare(args):
    import numpy as np
    args.output.mkdir(parents=True, exist_ok=False)
    for model in ('sft', 'scale'):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), 'worker', '--model', model,
                        '--plus', str(args.plus), '--directory', str(args.directory), '--output', str(args.output)], check=True)
    models = [json.loads((args.output / f'{m}-model.json').read_text()) for m in ('sft', 'scale')]
    report = {'weights_equal': models[0]['weights'] == models[1]['weights'],
              'norm_stats_equal': models[0]['norm_stats'] == models[1]['norm_stats'],
              'unnorm_key_equal': models[0]['unnorm_key'] == models[1]['unnorm_key'],
              'relative_definition': 'max(abs(a-b)/max(abs(a),abs(b),1e-8))', 'observations': {}}
    def metrics(a, b):
        if a.shape != b.shape:
            return {'shape_mismatch': [list(a.shape), list(b.shape)]}
        a, b = a.astype('float64'), b.astype('float64')
        delta = np.abs(a - b)
        return {'exact': bool(np.array_equal(a, b)), 'finite': bool(np.isfinite(a).all() and np.isfinite(b).all()),
                'max_abs': float(delta.max()), 'max_rel': float((delta / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-8)).max())}
    for path in sorted(args.directory.glob('*.npz')):
        comparisons = {}
        for label, left, right in [('native', 'sft-native', 'scale-native'),
                                   ('full_control', 'sft-native', 'scale-full'),
                                   ('scale_head_control', 'scale-native', 'scale-scale_perturbed')]:
            with np.load(args.output / f'{path.stem}-{left}.npz') as a, np.load(args.output / f'{path.stem}-{right}.npz') as b:
                comparisons[label] = {k: metrics(a[k], b[k]) for k in a.files if k in b.files}
                order = ('input_input_ids', 'input_attention_mask', 'input_pixel_values', 'input_proprio',
                         'multimodal', 'multimodal_mask', 'layer_0', 'layer_1', 'action_hidden',
                         'normalized_mean', 'unnormalized', 'executed')
                comparisons[label]['first_recorded_difference'] = next(
                    (k for k in order if k in comparisons[label] and not comparisons[label][k].get('exact', False)), None)
        with np.load(path, allow_pickle=False) as obs:
            metadata = str(obs['metadata']) if 'metadata' in obs else None
        report['observations'][path.stem] = {'metadata': metadata, 'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'comparisons': comparisons}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('preflight', 'capture', 'compare', 'worker'))
    p.add_argument('--plus', type=Path, default=ROOT / 'vendor/LIBERO-plus')
    p.add_argument('--directory', type=Path, default=ROOT / 'outputs/action-equivalence/observations')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/action-equivalence/comparison')
    p.add_argument('--model', choices=('sft', 'scale'))
    args = p.parse_args()
    if args.command == 'preflight':
        print(json.dumps(preflight(args.plus), indent=2))
    elif args.command == 'capture':
        capture(args)
    else:
        status = preflight(args.plus)
        if not status['ready']:
            p.error(json.dumps(status))
        if args.command == 'worker':
            if not args.model:
                p.error('worker requires --model')
            worker(args)
        else:
            compare(args)


if __name__ == '__main__':
    main()
