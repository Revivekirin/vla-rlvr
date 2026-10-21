"""Measure genuine Goal SFT vs scale-only actions, sequentially in isolated workers.

Default PASS means a valid finite comparison, not numerical equivalence. Supply
--max-abs-error to apply an explicit application-specific compatibility tolerance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import traceback

from validation_utils import ROOT, bounded_worker, configure, digest, provenance, write_json


def compare_actions(left, right):
    import numpy as np
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    result = {'sft_shape': list(a.shape), 'scale_shape': list(b.shape), 'finite': finite}
    if a.shape != (8, 7) or b.shape != (8, 7) or not finite:
        raise ValueError(f'Invalid comparison inputs: {result}')
    diff = np.abs(a - b)
    result.update(mean_absolute_error=float(diff.mean()), maximum_absolute_error=float(diff.max()),
                  per_action_dimension_mean_absolute_error=diff.mean(axis=0).tolist(),
                  per_action_dimension_maximum_absolute_error=diff.max(axis=0).tolist(),
                  absolute_differences=diff.tolist(),
                  dimension_order=['dx', 'dy', 'dz', 'daxis_angle_x', 'daxis_angle_y', 'daxis_angle_z', 'gripper'])
    return result


def fingerprint(directory):
    # Metadata guards all checkpoint files; content hashes additionally guard files
    # the upstream loader would otherwise rewrite. No write access is requested.
    return {str(p.relative_to(directory)): {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns,
            'sha256': digest(p) if p.stat().st_size < 10_000_000 else None}
            for p in directory.rglob('*') if p.is_file()}


def checkpoint_copy(source, target):
    """Copy metadata; symlink immutable tensor payloads to avoid a 15 GiB duplicate."""
    target.mkdir()
    for path in source.iterdir():
        if path.is_file():
            if path.stat().st_size < 10_000_000:
                shutil.copy2(path, target / path.name)
            else:
                (target / path.name).symlink_to(path.resolve())


def worker(args):
    configure()
    import numpy as np
    import torch
    from experiments.robot.robot_utils import set_seed_everywhere
    from experiments.robot.libero.run_libero_eval import process_action
    from experiments.robot.openvla_utils import normalize_proprio

    result = {'status': 'RUNNING', 'mode': args.mode, 'provenance': provenance()}
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU or approximate policy fallback')
    set_seed_everywhere(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    base = ROOT / 'checkpoints/goal/oft'
    scale = ROOT / 'checkpoints/ript/openvla_oft/scale_header/LIBERO_GOAL_scale_header.pth'
    lock = json.loads((ROOT / 'stage0/models.lock.json').read_text())
    base_lock = lock['moojink/openvla-7b-oft-finetuned-libero-goal']
    # Full tensors have already passed smoke tests. Check presence/size here,
    # rather than rehashing all 15 GiB on every comparison.
    for item in base_lock['files']:
        name = item['rfilename']
        if name.startswith('model-') or name.startswith(('action_head--', 'proprio_projector--')):
            if not (base / name).is_file() or (base / name).stat().st_size != item['size']:
                raise ValueError(f'Missing or incorrect released component: {name}')
    if args.mode == 'scale':
        entry = next(f for f in lock['tanshh97/RIPT_VLA']['files'] if f['rfilename'].endswith('scale_header/LIBERO_GOAL_scale_header.pth'))
        if digest(scale) != entry['lfs']['sha256']:
            raise ValueError('SFT scale head does not match pinned official SHA256')
    before = fingerprint(base)
    result['checkpoint_revision'] = base_lock['revision']
    result['input_sha256'] = digest(args.input)
    scale_before = fingerprint(scale.parent) if args.mode == 'scale' else None
    torch.cuda.reset_peak_memory_stats()
    try:
        with tempfile.TemporaryDirectory(prefix='oft-libero-checkpoint-') as temp:
            staged = Path(temp) / 'oft'
            checkpoint_copy(base, staged)
            if args.mode == 'sft':
                from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model
                from experiments.robot.robot_utils import get_action
                cfg = GenerateConfig(pretrained_checkpoint=str(staged), task_suite_name='libero_goal',
                                     center_crop=True, seed=1)
                model, action, proprio, noisy, processor = initialize_model(cfg)
                result['forward_path'] = 'Official OFT: one multimodal language-model forward; past_key_values=None; use_cache=None (model default); no prompt-cache reuse'
            else:
                # Import only in this process: this module globally patches LlamaSdpaAttention.
                from ript.algos.rl_optimizers.openvla_oft_interface import OpenVLA_OFT_Policy
                from ript.env_runner.openvla_oft_libero_runner import get_vla_action_batch
                policy = OpenVLA_OFT_Policy(str(staged), str(scale), 'libero_goal', 32, 0.0, None,
                                           seed=1, fix_scale_head=True, log_scale_clip=[-2.0, 0.5])
                cfg, model, action, proprio, processor = policy.cfg, policy.model, policy.action_head, policy.proprio_projector, policy.processor
                # Official null-adapter branch initializes LoRA A but zero B.
                # Verify that no random nonzero adapter update alters the SFT model.
                b_weights = [p for n, p in model.named_parameters() if 'lora_B' in n]
                if not b_weights or any(torch.count_nonzero(p).item() for p in b_weights):
                    raise ValueError('Scale-only LoRA is not a zero update')
                result['zero_lora_update_verified'] = True
                result['forward_path'] = 'RIPT: no-grad prompt/vision pass with use_cache=True, followed by action-token pass using past_key_values and explicit positions; patched SDPA mask'
            for component in [model, action, proprio]:
                component.eval()
            if args.mode == 'scale':
                policy.scale_head.eval()
            if cfg.unnorm_key != 'libero_goal_no_noops':
                raise ValueError(f'Unexpected normalization key: {cfg.unnorm_key}')
            result['configuration'] = {k: getattr(cfg, k) for k in ['unnorm_key', 'center_crop', 'num_images_in_input',
                'use_proprio', 'use_l1_regression', 'use_diffusion', 'use_film', 'num_open_loop_steps']}
            result['determinism'] = {'seed': 1, 'eval_mode': True, 'sampling': False,
                'tf32': False, 'deterministic_algorithms': 'warn_only; warnings retained in worker log'}
            result['preprocessing'] = 'Shared OFT prepare_images_for_vla: center 90% crop and processor; two cameras; masked q01/q99 proprio normalization. SFT calls processor with scalar prompt/image, RIPT with lists (batch size one).'
            result['dataset_statistics_sha256'] = digest(base / 'dataset_statistics.json')
            result['observations'] = {}
            # Record actual processor outputs without altering their values or types.
            class RecordedProcessor:
                def __init__(self, wrapped):
                    self.wrapped, self.calls = wrapped, []

                def __call__(self, *values, **kwargs):
                    output = self.wrapped(*values, **kwargs)
                    self.calls.append({key: {'shape': list(value.shape), 'dtype': str(value.dtype),
                        'sha256': hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
                        for key, value in output.items() if torch.is_tensor(value)})
                    return output

            recorded = RecordedProcessor(processor)
            with np.load(args.input, allow_pickle=False) as data:
                for label in json.loads(str(data['labels'])):
                    obs = {key: data[f'{label}_{key}'].copy() for key in ['full_image', 'wrist_image', 'state']}
                    instruction = str(data[f'{label}_instruction'])
                    normalized_proprio = normalize_proprio(obs['state'].copy(), model.norm_stats[cfg.unnorm_key]['proprio'])
                    recorded.calls = []
                    with torch.inference_mode():
                        if args.mode == 'sft':
                            raw = np.asarray(get_action(cfg, model, obs, instruction, recorded, action, proprio, noisy))
                        else:
                            raw = get_vla_action_batch(cfg, model, recorded, [obs], instruction, action_head=action,
                                proprio_projector=proprio, scale_head=policy.scale_head, use_laplace_sampling=False,
                                scale_factor=1.0)[0][0]
                    physical = np.stack([process_action(a.copy(), 'openvla') for a in raw])
                    compare_actions(physical, physical)  # shape and finiteness validation
                    result['observations'][label] = {'instruction': instruction, 'actions': physical.tolist(),
                        'unnormalized_pre_gripper_actions': raw.tolist(), 'normalized_proprio': normalized_proprio.tolist(),
                        'processor_calls': recorded.calls}
            torch.cuda.synchronize()
            result['gpu_memory'] = {'device': torch.cuda.get_device_name(),
                'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
                'allocated_at_end_gib': torch.cuda.memory_allocated() / 2**30}
            result['status'] = 'PASS'
    finally:
        result['source_checkpoint_unchanged'] = fingerprint(base) == before and (
            scale_before is None or fingerprint(scale.parent) == scale_before)
        if not result['source_checkpoint_unchanged']:
            raise RuntimeError('Source checkpoint fingerprint changed during validation')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real-observation', type=Path, help='NPZ produced by check_libero_runtime.py')
    p.add_argument('--max-abs-error', type=float, help='Optional nonzero physical-action error tolerance')
    p.add_argument('--timeout-seconds', type=int, default=1800, help='Bound per policy worker')
    p.add_argument('--output', type=Path, default=ROOT / 'stage0/evidence/action_compatibility_goal.json')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--mode', choices=['sft', 'scale'], help=argparse.SUPPRESS)
    p.add_argument('--input', type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.timeout_seconds <= 0 or (args.max_abs_error is not None and not 0 < args.max_abs_error < float('inf')):
        p.error('Timeout and optional finite error tolerance must be positive')
    args.output = args.output.resolve()
    if args.worker:
        try:
            result = worker(args)
        except Exception:
            result = {'status': 'ERROR', 'mode': args.mode, 'error': traceback.format_exc()}
    else:
        result = {'status': 'RUNNING', 'suite': 'libero_goal', 'scope': 'Action compatibility only; no environment steps or success evaluation',
                  'coordinate_convention': 'Unnormalized OSC delta pose, binarized and inverted LIBERO gripper (official process_action)',
                  'workers': {}, 'comparisons': {}, 'compatibility': 'NOT_ASSESSED'}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            import numpy as np
            with tempfile.TemporaryDirectory(prefix='action-compatibility-') as tmp:
                grid = np.arange(224 * 224 * 3, dtype=np.uint32).reshape(224, 224, 3)
                data = {'synthetic_full_image': (grid % 256).astype(np.uint8),
                        'synthetic_wrist_image': ((grid + 67) % 256).astype(np.uint8),
                        'synthetic_state': np.array([0.4, 0.0, 0.3, 0, 0, 0, 0.02, 0.02], dtype=np.float32),
                        'synthetic_instruction': np.array('put the bowl on the plate')}
                labels = ['synthetic']
                if args.real_observation:
                    with np.load(args.real_observation, allow_pickle=False) as real:
                        if str(real['suite']) != 'libero_goal':
                            raise ValueError('Real observation must come from Goal')
                        for key in ['full_image', 'wrist_image', 'state', 'instruction']:
                            data['real_' + key] = real[key].copy()
                    labels.append('real')
                    result['real_observation_sha256'] = digest(args.real_observation)
                for label in labels:
                    for key in ['full_image', 'wrist_image']:
                        value = data[f'{label}_{key}']
                        if value.shape != (224, 224, 3) or value.dtype != np.uint8:
                            raise ValueError('Expected officially prepared 224x224 RGB uint8 images')
                    state = data[f'{label}_state']
                    if state.shape != (8,) or not np.isfinite(state).all():
                        raise ValueError('Invalid proprioception')
                data['labels'] = np.array(json.dumps(labels))
                inputs = Path(tmp) / 'inputs.npz'
                np.savez_compressed(inputs, **data)
                for mode in ['sft', 'scale']:
                    result['workers'][mode] = bounded_worker(Path(__file__).resolve(), ['--mode', mode, '--input', str(inputs)],
                        Path(tmp) / f'{mode}.json', args.output.with_name(f'action_compatibility_goal_{mode}.log'), args.timeout_seconds)
                    if result['workers'][mode]['status'] != 'PASS':
                        raise RuntimeError(f'{mode} worker failed; see its log/evidence')
                left, right = result['workers']['sft'], result['workers']['scale']
                if left['configuration'] != right['configuration'] or left['input_sha256'] != right['input_sha256'] or left['dataset_statistics_sha256'] != right['dataset_statistics_sha256']:
                    raise ValueError('Inputs or normalization/preprocessing settings differ')
                for label in labels:
                    a, b = left['observations'][label], right['observations'][label]
                    if a['processor_calls'] != b['processor_calls'] or a['normalized_proprio'] != b['normalized_proprio'] or a['instruction'] != b['instruction']:
                        raise ValueError(f'Actual preprocessed inputs differ for {label}')
                    result['comparisons'][label] = compare_actions(a['actions'], b['actions'])
                    result['comparisons'][label]['pre_gripper_conversion'] = compare_actions(a['unnormalized_pre_gripper_actions'], b['unnormalized_pre_gripper_actions'])
                result['status'] = 'PASS'
                result['compatibility'] = 'MEASURED_REVIEW_REQUIRED'
                result['pass_definition'] = 'Same inputs and finite valid actions; PASS does not assert policy equivalence or performance'
                if args.max_abs_error is not None:
                    result['max_abs_error_tolerance'] = args.max_abs_error
                    passed = all(v['maximum_absolute_error'] <= args.max_abs_error for v in result['comparisons'].values())
                    result['compatibility'] = 'WITHIN_USER_TOLERANCE' if passed else 'OUTSIDE_USER_TOLERANCE'
                    result['status'] = 'PASS' if passed else 'FAIL'
        except Exception:
            result.update(status='ERROR', error=traceback.format_exc())
    write_json(args.output, result)
    print(json.dumps({'status': result['status'], 'output': str(args.output), 'compatibility': result.get('compatibility')}, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
