"""Goal-only LIBERO-Plus evaluation; no training or nominal evaluator changes."""
import argparse
import ast
from collections import Counter, deque
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REVISION = '4976dc30028e805ff8094b55501d532c48fec182'
SUITE = 'libero_goal'
MODELS = ('sft', 'scale', 'ript')


def write_result(path, value):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def task_mapping(plus):
    folder = plus / 'libero/libero/benchmark'
    tree = ast.parse((folder / 'libero_suite_task_map.py').read_text())
    names = next(ast.literal_eval(n.value)[SUITE] for n in tree.body
                 if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'libero_task_map' for t in n.targets))
    entries = json.loads((folder / 'task_classification.json').read_text())[SUITE]
    assert len(names) == len(entries) == 2591
    assert len(set(names)) == len(names)
    for i, entry in enumerate(entries):
        assert entry['id'] == i + 1 and entry['name'] == names[i], (i, entry)
    return entries


def checkpoints(model):
    result = {'base': 'checkpoints/goal/oft'}
    if model != 'sft':
        result['scale'] = 'checkpoints/ript/openvla_oft/scale_header/LIBERO_GOAL_scale_header.pth'
    if model == 'ript':
        result['adapter'] = 'checkpoints/ript/openvla_oft/ript_adaptors/LIBERO_GOAL_lora'
        result['rl_heads'] = result['adapter'] + '/openvla_headers.pt'
    return result


def configure(plus):
    revision = subprocess.check_output(['git', '-C', str(plus), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError(f'Expected LIBERO-Plus {REVISION}, got {revision}')
    if subprocess.check_output(['git', '-C', str(plus), 'diff', 'HEAD', '--'], text=True):
        raise ValueError('LIBERO-Plus tracked source has local changes')
    config_dir = ROOT / 'runtime/libero-plus-config'
    config_dir.mkdir(parents=True, exist_ok=True)
    base = plus / 'libero/libero'
    paths = dict(benchmark_root=str(base), bddl_files=str(base / 'bddl_files'),
                 init_states=str(base / 'init_files'), assets=str(base / 'assets'), datasets=str(plus / 'datasets'))
    # JSON is valid YAML; keep the original LIBERO config untouched.
    (config_dir / 'config.yaml').write_text(json.dumps(paths, indent=2) + '\n')
    os.environ.update(LIBERO_CONFIG_PATH=str(config_dir), MUJOCO_GL='egl',
                      TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    sys.path[:0] = [str(plus), str(ROOT / 'runtime/openvla-oft'), str(ROOT / 'runtime/ript-vla')]


def run(args, entries):
    import libero.libero as libero
    from libero.libero import benchmark
    assert Path(libero.__file__).resolve().is_relative_to(args.plus)
    suite = benchmark.get_benchmark_dict()[SUITE](task_order_index=0)
    assert [suite.get_task(i).name for i in range(suite.n_tasks)] == [e['name'] for e in entries]
    print('benchmark import:', libero.__file__, 'tasks:', suite.n_tasks)
    if args.check_import:
        return  # No environment instance and no checkpoint loading.

    out = args.output / args.model / f'seed-{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    # A process lock prevents two concurrent writers to the same evaluation.
    import fcntl
    lock = (out / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = dict(revision=REVISION, suite=SUITE, model=args.model, seed=args.seed,
                    checkpoints=checkpoints(args.model), tasks=entries, horizon=300,
                    wait_steps=10, env_seed=0, initial_state_index=0, trials_per_task=1,
                    image_resolution=256, policy_resolution=224, center_crop=True,
                    action_chunk=8, sampling=False, protocol='goal-plus-v1',
                    evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Resume manifest mismatch; use a new output directory')
    else:
        if any(out.glob('task-*.json')):
            raise ValueError('Existing results have no manifest; use a new output directory')
        write_result(manifest_path, manifest)
    pending = []
    for entry in entries:
        path = out / f"task-{entry['id']:04d}.json"
        if path.exists():
            row = json.loads(path.read_text())
            assert all(row[k] == v for k, v in dict(task_id=entry['id'], task_name=entry['name'],
                       shift=entry['category'], model=args.model, seed=args.seed, revision=REVISION).items())
            assert type(row['success']) is bool and row['failure'] == (not row['success'])
        else:
            pending.append(entry)
    if not pending:
        print('All tasks already complete')
        return

    import torch
    from experiments.robot.libero import run_libero_eval as oft
    # Existing loaders sync configuration/code but never save weights. Disable those
    # sync side effects: use the already prepared Stage 0 checkpoint as released.
    from experiments.robot import openvla_utils
    openvla_utils.update_auto_map = lambda _: None
    openvla_utils.check_model_logic_mismatch = lambda _: None
    oft.set_seed_everywhere(args.seed)
    ckpt = checkpoints(args.model)
    if args.model == 'sft':
        cfg = oft.GenerateConfig(pretrained_checkpoint=str(ROOT / ckpt['base']),
                                 task_suite_name=SUITE, seed=args.seed, num_trials_per_task=1)
        model, head, proprio, noisy, processor = oft.initialize_model(cfg)
        modules = [model, head, proprio, noisy]
        def predict(observation, language):
            return oft.get_action(cfg, model, observation, language, processor=processor,
                                  action_head=head, proprio_projector=proprio, noisy_action_projector=noisy)
    else:
        from ript.algos.rl_optimizers.openvla_oft_interface import OpenVLA_OFT_Policy
        from ript.env_runner.openvla_oft_libero_runner import get_vla_action_batch
        policy = OpenVLA_OFT_Policy(str(ROOT / ckpt['base']), str(ROOT / ckpt['scale']),
                    SUITE, 32, 0.0, str(ROOT / ckpt['adapter']) if args.model == 'ript' else None,
                    seed=args.seed, fix_scale_head=True)
        modules = [policy.model, policy.action_head, policy.scale_head, policy.proprio_projector]
        def predict(observation, language):
            return get_vla_action_batch(policy.cfg, policy.model, policy.processor,
                    [observation], language, action_head=policy.action_head,
                    proprio_projector=policy.proprio_projector, scale_head=policy.scale_head,
                    use_laplace_sampling=False, scale_factor=1.0)[0][0]
    for module in modules:
        if module is not None:
            module.eval()
            module.requires_grad_(False)
    for entry in pending:
        # Per-task reseeding makes resume independent of previously completed tasks.
        oft.set_seed_everywhere(args.seed)
        task = suite.get_task(entry['id'] - 1)
        env, language = oft.get_libero_env(task, 'openvla', resolution=256)
        try:
            env.reset()
            obs = env.set_init_state(suite.get_task_init_states(entry['id'] - 1)[0])
            for _ in range(10):
                obs, _, _, _ = env.step(oft.get_libero_dummy_action('openvla'))
            queue = deque()
            success = False
            with torch.inference_mode():
                for step in range(300):
                    if not queue:
                        observation, _ = oft.prepare_observation(obs, 224)
                        actions = predict(observation, language)
                        assert len(actions) == 8
                        queue.extend(actions)
                    action = oft.process_action(queue.popleft(), 'openvla')
                    obs, _, done, _ = env.step(action.tolist())
                    if done:
                        success = True
                        break
            row = dict(revision=REVISION, suite=SUITE, model=args.model, seed=args.seed,
                       task_id=entry['id'], task_index=entry['id'] - 1, task_name=entry['name'],
                       shift=entry['category'], difficulty=entry['difficulty_level'],
                       success=bool(success), failure=not bool(success), steps=step + 1)
            path = out / f"task-{entry['id']:04d}.json"
            write_result(path, row)
            print(json.dumps(row), flush=True)
        finally:
            env.close()  # Exceptions abort; unfinished task is retried on resume.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, default='sft')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--plus', type=Path, default=ROOT / 'vendor/LIBERO-plus')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/libero-plus')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--check-import', action='store_true')
    args = parser.parse_args()
    args.plus = args.plus.resolve()
    if not args.execute and not args.check_import:
        print(json.dumps(dict(revision=REVISION, suite=SUITE, model=args.model, seed=args.seed,
              checkpoints=checkpoints(args.model), tasks=2591, trials_per_task=1,
              plus=str(args.plus), output=str(args.output), execute=False), indent=2))
        return
    configure(args.plus)
    entries = task_mapping(args.plus)
    print('shift counts:', dict(Counter(e['category'] for e in entries)))
    run(args, entries)


if __name__ == '__main__':
    main()
