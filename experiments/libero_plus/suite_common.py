"""Long/Spatial evaluation using the Goal evaluator's model and action paths.

Goal remains owned by evaluate.py. This module has no training entrypoint.
"""
import argparse
import ast
from collections import Counter, deque
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SUITES = {'long': ('libero_10', 'libero_10_no_noops', 520, 2519),
          'spatial': ('libero_spatial', 'libero_spatial_no_noops', 220, 2402)}
MODELS = ('sft', 'scale', 'ript')
PLUS_REVISION = '4976dc30028e805ff8094b55501d532c48fec182'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checkpoint_paths(suite, model):
    require(suite in SUITES and model in MODELS, 'Unknown suite or model')
    base = ROOT / 'checkpoints' / suite / 'oft'
    result = {'base': str(base)}
    if model != 'sft':
        root = ROOT / 'checkpoints/ript/openvla_oft'
        label = suite.upper()
        result['scale'] = str(root / f'scale_header/LIBERO_{label}_scale_header.pth')
        if model == 'ript':
            result['adapter'] = str(root / f'ript_adaptors/LIBERO_{label}_lora')
            result['rl_heads'] = str(Path(result['adapter']) / 'openvla_headers.pt')
    return result


def checkpoint_check(suite, model):
    paths = checkpoint_paths(suite, model)
    base = Path(paths['base'])
    norm_key = SUITES[suite][1]
    errors = []
    info = {'paths': paths, 'norm_key': norm_key, 'missing': errors}
    try:
        config = json.loads((base/'config.json').read_text())
        stats = json.loads((base/'dataset_statistics.json').read_text())
        index = json.loads((base/'model.safetensors.index.json').read_text())
        require(config['architectures'] == ['OpenVLAForActionPrediction'], 'Unexpected architecture')
        require(config['llm_backbone_id'] == 'llama2-7b-pure', 'Unexpected language backbone')
        require(len(config['timm_model_ids']) == 2, 'Expected two vision encoders')
        require(config['image_sizes'] == [224, 224], 'Unexpected image sizes')
        # OFT keeps LIBERO continuous-action statistics in dataset_statistics.json;
        # config.norm_stats contains unrelated pretrained OpenVLA datasets.
        require(norm_key in stats, 'Normalization key absent')
        for name, dim in [('action', 7), ('proprio', 8)]:
            for field in ('mean', 'std'):
                require(len(stats[norm_key][name][field]) == dim, f'{name} {field} dimension mismatch')
        require(len(stats[norm_key]['action']['mask']) == 7, 'Action mask dimension mismatch')
        shards = sorted(set(index['weight_map'].values()))
        require(len(shards) == 4, 'Unexpected model shard count')
        required = shards + ['config.json', 'dataset_statistics.json', 'model.safetensors.index.json',
                             'preprocessor_config.json', 'processor_config.json', 'tokenizer.json']
        for kind in ('action_head', 'proprio_projector'):
            matches = list(base.glob(f'{kind}--*_checkpoint.pt'))
            require(len(matches) <= 1, f'Multiple {kind} checkpoints')
            required.append(matches[0].name if matches else f'{kind}--150000_checkpoint.pt')
        errors.extend(str(base/name) for name in required if not (base/name).is_file())
        info.update(architecture=config['architectures'][0], vision_encoders=config['timm_model_ids'],
                    language_backbone=config['llm_backbone_id'], shards=shards,
                    action_head=next(n for n in required if n.startswith('action_head--')),
                    proprio_projector=next(n for n in required if n.startswith('proprio_projector--')),
                    normalization_sha256=sha(base/'dataset_statistics.json'))
    except (OSError, KeyError, ValueError) as exc:
        errors.append(str(exc))
    if model != 'sft':
        if not Path(paths['scale']).is_file(): errors.append(paths['scale'])
        if model == 'ript':
            adapter = Path(paths['adapter'])
            for name in ('adapter_config.json', 'adapter_model.safetensors', 'openvla_headers.pt'):
                if not (adapter/name).is_file(): errors.append(str(adapter/name))
            if (adapter/'adapter_config.json').is_file():
                c = json.loads((adapter/'adapter_config.json').read_text())
                expected = 'libero-10' if suite == 'long' else 'libero-spatial'
                require(expected in c['base_model_name_or_path'], 'Cross-suite adapter base')
                require(c['r'] == 32 and c['lora_alpha'] == 16 and c['lora_dropout'] == 0,
                        'LoRA hyperparameter mismatch')
                info['lora'] = {'rank': c['r'], 'alpha': c['lora_alpha'],
                                'targets': c['target_modules']}
    info['ready'] = not errors
    return info


def names_from_map(folder, suite):
    tree = ast.parse((folder/'libero_suite_task_map.py').read_text())
    mapping = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'libero_task_map' for t in node.targets))
    return mapping[suite]


def task_plan(suite, benchmark, dataset):
    name = SUITES[suite][0]
    folder = benchmark/'libero/libero/benchmark'
    names = names_from_map(folder, name)
    if dataset == 'plus':
        classes = json.loads((folder/'task_classification.json').read_text())
        require(name in classes, f'{name} has no official Plus classification')
        entries = classes[name]
        require(len(entries) == len(names) == SUITES[suite][3], 'Plus task count mismatch')
        for i, entry in enumerate(entries):
            require(entry['id'] == i+1 and entry['name'] == names[i], f'Plus task map mismatch at {i+1}')
        require(len(set(names)) == len(names), 'Duplicate Plus task names')
        return entries
    require(len(names) == 10, 'Original suite does not contain 10 tasks')
    tree = ast.parse((folder/'__init__.py').read_text())
    order = next(ast.literal_eval(node.value)[0] for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'task_orders' for t in node.targets))
    require(sorted(order) == list(range(10)), 'Unexpected nominal task order')
    return [dict(id=i+1, name=names[j], category='Nominal', difficulty_level=None) for i,j in enumerate(order)]


def checkpoint_provenance(paths):
    files = []
    for value in paths.values():
        p = Path(value)
        if p.is_dir():
            files.extend(x for x in p.rglob('*') if x.is_file() and '.cache' not in x.parts)
        elif p.is_file(): files.append(p)
    return {str(p): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(set(files))}


def preflight(suite, model, dataset, benchmark):
    require(suite in SUITES and model in MODELS and dataset in ('nominal','plus'), 'Invalid request')
    errors = []
    info = {'suite': suite, 'model': model, 'dataset': dataset,
            'checkpoint': checkpoint_check(suite, model)}
    if not info['checkpoint']['ready']:
        errors.extend('Missing/incompatible checkpoint: '+x for x in info['checkpoint']['missing'])
    try:
        expected = PLUS_REVISION if dataset == 'plus' else json.loads((ROOT/'stage0/repos.lock.json').read_text())['LIBERO']['revision']
        actual = subprocess.check_output(['git','-C',str(benchmark),'rev-parse','HEAD'], text=True).strip()
        require(actual == expected, f'Benchmark revision mismatch: {actual}')
        if dataset == 'plus':
            require(not subprocess.check_output(['git','-C',str(benchmark),'diff','HEAD','--']),
                    'LIBERO-Plus tracked source changed')
        for folder_name in ('assets','bddl_files','init_files'):
            require((benchmark/'libero/libero'/folder_name).is_dir(),
                    f'Missing benchmark {folder_name}')
        if dataset == 'nominal':
            require((ROOT/'runtime/libero-config/config.yaml').is_file(),
                    'Original LIBERO runtime config missing')
        entries = task_plan(suite, benchmark, dataset)
        info.update(tasks=len(entries), categories=dict(Counter(e['category'] for e in entries)),
                    mapping_sha256=sha(benchmark/'libero/libero/benchmark/libero_suite_task_map.py'),
                    benchmark_revision=actual)
    except (OSError, subprocess.CalledProcessError, KeyError, ValueError) as exc:
        errors.append(str(exc))
    info['errors'] = errors
    info['ready'] = not errors
    return info


def configure(dataset, benchmark):
    sys.path[:0] = [str(benchmark), str(ROOT/'runtime/openvla-oft'), str(ROOT/'runtime/ript-vla')]
    os.environ.update(MUJOCO_GL='egl', TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if dataset == 'plus':
        from experiments.libero_plus import evaluate as goal
        goal.configure(benchmark)
    else:
        os.environ['LIBERO_CONFIG_PATH'] = str(ROOT/'runtime/libero-config')


def predict_fn(suite, model, seed):
    import torch
    from experiments.robot.libero import run_libero_eval as oft
    from experiments.robot import openvla_utils
    openvla_utils.update_auto_map = lambda _: None
    openvla_utils.check_model_logic_mismatch = lambda _: None
    paths = checkpoint_paths(suite, model)
    task_suite = SUITES[suite][0]
    if model == 'sft':
        cfg = oft.GenerateConfig(pretrained_checkpoint=paths['base'], task_suite_name=task_suite,
                                 seed=seed, num_trials_per_task=1, center_crop=True)
        base, head, proprio, noisy, processor = oft.initialize_model(cfg)
        require(cfg.unnorm_key == SUITES[suite][1], 'SFT normalization key mismatch')
        modules = (base, head, proprio, noisy)
        def predict(obs, language):
            return oft.get_action(cfg, base, obs, language, processor=processor,
                                  action_head=head, proprio_projector=proprio, noisy_action_projector=noisy)
    else:
        from ript.algos.rl_optimizers.openvla_oft_interface import OpenVLA_OFT_Policy
        from ript.env_runner.openvla_oft_libero_runner import get_vla_action_batch
        policy = OpenVLA_OFT_Policy(paths['base'], paths['scale'], task_suite, 32, 0.,
                    paths.get('adapter'), seed=seed, fix_scale_head=True)
        modules = (policy.model, policy.action_head, policy.scale_head, policy.proprio_projector)
        require(policy.cfg.unnorm_key == SUITES[suite][1], 'Cached inference normalization key mismatch')
        lora_b = [param for name,param in policy.model.named_parameters() if 'lora_B' in name]
        require(bool(lora_b), 'No LoRA B matrices found')
        has_trained_delta = any(bool(torch.count_nonzero(p).item()) for p in lora_b)
        require(has_trained_delta == (model == 'ript'), 'Scale/RIPT LoRA state mismatch')
        def predict(obs, language):
            return get_vla_action_batch(policy.cfg, policy.model, policy.processor, [obs], language,
                action_head=policy.action_head, proprio_projector=policy.proprio_projector,
                scale_head=policy.scale_head, use_laplace_sampling=False, scale_factor=1.0)[0][0]
    for module in modules:
        if module is not None:
            module.eval()
            module.requires_grad_(False)
    return predict


def result_path(out, task_id, trial, dataset):
    return out / (f'task-{task_id:04d}-trial-{trial:02d}.json' if dataset == 'nominal'
                  else f'task-{task_id:04d}.json')


def run(args, dataset):
    from libero.libero import benchmark
    import libero.libero as libero
    require(Path(libero.__file__).resolve().is_relative_to(args.benchmark), 'Wrong LIBERO import path')
    task_suite_name = SUITES[args.suite][0]
    entries = task_plan(args.suite, args.benchmark, dataset)
    suite = benchmark.get_benchmark_dict()[task_suite_name](task_order_index=0)
    require(suite.n_tasks == len(entries), 'Benchmark task count mismatch')
    require([suite.get_task(i).name for i in range(suite.n_tasks)] == [e['name'] for e in entries],
            'Imported benchmark task order differs from manifest')
    if args.check_import:
        print(json.dumps({'import': str(libero.__file__), 'tasks': suite.n_tasks}))
        return
    import numpy as np
    import torch
    from experiments.robot.libero import run_libero_eval as oft
    trials = 50 if dataset == 'nominal' else 1
    horizon = SUITES[args.suite][2] if dataset == 'nominal' or args.suite == 'long' else 280
    output = args.output/args.suite/args.model/f'seed-{args.seed}'
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (output/'.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        import experiments.libero_plus.evaluate as goal
        manifest = {'suite': task_suite_name, 'suite_label': args.suite, 'dataset': dataset,
                    'model': args.model, 'seed': args.seed, 'trials_per_task': trials,
                    'horizon': horizon, 'wait_steps': 10, 'env_seed': 0,
                    'image_resolution': 256, 'policy_resolution': 224, 'center_crop': True,
                    'action_chunk': [8, 7], 'sampling': False,
                    'task_mapping': entries, 'benchmark_revision': preflight(args.suite,args.model,dataset,args.benchmark)['benchmark_revision'],
                    'evaluator_sha256': sha(__file__), 'checkpoint': checkpoint_provenance(checkpoint_paths(args.suite,args.model)),
                    'python': platform.python_version()}
        mp = output/'manifest.json'
        if mp.exists():
            require(json.loads(mp.read_text()) == manifest, 'Resume manifest mismatch')
        else:
            require(not list(output.glob('task-*.json')), 'Results without manifest')
            goal.write_result(mp, manifest)
        pending = [(e, t) for e in entries for t in range(trials)
                   if args.task_ids is None or e['id'] in args.task_ids]
        if args.max_trials is not None:
            pending = [(e,t) for e,t in pending if t < args.max_trials]
        for entry, trial in pending:
            dest = result_path(output, entry['id'], trial, dataset)
            if dest.exists():
                old = json.loads(dest.read_text())
                require(all(old[k] == v for k,v in {'suite':task_suite_name,'model':args.model,'task_id':entry['id'],
                        'task_name':entry['name'],'trial_id':trial,'seed':args.seed}.items()), 'Existing result identity mismatch')
                require(type(old['success']) is bool and 0 < old['steps'] <= horizon, 'Invalid existing result')
        pending = [(e,t) for e,t in pending if not result_path(output,e['id'],t,dataset).exists()]
        if args.max_tasks is not None:
            ids = set(e['id'] for e,_ in pending)
            ids = set(sorted(ids)[:args.max_tasks])
            pending = [(e,t) for e,t in pending if e['id'] in ids]
        if not pending:
            print('All selected episodes complete')
            return
        oft.set_seed_everywhere(args.seed)
        predict = predict_fn(args.suite,args.model,args.seed)
        runtime = {'torch':torch.__version__, 'cuda':torch.version.cuda,
                   'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                   'numpy':np.__version__}
        for entry, trial in pending:
            oft.set_seed_everywhere(args.seed)
            idx = entry['id'] - 1
            task = suite.get_task(idx)
            states = suite.get_task_init_states(idx)
            state_idx = trial if dataset == 'nominal' else 0
            require(state_idx < len(states), f'Missing initial state {state_idx} for task {entry["id"]}')
            env, language = oft.get_libero_env(task,'openvla',resolution=256)
            try:
                env.reset()
                obs = env.set_init_state(states[state_idx])
                for _ in range(10):
                    obs,_,_,_ = env.step(oft.get_libero_dummy_action('openvla'))
                queue = deque()
                success = False
                with torch.inference_mode():
                    for step in range(horizon):
                        if not queue:
                            prepared,_ = oft.prepare_observation(obs,224)
                            require(prepared['full_image'].shape[:2] == (224,224) and
                                    prepared['wrist_image'].shape[:2] == (224,224) and
                                    np.asarray(prepared['state']).shape == (8,) and
                                    np.isfinite(prepared['state']).all(), 'Invalid prepared observation')
                            actions = np.asarray(predict(prepared,language))
                            require(actions.shape == (8,7) and np.isfinite(actions).all(), 'Invalid action chunk')
                            queue.extend(actions)
                        action = oft.process_action(queue.popleft(),'openvla')
                        require(action.shape == (7,) and np.isfinite(action).all() and
                                np.max(np.abs(action)) <= 1.0001, 'Invalid executed action')
                        obs,_,done,_ = env.step(action.tolist())
                        if done:
                            success = True
                            break
                row = {'suite':task_suite_name,'dataset':dataset,'model':args.model,'seed':args.seed,
                       'task_id':entry['id'],'task_index':idx,'task_name':entry['name'],
                       'shift':entry['category'],'difficulty':entry['difficulty_level'],
                       'trial_id':trial,'initial_state_index':state_idx,'success':bool(success),
                       'failure':not bool(success),'steps':step+1,'language':language,
                       'benchmark_revision':manifest['benchmark_revision'],
                       'evaluator_sha256':manifest['evaluator_sha256'], 'runtime':runtime,
                       'host':platform.node()}
                dest = result_path(output,entry['id'],trial,dataset)
                require(not dest.exists(), 'Refusing to overwrite result')
                goal.write_result(dest,row)
                require(json.loads(dest.read_text()) == row, 'Result save/reload mismatch')
                print(json.dumps({'task_id':entry['id'],'trial':trial,'success':success,'steps':step+1}),flush=True)
            finally:
                env.close()


def cli(dataset):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=tuple(SUITES), required=True)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--benchmark', type=Path, default=ROOT/'vendor'/('LIBERO-plus' if dataset=='plus' else 'LIBERO'))
    parser.add_argument('--output', type=Path, default=ROOT/'outputs'/('libero-plus' if dataset=='plus' else 'libero-nominal'))
    parser.add_argument('--task-ids', type=int, nargs='+')
    parser.add_argument('--max-tasks', type=int)
    parser.add_argument('--max-trials', type=int, help='Bounded subset of trials per selected task')
    parser.add_argument('--check-import', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Explicit CPU-only preflight (also the default)')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    args.benchmark = args.benchmark.resolve()
    require(args.max_tasks is None or args.max_tasks > 0, 'max-tasks must be positive')
    require(args.max_trials is None or 0 < args.max_trials <= (50 if dataset=='nominal' else 1),
            'max-trials outside protocol')
    require(not (args.dry_run and (args.execute or args.check_import)),
            'dry-run cannot be combined with execution or import check')
    if args.task_ids is not None:
        require(len(args.task_ids)==len(set(args.task_ids)) and all(n>0 for n in args.task_ids), 'Invalid task IDs')
    plan = preflight(args.suite,args.model,dataset,args.benchmark)
    plan.update(output=str(args.output),seed=args.seed,requested_task_ids=args.task_ids,
                max_tasks=args.max_tasks,max_trials=args.max_trials)
    print(json.dumps(plan,indent=2))
    if not args.execute and not args.check_import: return 0 if plan['ready'] else 2
    if args.execute:
        require(plan['ready'], 'Preflight failed')
    else:
        require('tasks' in plan and 'benchmark_revision' in plan,
                'Benchmark mapping/revision preflight failed')
    if args.task_ids is not None:
        require(set(args.task_ids) <= {e['id'] for e in task_plan(args.suite,args.benchmark,dataset)}, 'Unknown task IDs')
    configure(dataset,args.benchmark)
    run(args,dataset)
    return 0
