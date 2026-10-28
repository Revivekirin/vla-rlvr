"""P3-A1: two-case Camera smoke, captured trajectories and isolated cached replay.

CPU operations: dry-run, validate. GPU operations: capture, compare (workers).
Only run-local outputs are written; existing runs/artifacts are never replaced.
"""
import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True

from evaluate import ROOT, REVISION, SUITE, checkpoints, task_mapping

P3 = ROOT / 'outputs/rlvr_analysis/p3'
MODELS = ('scale', 'ript')
SHIFT = 'Camera Viewpoints'
IMPL = ('experiments/libero_plus/evaluate.py',
        'experiments/libero_plus/p3_action_analysis.py',
        'runtime/ript-vla/ript/algos/rl_optimizers/openvla_oft_interface.py',
        'runtime/ript-vla/ript/env_runner/openvla_oft_libero_runner.py',
        'runtime/openvla-oft/experiments/robot/openvla_utils.py',
        'runtime/openvla-oft/experiments/robot/robot_utils.py',
        'runtime/openvla-oft/experiments/robot/libero/run_libero_eval.py',
        'runtime/openvla-oft/experiments/robot/libero/libero_utils.py',
        'runtime/openvla-oft/prismatic/vla/constants.py',
        'runtime/openvla-oft/prismatic/models/action_heads.py',
        'runtime/openvla-oft/prismatic/vla/datasets/rlds/oxe/transforms.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def save_npz(path, **arrays):
    import numpy as np
    with path.open('xb') as f:
        np.savez_compressed(f, **arrays)


def run_path(run_id):
    require(bool(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id)), 'Invalid run ID')
    path = P3 / run_id
    require(path.resolve().parent == P3.resolve(), 'Run path escapes P3')
    require(not path.is_symlink(), 'Run directory must not be a symlink')
    return path


def check_revision(plus):
    require(plus.is_dir(), f'Missing LIBERO-Plus checkout: {plus}')
    revision = subprocess.check_output(['git', '-C', str(plus), 'rev-parse', 'HEAD'], text=True).strip()
    require(revision == REVISION, f'Wrong Plus revision: {revision}')
    require(not subprocess.check_output(['git', '-C', str(plus), 'diff', 'HEAD', '--'], text=True), 'Dirty Plus source')


def norm_spec(base):
    import numpy as np
    stats = read_json(base / 'dataset_statistics.json')
    key = SUITE if SUITE in stats else SUITE + '_no_noops'
    require(key in stats, 'Missing normalization key')
    stats = stats[key]
    for field, dim in [('action', 7), ('proprio', 8)]:
        for name in ('q01', 'q99'):
            value = np.asarray(stats[field][name])
            require(value.shape == (dim,) and np.isfinite(value).all(), f'Invalid {field}/{name}')
        mask = np.asarray(stats[field].get('mask', [True] * dim))
        require(mask.shape == (dim,) and mask.dtype == bool, f'Invalid {field} mask')
        require(np.all(np.asarray(stats[field]['q99']) > np.asarray(stats[field]['q01'])), f'Invalid {field} bounds')
    require(stats['action'].get('mask') == [True] * 6 + [False], 'Unexpected action normalization mask')
    return key, stats


def safetensor_header(path):
    """Validate tensor storage bounds on CPU without deserializing model weights."""
    import struct
    with path.open('rb') as f:
        length = f.read(8)
        require(len(length) == 8, f'Truncated safetensors: {path}')
        size = struct.unpack('<Q', length)[0]
        require(0 < size <= 128 * 1024 * 1024, f'Invalid safetensors header: {path}')
        header = json.loads(f.read(size))
    available = path.stat().st_size - 8 - size
    widths = {'BOOL': 1, 'U8': 1, 'I8': 1, 'I16': 2, 'U16': 2, 'F16': 2, 'BF16': 2,
              'I32': 4, 'U32': 4, 'F32': 4, 'F64': 8, 'I64': 8, 'U64': 8}
    intervals = []
    for name, item in header.items():
        if name == '__metadata__':
            continue
        low, high = item['data_offsets']
        count = 1
        for dim in item['shape']:
            require(type(dim) is int and dim >= 0, f'Invalid tensor shape: {path}/{name}')
            count *= dim
        require(item['dtype'] in widths and 0 <= low <= high <= available
                and high - low == count * widths[item['dtype']], f'Invalid tensor storage: {path}/{name}')
        intervals.append((low, high))
    cursor = 0
    for low, high in sorted(intervals):
        require(low == cursor, f'Gapped/overlapping safetensor: {path}')
        cursor = high
    require(cursor == available and bool(intervals), f'Truncated/trailing safetensor data: {path}')
    return set(header) - {'__metadata__'}


def checkpoint_files():
    """Require shards and exactly the component files the original loader selects."""
    result = {}
    for model in MODELS:
        ck = checkpoints(model)
        base = ROOT / ck['base']
        index = read_json(base / 'model.safetensors.index.json')
        required = ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'preprocessor_config.json',
                    'dataset_statistics.json', 'modeling_prismatic.py')
        for name in required:
            require((base / name).is_file(), f'Missing checkpoint configuration: {base / name}')
        files = list(base.glob('*.json')) + list(base.glob('*.py')) + list(base.glob('tokenizer.*'))
        files += [base / f for f in sorted(set(index['weight_map'].values()))]
        for component in ('action_head', 'proprio_projector'):
            matches = list(base.glob(f'*{component}*.pt'))
            require(len(matches) == 1, f'Ambiguous/missing {component}: {matches}')
            files += matches
        files.append(ROOT / ck['scale'])
        if model == 'ript':
            adapter = ROOT / ck['adapter']
            files += [adapter / 'adapter_config.json', ROOT / ck['rl_heads']]
            weights = [p for p in (adapter / 'adapter_model.safetensors', adapter / 'adapter_model.bin') if p.is_file()]
            require(len(weights) == 1, 'Missing/ambiguous adapter weights')
            files += weights
        for path in files:
            require(path.is_file() and path.stat().st_size > 0, f'Missing checkpoint file: {path}')
        for shard in set(index['weight_map'].values()):
            names = safetensor_header(base / shard)
            require(names == {k for k, v in index['weight_map'].items() if v == shard}, f'Shard/index tensor mismatch: {shard}')
        if model == 'ript' and weights[0].suffix == '.safetensors':
            safetensor_header(weights[0])
        result[model] = sorted(set(str(p.resolve()) for p in files))
    return result


def select_cases(candidate, entries, results, seed, id_column=None):
    with candidate.open(newline='') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        if id_column is None:
            matches = [k for k in ('task_id', 'id') if k in columns]
            require(len(matches) == 1, f'Specify --id-column; CSV columns: {columns}')
            id_column = matches[0]
        require(id_column in columns, f'Missing CSV column {id_column}')
        rows = list(reader)
    by_id = {e['id']: e for e in entries}
    selected, seen = {}, set()
    for row in rows:
        task_id = int(row[id_column])
        require(task_id not in seen, f'Duplicate candidate task ID {task_id}')
        seen.add(task_id)
        require(task_id in by_id, f'Unknown candidate ID {task_id}')
        entry = by_id[task_id]
        for column, expected in [('task_name', entry['name']), ('shift', entry['category'])]:
            if row.get(column):
                require(row[column] == expected, f'Candidate {column} disagrees with mapping: {task_id}')
        if entry['category'] != SHIFT:
            continue
        records = {}
        for model in MODELS:
            path = results / model / f'seed-{seed}' / f'task-{task_id:04d}.json'
            record = read_json(path)
            expected = dict(task_id=task_id, task_name=entry['name'], shift=SHIFT, model=model, seed=seed, revision=REVISION)
            require(all(record.get(k) == v for k, v in expected.items()), f'Invalid evaluation identity: {path}')
            require(type(record.get('success')) is bool and record.get('failure') == (not record['success']), f'Invalid outcome: {path}')
            records[model] = dict(path=str(path.resolve()), sha256=sha(path), record=record)
        s, r = (records[m]['record']['success'] for m in MODELS)
        case = 'lost' if s and not r else 'gained' if r and not s else 'unchanged'
        for column in ('case', 'transition', 'pair_type'):
            if row.get(column) in ('lost', 'gained'):
                require(row[column] == case, f'CSV transition conflicts with results: {task_id}')
        if case in ('lost', 'gained') and case not in selected:
            selected[case] = dict(task=entry, historical_case=case, evaluation=records, candidate_row=row)
    require(set(selected) == {'lost', 'gained'}, 'Need one Camera lost and one Camera gained candidate')
    return [selected[k] for k in ('lost', 'gained')]


def verify_ordering():
    """Read installed controller source without importing robosuite or initializing GL."""
    spec = importlib.util.find_spec('robosuite')
    require(spec is not None and spec.origin, 'robosuite unavailable for action-order audit')
    package = Path(spec.origin).parent
    osc = package / 'controllers/osc.py'
    arm = package / 'robots/single_arm.py'
    source = re.sub(r'\s+', '', osc.read_text())
    robot = re.sub(r'\s+', '', arm.read_text())
    require('delta=action' in source and 'scaled_delta=self.scale_action(delta)' in source
            and 'set_goal_position(scaled_delta[:3],' in source
            and 'set_goal_orientation(scaled_delta[3:],' in source, 'Unrecognized OSC position/rotation ordering')
    require('arm_action=action[:self.controller.control_dim]' in robot and
            'gripper_action=action[self.controller.control_dim:]' in robot, 'Unrecognized robot arm/gripper ordering')
    return dict(translation=[0, 1, 2], rotation=[3, 4, 5], gripper=[6],
                evidence={str(p): sha(p) for p in (osc, arm)})


def physical_task_inputs(plus, folder, bddl_file, init_file):
    """Resolve Camera virtual filenames for validation only; never alter the Task.

    Mirrors pinned Plus ControlEnv.__init__ and Benchmark.get_task_init_states.
    The Camera smoke uses the _view_ branch, not the other perturbation branches.
    """
    if '_view_' in bddl_file:
        require('_initstate_' in bddl_file, 'Malformed Camera BDDL identifier')
        bddl_file = bddl_file.split('_view_')[0] + '.bddl'
    if '_view_' in init_file:
        require('_language_' not in init_file, 'Mixed Language/Camera task outside this smoke')
        init_file = init_file.split('_view_')[0] + '.' + init_file.split('.')[-1]
    root = plus / 'libero/libero'
    paths = [root / 'bddl_files' / folder / bddl_file,
             root / 'init_files' / folder / init_file]
    for path in paths:
        require(path.is_file(), f'Missing physical task input: {path}')
    return paths


def preflight(args):
    """No torch/model/environment imports. Aggregate failures for a useful dry run."""
    result = dict(ready=False, errors=[], seed=args.seed, source=args.source, horizon=args.horizon)
    checks = [
        ('mapping', lambda: (check_revision(args.plus), task_mapping(args.plus))[1]),
        ('checkpoints', checkpoint_files),
        ('normalization', lambda: norm_spec(ROOT / checkpoints('scale')['base'])),
        ('action_order', verify_ordering),
    ]
    for key, fn in checks:
        try:
            result[key] = fn()
        except Exception as e:
            result['errors'].append(f'{key}: {e}')
    for dep in ('numpy', 'torch', 'transformers', 'peft', 'tensorflow', 'mujoco', 'wand', 'skimage'):
        if importlib.util.find_spec(dep) is None:
            result['errors'].append(f'Missing dependency: {dep}')
    try:
        require(args.candidates.is_file(), f'Missing candidate CSV: {args.candidates}')
        if 'mapping' in result:
            result['cases'] = select_cases(args.candidates, result['mapping'], args.results, args.seed, args.id_column)
            require((args.plus / 'libero/libero/assets').is_dir(), 'Missing Plus assets directory')
            result['physical_task_inputs'] = {}
            for case in result['cases']:
                name = case['task']['name']
                paths = physical_task_inputs(args.plus, SUITE, name + '.bddl', name + '.pruned_init')
                result['physical_task_inputs'][str(case['task']['id'])] = [str(path) for path in paths]
            for model in MODELS:
                path = args.results / model / f'seed-{args.seed}' / 'manifest.json'
                m = read_json(path)
                expected = dict(revision=REVISION, suite=SUITE, model=model, seed=args.seed,
                                horizon=300, wait_steps=10, env_seed=0, initial_state_index=0,
                                image_resolution=256, policy_resolution=224, center_crop=True,
                                action_chunk=8, sampling=False, protocol='goal-plus-v1')
                require(all(m.get(k) == v for k, v in expected.items()), f'Manifest protocol mismatch: {path}')
                require(m['tasks'] == result['mapping'], 'Manifest mapping mismatch')
                require(m['checkpoints'] == checkpoints(model), 'Manifest checkpoint mismatch')
                require(m['evaluator_sha256'] == sha(ROOT / IMPL[0]), 'Evaluation source differs from original manifest')
    except Exception as e:
        result['errors'].append(str(e))
    result.pop('mapping', None)
    result['ready'] = not result['errors']
    return result


def input_sha(obs, language):
    import numpy as np
    h = hashlib.sha256()
    for key in ('full_image', 'wrist_image', 'state'):
        a = np.ascontiguousarray(obs[key])
        h.update(json.dumps([key, a.dtype.str, list(a.shape)], separators=(',', ':')).encode())
        h.update(a.tobytes())
    h.update(json.dumps(language, ensure_ascii=False).encode())
    return h.hexdigest()


def check_obs(obs, language):
    import numpy as np
    require(isinstance(language, str) and bool(language.strip()), 'Empty/non-string language')
    for key in ('full_image', 'wrist_image'):
        a = np.asarray(obs[key])
        require(a.shape == (224, 224, 3) and a.dtype == np.uint8, f'Invalid {key}')
    a = np.asarray(obs['state'])
    require(a.shape == (8,) and a.dtype.kind == 'f' and np.isfinite(a).all(), 'Invalid raw proprio')


def check_actions(mean, raw, executed, stats):
    import numpy as np
    for name, a in [('mean', mean), ('unnormalized', raw), ('executed', executed)]:
        require(a.shape[-2:] == (8, 7) and a.dtype.kind == 'f' and np.isfinite(a).all(), f'Invalid action {name}')
    require(mean.shape == raw.shape == executed.shape, 'Action shapes differ')
    s = stats['action']
    expected = np.where(s['mask'], .5 * (mean + 1) * (np.array(s['q99']) - s['q01'] + 1e-8) + s['q01'], mean)
    require(np.allclose(raw, expected, rtol=1e-6, atol=1e-7), 'Action unnormalization mismatch')
    final = raw.copy()
    final[..., -1] = -np.sign(2 * raw[..., -1] - 1)
    require(np.array_equal(final, executed), 'Executed action postprocessing mismatch')


def setup(run, metadata, role):
    """Equivalent evaluator configuration, with all generated files inside this run."""
    plus = Path(metadata['plus'])
    check_revision(plus)
    config = run / ('config-' + role)
    config.mkdir(exist_ok=False)
    base = plus / 'libero/libero'
    paths = dict(benchmark_root=str(base), bddl_files=str(base / 'bddl_files'),
                 init_states=str(base / 'init_files'), assets=str(base / 'assets'), datasets=str(plus / 'datasets'))
    save_json(config / 'config.yaml', paths)  # JSON is YAML
    os.environ.update(LIBERO_CONFIG_PATH=str(config), MUJOCO_GL='egl',
                      TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_MODULES_CACHE=str(run / ('hf-modules-' + role)))
    sys.path[:0] = [str(plus), str(ROOT / 'runtime/openvla-oft'), str(ROOT / 'runtime/ript-vla')]
    os.chdir(run)
    print(f'[{role}] Config saved; importing torch and evaluator dependencies', flush=True)
    import torch
    from experiments.robot.libero import run_libero_eval as oft
    from experiments.robot import openvla_utils
    openvla_utils.update_auto_map = lambda _: None
    openvla_utils.check_model_logic_mismatch = lambda _: None
    print(f'[{role}] Imports complete; checking CUDA visibility', flush=True)
    require(torch.cuda.is_available(), 'CUDA required; refusing CPU model load')
    oft.set_seed_everywhere(metadata['seed'])
    return torch, oft


def load_policy(model, metadata):
    import numpy as np
    import torch
    from ript.algos.rl_optimizers.openvla_oft_interface import OpenVLA_OFT_Policy
    from ript.env_runner.openvla_oft_libero_runner import get_vla_action_batch
    from experiments.robot.libero import run_libero_eval as oft
    from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_DIM, NUM_ACTIONS_CHUNK
    ck = checkpoints(model)
    policy = OpenVLA_OFT_Policy(str(ROOT / ck['base']), str(ROOT / ck['scale']), SUITE, 32, 0.,
                               str(ROOT / ck['adapter']) if model == 'ript' else None,
                               seed=metadata['seed'], fix_scale_head=True)
    for module in (policy.model, policy.action_head, policy.scale_head, policy.proprio_projector):
        module.eval().requires_grad_(False)
    require(ACTION_DIM == 7 and NUM_ACTIONS_CHUNK == 8 and ACTION_PROPRIO_NORMALIZATION_TYPE.value == 'bounds_q99', 'Wrong platform constants')
    require(policy.cfg.unnorm_key == metadata['normalization_key'], 'Normalization key mismatch')
    require(policy.model.norm_stats[policy.cfg.unnorm_key] == metadata['normalization'], 'Loaded normalization mismatch')
    require(policy.cfg.center_crop and policy.cfg.num_images_in_input == 2 and policy.cfg.use_proprio, 'Wrong preprocessing config')
    if model == 'scale':
        b = [p for n, p in policy.model.named_parameters() if 'lora_B' in n]
        require(bool(b) and all(torch.count_nonzero(p).item() == 0 for p in b), 'Scale LoRA B is not zero')
    info = dict(model=model, dtype=str(next(policy.action_head.parameters()).dtype),
                normalization_key=policy.cfg.unnorm_key, normalization_type='bounds_q99',
                attention_implementation=policy.model.language_model.config._attn_implementation,
                cached_inference=True, sampling=False, process_id=os.getpid(),
                versions={k: __import__(k).__version__ for k in ('torch', 'transformers', 'peft', 'numpy')},
                implementation_paths={name: inspect.getfile(obj) for name, obj in
                                      [('policy', OpenVLA_OFT_Policy), ('predict', get_vla_action_batch), ('model', type(policy.model.get_base_model()))]})
    def predict(obs, language):
        check_obs(obs, language)
        before = input_sha(obs, language)
        with torch.inference_mode():
            result = get_vla_action_batch(policy.cfg, policy.model, policy.processor,
                        [{k: v.copy() for k, v in obs.items()}], language,
                        action_head=policy.action_head, proprio_projector=policy.proprio_projector,
                        scale_head=policy.scale_head, use_laplace_sampling=False, scale_factor=1.)
        raw, sampled, log_prob, mean, log_scale = result
        require(all(torch.isfinite(v).all().item() for v in (sampled, log_prob, mean, log_scale)), 'Non-finite policy output')
        require(tuple(mean.shape) == (1, 8, 7) and raw.shape == (1, 8, 7), 'Invalid batch action shape')
        require(torch.equal(sampled, mean), 'Sampling enabled or mean changed')
        mean, raw = mean[0].float().cpu().numpy().copy(), raw[0].copy()
        executed = np.stack([oft.process_action(a.copy(), 'openvla') for a in raw])
        check_actions(mean, raw, executed, metadata['normalization'])
        require(input_sha(obs, language) == before, 'Prediction mutated saved input')
        return mean, raw, executed
    return predict, info


def reproduction(success, steps, horizon, expected):
    if not success and horizon < 300:
        return 'not_verified_truncated'
    return 'matched' if success == expected else 'reproduction_mismatch'


def capture_worker(args, run, metadata):
    import numpy as np
    _, oft = setup(run, metadata, 'capture')
    import libero.libero as libero
    from libero.libero import benchmark
    require(Path(libero.__file__).resolve().is_relative_to(Path(metadata['plus'])), 'Wrong LIBERO import')
    suite = benchmark.get_benchmark_dict()[SUITE](task_order_index=0)
    entries = task_mapping(Path(metadata['plus']))
    require([suite.get_task(i).name for i in range(suite.n_tasks)] == [e['name'] for e in entries], 'Suite mapping mismatch')
    task_files = {}
    for case in metadata['cases']:
        task = suite.get_task(case['task']['id'] - 1)
        for path in physical_task_inputs(Path(metadata['plus']), task.problem_folder,
                                         task.bddl_file, task.init_states_file):
            task_files[str(path)] = sha(path)
    predict, info = load_policy(metadata['source'], metadata)
    rows, episodes = [], []
    for case in metadata['cases']:
        task_id = case['task']['id']
        oft.set_seed_everywhere(metadata['seed'])
        task = suite.get_task(task_id - 1)
        env, language = oft.get_libero_env(task, 'openvla', resolution=256)
        try:
            robot = env.env.robots[0]
            require(robot.controller.control_dim == 6 and robot.action_dim == 7
                    and robot.controller.impedance_mode == 'fixed' and robot.controller.use_delta
                    and robot.controller.use_ori, 'Expected fixed 6D delta OSC + one gripper')
            env.reset()
            initial = suite.get_task_init_states(task_id - 1)[0]
            raw_obs = env.set_init_state(initial)
            for _ in range(10):
                raw_obs, _, _, _ = env.step(oft.get_libero_dummy_action('openvla'))
            success, step = False, 0
            while step < metadata['horizon'] and not success:
                obs, _ = oft.prepare_observation(raw_obs, 224)
                check_obs(obs, language)
                mean, raw, executed = predict(obs, language)
                row = dict(**{k: v.copy() for k, v in obs.items()}, language=language,
                           task_id=task_id, step=step, input_sha256=input_sha(obs, language),
                           normalized_mean=mean, unnormalized=raw, executed=executed)
                used = 0
                for action in executed:
                    if step == metadata['horizon']:
                        break
                    raw_obs, _, done, _ = env.step(action.tolist())
                    step += 1
                    used += 1
                    if done:
                        success = True
                        break
                row['executed_count'] = used
                rows.append(row)
            expected = case['evaluation'][metadata['source']]['record']
            status = reproduction(success, step, metadata['horizon'], expected['success'])
            episodes.append(dict(task_id=task_id, source=metadata['source'], shift=SHIFT,
                                 historical_case=case['historical_case'], analysis_case=case['historical_case'] if status == 'matched' else status,
                                 reproduction=status, success=success, steps=step,
                                 expected_success=expected['success'], expected_steps=expected.get('steps'),
                                 steps_match=step == expected.get('steps'),
                                 initial_state_sha256=hashlib.sha256(np.ascontiguousarray(initial).tobytes()).hexdigest()))
        finally:
            env.close()
    arrays = {key: np.asarray([r[key] for r in rows]) for key in rows[0]}
    save_npz(run / 'observations.npz', **arrays)
    save_json(run / 'capture.json', dict(episodes=episodes, policy=info, observations_sha256=sha(run / 'observations.npz'),
                                              task_files=task_files, dtypes={k: str(v.dtype) for k, v in arrays.items()}))


def load_observations(run, metadata):
    import numpy as np
    capture = read_json(run / 'capture.json')
    require(sha(run / 'observations.npz') == capture['observations_sha256'], 'Observation archive checksum mismatch')
    for path, digest in capture.get('task_files', {}).items():
        require(sha(path) == digest, f'Task file changed: {path}')
    with np.load(run / 'observations.npz', allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    needed = {'full_image', 'wrist_image', 'state', 'language', 'task_id', 'step', 'input_sha256',
              'normalized_mean', 'unnormalized', 'executed', 'executed_count'}
    require(set(data) == needed, 'Invalid observation archive keys')
    n = len(data['step'])
    require(0 < n <= 2 * ((metadata['horizon'] + 7) // 8), 'Invalid observation count')
    require(all(len(v) == n for v in data.values()), 'Observation array lengths differ')
    require(all(data[k].dtype.kind in 'iu' for k in ('task_id', 'step', 'executed_count')), 'Invalid integer metadata')
    expected_ids = {c['task']['id'] for c in metadata['cases']}
    require(set(data['task_id'].tolist()) == expected_ids, 'Observation tasks differ from selected cases')
    require({e['task_id'] for e in capture['episodes']} == expected_ids and len(capture['episodes']) == 2, 'Invalid episodes')
    for i in range(n):
        obs = {k: data[k][i] for k in ('full_image', 'wrist_image', 'state')}
        check_obs(obs, str(data['language'][i]))
        require(input_sha(obs, str(data['language'][i])) == data['input_sha256'][i], 'Input checksum mismatch')
        require(1 <= data['executed_count'][i] <= 8, 'Invalid execution count')
    for episode in capture['episodes']:
        require(type(episode['success']) is bool and 0 < episode['steps'] <= metadata['horizon'], 'Invalid episode outcome')
        require(episode['success'] or episode['steps'] == metadata['horizon'], 'Failed episode stopped before horizon')
        indexes = np.flatnonzero(data['task_id'] == episode['task_id'])
        require(np.array_equal(data['step'][indexes], np.arange(len(indexes)) * 8), 'Invalid chunk steps/order')
        require(np.all(data['executed_count'][indexes[:-1]] == 8), 'Incomplete non-final chunk')
        require(int(data['executed_count'][indexes].sum()) == episode['steps'], 'Episode step accounting mismatch')
        case = next(c for c in metadata['cases'] if c['task']['id'] == episode['task_id'])
        expected = case['evaluation'][metadata['source']]['record']['success']
        status = reproduction(episode['success'], episode['steps'], metadata['horizon'], expected)
        require(status == episode['reproduction'], 'Invalid reproduction status')
        require(episode['analysis_case'] == (case['historical_case'] if status == 'matched' else status), 'Mislabelled reproduction')
    check_actions(data['normalized_mean'], data['unnormalized'], data['executed'], metadata['normalization'])
    return data, capture


def replay_worker(args, run, metadata):
    import numpy as np
    data, _ = load_observations(run, metadata)
    setup(run, metadata, 'replay-' + args.model)
    predict, info = load_policy(args.model, metadata)
    outputs = [predict({k: data[k][i] for k in ('full_image', 'wrist_image', 'state')}, str(language))
               for i, language in enumerate(data['language'])]
    save_npz(run / f'{args.model}_actions.npz', normalized_mean=np.stack([o[0] for o in outputs]),
             unnormalized=np.stack([o[1] for o in outputs]), executed=np.stack([o[2] for o in outputs]),
             input_sha256=data['input_sha256'], task_id=data['task_id'], step=data['step'])
    info['artifact_sha256'] = sha(run / f'{args.model}_actions.npz')
    info['output_dtypes'] = {k: str(v.dtype) for k, v in zip(('normalized_mean', 'unnormalized', 'executed'), outputs[0])}
    save_json(run / f'{args.model}_worker.json', info)


def statistics(values):
    import numpy as np
    a = np.asarray(values, dtype=np.float64)
    require(a.size > 0 and np.isfinite(a).all(), 'Invalid metric values')
    return dict(count=int(a.size), mean=float(a.mean()), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), max=float(a.max()))


def distances(a, b):
    import numpy as np
    d = a.astype(np.float64) - b.astype(np.float64)
    return dict(action_l2=np.linalg.norm(d, axis=-1), translation_l2=np.linalg.norm(d[..., :3], axis=-1),
                rotation_l2=np.linalg.norm(d[..., 3:6], axis=-1), gripper_abs=np.abs(d[..., 6]))


def summarize(run, metadata, write=False):
    import numpy as np
    data, capture = load_observations(run, metadata)
    actions = {}
    for model in MODELS:
        info = read_json(run / f'{model}_worker.json')
        require(info['artifact_sha256'] == sha(run / f'{model}_actions.npz'), 'Action archive checksum mismatch')
        require(info['model'] == model and info['normalization_key'] == metadata['normalization_key'] and info['cached_inference'] and not info['sampling'], 'Wrong replay policy metadata')
        with np.load(run / f'{model}_actions.npz', allow_pickle=False) as f:
            a = {k: f[k] for k in f.files}
        for k in ('input_sha256', 'task_id', 'step'):
            require(np.array_equal(a[k], data[k]), f'Replay inputs/order differ: {model}/{k}')
        require(a['normalized_mean'].shape == (len(data['step']), 8, 7), 'Replay action count mismatch')
        check_actions(a['normalized_mean'], a['unnormalized'], a['executed'], metadata['normalization'])
        actions[model] = a
    metrics = {space: distances(actions['scale'][space], actions['ript'][space])
               for space in ('normalized_mean', 'unnormalized', 'executed')}
    disagreement = (actions['scale']['executed'][..., 6] != actions['ript']['executed'][..., 6]).astype(float)
    groups, csv_rows = {}, []
    for i, task_id in enumerate(data['task_id']):
        episode = next(e for e in capture['episodes'] if e['task_id'] == int(task_id))
        group = f"{metadata['source']}|{episode['shift']}|{episode['analysis_case']}"
        groups.setdefault(group, []).append(i)
        for offset in range(8):
            row = dict(observation_index=i, input_sha256=str(data['input_sha256'][i]), source=metadata['source'],
                       task_id=int(task_id), shift=episode['shift'], historical_case=episode['historical_case'],
                       analysis_case=episode['analysis_case'], reproduction=episode['reproduction'],
                       observation_step=int(data['step'][i]), action_offset=offset,
                       trajectory_step=int(data['step'][i]) + offset, source_executed=offset < data['executed_count'][i],
                       gripper_disagreement=int(disagreement[i, offset]))
            for space, values in metrics.items():
                for name, value in values.items():
                    row[f'{space}_{name}'] = float(value[i, offset])
                row[f'{space}_chunk_l2'] = float(np.linalg.norm(values['action_l2'][i]))
            csv_rows.append(row)
    def aggregate(indices):
        output = {}
        for space, values in metrics.items():
            output[space] = {name + '_per_step': statistics(value[indices]) for name, value in values.items()}
            for name in ('action_l2', 'translation_l2', 'rotation_l2'):
                output[space][name + '_per_chunk'] = statistics(np.linalg.norm(values[name][indices], axis=1))
            for name, value in values.items():
                output[space][name + '_observation_mean'] = statistics(value[indices].mean(axis=1))
        output['gripper_disagreement_rate'] = float(disagreement[indices].mean())
        output['gripper_disagreement_per_observation'] = statistics(disagreement[indices].mean(axis=1))
        mask = np.arange(8)[None, :] < data['executed_count'][indices, None]
        output['source_executed_steps'] = {'gripper_disagreement_rate': float(disagreement[indices][mask].mean()),
                                         **{name: statistics(v[indices][mask]) for name, v in metrics['executed'].items()}}
        return output
    replay_delta = distances(data['executed'], actions[metadata['source']]['executed'])['action_l2']
    summary = dict(schema_version=1, observations=len(data['step']), source=metadata['source'], episodes=capture['episodes'],
                   groups={k: aggregate(v) for k, v in groups.items()}, overall=aggregate(list(range(len(data['step'])))),
                   source_replay_executed_l2=statistics(replay_delta),
                   source_replay={space: dict(exact=bool(np.array_equal(data[space], actions[metadata['source']][space])),
                                             action_l2=statistics(distances(data[space], actions[metadata['source']][space])['action_l2']))
                                  for space in ('normalized_mean', 'unnormalized', 'executed')},
                   source_replay_exact=bool(np.array_equal(data['executed'], actions[metadata['source']]['executed'])),
                   interpretation='Descriptive paired action differences; not evidence of representation information loss or a causal OOD explanation.',
                   scope='All 8 predicted actions per observation; source_executed_steps excludes unused final-chunk slots. Rotation values are controller commands, not physical orientation distances.')
    if write:
        with (run / 'actions.csv').open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        summary['actions_csv_sha256'] = sha(run / 'actions.csv')
        save_json(run / 'summary.json', summary)
    return summary


def verify_provenance(metadata):
    for path, digest in metadata['file_sha256'].items():
        require(sha(path) == digest, f'Provenance changed: {path}')
    check_revision(Path(metadata['plus']))
    require(task_mapping(Path(metadata['plus'])) == metadata['task_mapping'], 'Mapping changed')


def run_status(run):
    required = ('metadata.json', 'observations.npz', 'capture.json')
    missing = [name for name in required if not (run / name).is_file()]
    return dict(run=str(run), exists=run.exists(), capture_artifacts_present=not missing,
                missing_capture_files=missing,
                note='Artifact presence only; this does not prove validity or whether a worker is still running.',
                logs=[str(p) for p in sorted(run.glob('*.log'))],
                worker_exits={p.name: read_json(p) for p in sorted(run.glob('*-exit.json'))})


def require_capture(run):
    status = run_status(run)
    require(status['capture_artifacts_present'],
            f"Capture incomplete: missing {', '.join(status['missing_capture_files'])}. "
            f"Check the running capture process and {run / 'capture.log'} "
            "(older versions did not save logs). Do not run compare yet. "
            "After a failed capture, preserve this directory and use a new run ID.")


def run_logged(cmd, run, label):
    log = run / f'{label}.log'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    print(f'Worker log: {log}', flush=True)
    with log.open('x', buffering=1) as stream:
        stream.write('Command: ' + json.dumps(cmd) + '\n')
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors='replace', env=env, cwd=run) as child:
            try:
                for line in child.stdout:
                    stream.write(line)
                    print(line, end='', flush=True)
                code = child.wait()
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
    save_json(run / f'{label}-exit.json', dict(returncode=code, log=str(log)))
    require(code == 0, f'{label} worker exited with code {code}; see {log}. Preserve this run and diagnose the log before retrying.')


def launch(args, run, command, model=None):
    cmd = [sys.executable, '-u', str(Path(__file__).resolve()), command, '--run-id', args.run_id]
    if model:
        cmd += ['--model', model]
    run_logged(cmd, run, 'capture' if command == '_capture' else 'replay-' + model)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('dry-run', 'status', 'capture', 'compare', 'validate', '_capture', '_replay'))
    p.add_argument('--run-id', required=True)
    p.add_argument('--source', choices=MODELS, default='scale')
    p.add_argument('--model', choices=MODELS)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--horizon', type=int, default=300)
    p.add_argument('--plus', type=Path, default=ROOT / 'vendor/LIBERO-plus')
    p.add_argument('--candidates', type=Path, default=P3 / 'p3_case_candidates.csv')
    p.add_argument('--results', type=Path, default=ROOT / 'outputs/libero-plus')
    p.add_argument('--id-column', help='One-based Plus task ID column; auto-detect task_id or id')
    args = p.parse_args()
    args.plus, args.candidates, args.results = (x.resolve() for x in (args.plus, args.candidates, args.results))
    require(1 <= args.horizon <= 300, 'Horizon must be 1..300')
    run = run_path(args.run_id)
    if args.operation == 'status':
        print(json.dumps(run_status(run), indent=2))
        return 0
    if args.operation == 'capture':
        require(not run.exists(), f'Run exists: {run}. Use status to inspect it; preserve it and choose a new run ID after a failed capture.')
    if args.operation in ('dry-run', 'capture'):
        plan = preflight(args)
        if args.operation == 'dry-run':
            plan['ready_meaning'] = 'CPU prerequisites only; not evidence of a completed capture or GPU runtime readiness'
            plan['run_state'] = run_status(run)
            plan['capture_can_start'] = plan['ready'] and not run.exists()
            print(json.dumps(plan, indent=2))
            return 0 if plan['ready'] else 2
        require(plan['ready'], json.dumps(plan['errors']))
        require(not run.exists(), 'Run exists; choose a new run ID')
        paths = set(plan['action_order']['evidence']) | set(sum(plan['checkpoints'].values(), [])) | {str(ROOT / f) for f in IMPL} | {str(args.candidates)}
        for files in plan['physical_task_inputs'].values():
            paths.update(files)
        for case in plan['cases']:
            paths.update(v['path'] for v in case['evaluation'].values())
        for model in MODELS:
            paths.add(str(args.results / model / f'seed-{args.seed}' / 'manifest.json'))
        print('CPU preflight passed; hashing checkpoint provenance before any GPU load.', flush=True)
        metadata = dict(schema_version=1, revision=REVISION, suite=SUITE, plus=str(args.plus),
                        seed=args.seed, source=args.source, horizon=args.horizon, cases=plan['cases'],
                        normalization_key=plan['normalization'][0], normalization=plan['normalization'][1],
                        checkpoint_paths=plan['checkpoints'], action_order=plan['action_order'],
                        physical_task_inputs=plan['physical_task_inputs'],
                        task_mapping=task_mapping(args.plus), file_sha256={path: sha(path) for path in sorted(paths)},
                        protocol=dict(initial_state_index=0, wait_steps=10, env_seed=0, image_resolution=256,
                                      policy_resolution=224, center_crop=True, sampling=False, chunk_shape=[8, 7]),
                        observation_stage='After prepare_observation(224), before policy center crop and proprio normalization')
        run.mkdir(parents=True, exist_ok=False)
        save_json(run / 'metadata.json', metadata)
        launch(args, run, '_capture')
        data, captured = load_observations(run, metadata)
        print(json.dumps(dict(observations=len(data['step']), episodes=captured['episodes']), indent=2))
        return 0 if all(e['reproduction'] == 'matched' for e in captured['episodes']) else 3
    metadata = read_json(run / 'metadata.json')
    if args.operation == '_capture':
        capture_worker(args, run, metadata)
    elif args.operation == '_replay':
        require(args.model in MODELS, '_replay requires --model')
        replay_worker(args, run, metadata)
    else:
        require_capture(run)
        verify_provenance(metadata)
        load_observations(run, metadata)  # CPU validation precedes either GPU worker.
        if args.operation == 'compare':
            require(not any((run / name).exists() for name in ('scale_actions.npz', 'ript_actions.npz', 'summary.json', 'actions.csv', 'config-replay-scale', 'config-replay-ript')), 'Comparison exists/partial; refusing overwrite')
            for model in MODELS:
                launch(args, run, '_replay', model)
            summary = summarize(run, metadata, write=True)
        else:
            summary = summarize(run, metadata)
            saved = read_json(run / 'summary.json')
            require(saved.pop('actions_csv_sha256') == sha(run / 'actions.csv'), 'CSV checksum mismatch')
            require(saved == summary, 'Summary differs from recomputation')
        print(json.dumps(dict(valid=True, observations=summary['observations'], episodes=summary['episodes'],
                              source_replay_exact=summary['source_replay_exact']), indent=2))
        return 0 if all(e['reproduction'] == 'matched' for e in summary['episodes']) else 3
    return 0


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f'P3-A1 failed: {exc}', file=sys.stderr)
        sys.exit(2)
