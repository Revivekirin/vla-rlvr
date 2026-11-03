import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[2]
SUITES = {'goal': 'libero_goal', 'spatial': 'libero_spatial', 'long': 'libero_10'}
REVISION = 'a7c9ae18499b6eea8a32f78a9302327b752b1b5f'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def task_names(suite):
    require(suite in SUITES, 'Unknown suite')
    tree = ast.parse((ROOT/'vendor/LIBERO/libero/libero/benchmark/libero_suite_task_map.py').read_text())
    mapping = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'libero_task_map' for t in n.targets))
    names = mapping[SUITES[suite]]
    require(len(names) == len(set(names)) == 10, 'Expected ten distinct tasks')
    return names


def config(overrides=()):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    # Context is closed before any subprocess can enter the RIPT Hydra trainer.
    with initialize_config_dir(config_dir=str(Path(__file__).parent/'conf'), version_base='1.3'):
        c = OmegaConf.to_container(compose(config_name='config', overrides=list(overrides)),
                                  resolve=True, throw_on_missing=True)
    require(c['suite']['name'] in SUITES, 'Unknown suite')
    require(c['suite']['benchmark'] == SUITES[c['suite']['name']] and
            c['suite']['dataset'] == c['suite']['benchmark']+'_no_noops', 'Suite mapping mismatch')
    require(type(c['shots']) is int and c['shots'] in (1, 2, 5, 10, 20) or c['shots'] == 'Full',
            'shots must be 1/2/5/10/20/Full')
    require(c['dataset'] == dict(repo='openvla/modified_libero_rlds', revision=REVISION, split='train'),
            'Dataset identity/revision is pinned')
    for k in ('selection_seed', 'train_seed', 'eval_seed'):
        require(type(c[k]) is int and c[k] >= 0, 'Seeds must be nonnegative integers')
    for k, v in dict(base='openvla/openvla-7b', loss='l1', cameras=2, proprio=True,
                     chunk=8, action_dim=7, lora_rank=32).items():
        require(c['sft'][k] == v, f'Fixed architecture/base mismatch: {k}')
    for block, keys in [('sft', ('batch_size', 'accumulation', 'max_steps', 'save_freq')),
                        ('scale', ('max_steps',)), ('ript', ('max_updates',)),
                        ('evaluation', ('max_tasks', 'max_trials'))]:
        for k in keys:
            require(type(c[block][k]) is int and c[block][k] > 0, f'Invalid bound: {block}.{k}')
    require(0 < c['sft']['learning_rate'] < 1 and 0 < c['ript']['learning_rate'] < 1, 'Invalid learning rate')
    require(c['sft']['save_freq'] <= c['sft']['max_steps'] <= 6000, 'Invalid SFT step bounds')
    require(c['execution']['world_size'] == 1 and type(c['execution']['gpu']) is int and
            c['execution']['gpu'] >= 0, 'Only one explicitly selected GPU is supported')
    require(c['evaluation']['dataset'] in ('nominal', 'plus'), 'Unknown evaluation dataset')
    require(c['evaluation']['inference'] in ('sft_full', 'd0_cached', 'd1_cached'), 'Unknown inference mode')
    require(c['scale']['loss'] == 'laplace_nll' and c['ript']['optimizer'] == 'official_ppo', 'Unknown trainer')
    output = Path(c['output'])
    output = (ROOT/output).resolve() if not output.is_absolute() else output.resolve()
    require(output.is_relative_to(ROOT/'outputs/shot_scaling'), 'Output must be inside outputs/shot_scaling')
    c['output'] = str(output)
    return c


def runtime():
    versions = {}
    for name in ('hydra-core', 'omegaconf', 'numpy', 'matplotlib', 'torch', 'tensorflow', 'tensorflow-datasets'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, capture_output=True)
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, text=True, capture_output=True)
    return dict(python=platform.python_version(), platform=platform.platform(), versions=versions,
                git_revision=git.stdout.strip() or None, git_status=dirty.stdout.splitlines(),
                gpu=None, gpu_note='CPU-only; no CUDA probe or execution',
                implementation={str(p.relative_to(ROOT)): sha(p) for p in
                    sorted((ROOT/'experiments/shot_scaling').rglob('*'))
                    if p.is_file() and p.suffix in ('.py', '.yaml', '.txt')})


def run_dir(c, command, inputs):
    identity = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:12]
    out = Path(c['output'])/identity
    out.mkdir(parents=True, exist_ok=False)
    write(out/'manifest.json', dict(schema=1, experiment_id=identity, command=command,
          config=c, inputs={str(Path(p).resolve()): sha(p) for p in inputs}, runtime=runtime()))
    return out


def checkpoint_identity(path):
    path = Path(path)
    require(path.is_dir(), 'Checkpoint directory missing')
    files = {str(p.relative_to(path)): sha(p) for p in sorted(path.rglob('*')) if p.is_file()}
    require(bool(files), 'Empty checkpoint')
    return dict(files=files, sha256=digest(files))


def compatible(d0, d1):
    """Contract only; does not certify cached-model numerical equivalence."""
    for k in ('suite', 'shots', 'selection_seed', 'train_seed', 'dataset_sha256',
              'base_sha256', 'sft_sha256', 'scale_sha256', 'normalization_sha256', 'architecture'):
        require(k in d0 and d0[k] is not None and d0[k] == d1.get(k), f'D0/D1 mismatch: {k}')
    for k in ('dataset_sha256', 'base_sha256', 'sft_sha256', 'scale_sha256', 'normalization_sha256'):
        require(isinstance(d0[k], str) and len(d0[k]) == 64 and
                all(x in '0123456789abcdef' for x in d0[k]), f'Invalid digest: {k}')
    require(d0.get('stage') == 'd0_cached' and d1.get('stage') == 'd1_cached', 'Cached D0/D1 required')
    require(d1.get('parent_d0_sha256') == digest(d0), 'D1 must reference exact D0 snapshot')
    return True
