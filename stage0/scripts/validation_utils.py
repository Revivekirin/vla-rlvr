"""Shared bounded-validation plumbing; no model/simulator imports at module load."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def configure():
    for path in reversed([ROOT / 'runtime/openvla-oft', ROOT / 'runtime/ript-vla', ROOT / 'vendor/LIBERO']):
        if not path.is_dir():
            raise FileNotFoundError(f'Existing pinned runtime required: {path}')
        sys.path.insert(0, str(path))
    os.environ.setdefault('LIBERO_CONFIG_PATH', str(ROOT / 'runtime/libero-config'))
    config = Path(os.environ['LIBERO_CONFIG_PATH']) / 'config.yaml'
    if not config.is_file():
        raise FileNotFoundError(f'LIBERO config missing: {config}; configure it before running')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
                      TOKENIZERS_PARALLELISM='false')
    os.chdir(ROOT / 'runtime/openvla-oft')


def provenance():
    lock = json.loads((ROOT / 'stage0/repos.lock.json').read_text())
    for name, info in lock.items():
        if info.get('reference_only'):
            continue
        actual = subprocess.check_output(['git', '-C', str(ROOT / 'vendor' / name), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != info['revision']:
            raise RuntimeError(f'{name}: expected {info["revision"]}, got {actual}')
        subprocess.run(['git', '-C', str(ROOT / 'vendor' / name), 'diff', '--exit-code', 'HEAD'], check=True,
                       stdout=subprocess.DEVNULL)
    files = {
        'openvla-oft': ['experiments/robot/openvla_utils.py', 'experiments/robot/libero/run_libero_eval.py',
                        'experiments/robot/libero/libero_utils.py', 'prismatic/extern/hf/modeling_prismatic.py'],
        'ript-vla': ['ript/algos/rl_optimizers/openvla_oft_interface.py', 'ript/env_runner/openvla_oft_libero_runner.py'],
    }
    hashes = {}
    for repo, names in files.items():
        for name in names:
            original = digest(ROOT / 'vendor' / repo / name)
            actual = digest(ROOT / 'runtime' / repo / name)
            if original != actual:
                raise RuntimeError(f'Policy source differs from pinned upstream: {repo}/{name}')
            hashes[f'{repo}/{name}'] = actual
    versions = {}
    for name in ['torch', 'transformers', 'peft', 'mujoco', 'robosuite', 'numpy', 'tensorflow']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'repositories': lock, 'policy_source_sha256': hashes, 'package_versions': versions,
            'python': sys.version, 'mujoco_gl': os.environ.get('MUJOCO_GL')}


def bounded_worker(script, arguments, output, log, timeout):
    """Isolate global patches and GPU allocations; terminate all child processes on timeout."""
    start = time.monotonic()
    with Path(log).open('w') as stream:
        process = subprocess.Popen([sys.executable, str(script), '--worker', *arguments, '--output', str(output)],
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return {'status': 'TIMEOUT' if isinstance(exc, subprocess.TimeoutExpired) else 'INTERRUPTED',
                    'elapsed_seconds': time.monotonic() - start, 'log': str(log),
                    'cleanup': 'Worker process group terminated; graceful simulator close not confirmed'}
    try:
        result = json.loads(Path(output).read_text()) if Path(output).exists() else {'status': 'ERROR', 'error': 'Worker produced no evidence'}
        if not isinstance(result, dict) or 'status' not in result:
            raise ValueError('Worker evidence has no status')
    except (ValueError, OSError) as exc:
        result = {'status': 'ERROR', 'error': f'Invalid worker evidence: {exc}'}
    result.update(exit_code=code, elapsed_seconds=time.monotonic() - start, log=str(log))
    if code != 0:
        result['status'] = 'ERROR'
    return result
