import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/fewshot_sft'
OFT = ROOT / 'runtime/openvla-oft'
NAME = 'libero_goal_no_noops'
BASE = 'openvla/openvla-7b'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def destination(path):
    path = Path(path).resolve()
    require(path.is_relative_to(OUT.resolve()) and path != OUT.resolve(),
            'Destination must be inside outputs/fewshot_sft')
    return path


def write(path, value, replace=False):
    path = destination(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w' if replace else 'x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')


def tasks():
    path = ROOT / 'vendor/LIBERO/libero/libero/benchmark/libero_suite_task_map.py'
    tree = ast.parse(path.read_text())
    mapping = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'libero_task_map' for t in n.targets))
    result = mapping['libero_goal']
    require(len(result) == len(set(result)) == 10, 'Expected ten distinct Goal tasks')
    return result


def inventory(directory):
    directory = Path(directory)
    return {str(p.relative_to(directory)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
            for p in sorted(directory.rglob('*')) if p.is_file() and '.git' not in p.parts
            and '.cache' not in p.parts and not p.name.startswith('dataset_statistics_')}
