"""Offline artifact integrity, static checks, and eval-exit regression check."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace as NS

root = Path(__file__).resolve().parents[2]
lock = json.loads((root / 'stage0/models.lock.json').read_text())
results = {'artifacts': {}, 'normalization': {}}
for suite in ['goal', 'long', 'object', 'spatial']:
    name = '10' if suite == 'long' else suite
    repo = f'moojink/openvla-7b-oft-finetuned-libero-{name}'
    base = root / 'checkpoints' / suite / 'oft'
    stats = json.loads((base / 'dataset_statistics.json').read_text())
    key = f'libero_{name}_no_noops'
    assert len(stats[key]['action']['q01']) == 7 and len(stats[key]['proprio']['q01']) == 8
    results['normalization'][suite] = key
    for r, folder in [(repo, base), ('tanshh97/RIPT_VLA', root / 'checkpoints/ript')]:
        for f in lock[r]['files']:
            path = folder / f['rfilename']
            if not path.exists() or str(path) in results['artifacts']:
                continue
            assert path.stat().st_size == f['size'], path
            sha = f.get('lfs', {}).get('sha256')
            h = hashlib.sha256() if sha else hashlib.sha1()
            if not sha:
                h.update(f'blob {f["size"]}\0'.encode())
                sha = f['blobId']
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(block)
            assert h.hexdigest() == sha, path
            results['artifacts'][str(path.relative_to(root))] = {'bytes': f['size'], 'digest': sha, 'verified': True}

for p in (root / 'stage0/scripts').glob('*.py'):
    ast.parse(p.read_text())
for p in (root / 'vendor/ript-vla/scripts/openvla_oft').rglob('*.sh'):
    subprocess.run(['bash', '-n', str(p)], check=True)
subprocess.run(['bash', '-n', str(root / 'stage0/scripts/install_linux.sh')], check=True)

# Execute the real evaluation/control-flow suffix with a stub environment runner.
# No model loading, simulator, or optimization happens in this regression check.
events = []
tree = ast.parse((root / 'runtime/ript-vla/train_ript_openvla_oft.py').read_text())
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
start = next(i for i, n in enumerate(main.body) if isinstance(n, ast.If) and ast.unparse(n.test) == 'cfg.rollout.enabled')
main.body = main.body[start:]
main.decorator_list = []
cfg = NS(rollout=NS(enabled=True), algo=NS(eval_only=True, env_runner=NS(num_parallel_envs=1)))
namespace = dict(instantiate=lambda *a, **k: NS(run=lambda *a, **k: events.append('eval') or {}),
    model=NS(model=NS(eval=lambda: None)), rank=0, local_eval_tasks=[], logger=None,
    train_cfg=NS(use_tqdm=False), sync_rollout_results_via_file=lambda *a: events.append('aggregate'),
    dist=NS(barrier=lambda: events.append('barrier'), destroy_process_group=lambda: events.append('destroy')),
    wandb=NS(finish=lambda: events.append('finish')))
exec(compile(ast.Module(body=[main], type_ignores=[]), '<eval-exit-regression>', 'exec'), namespace)
namespace['main'](cfg)
assert events == ['eval', 'aggregate', 'barrier', 'finish', 'destroy'], events
results['eval_exit_regression'] = events
results['python_parse_and_shell_syntax'] = True
results['scope'] = 'Static, integrity, and stubbed control flow only; not a CUDA runtime test.'
(root / 'stage0/evidence/local_audit.json').write_text(json.dumps(results, indent=2) + '\n')
print(f'PASS: {len(results["artifacts"])} artifact hashes, 4 normalization mappings, script syntax, eval-only exit regression')
