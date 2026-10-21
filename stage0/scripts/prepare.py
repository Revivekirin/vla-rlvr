"""Materialize pinned runtime copies and documented compatibility patches."""
import difflib
import json
import shutil
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[2]
lock = json.loads((root / 'stage0/repos.lock.json').read_text())
for name, info in lock.items():
    if info.get('reference_only'):
        continue
    src = root / 'vendor' / name
    if not src.exists():
        subprocess.run(['git', 'clone', info['url'], str(src)], check=True)
        subprocess.run(['git', '-C', str(src), 'checkout', '--detach', info['revision']], check=True)
    actual = subprocess.check_output(['git', '-C', str(src), 'rev-parse', 'HEAD'], text=True).strip()
    assert actual == info['revision'], (name, actual)
    assert not subprocess.check_output(['git', '-C', str(src), 'status', '--porcelain'], text=True).strip(), name

for name in ['ript-vla', 'openvla-oft']:
    src, dst = root / 'vendor' / name, root / 'runtime' / name
    if dst.exists():
        raise SystemExit(f'{dst} already exists; retain it or move it aside before preparing again.')
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('.git', '__pycache__'))

def patch(name, filename, old, new, label):
    path = root / 'runtime' / name / filename
    before = path.read_text()
    assert before.count(old) == 1, (filename, old)
    after = before.replace(old, new)
    path.write_text(after)
    diff = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='a/' + filename, tofile='b/' + filename))
    (root / 'stage0/patches' / label).write_text(diff)

patch('ript-vla', 'train_ript_openvla_oft.py',
      '    # Synchronize all processes before training starts\n',
      '    # Stage 0 compatibility fix: evaluation must not enter training setup.\n'
      '    if cfg.algo.eval_only:\n'
      '        dist.barrier()\n'
      '        if rank == 0:\n'
      '            wandb.finish()\n'
      '        dist.destroy_process_group()\n'
      '        return\n\n'
      '    # Synchronize all processes before training starts\n', 'eval-only-exit.patch')
patch('ript-vla', 'requirements.txt', 'diffusers==0.28.0', 'diffusers==0.30.3', 'diffusers-reconciliation.patch')
patch('openvla-oft', 'pyproject.toml',
      'git+https://github.com/moojink/transformers-openvla-oft.git',
      'git+https://github.com/moojink/transformers-openvla-oft.git@bc339d9ad707454c0c115970db43c260067c61ab', 'transformers-pin.patch')
patch('openvla-oft', 'pyproject.toml', 'git+https://github.com/moojink/dlimp_openvla"',
      'git+https://github.com/moojink/dlimp_openvla@040105d256bd28866cc6620621a3d5f7b6b91b46"', 'dlimp-pin.patch')

libero = root / 'vendor/LIBERO/libero/libero'
cfg = root / 'runtime/libero-config'
cfg.mkdir(parents=True, exist_ok=True)
(root / 'data').mkdir(exist_ok=True)
paths = {'benchmark_root': str(libero), 'bddl_files': str(libero / 'bddl_files'),
         'init_states': str(libero / 'init_files'), 'assets': str(libero / 'assets'), 'datasets': str(root / 'data')}
(cfg / 'config.yaml').write_text(json.dumps(paths, indent=2) + '\n')
print('Prepared pinned runtime copies, explicit patches, and local LIBERO configuration.')
