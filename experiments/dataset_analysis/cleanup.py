"""Two-phase, exact-inventory cleanup. No source or checkpoint directory is deleted."""
import argparse
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
from experiments.shot_scaling.core import ROOT, digest, read, require, sha, write

TARGETS = ('dataset_analysis', 'research_dashboard', 'fewshot_sft', 'rlvr_analysis')
GUARDS = ('checkpoints', 'outputs/goal', 'outputs/libero', 'data', 'vendor', 'runtime')


def safe(root, path):
    root, path = Path(root).absolute(), Path(path).absolute()
    require(path.is_relative_to(root), f'Path escape: {path}')
    for p in (root, *reversed(path.parents), path):
        require(not p.is_symlink(), f'Symlink rejected: {p}')
    require(path.resolve() == path, f'Noncanonical path: {path}')
    return path


def inventory(root, path, *, protected=False):
    path = safe(root, path)
    require(path.is_dir(), f'Missing directory: {path}')
    result = {}
    for p in sorted(path.rglob('*')):
        if protected and p.is_symlink():
            safe(root, p.parent)
            link = os.readlink(p)
            target = p.resolve(strict=True)
            require(target.is_relative_to(path), f'Protected link escapes its tree: {p} -> {target}')
            safe(root, target)
            require(target.is_file(), f'Protected link must resolve to a regular file: {p}')
            result[str(p.relative_to(path))] = dict(type='symlink', link=link,
                resolved=str(target), bytes=target.stat().st_size, sha256=sha(target))
            continue
        safe(root, p)
        mode = p.lstat().st_mode
        require(stat.S_ISREG(mode) or stat.S_ISDIR(mode), f'Special file: {p}')
        result[str(p.relative_to(path))] = ({'type': 'directory'} if p.is_dir() else
            {'type': 'file', 'bytes': p.stat().st_size, 'sha256': sha(p)})
    return result


def process_check(root):
    # Check command lines plus open files/cwd. Any incomplete inspection blocks apply.
    checks = []
    for command in (['ps', '-axo', 'pid=,command='], ['lsof', '-nP', '+D', str(root/'outputs')]):
        try:
            p = subprocess.run(command, capture_output=True, text=True, timeout=60)
            lines = p.stdout.splitlines()
            if command[0] == 'ps':
                matches = [s for s in lines if any('outputs/'+t in s for t in TARGETS)
                           and s.strip().split()[0] != str(os.getpid())]
                ok = p.returncode == 0
            else:
                matches = [s for s in lines if any('outputs/'+t in s for t in TARGETS)]
                ok = p.returncode in (0, 1) and not p.stderr.strip()
            checks.append(dict(command=command, complete=ok, matches=matches, stderr=p.stderr))
        except (OSError, subprocess.TimeoutExpired) as e:
            checks.append(dict(command=command, complete=False, matches=[], error=str(e)))
    return dict(clear=all(c['complete'] and not c['matches'] for c in checks), checks=checks,
                scope='Local host only; cannot establish remote server inactivity')


def plan(root):
    root = Path(root).resolve()
    targets = {str(root/'outputs'/t): inventory(root, root/'outputs'/t)
               for t in TARGETS if (root/'outputs'/t).exists()}
    guards = {str(root/g): inventory(root, root/g, protected=True) for g in GUARDS}
    return dict(schema=1, root=str(root), targets=targets, protected=guards,
                processes=process_check(root), deletion_set_sha256=digest(targets),
                status='PROPOSAL_ONLY', source_code_deleted=False)


def unchanged(p):
    root = Path(p['root'])
    require(p['deletion_set_sha256'] == digest(p['targets']), 'Invalid deletion digest')
    require(set(p['targets']) <= {str(root/'outputs'/t) for t in TARGETS}, 'Unexpected target')
    require(set(p['protected']) == {str(root/g) for g in GUARDS}, 'Missing guards')
    for path, expected in {**p['targets'], **p['protected']}.items():
        require(inventory(root, path, protected=path in p['protected']) == expected,
                f'Changed since proposal: {path}')


def save_plan(root, destination):
    """Publish a complete proposal atomically, without overwriting an earlier one."""
    root = Path(root).resolve()
    destination = safe(root, destination)
    require(not any(destination.is_relative_to(root/p) for p in
                    (*GUARDS, *('outputs/'+t for t in TARGETS))), 'Plan must be outside inventoried trees')
    require(not destination.exists(), f'Plan already exists: {destination}')
    payload = json.dumps(plan(root), indent=2, sort_keys=True) + '\n'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=destination.parent,
                                         prefix='.cleanup-plan-', delete=False) as f:
            temporary = Path(f.name)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, destination)  # Atomic publication; fails if destination appeared.
    finally:
        if temporary is not None: temporary.unlink()


def archive(p, destination):
    unchanged(p)
    root = Path(p['root']); destination = safe(root, destination)
    require(not destination.exists(), 'Archive exists')
    require(not any(destination.is_relative_to(root/'outputs'/t) for t in TARGETS), 'Archive inside deletion target')
    require(not any(destination.is_relative_to(root/g) for g in GUARDS), 'Archive inside protected path')
    destination.mkdir(parents=True)
    for target in p['targets']:
        shutil.copytree(target, destination/'artifacts'/Path(target).name)
    write(destination/'archive_manifest.json', dict(plan=p, reference_only=True,
          note='Historical outputs; not current Stage 1 or new RLVR evidence'))
    verify(destination)
    unchanged(p)
    return sha(destination/'archive_manifest.json')


def verify(destination):
    destination = Path(destination).absolute()
    safe(destination, destination)
    m = read(destination/'archive_manifest.json')
    for target, expected in m['plan']['targets'].items():
        require(inventory(destination, destination/'artifacts'/Path(target).name) == expected,
                f'Archive integrity mismatch: {target}')
    return m


def apply(p, destination, approval):
    require(approval == digest(p), 'Approval must be the SHA256 digest of the exact complete proposal')
    require(verify(destination)['plan'] == p, 'Archive does not match approved proposal')
    unchanged(p)
    require(process_check(Path(p['root']))['clear'], 'Process use unknown or active; stop destructive part')
    # Unlink only inventoried files, then empty directories. Unexpected new files prevent rmdir.
    for target, entries in p['targets'].items():
        base = Path(target)
        for rel, info in entries.items():
            if info['type'] == 'file':
                path = safe(Path(p['root']), base/rel)
                require(sha(path) == info['sha256'], f'File changed: {path}')
                path.unlink()
        for rel, info in sorted(entries.items(), key=lambda x: len(Path(x[0]).parts), reverse=True):
            if info['type'] == 'directory': safe(Path(p['root']), base/rel).rmdir()
        base.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan-cleanup', 'archive', 'verify', 'apply-cleanup'])
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--approved-plan-sha256')
    parser.add_argument('--output', type=Path, help='Atomically save plan-cleanup JSON; refuses overwrite')
    a = parser.parse_args()
    if a.command == 'plan-cleanup':
        if a.output:
            save_plan(ROOT, a.output)
            print(a.output.resolve())
        else: print(json.dumps(plan(ROOT), indent=2, sort_keys=True))
    elif a.command == 'verify':
        verify(a.archive); print(sha(a.archive/'archive_manifest.json'))
    elif a.command == 'archive': print(archive(read(a.plan), a.archive))
    else: apply(read(a.plan), a.archive, a.approved_plan_sha256)


if __name__ == '__main__': main()
