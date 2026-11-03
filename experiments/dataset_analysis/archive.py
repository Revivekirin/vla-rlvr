"""Verified copies, not moves: retain all checkpoint-referenced originals."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
from experiments.shot_scaling.core import require, sha, read, write


def files(root):
    root=Path(root).resolve()
    require(root.is_dir(),f'Missing directory: {root}')
    result={}
    for p in sorted(root.rglob('*')):
        require(not p.is_symlink(),f'Symlink requires explicit archive review: {p}')
        if p.is_file(): result[str(p.relative_to(root))]=dict(bytes=p.stat().st_size,sha256=sha(p))
    return result


def create(source, destination, protected=()):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    require(not destination.exists(),'Archive destination already exists')
    require(not destination.is_relative_to(source),'Archive cannot be inside its source')
    require(not any(destination.is_relative_to(Path(p).resolve()) for p in protected),
            'Archive cannot be inside a protected checkpoint/result directory')
    original=files(source)
    require(bool(original),'Refusing an empty archive')
    guards={str(Path(p).resolve()):files(p) for p in protected}
    manifests={name:read(source/name) for name in original if name.endswith(('manifest.json','selection.json','fewshot_checkpoint.json'))}
    destination.mkdir(parents=True)
    shutil.copytree(source,destination/'artifacts')
    require(files(destination/'artifacts')==original,'Archive copy mismatch')
    require(files(source)==original,'Source changed during archive')
    require(all(files(p)==v for p,v in guards.items()),'Protected checkpoint/result files changed')
    write(destination/'archive_manifest.json',dict(schema=1,status='VERIFIED_COPY',created_utc=datetime.now(timezone.utc).isoformat(),
        source=str(source),files=original,manifest_identities=manifests,protected_files=guards,
        originals_retained=True,immutability='Files chmod 0444; directories 0555. Detect changes via SHA256; not WORM storage.'))
    for p in sorted(destination.rglob('*'),key=lambda p:len(p.parts),reverse=True):
        p.chmod(0o555 if p.is_dir() else 0o444)
    destination.chmod(0o555)
    return destination


def verify(destination):
    destination=Path(destination)
    m=read(destination/'archive_manifest.json')
    require(m['status']=='VERIFIED_COPY' and files(destination/'artifacts')==m['files'],'Archive integrity mismatch')
    return m
