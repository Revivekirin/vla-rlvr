"""Download pinned official artifacts with size and LFS SHA256 verification."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser()
p.add_argument('--suite', choices=['goal', 'long', 'object', 'spatial'], required=True)
p.add_argument('--level', choices=['metadata', 'components', 'full'], default='metadata')
a = p.parse_args()
lock = json.loads((root / 'stage0/models.lock.json').read_text())
base = 'moojink/openvla-7b-oft-finetuned-libero-' + ('10' if a.suite == 'long' else a.suite)
for repo in [base, 'tanshh97/RIPT_VLA']:
    item = lock[repo]
    for f in item['files']:
        name = f['rfilename']
        if repo == base:
            if '/' in name or name == '.gitattributes':
                continue  # SFT LoRA is already merged into full model.
            folder = root / 'checkpoints' / a.suite / 'oft'
        else:
            if not (name.startswith(f'openvla_oft/ript_adaptors/LIBERO_{a.suite.upper()}_lora/') or
                    name == f'openvla_oft/scale_header/LIBERO_{a.suite.upper()}_scale_header.pth'):
                continue
            folder = root / 'checkpoints' / 'ript'
        if a.level == 'metadata' and f.get('size', 0) > 5_000_000:
            continue
        if a.level == 'components' and name.startswith('model-'):
            continue
        dest = folder / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        def valid(path):
            if not path.exists() or path.stat().st_size != f['size']:
                return False
            sha = f.get('lfs', {}).get('sha256')
            if sha:
                h = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        h.update(block)
                return h.hexdigest() == sha
            return True
        if valid(dest):
            print('Verified existing', dest, flush=True)
            continue
        url = f'https://huggingface.co/{repo}/resolve/{item["revision"]}/{name}'
        print('Downloading', name, f['size'], flush=True)
        tmp = dest.with_name(dest.name + '.partial')
        with urllib.request.urlopen(url, timeout=120) as response, tmp.open('wb') as stream:
            while block := response.read(8 * 1024 * 1024):
                stream.write(block)
        if not valid(tmp):
            raise RuntimeError(f'Integrity check failed: {tmp}')
        tmp.replace(dest)
        print('Verified', dest, flush=True)
