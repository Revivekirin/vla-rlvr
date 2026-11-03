"""Check exact local Goal TFDS paths; optionally fetch only the pinned Goal subset."""
import argparse
import json
import os
from pathlib import Path
import shutil

from .common import NAME, ROOT, require

REPO = 'openvla/modified_libero_rlds'
REVISION = 'a7c9ae18499b6eea8a32f78a9302327b752b1b5f'
VERSION = '1.0.0'
DESTINATION = ROOT / 'data/modified_libero_rlds'
SOURCE = DESTINATION / NAME / VERSION


def candidates():
    """Inspect known exact directories only, without traversing shared storage."""
    result = [SOURCE, ROOT/'data'/NAME/VERSION]
    for key in ('RLDS_SOURCE', 'TFDS_DATA_DIR'):
        raw = os.environ.get(key)
        if raw:
            path = Path(raw).expanduser()
            result.extend((path, path/NAME/VERSION))
    # Hugging Face's default/overridden cache layout has one snapshot level.
    hub = os.environ.get('HF_HUB_CACHE') or os.environ.get('HUGGINGFACE_HUB_CACHE')
    if hub is None:
        hub = str(Path(os.environ.get('HF_HOME', str(Path.home()/'.cache/huggingface')))/'hub')
    snapshots = Path(hub)/'datasets--openvla--modified_libero_rlds/snapshots'
    if snapshots.is_dir():
        result.extend(p/NAME/VERSION for p in snapshots.iterdir() if p.is_dir())
    return list(dict.fromkeys(p.resolve() for p in result))


def ready(path):
    return ((path/'dataset_info.json').is_file()
            and (path/'features.json').is_file()
            and len(list(path.glob('*-train.tfrecord-*-of-00016'))) == 16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Download only the pinned Goal dataset if absent')
    args = parser.parse_args()
    found = [p for p in candidates() if ready(p)]
    info = dict(repo=REPO, revision=REVISION, expected_version=VERSION,
                expected_shards=16, found=[str(p) for p in found],
                destination=str(SOURCE), free_bytes=shutil.disk_usage(ROOT).free)
    print(json.dumps(info, indent=2))
    if found:
        source = found[0]
    elif args.execute:
        from huggingface_hub import HfApi, snapshot_download
        remote = HfApi().dataset_info(REPO, revision=REVISION, files_metadata=True)
        require(remote.sha == REVISION, 'Pinned Hugging Face revision changed')
        selected = [f for f in remote.siblings if f.rfilename.startswith(f'{NAME}/{VERSION}/')]
        shards = [f for f in selected if '-train.tfrecord-' in f.rfilename]
        require(len(shards) == 16 and all(f.size and f.size > 0 for f in shards),
                'Unexpected Goal dataset shard inventory')
        total = sum(f.size or 0 for f in selected)
        require(shutil.disk_usage(ROOT).free > 3*total, 'Insufficient storage for Goal download and subset')
        print(json.dumps(dict(remote_bytes=total, selected_files=[f.rfilename for f in selected]), indent=2), flush=True)
        snapshot_download(repo_id=REPO, repo_type='dataset', revision=REVISION,
                          allow_patterns=[f'{NAME}/{VERSION}/*'], local_dir=str(DESTINATION),
                          max_workers=4)
        require(ready(SOURCE), 'Goal download incomplete or TFDS shards missing')
        source = SOURCE
    else:
        print('No complete local Goal TFDS directory in the exact known paths. '
              'Run with --execute to fetch only Goal after reviewing the pinned revision and storage.')
        return
    print('Verified local Goal directory:', source)
    print('Dataset audit command:')
    print('bash experiments/server_env.sh original python -m experiments.fewshot_sft.prepare_dataset '
          f'--source {source} --source-revision {REVISION} --shots 5 --selection-seed 42 --dry-run')


if __name__ == '__main__':
    main()
