"""Separate held-out expert trajectories; never recompute normalization on validation."""
import argparse
from collections import Counter
import io
from pathlib import Path
import random

from .common import NAME, destination, digest, inventory, read, require, sha, tasks, write
from .prepare_dataset import episode_record, episodes, validate_manifest


def select_validation(records, train, per_task=1, seed=43):
    require(per_task > 0, 'Validation episodes per task must be positive')
    require(len({r['episode_id'] for r in records}) == len(records), 'Duplicate source IDs')
    train_ids = {r['episode_id'] for r in train['episodes']}
    train_hashes = {r['sha256'] for r in train['episodes']}
    rng = random.Random(seed)
    chosen, used = [], set(train_hashes)
    for task in sorted(tasks()):
        candidates = sorted((r for r in records if r['task'] == task and
                             r['episode_id'] not in train_ids and r['sha256'] not in used),
                            key=lambda r: r['episode_id'])
        rng.shuffle(candidates)
        selected = []
        for row in candidates:
            if row['sha256'] not in used:
                selected.append(row)
                used.add(row['sha256'])
            if len(selected) == per_task:
                break
        require(len(selected) == per_task, f'Insufficient disjoint experts for {task}')
        chosen.extend(selected)
    return sorted(chosen, key=lambda r: r['episode_id'])


def check_validation(path, train_path, verify_files=True):
    path, train_path = Path(path), Path(train_path)
    m, train = read(path/'validation_manifest.json'), read(train_path/'selection_manifest.json')
    validate_manifest(train)
    require(m['train_manifest_sha256'] == sha(train_path/'selection_manifest.json'), 'Validation train manifest changed')
    require(m['source'] == train['source'], 'Validation source differs')
    rows = m['episodes']
    require(m['selection_sha256'] == digest(rows), 'Validation selection checksum changed')
    require(len({r['episode_id'] for r in rows}) == len(rows), 'Duplicate validation IDs')
    require(len({r['sha256'] for r in rows}) == len(rows), 'Duplicate validation content')
    for key in ('episode_id', 'sha256'):
        require(not ({r[key] for r in rows} & {r[key] for r in train['episodes']}), f'Train/validation overlap: {key}')
    require(Counter(r['task'] for r in rows) == Counter({t:m['per_task'] for t in tasks()}), 'Validation task coverage mismatch')
    require(all(r['complete'] and r['steps'] > 0 for r in rows), 'Incomplete validation trajectory')
    a = read(path/'validation_audit.json')
    require(a['status'] == 'verified_materialization' and a['selection_sha256'] == m['selection_sha256'], 'Validation not verified')
    require(a['train_normalization_sha256'] == sha(train_path/'normalization.json'), 'Train normalization changed')
    if verify_files:
        require(inventory(path/'rlds') == a['subset_files'], 'Held-out RLDS changed')
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--train-dataset', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--per-task', type=int, default=1)
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    train = read(a.train_dataset/'selection_manifest.json')
    validate_manifest(train)
    require(inventory(a.source) == train['source']['files'], 'Source differs from original train source')
    out = destination(a.output)
    require(not out.exists(), 'Validation output exists; do not overwrite')
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    import tensorflow_datasets as tfds
    builder = tfds.builder_from_directory(str(a.source.resolve()))
    records = [episode_record(e, eid) for eid,e in episodes(builder)]
    chosen = select_validation(records, train, a.per_task, a.seed)
    manifest = dict(schema=1, source=train['source'], episodes=chosen, per_task=a.per_task,
                    seed=a.seed, selection_sha256=digest(chosen),
                    train_manifest_sha256=sha(a.train_dataset/'selection_manifest.json'))
    print(f'Held-out experts: {len(chosen)} trajectories, {sum(r["steps"] for r in chosen)} transitions; train normalization only')
    if not a.execute:
        return
    write(out/'validation_manifest.json', manifest)
    wanted = {r['episode_id']:r for r in chosen}
    class LiberoGoalNoNoops(tfds.core.GeneratorBasedBuilder, skip_registration=True):
        VERSION = tfds.core.Version(str(builder.version))
        def _info(self):
            return tfds.core.DatasetInfo(builder=self, features=builder.info.features,
                                        description='Disjoint held-out experts; never used for training')
        def _split_generators(self, dl_manager):
            return {'val': self._generate_examples()}
        def _generate_examples(self):
            for eid,e in episodes(builder):
                if eid in wanted:
                    require(episode_record(e,eid) == wanted[eid], 'Source changed during validation preparation')
                    for step in e['steps']:
                        for key in ('image','wrist_image'):
                            step['observation'][key] = io.BytesIO(step['observation'][key])
                    yield eid,e
    subset = LiberoGoalNoNoops(data_dir=str(out/'rlds'))
    subset.download_and_prepare()
    actual = Counter(episode_record(e,eid)['sha256'] for eid,e in episodes(subset, 'val'))
    require(actual == Counter(r['sha256'] for r in chosen), 'Held-out materialization changed content')
    write(out/'validation_audit.json', dict(status='verified_materialization',
        selection_sha256=manifest['selection_sha256'], subset_files=inventory(out/'rlds'),
        train_normalization_sha256=sha(a.train_dataset/'normalization.json')))
    check_validation(out, a.train_dataset)


if __name__ == '__main__':
    main()
