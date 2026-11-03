"""Audit local TFDS, select complete episodes, materialize only the selected subset."""
import argparse
from collections import Counter
import hashlib
import io
from pathlib import Path
import random
import shutil

from .common import NAME, OUT, destination, digest, inventory, read, require, tasks, write


def select(records, shots, seed, task_names):
    require(shots in (1, 5), 'Only 1-shot and 5-shot are supported')
    ids = [r['episode_id'] for r in records]
    require(len(ids) == len(set(ids)), 'Duplicate source episode IDs')
    require(set(r['task'] for r in records) == set(task_names), 'Source task coverage mismatch')
    rng = random.Random(seed)
    selected = []
    for task in sorted(task_names):
        candidates = sorted((r for r in records if r['task'] == task), key=lambda r: r['episode_id'])
        require(len(candidates) >= shots, f'Insufficient episodes: {task}')
        selected.extend(rng.sample(candidates, shots))
    return sorted(selected, key=lambda r: r['episode_id'])


def validate_manifest(m, provenance=None):
    require(m['schema'] == 1 and m['dataset_name'] == NAME, 'Wrong manifest schema/dataset')
    require(m['shots'] in (1, 5), 'Invalid shot count')
    rows = m['episodes']
    require(len({r['episode_id'] for r in rows}) == len(rows), 'Duplicate selected IDs')
    require(Counter(r['task'] for r in rows) == Counter({t: m['shots'] for t in tasks()}),
            'Expected exactly K episodes for each original Goal task')
    require(all(r['steps'] > 0 and r['complete'] and len(r['sha256']) == 64 for r in rows),
            'Incomplete or unhashed episode')
    require(m['selection_sha256'] == digest(rows), 'Manifest selection checksum mismatch')
    if provenance is not None:
        require(m['source'] == provenance, 'Source content/revision changed')


def payload_hash(value):
    """Hash decoded values with paths, types, shapes and temporal order."""
    import numpy as np
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, dict):
            for k in sorted(v):
                h.update(k.encode() + b'\0')
                visit(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(str(len(v)).encode() + b':')
            for item in v:
                visit(item)
        else:
            a = np.asarray(v)
            h.update(str(a.dtype).encode() + str(a.shape).encode())
            if a.dtype.kind in 'OSU':
                for item in a.reshape(-1):
                    b = item if isinstance(item, bytes) else str(item).encode()
                    h.update(str(len(b)).encode() + b':' + b)
            else:
                h.update(a.tobytes())
    visit(value)
    return h.hexdigest()


def episode_record(episode, episode_id):
    import numpy as np
    steps = episode['steps']
    require(bool(steps), 'Empty episode')
    first = [bool(s['is_first']) for s in steps]
    last = [bool(s['is_last']) for s in steps]
    terminal = [bool(s['is_terminal']) for s in steps]
    require(first == [True] + [False] * (len(steps)-1), 'Broken episode start')
    require(last == [False] * (len(steps)-1) + [True], 'Broken episode end')
    require(terminal == last, 'Nonterminal/truncated expert episode')
    languages = {s['language_instruction'].decode() for s in steps}
    require(len(languages) == 1, 'Task instructions change inside trajectory')
    language = next(iter(languages))
    task = language.replace(' ', '_')
    require(task in tasks(), f'Unknown Goal language/task: {language!r}')
    for s in steps:
        a, state = np.asarray(s['action']), np.asarray(s['observation']['state'])
        require(a.shape == (7,) and a.dtype == np.float32 and np.isfinite(a).all(), 'Invalid action')
        require(state.shape == (8,) and state.dtype == np.float32 and np.isfinite(state).all(), 'Invalid proprio')
        for key in ('image', 'wrist_image'):
            value = s['observation'][key]
            if isinstance(value, bytes):
                from PIL import Image
                with Image.open(io.BytesIO(value)) as image:
                    im = np.asarray(image.convert('RGB'))
            else:
                im = np.asarray(value)
            require(im.ndim == 3 and im.shape[-1] == 3 and im.dtype == np.uint8, 'Invalid RGB image')
    return dict(episode_id=episode_id, task=task, steps=len(steps), complete=True,
                sha256=payload_hash(episode))


def episodes(builder):
    import tensorflow_datasets as tfds
    # Preserve original JPEG bytes. Decoding and re-encoding would change pixels.
    decoders = {'steps': {'observation': {
        'image': tfds.decode.SkipDecoding(), 'wrist_image': tfds.decode.SkipDecoding()}}}
    ds = builder.as_dataset(split='train', shuffle_files=False,
                           read_config=tfds.ReadConfig(add_tfds_id=True), decoders=decoders)
    for e in tfds.as_numpy(ds):
        eid = e.pop('tfds_id').decode()
        e['steps'] = list(e['steps'])
        yield eid, e


def prepare(args):
    source = args.source.resolve()
    require(len(args.source_revision) == 40 and all(c in '0123456789abcdef' for c in args.source_revision),
            'Source revision must be a full lowercase 40-character HF commit SHA')
    require((source/'dataset_info.json').is_file(), f'Missing local TFDS version directory: {source}')
    require(NAME in source.parts, 'Source must be the original libero_goal_no_noops dataset')
    require(not source.is_relative_to(OUT.resolve()), 'Source cannot be a previously sampled dataset')
    files = inventory(source)
    provenance = dict(repo='openvla/modified_libero_rlds', revision=args.source_revision,
                      content_sha256=digest(files), files=files)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    import tensorflow_datasets as tfds
    builder = tfds.builder_from_directory(str(source))
    # Upstream 1.0.0 metadata says "libero_goal" although its directory is
    # "libero_goal_no_noops"; the latter is the official training dataset key.
    require(builder.info.name in ('libero_goal', NAME), 'Unexpected TFDS builder name')
    records = [episode_record(e, eid) for eid, e in episodes(builder)]
    require(len(records) == builder.info.splits['train'].num_examples, 'Source episode count mismatch')
    selected = select(records, args.shots, args.selection_seed, tasks())
    manifest = dict(schema=1, dataset_name=NAME, shots=args.shots, selection_seed=args.selection_seed,
                    source=provenance, episodes=selected, selection_sha256=digest(selected))
    if args.selection_manifest:
        prior = read(args.selection_manifest)
        validate_manifest(prior, provenance)
        require(prior == manifest, 'Selection manifest differs from deterministic selection')
        manifest = prior
    validate_manifest(manifest, provenance)
    audit = dict(source_episodes=len(records), source_counts=dict(Counter(r['task'] for r in records)),
                 selected_episodes=len(selected), selected_counts=dict(Counter(r['task'] for r in selected)),
                 selected_transitions=sum(r['steps'] for r in selected),
                 normalization='Official OFT BOUNDS_Q99 recomputed on selected episodes only',
                 no_noops='Source already filtered; no filtering, padding or transition insertion during preparation',
                 provenance=provenance)
    print(__import__('json').dumps(audit, indent=2))
    if args.dry_run:
        return
    out = destination(args.output or OUT/f'datasets/goal_{args.shots}shot_seed{args.selection_seed}')
    if out.exists():
        require(args.resume_existing, f'Refusing to overwrite dataset: {out}')
        require(read(out/'selection_manifest.json') == manifest, 'Existing selection/provenance differs')
        subset_path = out/'rlds'/NAME/str(builder.version)
        require((subset_path/'dataset_info.json').is_file(), 'Existing subset TFDS is incomplete')
        subset = tfds.builder_from_directory(str(subset_path))
        print(f'Resuming verification of existing subset: {subset_path}', flush=True)
    else:
        require(shutil.disk_usage(out.parent if out.parent.exists() else OUT.parent).free >
                sum(v['bytes'] for v in files.values()) * len(selected)/len(records) * 3,
                'Insufficient conservative subset storage budget')
        out.mkdir(parents=True)
        write(out/'selection_manifest.json', manifest)
        wanted = {r['episode_id']: r for r in selected}
        # Reuse the exact source FeaturesDict, including all episode and step metadata.
        class LiberoGoalNoNoops(tfds.core.GeneratorBasedBuilder, skip_registration=True):
            VERSION = tfds.core.Version(str(builder.version))
            def _info(self):
                return tfds.core.DatasetInfo(builder=self, features=builder.info.features,
                    description='Manifest-selected complete episodes from openvla/modified_libero_rlds')
            def _split_generators(self, dl_manager):
                return {'train': self._generate_examples()}
            def _generate_examples(self):
                for eid, e in episodes(builder):
                    if eid in wanted:
                        require(episode_record(e, eid) == wanted[eid], 'Source episode changed during preparation')
                        # Image accepts a file object containing already encoded JPEG.
                        for step in e['steps']:
                            for key in ('image', 'wrist_image'):
                                step['observation'][key] = io.BytesIO(step['observation'][key])
                        yield eid, e
        subset = LiberoGoalNoNoops(data_dir=str(out/'rlds'))
        subset.download_and_prepare()
    actual = Counter(episode_record(e, eid)['sha256'] for eid, e in episodes(subset))
    require(actual == Counter(r['sha256'] for r in selected), 'Materialized episode content mismatch')
    import numpy as np
    actions, proprios = [], []
    for _, e in episodes(subset):
        for s in e['steps']:
            action = s['action'].copy()
            action[-1] = 1 - np.clip(action[-1], 0, 1)  # official libero_dataset_transform
            actions.append(action)
            proprios.append(s['observation']['state'])
    stats = dict(num_trajectories=len(selected), num_transitions=len(actions))
    for key, values in [('action', actions), ('proprio', proprios)]:
        values = np.stack(values)
        stats[key] = dict(mean=values.mean(0).tolist(), std=values.std(0).tolist(),
                          min=values.min(0).tolist(), max=values.max(0).tolist(),
                          q01=np.quantile(values,.01,axis=0).tolist(), q99=np.quantile(values,.99,axis=0).tolist())
    if (out/'normalization.json').exists():
        require(read(out/'normalization.json') == stats, 'Existing normalization differs')
    else:
        write(out/'normalization.json', stats)
    audit.update(status='verified_materialization', subset_dir=str(Path(subset.data_dir).relative_to(out)),
                 subset_files=inventory(out/'rlds'), normalization_sha256=digest(stats))
    if (out/'dataset_audit.json').exists():
        require(read(out/'dataset_audit.json') == audit, 'Existing dataset audit differs')
    else:
        write(out/'dataset_audit.json', audit)
    print(f'Verified {len(selected)} complete episodes and {len(actions)} transitions: {out}', flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Local TFDS version directory containing dataset_info.json')
    p.add_argument('--source-revision', required=True, help='HF dataset commit; content hashes are independently recorded')
    p.add_argument('--shots', type=int, choices=(1, 5), default=5)
    p.add_argument('--selection-seed', type=int, default=42)
    p.add_argument('--selection-manifest', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume-existing', action='store_true', help='Verify/finalize an existing matching subset without rebuilding it')
    return p


if __name__ == '__main__':
    prepare(parser().parse_args())
