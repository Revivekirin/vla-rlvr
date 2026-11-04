"""Write complete selected RLDS episodes using the source TFDS feature schema."""
from collections import Counter
import io
from pathlib import Path

import numpy as np

from experiments.shot_scaling.core import ROOT, SUITES, digest, read, require, write
from experiments.shot_scaling.data import validate_selection
from experiments.dataset_analysis.analysis import train_inventory
from experiments.dataset_analysis.audit import record
from experiments.fewshot_sft.prepare_dataset import episodes
from .stage2 import OUT, checked_baseline, selection_path


def source_check(source, audit):
    source = Path(source).resolve()
    require(source == Path(audit['source']['path']).resolve(), 'Source path differs from Stage 1 pinned RLDS')
    require((source / 'dataset_info.json').is_file(), 'Source TFDS metadata is missing')
    for name, item in audit['source']['files'].items():
        p = source / name
        require(p.is_file() and p.stat().st_size == item['bytes'], f'Source file missing/size changed: {name}')
    return source


def statistics(episodes_iter, wanted, suite):
    acts, states = [], []
    expected_hashes = Counter(r['sha256'] for r in wanted.values())
    seen = Counter()
    count = 0
    for eid, episode in episodes_iter:
        r = record(episode, eid, suite, 'train')
        require(r['sha256'] in expected_hashes, 'Unselected or modified payload in RLDS')
        seen[r['sha256']] += 1
        count += 1
        for step in episode['steps']:
            action = np.asarray(step['action'], dtype=np.float32).copy()
            action[-1] = 1 - np.clip(action[-1], 0, 1)  # official libero_dataset_transform
            acts.append(action)
            states.append(np.asarray(step['observation']['state'], dtype=np.float32))
    require(count > 0 and len(acts) > 0 and seen == expected_hashes,
            'Materialized full episode/image/action/sequence integrity mismatch')
    result = dict(num_trajectories=count, num_transitions=len(acts))
    for name, values in [('action', acts), ('proprio', states)]:
        x = np.stack(values)
        result[name] = dict(mean=x.mean(0).tolist(), std=x.std(0).tolist(),
                            min=x.min(0).tolist(), max=x.max(0).tolist(),
                            q01=np.quantile(x, .01, axis=0).tolist(),
                            q99=np.quantile(x, .99, axis=0).tolist())
    return result, digest(sorted(h for h, n in seen.items() for _ in range(n)))


def install_statistics_cache(root, dataset_name, builder, expected):
    """Pin the statistics file looked up by the unmodified OFT training loader."""
    import hashlib
    import inspect
    import sys
    sys.path.insert(0, str(ROOT / 'runtime/openvla-oft'))
    from prismatic.vla.datasets.rlds.oxe.materialize import make_oxe_dataset_kwargs
    kwargs = make_oxe_dataset_kwargs(dataset_name, Path(root),
                                     load_camera_views=('primary', 'wrist'),
                                     load_proprio=True, load_language=True)
    deps = (str(builder.info), str(kwargs['state_obs_keys']),
            inspect.getsource(kwargs['standardize_fn']))
    key = hashlib.sha256(''.join(deps).encode(), usedforsecurity=False).hexdigest()
    cache = Path(builder.data_dir) / f'dataset_statistics_{key}.json'
    if cache.exists():
        require(read(cache) == expected, 'Existing OFT statistics cache differs from selected data')
    else:
        write(cache, expected)
    return cache


def check_train_loader(root, dataset_name, expected, *, use_cache):
    """Use the actual OFT RLDSDataset path, with verified subset statistics injected."""
    import sys
    sys.path.insert(0, str(ROOT / 'runtime/openvla-oft'))
    from prismatic.vla.datasets.rlds import dataset as rlds
    from prismatic.vla.datasets.datasets import RLDSDataset
    from prismatic.vla.datasets.rlds.traj_transforms import chunk_act_obs
    import tensorflow as tf
    old = rlds.get_dataset_statistics
    import copy
    if not use_cache:
        rlds.get_dataset_statistics = lambda *args, **kwargs: copy.deepcopy(expected)
    try:
        ds = RLDSDataset(Path(root), dataset_name, lambda sample: sample,
                         resize_resolution=(64, 64), shuffle_buffer_size=1,
                         train=True, image_aug=False)
        # make_interleaved_dataset returns statistics keyed by dataset name,
        # even for a single-dataset LIBERO mixture.
        require(set(ds.dataset_statistics) == {dataset_name}, 'Unexpected OFT dataset mixture')
        loader_stats = ds.dataset_statistics[dataset_name]
        require(int(loader_stats['num_trajectories']) == expected['num_trajectories'],
                'OFT loader normalization episode count mismatch')
        sample = next(iter(ds))
        require(np.asarray(sample['action']).shape[-2:] == (8, 7), 'OFT action chunk must be 8 x 7')
        require('proprio' in sample['observation'] and 'image_primary' in sample['observation']
                and 'image_wrist' in sample['observation'], 'OFT loader lacks images or proprio')
        require(np.isfinite(np.asarray(sample['action'])).all(), 'Nonfinite normalized OFT action')
        require(np.isin(np.asarray(sample['action'])[:, -1], [0.0, 1.0]).all(),
                'Official unnormalized gripper channel changed')
        for key in ('action', 'proprio'):
            for field in ('mean', 'std', 'min', 'max', 'q01', 'q99'):
                require(np.allclose(loader_stats[key][field], expected[key][field], rtol=1e-5, atol=1e-6),
                        f'OFT loader used different {key}/{field} normalization data')
        actions = tf.reshape(tf.range(63, dtype=tf.float32), (9, 7))
        fixture = dict(action=actions,
                       observation={'proprio': tf.zeros((9, 8), dtype=tf.float32)},
                       task={'language_instruction': tf.constant(['x'] * 9)},
                       dataset_name=tf.constant([dataset_name] * 9),
                       absolute_action_mask=tf.zeros((9, 7), dtype=tf.bool))
        chunks = chunk_act_obs(fixture, window_size=1, future_action_window_size=7)
        got = np.asarray(chunks['action'])
        require(got.shape == (2, 8, 7) and np.array_equal(got[0], np.asarray(actions)[:8])
                and np.array_equal(got[1], np.asarray(actions)[1:]),
                'OFT final chunk/terminal boundary behavior changed')
    finally:
        rlds.get_dataset_statistics = old


def materialize(suite, seed, shots, *, smoke=False):
    require(suite in SUITES and type(seed) is int and seed >= 0 and shots in (1, 2, 5, 10, 20, 'Full'),
            'Invalid Stage 2 condition')
    audits = checked_baseline()
    audit = audits[suite]
    inv = train_inventory(audit)
    target = selection_path(suite, seed, shots)
    selection = read(target / 'selection_manifest.json')
    validate_selection(selection, inv)
    if smoke:
        require(shots == 1, 'Bounded smoke test must be a one-shot condition')
    else:
        require((OUT / 'analysis' / 'smoke_validation.json').is_file(),
                'Run a bounded one-shot smoke materialization before all conditions')
    source = source_check(audit['source']['path'], audit)
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    import tensorflow_datasets as tfds
    builder = tfds.builder_from_directory(str(source))
    wanted = {r['episode_id']: r for r in selection['episodes']}
    require(len(wanted) == len(selection['episodes']), 'Duplicate episode IDs')
    dataset_name = selection['dataset_name']
    root = target / 'rlds'
    version = str(builder.version)
    target_version = root / dataset_name / version
    if shots == 'Full':
        # Full is already an actual loadable pinned RLDS dataset. Avoid copying it.
        require(builder.info.splits['train'].num_examples == len(wanted), 'Full source count mismatch')
        loader_root = source.parent.parent
        actual_builder = builder
        mode = 'pinned_original_rlds'
    else:
        if target_version.exists():
            require((target_version / 'dataset_info.json').is_file(),
                    f'Incomplete prior TFDS output; remove only this directory and retry: {target_version}')
            actual_builder = tfds.builder_from_directory(str(target_version))
            loader_root = root
            mode = 'materialized_subset'
        else:
            actual_builder = None
        found = set()
        class LiberoGoalNoNoops(tfds.core.GeneratorBasedBuilder, skip_registration=True):
            VERSION = tfds.core.Version(version)
            # The server TFDS version resolves namespace-package `experiments`
            # to MultiplexedPath, which has no `.parts`. Give it this real file.
            code_path = Path(__file__)
            def _info(self):
                return tfds.core.DatasetInfo(builder=self, features=builder.info.features,
                    description='Nested random complete LIBERO expert episodes from pinned RLDS')
            def _split_generators(self, dl_manager):
                return {'train': self._generate_examples()}
            def _generate_examples(self):
                for eid, episode in episodes(builder):
                    if eid not in wanted:
                        continue
                    r = record(episode, eid, suite, 'train')
                    require(all(r[k] == wanted[eid][k] for k in ('episode_id','task','steps','sha256')),
                            f'Source episode changed: {eid}')
                    found.add(eid)
                    for step in episode['steps']:
                        for camera in ('image', 'wrist_image'):
                            step['observation'][camera] = io.BytesIO(step['observation'][camera])
                    yield eid, episode
                require(found == set(wanted), 'Selected source episodes missing')
        class LiberoSpatialNoNoops(LiberoGoalNoNoops, skip_registration=True):
            pass
        class Libero10NoNoops(LiberoGoalNoNoops, skip_registration=True):
            # TFDS infers ``libero10_no_noops`` from this Python class name;
            # the official OFT dataset key has an underscore before 10.
            name = 'libero_10_no_noops'
        klass = {'goal': LiberoGoalNoNoops, 'spatial': LiberoSpatialNoNoops,
                 'long': Libero10NoNoops}[suite]
        if actual_builder is None:
            output_builder = klass(data_dir=str(root))
            require(output_builder.name == dataset_name, 'TFDS builder name does not match OFT config')
            output_builder.download_and_prepare(download_dir=str(target / 'tfds_downloads'))
            actual_builder = tfds.builder_from_directory(str(target_version))
            loader_root = root
            mode = 'materialized_subset'
    require(actual_builder.info.splits['train'].num_examples == len(wanted), 'Materialized count mismatch')
    stats, dataset_payload_sha256 = statistics(episodes(actual_builder), wanted, suite)
    total = stats['num_transitions']
    require(total == selection['selected_transitions'], 'Transition count mismatch')
    require(stats['num_trajectories'] == len(wanted) and stats['num_transitions'] == total,
            'Normalization source identity/count mismatch')
    cache = None
    if shots != 'Full':
        cache = install_statistics_cache(loader_root, dataset_name, actual_builder, stats)
    check_train_loader(loader_root, dataset_name, stats, use_cache=cache is not None)
    if (target / 'normalization.json').exists():
        require(read(target / 'normalization.json') == stats, 'Existing normalization changed')
    else:
        write(target / 'normalization.json', stats)
    result = dict(status='PASS', suite=suite, seed=seed, shots=shots,
                  materialization=mode, dataset_name=dataset_name, rlds_root=str(loader_root),
                  source_revision=audit['source']['revision'], source_file_hashes={k:v['sha256'] for k,v in audit['source']['files'].items()},
                  selected_episode_ids=sorted(wanted), selection_sha256=selection['selection_sha256'],
                  dataset_payload_sha256=dataset_payload_sha256,
                  episode_count=len(wanted), transition_count=total, normalization_sha256=digest(stats),
                  train_loader='OFT RLDSDataset: one image/proprio/8x7 action sample decoded',
                  training_statistics_cache=str(cache) if cache else None,
                  final_action_chunk='Original terminal action retained in RLDS; OFT window=1,future=7 excludes final 7 chunk starts (no terminal-padding training frames)')
    p = target / 'validation.json'
    if p.exists():
        require(read(p) == result, 'Existing validation differs')
    else:
        write(p, result)
    if smoke:
        p = OUT / 'analysis' / 'smoke_validation.json'
        if p.exists():
            require(read(p) == result, 'Existing smoke result differs')
        else:
            write(p, result)
    return result
