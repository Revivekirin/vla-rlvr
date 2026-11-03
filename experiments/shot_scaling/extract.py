"""Read-only Goal RLDS extraction reusing the historical decoder and payload validator."""
import os
from pathlib import Path
import sys

from .core import ROOT, REVISION, digest, require
from .data import validate_inventory, validate_selection


def collect(stream, expected_count, files, selection=None, limit=None):
    from experiments.fewshot_sft.prepare_dataset import episode_record
    records, trajectories = [], []
    import itertools
    for eid, episode in itertools.islice(stream, limit):
        record = episode_record(episode, eid)
        actions = [s['action'].tolist() for s in episode['steps']]
        initial = episode['steps'][0]['observation']['state'].tolist()
        records.append(record)
        trajectories.append(dict(episode_id=eid, task=record['task'], source_payload_sha256=record['sha256'],
            actions=actions, initial_proprio=initial, trajectory_sha256=digest([actions, initial])))
        if len(records) % 10 == 0:
            print(f'Validated {len(records)}/{expected_count} source episodes', file=sys.stderr, flush=True)
    require(bool(records), 'Source contains no episodes')
    if limit is not None:
        require(len(records) == min(limit, expected_count), 'Bounded source episode count mismatch')
        return dict(status='SMOKE_ONLY', episodes_checked=len(records), expected_source_count=expected_count,
                    complete_source=False, trainable=False), None
    require(len(records) == expected_count, 'Source split count does not match TFDS metadata')
    inventory = dict(schema=1, synthetic=False, suite='goal', dataset_name='libero_goal_no_noops',
        complete_source=True, source_episode_count=expected_count, episodes=records,
        episodes_sha256=digest(records), source=dict(repo='openvla/modified_libero_rlds',
            revision=REVISION, split='train', files={k: v['sha256'] for k, v in files.items()}))
    validate_inventory(inventory, 'goal')
    if selection is not None:
        validate_selection(selection, inventory)
    return inventory, dict(inventory_sha256=digest(inventory), episodes=trajectories)


def extract(source, revision, selection=None, limit=None):
    from experiments.fewshot_sft.common import inventory as file_inventory
    from experiments.fewshot_sft.prepare_dataset import episodes, validate_manifest
    source = Path(source).resolve()
    require(revision == REVISION, f'Expected pinned dataset revision {REVISION}')
    require(source.is_dir() and (source/'dataset_info.json').is_file(), 'Source must be a TFDS version directory')
    require('libero_goal_no_noops' in source.parts, 'Expected original Goal no-noops source')
    require(not source.is_relative_to(ROOT/'outputs'), 'Use original RLDS, not a materialized shot subset')
    require(limit is None or type(limit) is int and limit > 0, 'max-episodes must be positive')
    print('Hashing original source files (read-only)...', file=sys.stderr, flush=True)
    files = file_inventory(source)
    if selection is not None:
        validate_manifest(selection)
        require(selection['source']['revision'] == revision and selection['source']['files'] == files,
                'Historical selection source revision/files differ from this RLDS directory')
    # Hide CUDA before importing TensorFlow; do not affect a separate training process.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    import tensorflow_datasets as tfds
    builder = tfds.builder_from_directory(str(source))
    require(builder.info.name in ('libero_goal', 'libero_goal_no_noops'), 'Unexpected TFDS builder')
    require('train' in builder.info.splits, 'Missing train split')
    result = collect(episodes(builder), builder.info.splits['train'].num_examples, files, selection, limit)
    require(file_inventory(source) == files, 'Source files changed during extraction')
    return result
