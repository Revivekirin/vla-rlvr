"""Episode inventory contracts. No implicit download, TFDS extraction, or materialization."""
from collections import Counter
from .core import REVISION, SUITES, digest, require, task_names


def validate_inventory(data, suite):
    require(data['schema'] == 1 and data['suite'] == suite, 'Inventory suite/schema mismatch')
    require(data['dataset_name'] == SUITES[suite]+'_no_noops', 'Mislabeled dataset')
    source = data['source']
    require(source['repo'] == 'openvla/modified_libero_rlds' and source['revision'] == REVISION
            and source['split'] == 'train', 'Source identity mismatch')
    require(data['complete_source'] is True, 'An inventory must include the entire source train split')
    require(bool(source['files']) and all(isinstance(v, str) and len(v) == 64 and
            all(c in '0123456789abcdef' for c in v) for v in source['files'].values()),
            'Missing source content hashes')
    rows = data['episodes']
    require(len(rows) == data['source_episode_count'], 'Source count mismatch')
    require(all(isinstance(r['episode_id'], str) and r['episode_id'] for r in rows), 'Invalid episode IDs')
    require(len({r['episode_id'] for r in rows}) == len(rows), 'Duplicate source episode IDs')
    require(set(r['task'] for r in rows) == set(task_names(suite)), 'Task coverage/mapping mismatch')
    require(all(type(r['steps']) is int and r['steps'] > 0 and r['complete'] is True and
                isinstance(r['sha256'], str) and len(r['sha256']) == 64 and
                all(c in '0123456789abcdef' for c in r['sha256']) for r in rows),
            'Incomplete/unhashed episode')
    require(len({r['sha256'] for r in rows}) == len(rows),
            'Duplicate episode payload under different IDs; cannot certify held-out separation')
    require(data['episodes_sha256'] == digest(rows), 'Inventory checksum mismatch')
    return rows


def select(data, suite, shots, seed):
    rows = validate_inventory(data, suite)
    require(type(seed) is int and seed >= 0, 'Invalid selection seed')
    require(type(shots) is int and shots in (1, 2, 5, 10, 20) or shots == 'Full', 'Invalid shots')
    counts = Counter(r['task'] for r in rows)
    chosen = []
    for task in sorted(counts):
        # Per-task SHA order is independent of K, input ordering, and other tasks.
        ordered = sorted((r for r in rows if r['task'] == task), key=lambda r:
                         (digest(['nested-sha256-v1', seed, task, r['episode_id']]), r['episode_id']))
        k = len(ordered) if shots == 'Full' else shots
        require(len(ordered) >= k, f'Insufficient episodes for {task}: {len(ordered)} < {k}')
        chosen.extend(ordered[:k])
    return dict(schema=1, synthetic=bool(data.get('synthetic', False)), suite=suite,
                dataset_name=data['dataset_name'], source=data['source'],
                selection_algorithm='nested-sha256-v1', shots=shots, selection_seed=seed,
                inventory_sha256=digest(data), source_counts=dict(counts), episodes=chosen,
                selection_sha256=digest(chosen), selected_transitions=sum(r['steps'] for r in chosen),
                status='MANIFEST_ONLY', materialized=False,
                provenance_note='Source hashes/revision are supplied evidence, not independently authenticated HF bytes')


def validate_selection(selection, inventory):
    rows = validate_inventory(inventory, inventory['suite'])
    require(selection['dataset_name'] == inventory['dataset_name'], 'Selection dataset mismatch')
    require(selection['selection_sha256'] == digest(selection['episodes']), 'Selection checksum mismatch')
    lookup = {r['episode_id']: r for r in rows}
    ids = [r['episode_id'] for r in selection['episodes']]
    require(ids and len(ids) == len(set(ids)), 'Empty/duplicate selection')
    require(all(r == lookup.get(r['episode_id']) for r in selection['episodes']), 'Selection source/payload mismatch')
    if selection.get('selection_algorithm') == 'nested-sha256-v1':
        require(selection == select(inventory, inventory['suite'], selection['shots'],
                                    selection['selection_seed']), 'Nested selection identity mismatch')
    else:
        # Preserve the historical Goal selection verbatim; never replace with nested selection.
        from experiments.fewshot_sft.prepare_dataset import validate_manifest
        require(inventory['suite'] == 'goal', 'Only historical Goal manifests supported')
        validate_manifest(selection)
        require(selection['source']['revision'] == inventory['source']['revision'], 'Historical source revision mismatch')
        historical = {k: v['sha256'] for k, v in selection['source']['files'].items()}
        require(historical == inventory['source']['files'], 'Historical source content mismatch')
    return set(ids)


def held_out(selection, inventory):
    selected = validate_selection(selection, inventory)
    return [r for r in inventory['episodes'] if r['episode_id'] not in selected]
