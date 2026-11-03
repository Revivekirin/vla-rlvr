"""CPU artifact analysis; standardized here means z-score, not OFT BOUNDS_Q99."""
import csv
from pathlib import Path
import numpy as np
from .core import digest, require, write
from .data import validate_inventory, validate_selection


def oft_action(raw):
    a = np.asarray(raw, dtype=float).copy()
    require(a.ndim == 2 and a.shape[1] == 7 and len(a) > 0 and np.isfinite(a).all(), 'Invalid action array')
    a[:, -1] = 1 - np.clip(a[:, -1], 0, 1)
    return a


def stats(a):
    return {name: getattr(a, name)(axis=0).tolist() for name in ('mean', 'std', 'min', 'max')}


def directions(a):
    mag = np.linalg.norm(a, axis=1)
    unit = np.divide(a, mag[:, None], out=np.zeros_like(a), where=mag[:, None] > 0)
    counts, edges = np.histogram(np.arctan2(unit[mag > 0, 1], unit[mag > 0, 0]), bins=24, range=(-np.pi, np.pi))
    return dict(unit_xyz=unit.tolist(), zero_commands=int((mag == 0).sum()),
                azimuth_counts=counts.tolist(), azimuth_edges=edges.tolist(), magnitude=mag.tolist())


def analyze(inventory, trajectories, selection=None):
    rows = validate_inventory(inventory, inventory['suite'])
    require(trajectories['inventory_sha256'] == digest(inventory), 'Trajectory inventory identity mismatch')
    tr = trajectories['episodes']
    require(len(tr) == len(rows) and len({e['episode_id'] for e in tr}) == len(tr), 'Trajectory count/duplicates')
    lookup = {r['episode_id']: r for r in rows}
    raw, initial, ids = [], [], []
    for e in tr:
        r = lookup.get(e['episode_id'])
        require(r is not None and e['task'] == r['task'] and e['source_payload_sha256'] == r['sha256'],
                'Trajectory source ID/task/payload mismatch')
        require(e['trajectory_sha256'] == digest([e['actions'], e['initial_proprio']]), 'Trajectory checksum mismatch')
        a, p = np.asarray(e['actions'], dtype=float), np.asarray(e['initial_proprio'], dtype=float)
        oft_action(a)  # common shape/finite check, without changing raw commands
        require(len(a) == r['steps'] and p.shape == (8,) and np.isfinite(p).all(), 'Trajectory shape mismatch')
        raw.append(a)
        initial.append(p)
        ids.append(e['episode_id'])
    all_raw = np.concatenate(raw)
    mean, std = all_raw.mean(0), all_raw.std(0)
    standard = [(a-mean)/np.where(std > 0, std, 1) for a in raw]
    selected = validate_selection(selection, inventory) if selection else set()
    episodes = []
    descriptors = []
    for e, a, z in zip(tr, raw, standard):
        desc = np.concatenate([z.mean(0), z.std(0)])
        descriptors.append(desc)
        episodes.append(dict(episode_id=e['episode_id'], task=e['task'], length=len(a),
             selected=e['episode_id'] in selected, raw=stats(a), standardized=stats(z),
             translation=directions(a[:, :3]), rotation=directions(a[:, 3:6]),
             temporal_abs_change=np.abs(np.diff(a, axis=0)).mean(0).tolist() if len(a) > 1 else None,
             gripper_transitions=int(np.count_nonzero(np.diff(oft_action(a)[:, -1] >= .5))),
             initial_proprio=e['initial_proprio']))
    diversity, coverage = {}, {}
    descriptors = np.stack(descriptors)
    for task in sorted({e['task'] for e in tr}):
        idx = [i for i, e in enumerate(tr) if e['task'] == task]
        x = descriptors[idx]
        # Distances between mean/std trajectory descriptors, not physical trajectories.
        dist = np.linalg.norm(x[:, None]-x[None, :], axis=-1)
        upper = dist[np.triu_indices(len(x), 1)]
        diversity[task] = dict(episodes=len(idx), mean_descriptor_distance=float(upper.mean()) if len(upper) else None,
                              initial_proprio=stats(np.asarray(initial)[idx]))
        train = [i for i in idx if ids[i] in selected]
        held = [i for i in idx if ids[i] not in selected]
        distances = [float(np.linalg.norm(descriptors[train]-descriptors[i], axis=1).min()) for i in held] if train else []
        coverage[task] = dict(selected=len(train), held_out=len(held),
                mean_nearest_descriptor_distance=float(np.mean(distances)) if distances else None,
                distances=distances, status='COMPLETE' if distances else 'UNAVAILABLE')
    def weighted(arrays):
        means = np.stack([a.mean(0) for a in arrays])
        m = means.mean(0)
        second = np.stack([(a*a).mean(0) for a in arrays]).mean(0)
        return dict(transition_weighted=stats(np.concatenate(arrays)),
                    episode_weighted=dict(mean=m.tolist(), std=np.sqrt(np.maximum(second-m*m, 0)).tolist()))
    return dict(schema=1, kind='dataset_analysis', suite=inventory['suite'],
                synthetic=bool(inventory.get('synthetic', False)), inventory_sha256=digest(inventory),
                selection_sha256=digest(selection) if selection else None,
                episodes=episodes, raw=weighted(raw), standardized=weighted(standard),
                standardization=dict(mean=mean.tolist(), std=std.tolist(), fit='entire source, descriptive analysis only'),
                diversity=diversity, coverage=coverage,
                caveats=['Action commands are not physical end-effector displacements.',
                         'Coverage uses per-episode mean/std z-score descriptors; it is not a temporal alignment metric.',
                         'Held-out episodes are the analysis complement, not an official RLDS test split.',
                         'Source extraction is supplied evidence; this tool verifies artifact identity, not RLDS bytes.'])


def save(out, result):
    write(Path(out)/'analysis.json', result)
    with (Path(out)/'episodes.csv').open('x', newline='') as f:
        fields = ['episode_id', 'task', 'length', 'selected', 'gripper_transitions']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: e[k] for k in fields} for e in result['episodes'])
