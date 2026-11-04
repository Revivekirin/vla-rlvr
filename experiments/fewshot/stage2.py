"""Stage 2 planning and descriptive analysis from the read-only Stage 1 baseline."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import numpy as np

from experiments.shot_scaling.core import ROOT, REVISION, SUITES, digest, read, require, task_names, write
from experiments.shot_scaling.data import select, validate_selection
from experiments.dataset_analysis.analysis import train_inventory, align_path

BASELINE = ROOT / 'outputs/dataset_analysis'
OUT = ROOT / 'outputs/fewshot'
SHOTS = (1, 2, 5, 10, 20, 'Full')


def checked_baseline(path=BASELINE):
    path = Path(path).resolve()
    require(path == BASELINE.resolve(), 'Stage 2 requires outputs/dataset_analysis as its baseline')
    require((path / 'validation.json').is_file(), 'Missing verified outputs/dataset_analysis baseline')
    v = read(path / 'validation.json')
    require(v['status'] == 'PASS', 'Stage 1 baseline did not PASS')
    m = read(path / 'manifest.json')
    require(m['revision'] == REVISION and m['stage'] == 1, 'Stage 1 revision mismatch')
    audits = {}
    for suite in SUITES:
        audit = read(path / 'trajectory_data' / f'{suite}.json')
        require(audit['status'] == 'AUDITED' and audit['source']['revision'] == REVISION,
                f'Invalid {suite} source audit')
        require(audit['episodes_sha256'] == digest(audit['episodes']), 'Stage 1 audit checksum mismatch')
        require(audit['source']['files'] and audit['episodes'],
                'Incomplete Stage 1 audit')
        audits[suite] = audit
    return audits


def selection_path(suite, seed, shots):
    name = 'Full' if shots == 'Full' else f'{shots}shot'
    return OUT / 'datasets' / suite / f'seed{seed}' / name


def build_selections(audits, seeds):
    require(bool(seeds) and len(seeds) == len(set(seeds)) and all(type(s) is int and s >= 0 for s in seeds),
            'Selection seeds must be distinct nonnegative integers')
    selections = {}
    for suite, audit in audits.items():
        inv = train_inventory(audit)
        for seed in seeds:
            for shots in SHOTS:
                s = select(inv, suite, shots, seed)
                validate_selection(s, inv)
                selections[(suite, seed, shots)] = s
            ids = [{r['episode_id'] for r in selections[(suite, seed, k)]['episodes']}
                   for k in SHOTS]
            require(all(a < b for a, b in zip(ids, ids[1:])), 'Selections must nest strictly')
    return selections


def describe(audits, selections):
    rows = []
    for (suite, seed, shots), selection in selections.items():
        full = {e['episode_id']: e for e in audits[suite]['episodes'] if e['split'] == 'train'}
        groups = defaultdict(list)
        for r in selection['episodes']:
            e = full[r['episode_id']]
            require(e['sha256'] == r['sha256'] and e['steps'] == r['steps'] and e['task'] == r['task'],
                    'Selection differs from measured trajectory')
            groups[r['task']].append(e)
        for task, es in sorted(groups.items()):
            lengths = np.array([e['steps'] for e in es], dtype=float)
            xyz = [np.asarray(e['observation_state'], dtype=float)[:, :3] for e in es]
            require(all(x.shape == (e['steps'], 3) and np.isfinite(x).all() for x, e in zip(xyz, es)),
                    'Invalid selected XYZ')
            aligned = [align_path(x) for x in xyz]
            pair = [float(np.sqrt(np.mean(np.sum((a-b)**2, axis=1))))
                    for i, a in enumerate(aligned) for b in aligned[i+1:]]
            starts = np.array([x[0] for x in xyz])
            rows.append(dict(suite=suite, seed=seed, shots=shots, task=task,
                             episode_ids=[e['episode_id'] for e in es], episodes=len(es),
                             transitions=int(lengths.sum()), length_mean=float(lengths.mean()),
                             length_min=int(lengths.min()), length_max=int(lengths.max()),
                             initial_xyz_mean=starts.mean(0).tolist(),
                             path_pair_rms_mean_m=float(np.mean(pair)) if pair else None,
                             path_pair_count=len(pair)))
    return rows


def historical_manifest(audit, path):
    from experiments.fewshot_sft.prepare_dataset import select as historical_select
    from experiments.shot_scaling.core import digest as source_digest
    require(audit['suite'] == 'goal', 'Historical selection must be Goal 5-shot')
    inv = train_inventory(audit)
    rows = historical_select(inv['episodes'], 5, 42, task_names('goal'))
    files = audit['source']['files']
    source = dict(repo='openvla/modified_libero_rlds', revision=audit['source']['revision'],
                  content_sha256=source_digest(files), files=files)
    h = dict(schema=1, dataset_name=inv['dataset_name'], shots=5, selection_seed=42,
             source=source, episodes=rows, selection_sha256=digest(rows))
    if path is not None:
        require(read(path) == h, 'Supplied historical Goal5 differs from audited reconstruction')
    lookup = {r['episode_id']: r for r in inv['episodes']}
    require(len(h['episodes']) == 50, 'Historical Goal5 must contain 50 complete episodes')
    for r in h['episodes']:
        source = lookup.get(r['episode_id'])
        require(source is not None and all(source[k] == r[k] for k in ('task', 'steps', 'sha256')),
                'Historical Goal5 does not match audited source')
    require(h['selection_sha256'] == digest(h['episodes']), 'Historical manifest checksum mismatch')
    return h


def plan(seeds, historical=None):
    audits = checked_baseline()
    selections = build_selections(audits, seeds)
    rows = describe(audits, selections)
    hist = historical_manifest(audits['goal'], historical)
    for key, s in selections.items():
        path = selection_path(*key) / 'selection_manifest.json'
        if path.exists():
            require(read(path) == s, f'Existing manifest differs: {path}')
        else:
            write(path, s)
    if hist is not None:
        path = OUT / 'datasets' / 'historical_goal5' / 'selection_manifest.json'
        if path.exists():
            require(read(path) == hist, 'Historical manifest differs')
        else:
            write(path, hist)
    info = dict(schema=1, source_revision=REVISION, seeds=seeds, conditions=len(selections),
                historical_goal5='reconstructed_from_historical_seed42_algorithm_and_audited_source; kept separate',
                selection='deterministic per-task nested SHA order; no success or coverage optimization',
                path_diversity='32-point normalized-progress XYZ pair RMS, descriptive only', rows=rows)
    p = OUT / 'analysis' / 'selection_analysis.json'
    if p.exists():
        require(read(p) == info, 'Existing analysis differs')
    else:
        write(p, info)
    figures(audits, selections, rows)
    return info


def figures(audits, selections, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'Full': '#414141', 1: '#d55e00', 2: '#e69f00', 5: '#009e73',
              10: '#0072b2', 20: '#cc79a7'}
    first_seed = min(k[1] for k in selections)
    for suite, audit in audits.items():
        task = task_names(suite)[0]
        population = [e for e in audit['episodes'] if e['split'] == 'train' and e['task'] == task]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for ax, shots in zip(axes.flat, SHOTS):
            key = (suite, first_seed, shots)
            ids = {r['episode_id'] for r in selections[key]['episodes'] if r['task'] == task}
            for e in population:
                xyz = np.asarray(e['observation_state'])[:, :3]
                ax.plot(xyz[:, 0], xyz[:, 1], color='#bdbdbd', alpha=.17, lw=.5)
            for e in population:
                if e['episode_id'] in ids:
                    xyz = np.asarray(e['observation_state'])[:, :3]
                    ax.plot(xyz[:, 0], xyz[:, 1], color=colors[shots], alpha=.8, lw=.8)
                    ax.scatter(xyz[0, 0], xyz[0, 1], color=colors[shots], s=6)
            ax.set(title=f'{shots} shots/task ({len(ids)} paths)', xlabel='EEF X (m)', ylabel='EEF Y (m)')
            ax.grid(alpha=.2)
        fig.suptitle(f'{suite}: representative task 1 XY EEF paths | seed {first_seed}\n{task}', fontsize=12)
        fig.savefig(OUT / 'analysis' / f'{suite}_paths.png', dpi=145)
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    for ax, suite in zip(axes[0], SUITES):
        es = [e for e in audits[suite]['episodes'] if e['split'] == 'train']
        starts = np.array([e['observation_state'][0][:3] for e in es])
        ax.scatter(starts[:, 0], starts[:, 1], s=5, color='#bdbdbd', alpha=.4, label='Full')
        for shots in (20, 10, 5, 2, 1):
            ids = {r['episode_id'] for r in selections[(suite, first_seed, shots)]['episodes']}
            selected = np.array([e['observation_state'][0][:3] for e in es if e['episode_id'] in ids])
            ax.scatter(selected[:, 0], selected[:, 1], s=6 if shots > 2 else 12,
                       color=colors[shots], alpha=.75, label=f'{shots}-shot')
        ax.set(title=suite, xlabel='Initial X (m)', ylabel='Initial Y (m)'); ax.grid(alpha=.2)
    for ax, suite in zip(axes[1], SUITES):
        rs = [r for r in rows if r['suite'] == suite and r['seed'] == first_seed]
        for shots in SHOTS:
            vals = [r['path_pair_rms_mean_m'] for r in rs if r['shots'] == shots and r['path_pair_rms_mean_m'] is not None]
            if vals:
                ax.scatter([str(shots)] * len(vals), vals, s=15, alpha=.55, color=colors[shots])
        ax.set(title=f'{suite} task path variation', xlabel='Shots/task', ylabel='Pair RMS (m)'); ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle('Selected initial EEF positions and within-task path variation (descriptive)')
    fig.savefig(OUT / 'analysis' / 'selection_summary.png', dpi=145)
    plt.close(fig)
    index = ['# Stage 2 figure index', '',
             'Default figures use the first task in each suite and the first selection seed.',
             'Gray paths/points show the Full expert population. Colored paths/points show nested random selections.',
             'Path RMS summarizes measured within-task variation, not selection quality.', '']
    for suite in SUITES:
        index.append(f'- `{suite}_paths.png`: task 1, `{task_names(suite)[0]}`; 1/2/5/10/20/Full overlays.')
    index.extend(['- `selection_summary.png`: initial EEF XY and task-level path variation for all conditions.',
                  '', 'For another task, run `python -m experiments.fewshot plot-task --suite SUITE --task-index N --seed SEED`.',
                  'Task indices 1–10 follow the official LIBERO suite task order.'])
    (OUT / 'analysis' / 'figure_index.md').write_text('\n'.join(index)+'\n')
    repro = dict(
        source_audit_sha256={suite: digest(audit['episodes']) for suite, audit in audits.items()},
        selection_sha256={f'{suite}/seed{first_seed}/{shots}': selections[(suite, first_seed, shots)]['selection_sha256']
                          for suite in SUITES for shots in SHOTS},
        matplotlib_version=matplotlib.__version__,
        method='Raw recorded EEF XYZ; Full gray, selected colored; XY overlays; 32-point normalized-progress pair RMS',
        figures=['goal_paths.png','spatial_paths.png','long_paths.png','selection_summary.png'])
    p = OUT / 'analysis' / 'figure_reproducibility.json'
    if p.exists():
        require(read(p) == repro, 'Figure inputs or plotting version changed')
    else:
        write(p, repro)


def plot_task(suite, task_index, seed):
    require(suite in SUITES and type(task_index) is int and 1 <= task_index <= 10,
            'Invalid suite or task index')
    audit = checked_baseline()[suite]
    selections = build_selections({suite: audit}, [seed])
    task = task_names(suite)[task_index-1]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    population = [e for e in audit['episodes'] if e['split'] == 'train' and e['task'] == task]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, shots in zip(axes.flat, SHOTS):
        ids = {r['episode_id'] for r in selections[(suite, seed, shots)]['episodes'] if r['task'] == task}
        for e in population:
            x = np.asarray(e['observation_state'])[:, :3]
            ax.plot(x[:, 0], x[:, 1], color='#bbbbbb', alpha=.15, lw=.5)
        for e in population:
            if e['episode_id'] in ids:
                x = np.asarray(e['observation_state'])[:, :3]
                ax.plot(x[:, 0], x[:, 1], alpha=.8, lw=.8)
                ax.scatter(x[0, 0], x[0, 1], s=7)
        ax.set(title=f'{shots} shots/task ({len(ids)} paths)', xlabel='EEF X (m)', ylabel='EEF Y (m)')
        ax.grid(alpha=.2)
    fig.suptitle(f'{suite} task {task_index}: {task} | seed {seed}')
    path = OUT / 'analysis' / f'{suite}_task{task_index:02d}_seed{seed}.png'
    fig.savefig(path, dpi=145)
    plt.close(fig)
    return path
