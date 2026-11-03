"""Standalone artifact-only figures. Imports no dataset readers or model code."""
import os
from pathlib import Path
import tempfile
from .core import require


def plot(data, output):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'shot-scaling-mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    index = ['# Saved-artifact figures', '', 'Synthetic fixture; not experimental results.' if data.get('synthetic') else
             'Only saved metrics are shown. Missing metrics are not interpolated.', '']
    def save(fig, name):
        fig.tight_layout()
        for ext in ('png', 'svg'):
            fig.savefig(output/f'{name}.{ext}')
        plt.close(fig)
        index.append(f'- [{name}]({name}.png) · [SVG]({name}.svg)')
    if data['kind'] == 'dataset_analysis':
        episodes = data['episodes']
        require(bool(episodes), 'No episodes to plot')
        title = data['suite'] + (' — SYNTHETIC' if data.get('synthetic') else '')
        fig, ax = plt.subplots()
        ax.hist([e['length'] for e in episodes], bins='auto')
        ax.set(xlabel='Episode length (transitions)', ylabel='Episodes', title=title)
        save(fig, 'episode_lengths')
        for field in ('translation', 'rotation'):
            fig, ax = plt.subplots()
            counts = np.sum([e[field]['azimuth_counts'] for e in episodes], axis=0)
            edges = np.asarray(episodes[0][field]['azimuth_edges'])
            ax.bar((edges[:-1]+edges[1:])/2, counts, width=np.diff(edges))
            ax.set(xlabel=f'{field} command azimuth (rad; zero vectors excluded)', ylabel='Transitions', title=title)
            save(fig, field+'_directions')
        fig, ax = plt.subplots()
        ax.hist([e['gripper_transitions'] for e in episodes], bins='auto')
        ax.set(xlabel='Gripper transitions per episode', ylabel='Episodes', title=title)
        save(fig, 'gripper_transitions')
        fig, ax = plt.subplots()
        values = [(k, v['mean_nearest_descriptor_distance']) for k, v in data['coverage'].items()
                  if v['mean_nearest_descriptor_distance'] is not None]
        if values:
            ax.bar(range(len(values)), [v for _, v in values])
            ax.set_xticks(range(len(values)), [k for k, _ in values], rotation=90, fontsize=6)
        else:
            ax.text(.5, .5, 'UNAVAILABLE: no selected/held-out pair', ha='center', transform=ax.transAxes)
        ax.set(ylabel='Nearest mean/std descriptor distance', title=title+' — subset coverage')
        save(fig, 'shot_coverage')
    elif data['kind'] == 'metric_series':
        # Long-form saved metrics also cover CKA, effective rank, and same-observation action deltas.
        allowed = {'sft_loss', 'nominal_success', 'ript_success', 'mixed_success_group_rate',
                   'shot_success', 'shift_delta_pp', 'cka', 'effective_rank', 'action_difference'}
        for series in data['series']:
            require(series['metric'] in allowed, 'Unsupported plot metric')
            fig, ax = plt.subplots()
            complete = [p for p in series['points'] if p['status'] == 'COMPLETE' and p['y'] is not None]
            if complete:
                x, y = [p['x'] for p in complete], [p['y'] for p in complete]
                require(np.isfinite(x).all() and np.isfinite(y).all(), 'Nonfinite metric')
                ax.scatter(x, y)
                for p in complete:
                    if p.get('sem') is not None:
                        require(p.get('independent_seeds', 0) > 1 and p['sem'] >= 0, 'Unsupported error bar')
                        ax.errorbar(p['x'], p['y'], yerr=p['sem'], fmt='none', capsize=3)
            missing = len(series['points'])-len(complete)
            ax.set(xlabel=series['x_label'], ylabel=series['metric'],
                   title=series['label']+f' — {missing} incomplete/missing')
            save(fig, f'{len(index):02d}_{series["metric"]}')
    else:
        raise ValueError('Plot requires dataset_analysis or documented metric_series artifact')
    (output/'index.md').write_text('\n'.join(index)+'\n')
    return output/'index.md'


if __name__ == '__main__':
    import argparse
    from .core import read
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(plot(read(a.input), a.output))
