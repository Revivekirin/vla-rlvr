import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from experiments.shot_scaling.core import (REVISION, SUITES, checkpoint_identity, compatible,
    config, digest, task_names, write)
from experiments.shot_scaling.data import select, validate_inventory, held_out
from experiments.shot_scaling.analysis import analyze, oft_action
from experiments.shot_scaling.reporting import aggregate, paired


def fixture(suite='goal', count=21):
    rows, trajectories = [], []
    rng = np.random.default_rng(19)
    for task in task_names(suite):
        for i in range(count):
            actions = rng.normal(size=(i % 7+2, 7)).tolist()
            proprio = rng.normal(size=8).tolist()
            row = dict(episode_id=f'{task}/{i}', task=task, steps=len(actions), complete=True,
                       sha256=digest([actions, proprio, 'synthetic-payload']))
            rows.append(row)
            trajectories.append(dict(episode_id=row['episode_id'], task=task,
                source_payload_sha256=row['sha256'], actions=actions, initial_proprio=proprio,
                trajectory_sha256=digest([actions, proprio])))
    inventory = dict(schema=1, synthetic=True, suite=suite, dataset_name=SUITES[suite]+'_no_noops',
        complete_source=True, source_episode_count=len(rows), episodes=rows, episodes_sha256=digest(rows),
        source=dict(repo='openvla/modified_libero_rlds', revision=REVISION, split='train', files={'fixture': 'a'*64}))
    return inventory, dict(inventory_sha256=digest(inventory), episodes=trajectories)


def bundle(seed=1, train=7, inference='d0_cached'):
    protocol = dict(expected_episode_keys=[[0, 0, 0], [0, 1, 0]], benchmark_revision='fixture')
    condition = dict(base='synthetic', optimizer='synthetic', learning_rate=.0005)
    return dict(synthetic=True, manifest=dict(suite='goal', shots=5, selection_seed=42, train_seed=train, eval_seed=seed,
        condition=condition, condition_sha256=digest(condition),
        checkpoint_sha256='b'*64, inference=inference, dataset='plus', protocol=protocol,
        protocol_sha256=digest(protocol), expected_episode_keys=protocol['expected_episode_keys'], optimizer_step=1),
        episodes=[dict(task_id=0, condition_id=i, trial_id=0, success=bool(i), shift='camera', seed=seed) for i in range(2)])


def checkpoint():
    return dict(suite='goal', shots=5, selection_seed=42, train_seed=7, dataset_sha256='a'*64,
        base_sha256='b'*64, sft_sha256='c'*64, scale_sha256='d'*64, normalization_sha256='e'*64,
        architecture={'chunk': 8, 'action_dim': 7}, stage='d0_cached')


class PipelineTest(unittest.TestCase):
    def test_nested_all_suites_and_seeds(self):
        for suite in SUITES:
            inv, _ = fixture(suite)
            for seed in (0, 42, 123):
                previous = set()
                for k in (1, 2, 5, 10, 20, 'Full'):
                    m = select(inv, suite, k, seed)
                    chosen = {r['episode_id'] for r in m['episodes']}
                    self.assertTrue(previous <= chosen)
                    self.assertEqual(len(chosen), 210 if k == 'Full' else 10*k)
                    previous = chosen
                self.assertEqual(select(inv, suite, 'Full', seed)['source_counts'], {t:21 for t in task_names(suite)})

    def test_order_independence(self):
        inv, _ = fixture()
        before = select(inv, 'goal', 5, 42)['episodes']
        inv['episodes'].reverse()
        inv['episodes_sha256'] = digest(inv['episodes'])
        self.assertEqual(before, select(inv, 'goal', 5, 42)['episodes'])
        self.assertNotEqual(before, select(inv, 'goal', 5, 43)['episodes'])

    def test_insufficient_duplicates_mislabeled(self):
        inv, _ = fixture(count=2)
        with self.assertRaises(ValueError): select(inv, 'goal', 5, 42)
        with self.assertRaises(ValueError): validate_inventory(inv, 'long')
        inv['episodes'][1]['episode_id'] = inv['episodes'][0]['episode_id']
        inv['episodes_sha256'] = digest(inv['episodes'])
        with self.assertRaises(ValueError): validate_inventory(inv, 'goal')

    def test_no_leakage(self):
        inv, _ = fixture()
        m = select(inv, 'goal', 5, 42)
        a = {r['episode_id'] for r in m['episodes']}
        b = {r['episode_id'] for r in held_out(m, inv)}
        self.assertFalse(a & b)
        self.assertEqual(len(a | b), 210)
        inv['episodes'][1]['sha256'] = inv['episodes'][0]['sha256']
        inv['episodes_sha256'] = digest(inv['episodes'])
        with self.assertRaises(ValueError): held_out(m, inv)

    def test_historical_sampler_unchanged_and_distinct(self):
        from experiments.fewshot_sft.prepare_dataset import select as old
        inv, _ = fixture()
        a = old(inv['episodes'], 5, 42, task_names('goal'))
        self.assertNotEqual(a, select(inv, 'goal', 5, 42)['episodes'])
        with self.assertRaises(ValueError): old(inv['episodes'], 2, 42, task_names('goal'))

    def test_action_transform(self):
        x = np.zeros((4, 7)); x[:, -1] = [-1, 0, 1, 2]
        y = oft_action(x)
        np.testing.assert_array_equal(y[:, -1], [1, 1, 0, 0])
        np.testing.assert_array_equal(x[:, -1], [-1, 0, 1, 2])
        with self.assertRaises(ValueError): oft_action(np.ones((2, 8)))

    def test_analysis_weighting_and_coverage(self):
        inv, tr = fixture(count=2)
        result = analyze(inv, tr, select(inv, 'goal', 1, 42))
        self.assertTrue(all(v['held_out'] == v['selected'] == 1 for v in result['coverage'].values()))
        arrays = [np.asarray(e['actions']) for e in tr['episodes']]
        np.testing.assert_allclose(result['raw']['transition_weighted']['mean'], np.concatenate(arrays).mean(0))
        np.testing.assert_allclose(result['raw']['episode_weighted']['mean'], np.mean([x.mean(0) for x in arrays], axis=0))
        full = analyze(inv, tr, select(inv, 'goal', 'Full', 42))
        self.assertTrue(all(v['mean_nearest_descriptor_distance'] is None for v in full['coverage'].values()))
        tr['episodes'][0]['actions'][0][0] += 1
        with self.assertRaises(ValueError): analyze(inv, tr)

    def test_config_no_cwd_or_hydra_leak(self):
        cwd = Path.cwd()
        for suite in SUITES:
            c = config([f'suite={suite}', 'shots=Full', 'sft.save_freq=${sft.max_steps}'])
            self.assertEqual(c['suite']['benchmark'], SUITES[suite])
            self.assertEqual(c['sft']['save_freq'], 1)
        self.assertEqual(cwd, Path.cwd())
        from hydra.core.global_hydra import GlobalHydra
        self.assertFalse(GlobalHydra.instance().is_initialized())
        for override in ('shots=3', 'execution.world_size=2', 'sft.base=released', 'sft.chunk=4',
                         'selection_seed=-1', 'evaluation.max_trials=0', 'output=outputs/goal'):
            with self.assertRaises(ValueError): config([override])

    def test_resume_and_overwrite(self):
        from experiments.shot_scaling.__main__ import main
        inv, _ = fixture()
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)
            write(p/'inventory.json', inv)
            write(p/'selection.json', select(inv, 'goal', 5, 42))
            with self.assertRaises(FileExistsError): write(p/'selection.json', {})
            self.assertEqual(main(['prepare', '--inventory', str(p/'inventory.json'), '--resume', str(p/'selection.json')]), 0)
            with self.assertRaises(ValueError): main(['prepare', '--inventory', str(p/'inventory.json'),
                                                     '--resume', str(p/'selection.json'), 'selection_seed=43'])

    def test_checkpoint_and_pair_identity(self):
        a = checkpoint(); b = dict(a, stage='d1_cached', parent_d0_sha256=digest(a))
        self.assertTrue(compatible(a, b))
        b['sft_sha256'] = 'f'*64
        with self.assertRaises(ValueError): compatible(a, b)
        with tempfile.TemporaryDirectory() as t:
            p = Path(t)/'weight'; p.write_bytes(b'first')
            before = checkpoint_identity(t)
            p.write_bytes(b'second')
            self.assertNotEqual(before, checkpoint_identity(t))

    def test_aggregation_no_pseudoreplication(self):
        a, b, c = bundle(), bundle(seed=2), bundle(train=8)
        c['episodes'][0]['success'] = True
        result = aggregate([a, b, c])
        self.assertEqual(result['groups'][0]['rate'], .75)
        self.assertEqual(result['groups'][0]['training_seeds'], 2)
        self.assertAlmostEqual(result['groups'][0]['training_seed_sem'], .25)
        with self.assertRaises(ValueError): aggregate([a, a])
        different = copy.deepcopy(c)
        different['manifest']['condition']['learning_rate'] = .0001
        different['manifest']['condition_sha256'] = digest(different['manifest']['condition'])
        self.assertEqual(len(aggregate([a, different])['groups']), 2)
        a['episodes'].pop()
        self.assertEqual(aggregate([a])['runs'][0]['status'], 'INCOMPLETE')
        self.assertEqual(aggregate([a])['groups'], [])

    def test_matched_pair(self):
        a, b = bundle(), bundle(inference='d1_cached')
        a['checkpoint'] = checkpoint()
        b['checkpoint'] = dict(a['checkpoint'], stage='d1_cached', parent_d0_sha256=digest(a['checkpoint']))
        for r in (a, b): r['manifest']['checkpoint_sha256'] = digest(r['checkpoint'])
        b['episodes'][0]['success'] = True
        self.assertEqual(paired(a, b)['camera']['difference_pp'], 50)
        b['manifest']['eval_seed'] = 2
        with self.assertRaises(ValueError): paired(a, b)

    def test_plotting_saved_artifact(self):
        from experiments.shot_scaling.plotting import plot
        inv, tr = fixture(count=2)
        result = analyze(inv, tr, select(inv, 'goal', 1, 42))
        with tempfile.TemporaryDirectory() as t:
            path = plot(json.loads(json.dumps(result)), Path(t)/'plots')
            self.assertTrue(path.is_file())
            self.assertEqual(len(list(path.parent.glob('*.png'))), 5)
            self.assertEqual(len(list(path.parent.glob('*.svg'))), 5)
            with self.assertRaises(FileExistsError): plot(result, path.parent)

    def test_metric_plot_missing_and_error_bars(self):
        from experiments.shot_scaling.plotting import plot
        data = dict(kind='metric_series', synthetic=True, series=[dict(metric='cka', x_label='Layer', label='Fixture',
            points=[dict(x=0, y=.5, status='COMPLETE', sem=.1, independent_seeds=2), dict(x=1, y=None, status='INCOMPLETE')])])
        with tempfile.TemporaryDirectory() as t:
            self.assertTrue(plot(data, Path(t)/'plot').is_file())

    def test_blocked_commands_nonzero(self):
        from contextlib import redirect_stdout
        import io
        from experiments.shot_scaling.__main__ import BLOCKERS, main
        for command in BLOCKERS:
            with redirect_stdout(io.StringIO()): self.assertEqual(main([command, '--check']), 2)


if __name__ == '__main__': unittest.main()
