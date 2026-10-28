"""CPU-only P3-A1 tests. No policy, torch, simulator, or GPU imports."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import struct
from types import SimpleNamespace, ModuleType
from unittest.mock import patch
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3_action_analysis as p3


STATS = {'action': {'q01': [-1.] * 6 + [0.], 'q99': [1.] * 7, 'mask': [True] * 6 + [False]},
         'proprio': {'q01': [-1.] * 8, 'q99': [1.] * 8}}


def convert(mean):
    raw = mean.astype(np.float64).copy()
    raw[..., :6] = .5 * (mean[..., :6] + 1) * (2 + 1e-8) - 1
    executed = raw.copy()
    executed[..., -1] = -np.sign(2 * raw[..., -1] - 1)
    return raw, executed


def fixture(run, mismatch=False):
    first_n = 38 if mismatch else 1
    n = first_n + 38
    task_ids = np.array([1] * first_n + [2] * 38)
    steps = np.concatenate([np.arange(first_n) * 8, np.arange(38) * 8])
    counts = np.full(n, 8)
    counts[-1] = 4
    if mismatch:
        counts[first_n - 1] = 4
    language = np.array(['camera lost task'] * first_n + ['camera gained task'] * 38)
    obs = dict(full_image=np.zeros((n, 224, 224, 3), dtype=np.uint8),
               wrist_image=np.ones((n, 224, 224, 3), dtype=np.uint8), state=np.zeros((n, 8), dtype=np.float64))
    mean = np.zeros((n, 8, 7), dtype=np.float32)
    raw, executed = convert(mean)
    data = dict(**obs, language=language, task_id=task_ids, step=steps,
                input_sha256=np.array([p3.input_sha({k: v[i] for k, v in obs.items()}, str(language[i])) for i in range(n)]),
                normalized_mean=mean, unnormalized=raw, executed=executed, executed_count=counts)
    cases = [dict(task={'id': i + 1}, historical_case=case,
                  evaluation={'scale': {'record': {'success': success}}})
             for i, (case, success) in enumerate([('lost', True), ('gained', False)])]
    metadata = dict(horizon=300, source='scale', normalization=STATS, normalization_key='test', cases=cases)
    episodes = [dict(task_id=i + 1, steps=8 if i == 0 else 300, source='scale', shift=p3.SHIFT,
                     historical_case=case, analysis_case=case, reproduction='matched', success=success)
                for i, (case, success) in enumerate([('lost', True), ('gained', False)])]
    if mismatch:
        episodes[0].update(success=False, steps=300, reproduction='reproduction_mismatch', analysis_case='reproduction_mismatch')
    p3.save_npz(run / 'observations.npz', **data)
    p3.save_json(run / 'capture.json', dict(episodes=episodes, observations_sha256=p3.sha(run / 'observations.npz')))
    for model in p3.MODELS:
        out = mean.copy()
        if model == 'ript':
            out[..., 0] = .3
            out[..., 1] = .4
            out[..., 3] = 1.2
            out[0, :4, 6] = 1
        raw, executed = convert(out)
        p3.save_npz(run / f'{model}_actions.npz', normalized_mean=out, unnormalized=raw, executed=executed,
                    **{k: data[k] for k in ('input_sha256', 'task_id', 'step')})
        p3.save_json(run / f'{model}_worker.json', dict(model=model, normalization_key='test', cached_inference=True,
                                                     sampling=False, artifact_sha256=p3.sha(run / f'{model}_actions.npz')))
    return metadata, data


class AnalysisTests(unittest.TestCase):
    def test_hash_covers_pixels_state_language_dtype(self):
        obs = dict(full_image=np.zeros((224, 224, 3), np.uint8), wrist_image=np.zeros((224, 224, 3), np.uint8), state=np.zeros(8))
        p3.check_obs(obs, 'task')
        original = p3.input_sha(obs, 'task')
        for key in obs:
            changed = copy.deepcopy(obs)
            changed[key].flat[0] = 1
            self.assertNotEqual(original, p3.input_sha(changed, 'task'))
        self.assertNotEqual(original, p3.input_sha(obs, 'other task'))
        obs['state'] = obs['state'].astype(np.float32)
        self.assertNotEqual(original, p3.input_sha(obs, 'task'))

    def test_observation_rejections(self):
        obs = dict(full_image=np.zeros((224, 224, 3), np.uint8), wrist_image=np.zeros((224, 224, 3), np.uint8), state=np.zeros(8))
        for field, value in [('state', np.full(8, np.nan)), ('full_image', np.zeros((256, 256, 3), np.uint8)), ('wrist_image', np.zeros((224, 224, 3), np.float32))]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                p3.check_obs(dict(obs, **{field: value}), 'task')

    def test_action_validation_and_gripper_boundary(self):
        mean = np.zeros((8, 7), np.float32)
        mean[:, 6] = .5
        raw, executed = convert(mean)
        self.assertTrue(np.all(executed[:, 6] == 0))  # Preserve np.sign(0), do not silently force +/-1.
        p3.check_actions(mean, raw, executed, STATS)
        with self.assertRaises(ValueError):
            p3.check_actions(mean, raw + 1, executed, STATS)
        with self.assertRaises(ValueError):
            p3.check_actions(mean[:, :6], raw[:, :6], executed[:, :6], STATS)
        mean[0, 0] = np.inf
        with self.assertRaises(ValueError):
            p3.check_actions(mean, raw, executed, STATS)

    def test_known_metrics(self):
        a, b = np.zeros((2, 8, 7)), np.zeros((2, 8, 7))
        b[..., :3] = [3, 4, 0]
        b[..., 3:6] = [0, 0, 12]
        d = p3.distances(a, b)
        np.testing.assert_array_equal(d['translation_l2'], 5)
        np.testing.assert_array_equal(d['rotation_l2'], 12)
        np.testing.assert_array_equal(d['action_l2'], 13)
        self.assertEqual(p3.statistics([0, 1, 2, 3]), dict(count=4, mean=1.5, median=1.5, p95=2.8499999999999996, max=3.))

    def test_roundtrip_summary_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            metadata, data = fixture(run)
            summary = p3.summarize(run, metadata, write=True)
            self.assertEqual(summary['observations'], 39)
            self.assertEqual(summary['overall']['gripper_disagreement_rate'], 4 / (39 * 8))
            self.assertEqual(summary['groups']['scale|Camera Viewpoints|lost']['gripper_disagreement_rate'], .5)
            self.assertEqual(summary['groups']['scale|Camera Viewpoints|gained']['gripper_disagreement_rate'], 0.)
            self.assertTrue(summary['source_replay_exact'])
            self.assertAlmostEqual(summary['overall']['normalized_mean']['translation_l2_per_chunk']['mean'], np.sqrt(8) * .5, places=6)
            recomputed = p3.summarize(run, metadata)
            self.assertEqual(summary.pop('actions_csv_sha256'), p3.sha(run / 'actions.csv'))
            self.assertEqual(summary, recomputed)
            with (run / 'actions.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 39 * 8)
            with self.assertRaises(FileExistsError):
                p3.save_npz(run / 'observations.npz', x=np.zeros(1))
            with self.assertRaises(FileExistsError):
                p3.summarize(run, metadata, write=True)

    def test_corrupt_observation_rejected_before_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            metadata, _ = fixture(run)
            with (run / 'observations.npz').open('ab') as f:
                f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                p3.load_observations(run, metadata)

    def test_reordered_replay_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            metadata, _ = fixture(run)
            with np.load(run / 'ript_actions.npz') as archive:
                arrays = {k: archive[k][::-1] for k in archive.files}
            # Deliberate fixture corruption, never used on experiment artifacts.
            with (run / 'ript_actions.npz').open('wb') as f:
                np.savez(f, **arrays)
            info = p3.read_json(run / 'ript_worker.json')
            info['artifact_sha256'] = p3.sha(run / 'ript_actions.npz')
            (run / 'ript_worker.json').write_text(json.dumps(info))
            with self.assertRaisesRegex(ValueError, 'inputs/order'):
                p3.summarize(run, metadata)

    def test_reproduction_labels(self):
        self.assertEqual(p3.reproduction(False, 16, 16, False), 'not_verified_truncated')
        self.assertEqual(p3.reproduction(True, 9, 16, True), 'matched')
        self.assertEqual(p3.reproduction(False, 300, 300, True), 'reproduction_mismatch')
        with tempfile.TemporaryDirectory() as folder:
            metadata, _ = fixture(Path(folder), mismatch=True)
            summary = p3.summarize(Path(folder), metadata)
            self.assertNotIn('scale|Camera Viewpoints|lost', summary['groups'])
            self.assertIn('scale|Camera Viewpoints|reproduction_mismatch', summary['groups'])

    def test_selection_uses_original_pair_outcomes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            candidate = root / 'candidates.csv'
            candidate.write_text('task_id,shift,case\n2,Camera Viewpoints,gained\n1,Camera Viewpoints,lost\n')
            entries = [dict(id=i, name=f'task{i}', category=p3.SHIFT) for i in (1, 2)]
            for model in p3.MODELS:
                out = root / model / 'seed-1'
                out.mkdir(parents=True)
                for i in (1, 2):
                    success = (model == 'scale') == (i == 1)
                    p3.save_json(out / f'task-{i:04d}.json', dict(task_id=i, task_name=f'task{i}', shift=p3.SHIFT,
                                 model=model, seed=1, revision=p3.REVISION, success=success, failure=not success))
            cases = p3.select_cases(candidate, entries, root, 1)
            self.assertEqual([c['task']['id'] for c in cases], [1, 2])
            candidate.write_text('task_id,case\n1,gained\n2,gained\n')
            with self.assertRaisesRegex(ValueError, 'conflicts'):
                p3.select_cases(candidate, entries, root, 1)

    def test_safetensor_header_storage_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'weights.safetensors'
            header = json.dumps({'weight': {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 8]}}).encode()
            path.write_bytes(struct.pack('<Q', len(header)) + header + b'\x00' * 8)
            self.assertEqual(p3.safetensor_header(path), {'weight'})
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                p3.safetensor_header(path)

    def test_bounded_capture_chunk_order_and_truncation(self):
        # Test the real capture loop with a CPU fake simulator, never a GPU policy.
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            plus = root / 'plus'
            run = root / 'run'
            run.mkdir()
            entries = [dict(id=i, name=f'task{i}', category=p3.SHIFT) for i in (1, 2)]
            tasks = [SimpleNamespace(name=f'task{i}', problem_folder='libero_goal',
                                     bddl_file=f'{i}.bddl', init_states_file=f'{i}.pruned_init') for i in (1, 2)]
            for task in tasks:
                for kind, name in [('bddl_files', task.bddl_file), ('init_files', task.init_states_file)]:
                    path = plus / 'libero/libero' / kind / task.problem_folder / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('test fixture')
            suite = SimpleNamespace(n_tasks=2, get_task=lambda i: tasks[i], get_task_init_states=lambda i: [np.zeros(3)])
            parent = ModuleType('libero')
            child = ModuleType('libero.libero')
            child.__file__ = str(plus / 'libero/libero/__init__.py')
            child.benchmark = SimpleNamespace(get_benchmark_dict=lambda: {p3.SUITE: lambda **kw: suite})
            parent.libero = child
            environments = []
            obs = dict(full_image=np.zeros((224, 224, 3), np.uint8), wrist_image=np.zeros((224, 224, 3), np.uint8), state=np.zeros(8))
            class Env:
                def __init__(self, first):
                    self.first, self.actions, self.closed = first, [], False
                    self.env = SimpleNamespace(robots=[SimpleNamespace(action_dim=7, controller=SimpleNamespace(
                        control_dim=6, impedance_mode='fixed', use_delta=True, use_ori=True))])
                def reset(self):
                    pass
                def set_init_state(self, state):
                    return obs
                def step(self, action):
                    self.actions.append(action)
                    return obs, 0, self.first and len(self.actions) == 11, {}
                def close(self):
                    self.closed = True
            def make_env(task, family, resolution):
                self.assertEqual(resolution, 256)
                env = Env(task.name == 'task1')
                environments.append(env)
                return env, task.name
            oft = SimpleNamespace(set_seed_everywhere=lambda seed: None, get_libero_env=make_env,
                                  get_libero_dummy_action=lambda family: [0] * 6 + [-1],
                                  prepare_observation=lambda raw, size: (raw, None))
            mean = np.zeros((8, 7), np.float32)
            mean[:, 0] = np.arange(8) / 10
            raw, executed = convert(mean)
            metadata = dict(plus=str(plus), seed=1, source='scale', horizon=16, normalization=STATS,
                            cases=[dict(task=e, historical_case=case, evaluation={'scale': {'record': {'success': success, 'steps': 300}}})
                                   for e, case, success in zip(entries, ['lost', 'gained'], [True, False])])
            with patch.dict(sys.modules, {'libero': parent, 'libero.libero': child}), \
                 patch.object(p3, 'setup', return_value=(None, oft)), \
                 patch.object(p3, 'load_policy', return_value=(lambda o, l: (mean, raw, executed), {})), \
                 patch.object(p3, 'task_mapping', return_value=entries):
                p3.capture_worker(None, run, metadata)
            data, capture = p3.load_observations(run, metadata)
            self.assertEqual([len(e.actions) for e in environments], [11, 26])
            self.assertTrue(all(e.closed for e in environments))
            np.testing.assert_array_equal(environments[1].actions[10:18], executed)
            np.testing.assert_array_equal(environments[1].actions[18:26], executed)
            np.testing.assert_array_equal(data['step'], [0, 0, 8])
            np.testing.assert_array_equal(data['executed_count'], [1, 8, 8])
            self.assertEqual([e['reproduction'] for e in capture['episodes']], ['matched', 'not_verified_truncated'])

    def test_run_path_rejects_traversal(self):
        for name in ('../other', '/tmp/escape', '.', 'x/y'):
            with self.assertRaises(ValueError):
                p3.run_path(name)

    def test_preflight_does_not_import_torch(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('ript.algos.rl_optimizers.openvla_oft_interface', sys.modules)


if __name__ == '__main__':
    unittest.main()
