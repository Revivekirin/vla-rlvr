"""CPU checks for the bounded Stage 3 Full training audit."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import sys
import types
import unittest
from unittest import mock

from experiments.fewshot.train import ARCHITECTURE, checkpoint_schedule, configuration, loss_trend
from experiments.fewshot_sft.checkpoint import check
from experiments.fewshot_sft.common import inventory


class LossTrendTests(unittest.TestCase):
    def test_records_window_means_checkpoint_and_task_draws(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            rows = [dict(step=step, normalized_chunk_l1=loss, moving_mean=loss,
                         task_exposure={'task_a': step})
                    for step, loss in ((1, 3.0), (2, 2.0), (3, 1.0))]
            (run / 'diagnostics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            trend = loss_trend(run, 0, 3, [2, 3])
            self.assertEqual(trend['checkpoints'], {'2': 2.0, '3': 1.0})
            self.assertEqual(trend['final_task_frame_draws'], {'task_a': 3})
            self.assertEqual(trend['first_window_mean'], 2.0)
            self.assertEqual(trend['last_window_mean'], 2.0)

    def test_rejects_missing_optimizer_step(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'diagnostics.jsonl').write_text(
                json.dumps(dict(step=2, normalized_chunk_l1=1.0,
                                moving_mean=1.0, task_exposure={})) + '\n')
            with self.assertRaisesRegex(ValueError, 'Missing or duplicate'):
                loss_trend(run, 0, 2, [2])


class FullCheckpointTests(unittest.TestCase):
    def test_full_uses_recorded_episode_count_not_ten_times_shots(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            metadata = dict(base_identity='openvla/openvla-7b', merged=False,
                            architecture=ARCHITECTURE, shots='Full', episode_count=428,
                            selected_transitions=52042, step=1000)
            (checkpoint / 'fewshot_checkpoint.json').write_text(json.dumps(metadata))
            stats = {'libero_goal_no_noops': dict(
                num_trajectories=428, num_transitions=52042,
                action={**{field: [0] * 7 for field in ('mean', 'std', 'min', 'max', 'q01', 'q99')},
                        'mask': [True] * 6 + [False]},
                proprio={field: [0] * 8 for field in ('mean', 'std', 'min', 'max', 'q01', 'q99')})}
            (checkpoint / 'dataset_statistics.json').write_text(json.dumps(stats))
            (checkpoint / 'component_verification.json').write_text(json.dumps({'status': 'passed'}))
            for name in ('action_head', 'proprio_projector'):
                (checkpoint / f'{name}--1000_checkpoint.pt').write_bytes(b'fixture')
            (checkpoint / 'checkpoint_files.json').write_text(json.dumps(inventory(checkpoint)))
            self.assertEqual(check(checkpoint, merged=False)['episode_count'], 428)


class ShotScheduleTests(unittest.TestCase):
    def test_external_output_root_propagates_to_new_process(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, FEWSHOT_SFT_OUTPUT_ROOT=directory)
            result = subprocess.run([sys.executable, '-c',
                'from experiments.fewshot_sft.common import OUT, destination; '
                'print(destination(OUT / "runs" / "test"))'],
                env=env, capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.strip(), str(Path(directory).resolve() / 'runs' / 'test'))

    def test_default_and_explicit_optimizer_step_checkpoints(self):
        self.assertEqual(checkpoint_schedule('single_gpu_batch1_pilot', 52000),
                         [1000, 6000, 12000, 24000, 52000])
        self.assertEqual(checkpoint_schedule('single_gpu_batch1_pilot', 6344),
                         [1000, 6000, 6344])
        self.assertEqual(checkpoint_schedule('single_gpu_batch1_pilot', 6000, [6000, 1265, 3000]),
                         [1265, 3000, 6000])
        with self.assertRaisesRegex(ValueError, 'include final'):
            checkpoint_schedule('single_gpu_batch1_pilot', 6000, [1265, 3000])
        with self.assertRaisesRegex(ValueError, 'unique'):
            checkpoint_schedule('single_gpu_batch1_pilot', 6000, [6000, 6000])
        with self.assertRaisesRegex(ValueError, 'within max_steps'):
            checkpoint_schedule('single_gpu_batch1_pilot', 6000, [6000, 12000])

    def test_numeric_shot_uses_its_own_dataset_and_exposure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base-revision'
            base.mkdir()
            dataset = root / 'goal-5shot'
            validation = dict(rlds_root=str(dataset / 'rlds'), dataset_name='libero_goal_no_noops',
                              source_revision='source', selection_sha256='selection',
                              normalization_sha256='normalization', episode_count=50,
                              transition_count=6344)
            smoke = types.ModuleType('experiments.fewshot.train_smoke')
            smoke.preflight = lambda suite, seed, shots: (dataset, validation, {})
            hub = types.ModuleType('huggingface_hub')
            hub.snapshot_download = lambda *_, **__: str(base)
            torch = types.ModuleType('torch')
            torch.cuda = types.SimpleNamespace(is_available=lambda: True,
                                               is_bf16_supported=lambda: True)
            with (mock.patch.dict(sys.modules, {'experiments.fewshot.train_smoke': smoke,
                                                'huggingface_hub': hub, 'torch': torch}),
                  mock.patch('experiments.fewshot.train.OUT', root),
                  mock.patch('experiments.fewshot.train.destination', side_effect=lambda p: Path(p)),
                  mock.patch('experiments.fewshot.train.inventory', return_value={}),
                  mock.patch('experiments.fewshot.train.sha', return_value='sha')):
                config = configuration('goal', 42, 5, 6344, 'goal-5shot-test',
                                       profile='single_gpu_batch1_pilot',
                                       checkpoint_steps=[1265, 3000, 6344])
            self.assertEqual(config['shots'], 5)
            self.assertEqual(config['dataset'], str(dataset))
            self.assertEqual(config['episode_count'], 50)
            self.assertEqual(config['selected_transitions'], 6344)
            self.assertEqual(config['checkpoint_steps'], [1265, 3000, 6344])
            self.assertEqual(config['checkpoint_exposure']['6344']['nominal_dataset_passes'], 1)
            self.assertIsNone(config['resume'])

            checkpoint = root / 'own-5shot-step3000'
            checkpoint.mkdir()
            (checkpoint / 'optimizer.pt').write_bytes(b'fixture')
            previous = dict(config, step=3000)
            with (mock.patch.dict(sys.modules, {'experiments.fewshot.train_smoke': smoke,
                                                'huggingface_hub': hub, 'torch': torch}),
                  mock.patch('experiments.fewshot.train.OUT', root),
                  mock.patch('experiments.fewshot.train.destination', side_effect=lambda p: Path(p)),
                  mock.patch('experiments.fewshot.train.inventory', return_value={}),
                  mock.patch('experiments.fewshot.train.sha', return_value='sha'),
                  mock.patch('experiments.fewshot.train.check', return_value=previous),
                  mock.patch('experiments.fewshot_sft.resume_checks.verify_normalization')):
                continued = configuration('goal', 42, 5, 6344, 'goal-5shot-continued',
                                          resume=checkpoint, profile='single_gpu_batch1_pilot',
                                          checkpoint_steps=[1265, 3000, 6344])
            self.assertEqual(continued['resume_step'], 3000)
            self.assertEqual(continued['checkpoint_steps'], [6344])
            self.assertEqual(continued['dataset_exposure']['frame_draws'], 3344)
            self.assertFalse(continued['continuity']['bitwise_uninterrupted_equivalence'])


if __name__ == '__main__':
    unittest.main()
