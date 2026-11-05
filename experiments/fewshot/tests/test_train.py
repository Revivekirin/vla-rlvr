"""CPU checks for the bounded Stage 3 Full training audit."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.fewshot.train import ARCHITECTURE, loss_trend
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


if __name__ == '__main__':
    unittest.main()
