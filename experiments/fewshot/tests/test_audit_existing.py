"""Fixture for the read-only completed-run audit."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.fewshot import audit_existing


class AuditExistingTests(unittest.TestCase):
    def test_reads_losses_batch_and_all_checkpoint_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'runs/goal-full-seed42-1000'
            run.mkdir(parents=True)
            config = dict(suite='goal', shots='Full', max_steps=1000,
                          batch_size=1, gradient_accumulation=1, world_size=1,
                          effective_batch_size=1, selection_sha256='selection',
                          normalization_sha256='normalization', base_snapshot='snapshot',
                          dataset='dataset', rlds_root='rlds', selection_seed=42,
                          train_seed=7, base_revision='revision',
                          base_identity='openvla/openvla-7b', learning_rate=5e-4,
                          scheduler_config={'milestones': [100000]},
                          dataset_exposure={'frame_draws': 1000},
                          official_finetune_sha256='not-current')
            (run / 'training_config.json').write_text(json.dumps(config))
            (run / 'runtime.json').write_text(json.dumps(dict(status='passed', actual_optimizer_steps=1000)))
            with (run / 'diagnostics.jsonl').open('w') as stream:
                for step in range(1, 1001):
                    stream.write(json.dumps(dict(step=step, normalized_chunk_l1=1 / step,
                                                 moving_mean=1 / step, lr=[5e-4])) + '\n')
            for step in audit_existing.STEPS:
                folder = Path(str(run) + f'--{step}_chkpt')
                folder.mkdir()
                (folder / 'fewshot_checkpoint.json').write_text(json.dumps(dict(
                    step=step, selection_sha256='selection',
                    normalization_sha256='normalization', base_snapshot='snapshot')))
                (folder / 'component_verification.json').write_text(json.dumps({'status': 'passed'}))
                (folder / 'checkpoint_files.json').write_text('{}')
            with patch.object(audit_existing, 'OUT', root):
                result = audit_existing.audit(run)
            self.assertEqual(result['batch']['effective_global'], 1)
            self.assertEqual([entry['step'] for entry in result['checkpoints']], [200, 500, 1000])
            self.assertLess(result['losses']['last_100_mean'], result['losses']['first_100_mean'])


if __name__ == '__main__':
    unittest.main()
