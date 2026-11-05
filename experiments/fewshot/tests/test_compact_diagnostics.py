"""Keep accumulated training logs bounded per optimizer step."""
import json
from pathlib import Path
import tempfile
import types
import unittest

from experiments.fewshot_sft.diagnostics import Diagnostics


class CompactDiagnosticsTests(unittest.TestCase):
    def test_aggregates_microbatches_without_serializing_each_one(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            config = dict(output=str(run), loss_window=100, resume=None,
                          validation=None, diagnostics_format='compact')
            diagnostic = Diagnostics(config, lambda **_: None)
            diagnostic.pending = [
                dict(loss=1.0, samples=1, tasks=['a'], actions={'large': [1] * 100}),
                dict(loss=3.0, samples=3, tasks=['b'] * 3, actions={'large': [2] * 100}),
            ]
            diagnostic.after_step(dict(log_step=1, optimizer=types.SimpleNamespace(
                param_groups=[{'lr': 5e-4}])))
            row = json.loads((run / 'diagnostics.jsonl').read_text())
            self.assertEqual(row['microbatch_count'], 2)
            self.assertEqual(row['samples'], 4)
            self.assertEqual(row['normalized_chunk_l1'], 2.5)
            self.assertNotIn('microbatches', row)
            self.assertEqual(diagnostic.pending, [])


if __name__ == '__main__':
    unittest.main()
