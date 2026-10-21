"""CPU-only tests for metrics, immutable staging, and bounded-worker failures."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from check_action_compatibility import checkpoint_copy, compare_actions, fingerprint
from validation_utils import bounded_worker


class ValidationTests(unittest.TestCase):
    def test_metrics_keep_dimension_and_chunk_information(self):
        a = np.zeros((8, 7))
        b = a.copy()
        b[2, 3] = -0.56
        result = compare_actions(a, b)
        self.assertAlmostEqual(result['mean_absolute_error'], 0.01)
        self.assertAlmostEqual(result['maximum_absolute_error'], 0.56)
        self.assertAlmostEqual(result['per_action_dimension_mean_absolute_error'][3], 0.07)
        self.assertAlmostEqual(result['absolute_differences'][2][3], 0.56)

    def test_invalid_shapes_and_nonfinite_are_rejected(self):
        for bad in [np.zeros((1, 8, 7)), np.full((8, 7), np.nan), np.full((8, 7), np.inf)]:
            with self.assertRaises(ValueError):
                compare_actions(np.zeros((8, 7)), bad)

    def test_loader_metadata_writes_cannot_reach_original(self):
        with tempfile.TemporaryDirectory() as temp:
            src, dst = Path(temp) / 'source', Path(temp) / 'scratch'
            src.mkdir()
            (src / 'config.json').write_text('{"original": true}')
            with (src / 'weights.safetensors').open('wb') as stream:
                stream.truncate(10_000_001)
            before = fingerprint(src)
            checkpoint_copy(src, dst)
            (dst / 'config.json').write_text('{"loader_updated": true}')
            self.assertEqual(fingerprint(src), before)
            self.assertTrue((dst / 'weights.safetensors').is_symlink())

    def test_worker_crash_is_evidence_not_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            script = path / 'crash.py'
            script.write_text('raise RuntimeError("test failure")\n')
            result = bounded_worker(script, [], path / 'result.json', path / 'worker.log', 10)
            self.assertEqual(result['status'], 'ERROR')
            self.assertNotEqual(result['exit_code'], 0)

    def test_timeout_is_bounded_and_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            script = path / 'sleep.py'
            script.write_text('import time\ntime.sleep(30)\n')
            result = bounded_worker(script, [], path / 'result.json', path / 'worker.log', 0.1)
            self.assertEqual(result['status'], 'TIMEOUT')
            self.assertLess(result['elapsed_seconds'], 15)


if __name__ == '__main__':
    unittest.main()
