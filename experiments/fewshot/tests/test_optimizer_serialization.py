"""CPU checks for portable optimizer/scheduler checkpoint serialization."""
from collections import Counter
import hashlib
from pathlib import Path
import pickle
import sys
import tempfile
import types
import unittest
from unittest import mock

from experiments.fewshot_sft.resume_checks import load_optimizer_checkpoint, portable_scheduler_state


class OptimizerSerializationTests(unittest.TestCase):
    def test_new_scheduler_state_has_only_plain_milestone_dict(self):
        scheduler = types.SimpleNamespace(state_dict=lambda: {
            'milestones': Counter({100000: 1}), 'last_epoch': 6000})
        state = portable_scheduler_state(scheduler)
        self.assertIs(type(state['milestones']), dict)
        self.assertEqual(state['milestones'], {100000: 1})
        self.assertEqual(state['last_epoch'], 6000)

    def test_legacy_counter_fallback_only_after_file_hash_and_expected_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'optimizer.pt'
            path.write_bytes(b'locally verified checkpoint fixture')
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            calls = []
            torch = types.ModuleType('torch')

            def load(_path, *, map_location, weights_only):
                calls.append(weights_only)
                if weights_only:
                    raise pickle.UnpicklingError('Unsupported class collections.Counter')
                return {'scheduler': {'milestones': Counter({100000: 1})}}

            torch.load = load
            with mock.patch.dict(sys.modules, {'torch': torch}):
                with self.assertRaisesRegex(ValueError, 'identity changed'):
                    load_optimizer_checkpoint(path, 'wrong-sha256')
                self.assertEqual(calls, [])
                state = load_optimizer_checkpoint(path, expected)
            self.assertEqual(calls, [True, False])
            self.assertEqual(state['scheduler']['milestones'], Counter({100000: 1}))

    def test_other_unpickling_errors_do_not_enable_unsafe_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'optimizer.pt'
            path.write_bytes(b'fixture')
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            calls = []
            torch = types.ModuleType('torch')

            def load(_path, *, map_location, weights_only):
                calls.append(weights_only)
                raise pickle.UnpicklingError('Unsupported class arbitrary.Module')

            torch.load = load
            with mock.patch.dict(sys.modules, {'torch': torch}):
                with self.assertRaisesRegex(ValueError, 'Unexpected optimizer serialization'):
                    load_optimizer_checkpoint(path, expected)
            self.assertEqual(calls, [True])


if __name__ == '__main__':
    unittest.main()
