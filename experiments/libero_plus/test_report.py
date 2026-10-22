"""Two CPU-only checks: actual evidence parsing and strict episode aggregation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from evaluate import ROOT, REVISION
from report import aggregate, comparable, nominal


class EvidenceTests(unittest.TestCase):
    def test_nominal_counts(self):
        rows = nominal(ROOT)
        self.assertEqual([r['successful_episodes'] for r in rows], [486, 488, 488])
        for row in rows:
            tasks = json.loads(row['per_task_success_rates'])
            self.assertEqual(len(tasks), 10)
            self.assertEqual(sum(t['episodes'] for t in tasks.values()), 500)
            self.assertEqual(sum(t['successes'] for t in tasks.values()), row['successful_episodes'])

    def test_aggregation_and_matching(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            manifest = dict(revision=REVISION, suite='libero_goal', model='sft', seed=1,
                            tasks=[dict(id=1, name='a', category='Camera Viewpoints'),
                                   dict(id=2, name='b', category='Sensor Noise')])
            (folder / 'manifest.json').write_text(json.dumps(manifest))
            for i, success in [(1, True), (2, False)]:
                entry = manifest['tasks'][i - 1]
                row = dict(revision=REVISION, suite='libero_goal', model='sft', seed=1,
                           task_id=i, task_name=entry['name'], shift=entry['category'],
                           success=success, failure=not success)
                (folder / f'task-{i:04d}.json').write_text(json.dumps(row))
            left = aggregate(folder)
            self.assertEqual(sum(r['success'] for r in left[1].values()), 1)
            self.assertEqual(len(left[1]), 2)
            right = copy.deepcopy(left)
            right[0]['model'] = 'scale'
            self.assertTrue(comparable(left, right))
            right[1].pop(2)
            self.assertFalse(comparable(left, right))
            (folder / 'task-duplicate.json').write_text(json.dumps(left[1][1]))
            with self.assertRaises(ValueError):
                aggregate(folder)
            (folder / 'task-duplicate.json').unlink()
            row['shift'] = 'wrong'
            (folder / 'task-0002.json').write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                aggregate(folder)


if __name__ == '__main__':
    unittest.main()
