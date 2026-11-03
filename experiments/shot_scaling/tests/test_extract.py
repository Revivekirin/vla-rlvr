import copy
import unittest
import numpy as np
from experiments.shot_scaling.core import REVISION, digest, task_names
from experiments.shot_scaling.extract import collect
from experiments.shot_scaling.analysis import analyze
from experiments.fewshot_sft.prepare_dataset import episode_record, select


class ExtractTests(unittest.TestCase):
    def setUp(self):
        self.stream = []
        for task in task_names('goal'):
            for i in range(6):
                steps = [dict(action=np.full(7, i/10, dtype=np.float32),
                    observation=dict(state=np.zeros(8, dtype=np.float32),
                        image=np.zeros((2, 2, 3), dtype=np.uint8),
                        wrist_image=np.ones((2, 2, 3), dtype=np.uint8)),
                    language_instruction=task.replace('_', ' ').encode(),
                    is_first=j == 0, is_last=j == 1, is_terminal=j == 1) for j in range(2)]
                self.stream.append((f'train-{task}-{i}', dict(steps=steps)))
        self.files = {'shard': dict(bytes=123, sha256='a'*64)}
        records = [episode_record(e, eid) for eid, e in self.stream]
        chosen = select(records, 5, 42, task_names('goal'))
        self.selection = dict(schema=1, dataset_name='libero_goal_no_noops', shots=5, selection_seed=42,
            episodes=chosen, selection_sha256=digest(chosen), source=dict(revision=REVISION, files=self.files))

    def test_historical_roundtrip_and_analysis(self):
        inv, tr = collect(iter(self.stream), 60, self.files, self.selection)
        self.assertEqual(inv['source_episode_count'], 60)
        lookup = {r['episode_id']: r for r in inv['episodes']}
        for r in self.selection['episodes']:
            self.assertEqual(lookup[r['episode_id']], r)
        result = analyze(inv, tr, self.selection)
        self.assertEqual(sum(e['selected'] for e in result['episodes']), 50)
        self.assertTrue(all(v['held_out'] == 1 for v in result['coverage'].values()))

    def test_bounded_smoke_cannot_be_used_as_inventory(self):
        summary, tr = collect(iter(self.stream), 60, self.files, self.selection, limit=2)
        self.assertEqual(summary['episodes_checked'], 2)
        self.assertFalse(summary['complete_source'])
        self.assertIsNone(tr)
        self.assertNotIn('episodes', summary)

    def test_reject_count_and_historical_payload_mismatch(self):
        with self.assertRaisesRegex(ValueError, 'count'):
            collect(iter(self.stream), 61, self.files)
        changed = copy.deepcopy(self.selection)
        changed['episodes'][0]['sha256'] = 'b'*64
        changed['selection_sha256'] = digest(changed['episodes'])
        with self.assertRaisesRegex(ValueError, 'payload mismatch'):
            collect(iter(self.stream), 60, self.files, changed)


if __name__ == '__main__': unittest.main()
