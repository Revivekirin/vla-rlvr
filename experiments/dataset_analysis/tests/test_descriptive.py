"""Descriptive EEF semantics and sanity checks; fixtures are not research evidence."""
import unittest
import numpy as np
from experiments.dataset_analysis.semantics import documented_mapping, sanity
from experiments.dataset_analysis.analysis import analyze
from experiments.dataset_analysis.tests.test_dataset import fixture


class DescriptiveTests(unittest.TestCase):
    def test_source_documented_paths_do_not_claim_producer_verification(self):
        a=fixture(); a['physical_convention']=documented_mapping()
        d=analyze(a,seeds=(),documented=True)
        self.assertFalse(d['physical_convention']['bitwise_verified'])
        self.assertFalse(d['physical_convention']['exact_producer_commit_verified'])
        self.assertEqual(d['physical_convention']['status'],'SOURCE_DOCUMENTED')
        self.assertEqual(d['selections'],{})
        self.assertEqual(len(d['physical_pairs']),10*190)
        for e in d['episodes']:
            self.assertEqual(e['physical_status'],'SOURCE_DOCUMENTED')
            self.assertAlmostEqual(e['physical']['path_length'],.02,places=6)
        self.assertAlmostEqual(d['physical_pairs'][0]['aligned_xyz_rms_m'],
            abs(next(e for e in d['episodes'] if e['episode_id']==d['physical_pairs'][0]['episode_a'])['xyz'][0][1]-
                next(e for e in d['episodes'] if e['episode_id']==d['physical_pairs'][0]['episode_b'])['xyz'][0][1]),places=6)

    def test_numerical_flags_are_not_silently_clipped(self):
        e=dict(episode_id='fixture',steps=3,actions=np.zeros((3,7)).tolist(),
               observation_state=np.zeros((3,8)).tolist())
        result=sanity([e]);self.assertEqual(result['stationary_episodes'],['fixture'])
        e['observation_state'][1][0]=6
        result=sanity([e]);self.assertEqual(result['out_of_range_episodes'],['fixture'])
        self.assertEqual(result['large_jump_episodes'],['fixture'])
        self.assertEqual(e['observation_state'][1][0],6)
        e['observation_state'][1][0]=float('nan')
        with self.assertRaises(ValueError):sanity([e])

    def test_missing_or_misaligned_states_fail(self):
        for state in (None,[[0]*8]):
            with self.assertRaises(ValueError):sanity([dict(episode_id='fixture',steps=2,actions=[[0]*7]*2,observation_state=state)])


if __name__=='__main__':unittest.main()
