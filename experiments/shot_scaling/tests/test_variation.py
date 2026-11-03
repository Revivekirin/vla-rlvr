import copy
import unittest
import numpy as np
from experiments.shot_scaling.variation import (align,command_features,episode_features,rms_matrix,
    nearest,average_pair_distance,random_order,standardize,compute,band)


def episode(i=0, actions=None):
    if actions is None:
        t=np.linspace(0,1,12+i)
        actions=np.stack([np.sin(t*(i+1)),t*(i+1),np.cos(t*(i+1)),t*t,t/(i+1),t**3,np.where(t<.3+(i%3)*.1,-1,1)],1)
    return dict(episode_id=str(i),task='task',actions=np.asarray(actions).tolist(),
                initial_proprio=[float(i+j*i*.1) for j in range(8)])


class VariationTests(unittest.TestCase):
    def test_degenerate_random_band_not_spurious_extreme_percentile(self):
        b=band([.6981972684509141]*64,.6981972684509142)
        self.assertTrue(b['degenerate'])
        self.assertEqual(b['seed42_percentile'],50.)

    def test_zero_vectors_not_fake_reversals(self):
        f=command_features(np.zeros((5,3)))
        self.assertEqual(f['zero_count'],5)
        self.assertEqual(f['valid_successive_pairs'],0)
        self.assertIsNone(f['successive_cosine_mean'])
        self.assertEqual(np.sum(f['direction_histogram']),0)

    def test_reversals_exclude_zero_pairs(self):
        f=command_features(np.array([[1.,0,0],[-1,0,0],[0,0,0],[1,0,0]]))
        self.assertEqual(f['valid_successive_pairs'],1)
        self.assertEqual(f['reversal_count'],1)
        self.assertEqual(f['successive_cosine_mean'],-1)
        self.assertEqual(np.sum(f['direction_histogram']),3)

    def test_gripper_events_and_no_physical_reconstruction(self):
        actions=np.zeros((4,7));actions[:,-1]=[-1,1,1,-1]
        f=episode_features(episode(actions=actions),32)
        self.assertEqual([e['command'] for e in f['gripper_events']],['close','open'])
        self.assertEqual(f['gripper_closed_fraction'],.5)
        self.assertIsNone(f['eef_path_length'])
        self.assertIsNone(f['eef_net_displacement'])
        self.assertIsNone(f['success'])
        self.assertEqual(len(f['descriptor']),len(f['descriptor_names']))

    def test_temporal_distance_detects_order_hidden_by_mean_std(self):
        x=np.array([[0.,1],[1,0],[2,1],[3,0]])
        y=x[::-1]
        np.testing.assert_allclose(x.mean(0),y.mean(0));np.testing.assert_allclose(x.std(0),y.std(0))
        m=rms_matrix(np.array([align(x),align(y)]))
        self.assertGreater(m[0,1],0)
        np.testing.assert_allclose(np.diag(m),0)
        np.testing.assert_allclose(m,m.T)

    def test_alignment_keeps_endpoints(self):
        x=np.array([[1.,2],[3,4]])
        y=align(x,32)
        np.testing.assert_array_equal(y[0],x[0]);np.testing.assert_array_equal(y[-1],x[-1])
        np.testing.assert_array_equal(align(x[:1],32),np.repeat(x[:1],32,axis=0))

    def test_leakage_rejected_and_singleton_diversity_unavailable(self):
        m=rms_matrix(np.array([[0.,0],[1,0],[2,0]]))
        with self.assertRaises(ValueError):nearest(m,[0],[0,1])
        self.assertIsNone(average_pair_distance(m,[0]))
        self.assertLessEqual(nearest(m,[0,1],[2])[0],nearest(m,[0],[2])[0])

    def test_scaling_fits_candidate_only(self):
        x=np.array([[0.,1],[2,1],[100,999]])
        z,s=standardize(x,np.array([0,1]))
        self.assertEqual(s['mean'],[1.,1.])
        self.assertEqual(s['used_dimensions'],[0])
        self.assertEqual(z.shape,(3,1))
        self.assertEqual(z[2,0],99)

    def test_random_nested_and_order_independent(self):
        ids=list(map(str,range(20)))
        order=random_order(ids,7,'task')
        self.assertEqual(order,random_order(ids[::-1],7,'task'))
        self.assertNotEqual(order,random_order(ids,8,'task'))
        self.assertTrue(set(order[:5])<=set(order[:10])<=set(order[:20]))

    def test_compute_fixed_reference_random_degeneracy_and_pair_budget(self):
        trs=[episode(i) for i in range(24)]
        selections={k:set(map(str,range(k))) for k in (1,2,5,10,20)}
        fixed=set(map(str,range(20,24)))
        result,pairs=compute({},trs,selections,selections[5],fixed,seeds=16)
        self.assertEqual(len(pairs),24*23//2)
        for row in result['coverage']:
            self.assertEqual(set(row['reference_episode_ids']),fixed)
            if row['shots']==20:
                np.testing.assert_allclose(row['random_coverage_means'],row['nearest_mean'],atol=1e-12)
            if row['shots']==1:self.assertIsNone(row['selected_diversity'])
        for metric in ('temporal_action','initial_state','descriptor','gripper_pattern'):
            rows=[r for r in result['coverage'] if r['metric']==metric]
            self.assertTrue(all(x['nearest_mean']>=y['nearest_mean']-1e-12 for x,y in zip(rows,rows[1:])))
        with self.assertRaises(ValueError):compute({},trs,selections,selections[5],fixed,seeds=16,max_task_episodes=20)


if __name__=='__main__':unittest.main()
