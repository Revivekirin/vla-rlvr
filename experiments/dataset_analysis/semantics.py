"""Documented EEF semantics and numerical screening, not producer authentication."""
import numpy as np
from experiments.shot_scaling.core import require


def documented_mapping():
    return dict(
        status='SOURCE_DOCUMENTED', state_position_indices=[0,1,2],
        state_orientation_indices=[3,4,5], state_gripper_indices=[6,7],
        coordinate_frame='simulator_world', position_units='m',
        orientation_representation='axis_angle_radians', state_timing='pre_action',
        interpretation='Documented source semantics applied to stored RLDS observations',
        exact_producer_commit_verified=False, bitwise_verified=False,
        matched_hdf5_comparison_performed=False,
        limitations=['Exact producer commit and original simulation environment are not authenticated.',
                     'Meter/world-frame interpretation follows documented simulation conventions, not a calibration check.',
                     'No post-final-action state or physical timestamps are inferred.'],
        sources=[
            dict(path='vendor/openvla-oft/experiments/robot/libero/regenerate_libero_dataset.py',
                 url='https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/experiments/robot/libero/regenerate_libero_dataset.py',
                 claim='Records EEF position, quaternion-to-axis-angle and gripper before the retained action.'),
            dict(path='experiments/dataset_analysis/evidence_sources/builder_LIBERO_Goal_LIBERO_Goal_dataset_builder.py',
                 url='https://github.com/moojink/rlds_dataset_builder/tree/6174b0b6bb69df6361f1117944952bf14afb0cc3',
                 claim='Goal/Spatial/10 converters concatenate ee_states[i] and gripper_states[i] with action[i]. This is a reference revision.'),
            dict(path='experiments/dataset_analysis/evidence_sources/robosuite_robosuite_robots_single_arm.py',
                 claim='EEF position reads simulator site_xpos; quaternion is end-effector body world orientation.'),
            dict(url='https://github.com/openvla/openvla#libero-setup',
                 claim='Official documentation identifies the regeneration script and moojink RLDS builder as the dataset pipeline.')])


def sanity(episodes):
    require(bool(episodes), 'No episodes')
    xyz=[]; changes=[]; stationary=[]; outliers=[]; jumps=[]
    for e in episodes:
        x=np.asarray(e['observation_state'], dtype=float)
        require(x.shape==(e['steps'],8) and len(x)>0 and np.isfinite(x).all(),
                f"Missing, nonfinite or misaligned state: {e['episode_id']}")
        a=np.asarray(e['actions'],dtype=float)
        require(a.shape==(len(x),7) and np.isfinite(a).all(), 'Invalid action/state alignment')
        x=x[:,:3]; delta=np.linalg.norm(np.diff(x,axis=0),axis=1)
        xyz.append(x); changes.append(delta)
        if not np.any(delta>1e-8): stationary.append(e['episode_id'])
        if np.any(np.abs(x)>5): outliers.append(e['episode_id'])
        if np.any(delta>.25): jumps.append(e['episode_id'])
    x=np.concatenate(xyz); delta=np.concatenate(changes)
    warnings=[]
    if outliers: warnings.append('Coordinates exceed broad +/-5 m screening bounds')
    if jumps: warnings.append('Some adjacent retained observations differ by >0.25 m')
    if stationary: warnings.append('Some episodes have no XYZ change above 1e-8 m')
    return dict(status='PASS' if not warnings else 'REVIEW_REQUIRED', finite=True,
        episodes=len(episodes), observations=len(x), xyz_min=x.min(0).tolist(),xyz_max=x.max(0).tolist(),
        xyz_quantiles=np.quantile(x,[0,.01,.5,.99,1],axis=0).tolist(),
        delta_quantiles_m=np.quantile(delta,[0,.5,.95,.99,1]).tolist() if len(delta) else [],
        zero_change_fraction=float(np.mean(delta<=1e-8)) if len(delta) else None,
        stationary_episodes=stationary,out_of_range_episodes=outliers,large_jump_episodes=jumps,
        warnings=warnings,
        screening_note='Broad heuristic flags, not calibrated robot workspace bounds. No clipping or dropping samples. Adjacent retained-step displacement is not velocity.')
