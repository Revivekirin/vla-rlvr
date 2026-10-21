"""Original LIBERO Goal: one reset, 10 official settling steps, one checked step.

No policy loading, episode loop, success-rate evaluation, or training.
"""
import argparse
import json
from pathlib import Path
import tempfile
import time
import traceback

from validation_utils import ROOT, bounded_worker, configure, digest, provenance, write_json


def run_check(args):
    configure()
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from experiments.robot.libero.libero_utils import get_libero_env, get_libero_dummy_action
    from experiments.robot.libero.run_libero_eval import prepare_observation

    result = {'status': 'RUNNING', 'suite': 'libero_goal', 'provenance': provenance(),
              'task_index': args.task_index, 'initial_state_index': args.state_index,
              'scope': 'Bounded runtime validation; no episode or task-success rate',
              'action_steps': 0, 'cleanup': {'attempted': False, 'completed': False}}
    env = None
    start = time.monotonic()

    def inspect(obs):
        specs = {'agentview_image': (256, 256, 3), 'robot0_eye_in_hand_image': (256, 256, 3),
                 'robot0_eef_pos': (3,), 'robot0_eef_quat': (4,), 'robot0_gripper_qpos': (2,)}
        summary = {}
        for key, shape in specs.items():
            value = np.asarray(obs[key])
            finite = bool(np.isfinite(value).all())
            summary[key] = {'shape': list(value.shape), 'dtype': str(value.dtype), 'finite': finite}
            if value.shape != shape or not finite:
                raise ValueError(f'Invalid observation: {key}: {summary[key]}')
            if key.endswith('_image') and value.dtype != np.uint8:
                raise ValueError(f'Expected uint8 camera output: {key}')
            if not key.endswith('_image') and value.dtype.kind != 'f':
                raise ValueError(f'Expected floating point proprioception: {key}')
        return summary

    try:
        original = ROOT / 'vendor/LIBERO/libero/libero'
        for key, expected in [('benchmark_root', original), ('bddl_files', original / 'bddl_files'),
                              ('init_states', original / 'init_files'), ('assets', original / 'assets')]:
            if Path(get_libero_path(key)).resolve() != expected.resolve():
                raise ValueError(f'LIBERO config {key} does not point to pinned Original LIBERO: {get_libero_path(key)}')
        suite = benchmark.get_benchmark_dict()['libero_goal']()
        if not 0 <= args.task_index < suite.n_tasks:
            raise ValueError('Task index out of bounds')
        task = suite.get_task(args.task_index)
        states = suite.get_task_init_states(args.task_index)
        if not 0 <= args.state_index < len(states):
            raise ValueError('Initial state index out of bounds')
        state_file = Path(get_libero_path('init_states')) / task.problem_folder / task.init_states_file
        result.update(task_name=task.name, instruction=task.language, initial_state_file=str(state_file),
                      initial_state_sha256=digest(state_file))
        env, instruction = get_libero_env(task, 'openvla', resolution=256)
        result['environment'] = {'factory': 'experiments.robot.libero.libero_utils.get_libero_env',
            'controller': env.robots[0].controller.name, 'control_freq': env.env.control_freq,
            'camera_names': list(env.env.camera_names), 'resolution': 256, 'seed': 0,
            'renderer': 'official OffScreenRenderEnv defaults'}
        result['reset_observations'] = inspect(env.reset())
        obs = env.set_init_state(states[args.state_index])
        result['fixed_state_observations'] = inspect(obs)
        # Exactly the ten settling steps used in official run_episode.
        for _ in range(10):
            obs, _, _, _ = env.step(get_libero_dummy_action('openvla'))
            result['action_steps'] += 1
        result['settled_observations'] = inspect(obs)
        prepared, _ = prepare_observation(obs, 224)
        assert prepared['state'].shape == (8,) and np.isfinite(prepared['state']).all()
        snapshot = Path(args.observation_output).resolve()
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        # Reused unchanged by both policy workers; this is the pre-step real observation.
        np.savez_compressed(snapshot, **prepared, instruction=np.array(instruction),
                            suite=np.array('libero_goal'), task_index=np.array(args.task_index),
                            state_index=np.array(args.state_index))
        result['observation_snapshot'] = {'path': str(snapshot), 'sha256': digest(snapshot),
            'preprocessing': 'Official prepare_observation: rotate both cameras 180 degrees, resize to 224, pos+axisangle+gripper; no crop or normalization yet'}
        action = np.asarray(get_libero_dummy_action('openvla'), dtype=float)
        low, high = env.env.action_spec
        if action.shape != (7,) or not np.isfinite(action).all() or not ((action >= low).all() and (action <= high).all()):
            raise ValueError('Official dummy action outside environment action specification')
        before_time = float(env.sim.data.time)
        obs, reward, done, info = env.step(action.tolist())
        result['action_steps'] += 1
        result['post_step_observations'] = inspect(obs)
        success = bool(env.check_success())
        reward_scale = env.env.reward_scale
        expected_reward = float(success) * (1.0 if reward_scale is None else float(reward_scale))
        result['step'] = {'action': action.tolist(), 'convention': 'Environment OSC_POSE delta pose + gripper; official dummy action already in environment convention',
            'reward': float(reward) if np.isfinite(reward) else None, 'done': bool(done), 'check_success': success,
            'expected_sparse_reward': expected_reward, 'info_type': type(info).__name__,
            'sim_time_before': before_time, 'sim_time_after': float(env.sim.data.time)}
        if not isinstance(done, (bool, np.bool_)) or not np.isfinite(reward) or bool(done) != success or not np.isclose(reward, expected_reward):
            raise ValueError('Reward/done semantics differ from pinned Original LIBERO')
        if not env.sim.data.time > before_time:
            raise ValueError('Simulator time did not advance')
        result['status'] = 'PASS'
    except Exception:
        result.update(status='ERROR', error=traceback.format_exc())
    finally:
        if env is not None:
            result['cleanup']['attempted'] = True
            try:
                env.close()
                result['cleanup']['completed'] = True
            except Exception:
                result.update(status='ERROR', cleanup_error=traceback.format_exc())
        result['elapsed_seconds'] = time.monotonic() - start
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task-index', type=int, default=0)
    p.add_argument('--state-index', type=int, default=0)
    p.add_argument('--timeout-seconds', type=int, default=180)
    p.add_argument('--observation-output', type=Path, default=ROOT / 'stage0/evidence/libero_goal_observation.npz')
    p.add_argument('--output', type=Path, default=ROOT / 'stage0/evidence/libero_runtime_goal.json')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    args.output = args.output.resolve()
    args.observation_output = args.observation_output.resolve()
    if args.timeout_seconds <= 0:
        p.error('timeout must be positive')
    if args.worker:
        try:
            result = run_check(args)
        except Exception:
            result = {'status': 'ERROR', 'error': traceback.format_exc(), 'cleanup': 'Environment not created'}
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='libero-runtime-') as tmp:
            result = bounded_worker(Path(__file__).resolve(), ['--task-index', str(args.task_index),
                '--state-index', str(args.state_index), '--observation-output', str(args.observation_output)],
                Path(tmp) / 'result.json', args.output.with_suffix('.log'), args.timeout_seconds)
        result['scope'] = 'One reset, ten official settling steps, one checked action; no success-rate evaluation'
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
