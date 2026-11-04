"""Optional observation-only logger. Never invokes an extra simulator/policy step."""
import hashlib
import json
from pathlib import Path
import numpy as np


def pose(obs):
    if not isinstance(obs, dict):
        return None
    try:
        p = np.asarray(obs['robot0_eef_pos'], dtype=float)
        q = np.asarray(obs['robot0_eef_quat'], dtype=float)
        if p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p, q]).all():
            return None
        return np.r_[p, q].copy()
    except (KeyError, TypeError, ValueError):
        return None


class PoseLogger:
    """Single-environment proxy; retains pre/post observations and exact sent action.

    Missing observations are NaN plus validity flags, never fabricated positions.
    Returned observation/reward/done/info objects and action argument are untouched.
    """
    def __init__(self, env, destination, metadata):
        self.env = env
        self.destination = Path(destination)
        if self.destination.exists():
            raise ValueError(f'Pose output exists: {self.destination}')
        self.metadata = dict(metadata)
        self.previous = None
        self.rows = []
        self.phase = 'settling'
        self.finished = False

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        self.previous = pose(result)
        return result

    def set_init_state(self, state, *args, **kwargs):
        self.metadata['initial_state_sha256'] = hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest()
        result = self.env.set_init_state(state, *args, **kwargs)
        self.previous = pose(result)
        return result

    def step(self, action, *args, **kwargs):
        sent = np.asarray(action).copy()
        before = self.previous
        result = self.env.step(action, *args, **kwargs)
        after = pose(result[0])
        self.rows.append((before, after, sent, bool(result[2]), self.phase))
        self.previous = after
        return result

    def save(self):
        if self.destination.exists():
            raise ValueError(f'Pose output exists: {self.destination}')
        self.destination.mkdir(parents=True)
        n = len(self.rows)
        arrays = dict(
            pre_pose=np.array([r[0] if r[0] is not None else np.full(7, np.nan) for r in self.rows]).reshape(n, 7),
            post_pose=np.array([r[1] if r[1] is not None else np.full(7, np.nan) for r in self.rows]).reshape(n, 7),
            pre_valid=np.array([r[0] is not None for r in self.rows]),
            post_valid=np.array([r[1] is not None for r in self.rows]),
            executed_action=np.array([r[2] for r in self.rows]).reshape(n, 7),
            done=np.array([r[3] for r in self.rows]),phase=np.array([r[4] for r in self.rows],dtype='U16'))
        np.savez_compressed(self.destination/'poses.npz', **arrays)
        meta = dict(self.metadata, schema=1, complete=self.finished, simulator_steps=n,
                    columns=['x','y','z','qx','qy','qz','qw'],
                    observation_fields=['robot0_eef_pos','robot0_eef_quat'],
                    timing='pre_pose before exact env.step(action), post_pose from returned observation',
                    convention_status='Raw named simulator fields; bind environment source evidence before physical comparison',
                    array_sha256=hashlib.sha256((self.destination/'poses.npz').read_bytes()).hexdigest())
        (self.destination/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
