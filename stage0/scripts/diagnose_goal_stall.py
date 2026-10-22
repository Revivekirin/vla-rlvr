"""One bounded SFT reproduction: first Goal task, initial states 0..19.

No upstream file edits. Policy and environment settings remain official. Reset
exception handling is replaced IN MEMORY to expose exceptions suppressed by the
upstream finally/continue. Diagnostic aborts never become policy failures.
"""
import argparse
import faulthandler
import functools
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[2]


class DiagnosticAbort(BaseException):
    """Bypass upstream run_episode's broad except Exception handler."""


class DiagnosticComplete(BaseException):
    pass


def visible_reset(inner_reset, randomization_error, emit, max_attempts):
    for attempt in range(1, max_attempts + 1):
        emit('reset_attempt', attempt=attempt)
        try:
            return inner_reset()
        except randomization_error:
            emit('reset_randomization_error', attempt=attempt, traceback=traceback.format_exc())
            if attempt == max_attempts:
                raise DiagnosticAbort('Reset retry limit reached; this is not a policy failure')
        except Exception as exc:
            emit('reset_unexpected_error', attempt=attempt, traceback=traceback.format_exc())
            raise DiagnosticAbort('Underlying reset raised an exception') from exc
    raise AssertionError('Positive reset retry limit required')


def run_worker(args):
    local = args.local_dir
    faulthandler.enable(all_threads=True)
    # This C-level watchdog also reports stacks when ordinary Python signal
    # handling cannot proceed. exit=True exits the process after the stack dump.
    faulthandler.dump_traceback_later(300, exit=True)
    events = (local / 'events.jsonl').open('a', buffering=1)
    state = {'episode': 0, 'step': 0, 'inference': 0}
    envs = []

    def emit(event, **fields):
        row = {'time': time.time(), 'monotonic': time.monotonic(), **state, 'event': event, **fields}
        events.write(json.dumps(row) + '\n')
        print('[DIAG] ' + json.dumps(row), flush=True)

    def traced(name, fn, deadline=None):
        @functools.wraps(fn)
        def call(*values, **kwargs):
            faulthandler.dump_traceback_later(deadline or args.phase_timeout, exit=True)
            start = time.monotonic()
            emit('begin', phase=name)
            try:
                value = fn(*values, **kwargs)
            except BaseException:
                emit('error', phase=name, traceback=traceback.format_exc())
                raise DiagnosticAbort(f'Failed in {name}')
            else:
                emit('end', phase=name, seconds=time.monotonic() - start)
                return value
            finally:
                faulthandler.cancel_dump_traceback_later()
        return call

    result = {'status': 'RUNNING', 'scope': 'Diagnostic reproduction, not a completed suite score',
              'python': sys.executable, 'hostname': os.uname().nodename,
              'reset_handling': 'in-memory removal of exception suppression; bounded RandomizationError retries',
              'planned_episodes': args.episodes, 'initial_state_indices': [0, args.episodes - 1],
              'checkpoint': str(ROOT / 'checkpoints/goal/oft')}
    try:
        emit('begin', phase='imports')
        for path in reversed([ROOT / 'runtime/openvla-oft', ROOT / 'runtime/ript-vla', ROOT / 'vendor/LIBERO']):
            sys.path.insert(0, str(path))
        os.chdir(ROOT / 'runtime/openvla-oft')
        os.environ.setdefault('LIBERO_CONFIG_PATH', str(ROOT / 'runtime/libero-config'))
        os.environ.setdefault('MUJOCO_GL', 'egl')
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled',
                          TOKENIZERS_PARALLELISM='false')
        if not (Path(os.environ['LIBERO_CONFIG_PATH']) / 'config.yaml').is_file():
            raise FileNotFoundError('Existing LIBERO config missing; do not initialize a new environment')
        import experiments.robot.libero.run_libero_eval as evaluation
        from robosuite.utils.errors import RandomizationError
        emit('end', phase='imports')
        faulthandler.cancel_dump_traceback_later()

        original_factory = evaluation.get_libero_env

        def factory(*values, **kwargs):
            env, description = traced('env_create', original_factory)(*values, **kwargs)
            envs.append(env)
            # Preserve all actual simulator reset operations and randomization;
            # change only suppressed-error handling, without editing vendor files.
            env.reset = traced('reset', lambda: visible_reset(env.env.reset, RandomizationError, emit, args.reset_attempts))
            env.set_init_state = traced('set_init_state', env.set_init_state)
            original_step = env.step

            def step(*step_args, **step_kwargs):
                state['step'] += 1
                return traced('env_step', original_step)(*step_args, **step_kwargs)

            env.step = step
            return env, description

        evaluation.get_libero_env = factory
        evaluation.initialize_model = traced('model_load', evaluation.initialize_model, 300)
        evaluation.load_initial_states = traced('load_initial_states', evaluation.load_initial_states)
        evaluation.prepare_observation = traced('prepare_observation', evaluation.prepare_observation)
        evaluation.save_rollout_video = traced('video_save', evaluation.save_rollout_video)
        original_action, original_episode, original_task = evaluation.get_action, evaluation.run_episode, evaluation.run_task

        def action(*values, **kwargs):
            state['inference'] += 1
            return traced('policy_inference', original_action)(*values, **kwargs)

        def episode(*values, **kwargs):
            state.update(episode=state['episode'] + 1, step=0, inference=0)
            emit('episode_begin', initial_state_index=state['episode'] - 1)
            value = original_episode(*values, **kwargs)
            emit('episode_end', success=bool(value[0]))
            return value

        def task(*values, **kwargs):
            task_id = values[2] if len(values) > 2 else kwargs['task_id']
            if task_id != 0:
                raise DiagnosticAbort('Only first Goal task is permitted')
            counts = original_task(*values, **kwargs)
            emit('diagnostic_complete', completed_episodes=int(counts[0]), successes=int(counts[1]))
            result.update(status='COMPLETED_DIAGNOSTIC', completed_episodes=int(counts[0]), successes=int(counts[1]))
            raise DiagnosticComplete()

        evaluation.get_action = action
        evaluation.run_episode = episode
        evaluation.run_task = task
        sys.argv = [str(Path(evaluation.__file__)), '--pretrained_checkpoint', str(ROOT / 'checkpoints/goal/oft'),
                    '--task_suite_name', 'libero_goal', '--num_trials_per_task', str(args.episodes),
                    '--seed', '1', '--center_crop', 'True', '--use_wandb', 'False', '--local_log_dir', str(local)]
        emit('evaluation_start', argv=sys.argv, note='Original SFT precision defaults retained; no new TF32 override')
        evaluation.eval_libero()
    except DiagnosticComplete:
        pass
    except BaseException:
        result.update(status='ABORTED_DIAGNOSTIC', traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        faulthandler.cancel_dump_traceback_later()
        for env in envs:
            try:
                traced('env_close', env.close)()
            except BaseException:
                result.update(status='ABORTED_DIAGNOSTIC', cleanup_error=traceback.format_exc())
        (local / 'worker_result.json').write_text(json.dumps(result, indent=2) + '\n')
        events.close()
    return 0 if result['status'] == 'COMPLETED_DIAGNOSTIC' else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--phase-timeout', type=int, default=120)
    p.add_argument('--total-timeout', type=int, default=1800)
    p.add_argument('--reset-attempts', type=int, default=3)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--local-dir', type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if not 1 <= args.episodes <= 50 or min(args.phase_timeout, args.total_timeout, args.reset_attempts) <= 0:
        p.error('episodes must be 1..50 and all limits positive')
    if args.worker:
        return run_worker(args)
    local = Path(tempfile.mkdtemp(prefix='vla-goal-diagnostic-'))
    tag = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    archive = ROOT / 'stage0/evidence' / ('goal_stall_' + tag)
    print(f'Local diagnostic logs: {local}\nArchive after exit: {archive}', flush=True)
    cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', '--local-dir', str(local),
           '--episodes', str(args.episodes), '--phase-timeout', str(args.phase_timeout),
           '--reset-attempts', str(args.reset_attempts)]
    start = time.monotonic()
    with (local / 'console.log').open('w', buffering=1) as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                bufsize=1, start_new_session=True)

        def relay():
            for line in proc.stdout:
                log.write(line)
                print(line, end='', flush=True)

        reader = threading.Thread(target=relay, daemon=True)
        reader.start()
        reason = None
        try:
            code = proc.wait(timeout=args.total_timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            reason = 'TOTAL_TIMEOUT' if isinstance(exc, subprocess.TimeoutExpired) else 'USER_INTERRUPTED'
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            code = 124 if reason == 'TOTAL_TIMEOUT' else 130
        if code != 0:
            # The watchdog may exit while ffmpeg is active. Its descendants
            # belong to this private process group; do not leave them running.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        reader.join(timeout=5)
    events = local / 'events.jsonl'
    last = None
    if events.exists():
        for line in events.read_text().splitlines():
            try:
                last = json.loads(line)
            except ValueError:
                pass
    summary = {'exit_code': code, 'termination_reason': reason, 'last_event': last,
               'elapsed_seconds': time.monotonic() - start, 'command': cmd,
               'local_logs': str(local), 'archive': str(archive),
               'note': 'Nonzero exit/timeout is a diagnostic interruption, not a policy failure. See console.log for automatic stack dump.'}
    (local / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (local / 'exit_code').write_text(str(code) + '\n')
    try:
        shutil.copytree(local, archive)
    except OSError as exc:
        print(f'Archive failed: {exc}; local logs retained at {local}', flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return code if code >= 0 else 128 - code


if __name__ == '__main__':
    raise SystemExit(main())
