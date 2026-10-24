"""Restart the unchanged LIBERO-Plus evaluator periodically to bound leaked mesh FDs."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def completed(folder):
    # Atomic evaluator writes use .tmp until complete.
    return len(list(folder.glob('task-*.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('sft', 'scale', 'ript'), required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--tasks-per-process', type=int, default=100)
    parser.add_argument('--max-open-files', type=int, default=262144)
    parser.add_argument('--poll-seconds', type=float, default=2)
    args = parser.parse_args()
    if args.tasks_per_process < 1 or args.max_open_files < 1 or args.poll_seconds <= 0:
        parser.error('tasks-per-process, max-open-files and poll-seconds must be positive')
    folder = ROOT / 'outputs/libero-plus' / args.model / f'seed-{args.seed}'
    manifest = folder / 'manifest.json'
    if manifest.is_file():
        settings = json.loads(manifest.read_text())
        if settings['model'] != args.model or settings['seed'] != args.seed:
            parser.error('model/seed differs from existing manifest')
        total = len(settings['tasks'])
    else:
        if completed(folder):
            parser.error('task results exist without an evaluator manifest')
        total = 2591  # The evaluator creates its own manifest on first launch.
    command = [sys.executable, '-u', str(ROOT / 'experiments/libero_plus/evaluate.py'),
               '--model', args.model, '--seed', str(args.seed), '--execute']
    while completed(folder) < total:
        before = completed(folder)
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        fd_threshold = min(args.max_open_files, soft_limit // 2)
        if fd_threshold < 1:
            raise SystemExit(f'Invalid open-file soft limit: {soft_limit}')
        print(f'Starting {args.model}: {before}/{total} complete; '
              f'FD soft={soft_limit} hard={hard_limit} restart={fd_threshold}', flush=True)
        child = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
        interrupted_for_restart = False
        try:
            while child.poll() is None:
                time.sleep(args.poll_seconds)
                now = completed(folder)
                try:
                    open_files = len(os.listdir(f'/proc/{child.pid}/fd'))
                except (FileNotFoundError, ProcessLookupError):
                    open_files = 0  # Child has exited; status is checked below.
                threshold = now - before >= args.tasks_per_process or open_files >= fd_threshold
                if threshold and now < total and child.poll() is None:
                    if now == before:
                        os.killpg(child.pid, signal.SIGTERM)
                        child.wait()
                        raise SystemExit(f'FD restart threshold {fd_threshold} reached before a task completed; '
                                         'cannot make progress safely')
                    print(f'Restarting to release mesh FDs: {now}/{total} complete; '
                          f'open FDs={open_files}', flush=True)
                    # Finished JSON files survive; a task in flight will be retried.
                    os.killpg(child.pid, signal.SIGTERM)
                    interrupted_for_restart = True
                    break
            status = child.wait()
        except KeyboardInterrupt:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait()
            raise SystemExit(130)
        if interrupted_for_restart:
            continue
        if status != 0:
            raise SystemExit(f'Evaluator exited {status}; completed results retained at {folder}')
        if completed(folder) < total:
            raise SystemExit('Evaluator exited successfully before completing all tasks')
    print(f'{args.model}: {total}/{total} complete', flush=True)


if __name__ == '__main__':
    main()
