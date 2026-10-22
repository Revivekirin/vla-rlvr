# Goal Episode 20 diagnostic

The repeated stall is not yet attributed to reset, CUDA, rendering, or video I/O.
The pinned LIBERO reset has `finally: continue`, which suppresses unexpected
exceptions including KeyboardInterrupt. This is verified as a source defect,
not established as the cause of this particular stall.

Copy `scripts/diagnose_goal_stall.py` into the existing server repository. It is
self-contained (standard-library orchestration plus the existing installed
evaluation dependencies). No new clone, dependency installation, or smoke test
is required. The assistant has not executed anything on the remote server.

Run on server4, once no previous evaluation is running:

```bash
cd /nfs4/jhkim/repos/vla-rlvr
/home/jhkim/miniconda3/envs/ript_vla_openvla_oft_stage0/bin/python -u \
  stage0/scripts/diagnose_goal_stall.py
```

The explicit interpreter prevents the previous tdmpc2 environment mismatch.
The script tees stdout/stderr itself; no shell quoting, `tee`, or conda activation
is needed. It runs only the first Goal task with the official initial states
0 through 19, seed 1, 10 settling steps and normal policy/controller/camera/video
paths. It does not skip straight to state 19, since previous resets or episodes
may matter. No model precision or policy parameter changes are introduced.

Each reset, initial-state application, policy query, observation preparation,
environment step, and video save has a begin/end record. A begin without end
identifies the active call. Reset attempts and suppressed underlying exceptions
are exposed through a replacement wrapper in memory; RandomizationError has
three retries, while unexpected exceptions abort immediately. Diagnostic aborts
inherit BaseException so the official episode exception handler cannot count
them as ordinary policy failures. Upstream files are not patched. The original
OFT loader still performs its usual local config/modeling-file synchronization.

Calls have a 120-second watchdog; imports and model loading have 300 seconds.
The C-level faulthandler watchdog dumps all Python thread stacks and exits on
expiry; native library internals may require further investigation if a dump
only identifies a Python-to-C call. The parent enforces a separate 1800-second
total limit and kills the private process group on failure. Timeouts are
diagnostic interruptions, not benchmark failures. This is a diagnostic prefix
run, not a completed official suite evaluation.

Logs are written to a printed `/tmp/vla-goal-diagnostic-*` directory first, so
diagnostic writes do not depend on NFS. The official replay videos still use
their usual location and are traced to diagnose that path. On exit, logs are
copied to a unique `stage0/evidence/goal_stall_TIMESTAMP_ID/` directory; the local
copy remains available even if archiving fails. Files include `console.log`,
`events.jsonl`, `summary.json`, `exit_code`, and, when graceful exit is possible,
`worker_result.json`. A watchdog hard exit may omit `worker_result.json`; the
parent summary is still written. No prior evaluation logs are overwritten.

Return the last `[DIAG]` records and the timeout/exception traceback from
`console.log`, plus `summary.json`. If all 20 episodes finish, report that the
diagnostic run did not reproduce the stall; do not conclude the 500-episode
evaluation or original implementation is fixed.

Local checks: five unit tests cover reset success, retry order, bounded retries,
unexpected exception propagation, and KeyboardInterrupt propagation. No remote
model inference or simulation has been run by the assistant.
