# Stage 0 readiness — Goal action compatibility and simulator validation

**Overall: PARTIAL.** Existing genuine GPU inference evidence is accepted. This continuation implements the missing validation paths without rerunning those smoke tests, starting training, or launching an evaluation. The assistant edited the existing local checkout at `/Users/jihyekim/Desktop/vla-rlvr`; `/nfs4/jhkim/repos/vla-rlvr` is not mounted here. No SSH connection or remote execution was attempted. The user runs the new checks in the existing server repository.

| Readiness area | Current conclusion | Evidence / remaining requirement |
|---|---|---|
| GPU inference | **Verified for Goal, all three configurations** | Existing `gpu-goal-sft.json`, `gpu-goal-scale.json`, `gpu-goal-ript.json`; NVIDIA RTX A5000. Peak allocated: 14.9664, 15.5817, 15.5817 GiB respectively. These are inference checks, not simulator results. |
| LIBERO evaluation | **Pending** | New action comparison and actual simulator check have not been run remotely. `action_compatibility_goal.json` and `libero_runtime_goal.json` deliberately contain `NOT_RUN`, not invented measurements. |
| RL training | **Not ready** | Previously identified constructor mismatch, sampling/log-probability scale mismatch, padding/masking and accumulation issues remain unresolved. Inference and environment stepping cannot validate RLOO/PPO updates or trainable-gradient scope. |

## User-run commands on the existing server

Copy the changed `stage0/` scripts and documentation into the existing server checkout using your normal synchronization process. Do not replace the server's pinned sources, runtime configuration, checkpoint files, or existing GPU evidence. Activate the same environment that passed the A5000 smoke tests. No new repository, installation, or `prepare.py` run is required.

```bash
cd /nfs4/jhkim/repos/vla-rlvr

# Optional CPU-only implementation tests (does not load models or LIBERO).
python stage0/scripts/test_validation.py

# Actual Original LIBERO runtime: fixed Goal task 0 / initial state 0.
python stage0/scripts/check_libero_runtime.py --timeout-seconds 180

# Always compares a fixed synthetic observation first; also reuse the real
# observation saved by the successful runtime check. Two sequential policy loads.
python stage0/scripts/check_action_compatibility.py \
  --real-observation stage0/evidence/libero_goal_observation.npz \
  --timeout-seconds 1800
```

For a synthetic-only comparison, omit `--real-observation`. These are standalone commands: the existing `run.py` launcher still supports only `smoke` and `eval`. Do not pass invented actions to it. Existing `LIBERO_CONFIG_PATH` and `MUJOCO_GL` environment settings are respected; default paths are derived from the checkout, and EGL is the default when unspecified. The simulator validator refuses configurations whose assets, BDDL, or initial-state paths point outside pinned Original LIBERO. Existing CUDA device visibility settings are respected.

The simulator limit applies to its whole worker, including imports and reset. The policy limit applies separately to each worker; total runtime is bounded by two such limits plus orchestration/cleanup. A timeout terminates the worker process group and records that graceful cleanup could not be confirmed. A nonzero exit, import error, allocation failure, or signal is recorded as an error, never a passing result. Each script writes JSON and worker logs under `stage0/evidence/`.

## Action-compatibility interpretation

`check_action_compatibility.py` uses the existing pinned Goal full model, OFT action/proprio heads, and official RIPT SFT scale head. It checks source revisions and the critical runtime source files against the pinned vendor copies. It verifies released tensor presence/sizes and the SFT scale-head SHA256. It does not re-download models or redo the previous GPU smoke suite.

The official SFT loader and official scale-only policy constructor run in separate, sequential child processes. Thus both models never occupy the GPU simultaneously, the OS releases each worker's CUDA context before the next load, and RIPT's global `LlamaSdpaAttention` patch cannot contaminate the OFT worker. Only CPU-side actions and metadata are compared afterward. The official null-adapter branch creates zero-update LoRA wrappers; the script verifies every LoRA B tensor is zero. It does not substitute randomly initialized action or scale heads.

The upstream OFT loader normally edits config/modeling files. Each worker instead creates temporary metadata copies and links the read-only-used tensor payloads. Loader writes affect only the temporary copies; source checkpoint fingerprints are checked before and after. No optimizer, parameter update, checkpoint save, or source modification occurs.

Both workers receive byte-identical image/state snapshots and the same instruction. Both use the same q01/q99 statistics, `libero_goal_no_noops` key, center crop, two images, proprioception, eight-action chunk and deterministic mean inference. Fresh observation copies prevent OFT's in-place proprio normalization from affecting the other path. Actual processor tensor hashes and normalized proprioception are compared, so a preprocessing difference is surfaced as an error rather than silently corrected.

The forward paths remain distinct:

- OFT: official `get_action` / `predict_action`; one multimodal language-model forward, no prior KV cache, and the model-default cache setting.
- RIPT scale-only: official `get_vla_action_batch`; no-grad prompt/vision pass, cached keys/values reused by the action-token pass, explicit positions and patched SDPA attention masks. Laplace sampling is disabled.

The primary metrics use unnormalized robot delta-pose coordinates plus the official binarized/sign-inverted LIBERO gripper convention. Evidence also includes raw unnormalized pre-gripper metrics, preventing gripper binarization from hiding continuous-output differences. It stores action shapes, finiteness, mean/max absolute error, per-dimension mean/max errors, every chunk/dimension absolute difference, actions, input hashes, source/package provenance, and GPU allocated/reserved peaks for each worker. A maximum across mixed pose/gripper dimensions is diagnostic, not a physical safety bound.

**`status=PASS` means the comparison ran successfully with matching inputs and finite valid outputs. It does not mean numerical equality or equivalent task performance.** By default, `compatibility=MEASURED_REVIEW_REQUIRED`; review the measured discrepancies. Optional `--max-abs-error VALUE` applies a user-chosen positive tolerance in the documented output convention, reporting `WITHIN_USER_TOLERANCE` or `OUTSIDE_USER_TOLERANCE`. There is no arbitrary default tolerance and no exact-equality gate for policy actions.

## Simulator-validation interpretation

`check_libero_runtime.py` calls official OFT `get_libero_env(task, 'openvla', resolution=256)`, preserving the Original LIBERO controller/camera defaults, fixed environment seed 0, and the selected benchmark `.pruned_init` state. It checks reset and initialized observations, performs the same ten dummy settling steps as official evaluation, then one bounded official dummy action. This action is already in the simulator's OSC_POSE/gripper convention; it is not normalized or sign-flipped again.

Front/wrist images must be 256×256×3 uint8; end-effector position, quaternion and gripper state must have their expected shapes and finite floating-point values. The actual offscreen camera observations verify rendering through the real environment. The script checks simulator time advancement, finite sparse reward, `done == check_success()`, reward scaling, and `env.close()` completion. A single `done` value is a semantics check; no success rate is calculated or claimed. It saves a pre-check-step observation using official `prepare_observation` (180-degree camera rotation, resize, axis-angle proprio conversion) for the second policy comparison.

Outputs:

- `evidence/libero_runtime_goal.json` and `.log`
- `evidence/libero_goal_observation.npz` (created only by execution)
- `evidence/action_compatibility_goal.json`
- `evidence/action_compatibility_goal_sft.log`, `..._scale.log`

## Verification performed in this continuation

Five CPU-only tests passed: known-error metric calculation, nonfinite/shape rejection, preservation of original metadata during temporary checkpoint staging, worker crash reporting, and timeout termination. Both command-line help paths and Python parsing passed. No remote, GPU, or actual simulator validation was executed. Existing pinned upstream sources, reward/sampling/PPO code, and GPU evidence were left unchanged.

## Can Stage 1 begin?

**Not yet on the evidence currently available.** Goal GPU inference is ready; actual LIBERO evaluation readiness is still pending the user-run runtime result and action-comparison review. If the runtime check passes (including rendering, stepping, and cleanup), both comparison workers pass with matching preprocessing/normalization, and the measured discrepancies are reviewed, Stage 1 evaluation of the released Goal SFT and RIPT policies can begin with the existing evaluation commands. Numerical differences between OFT's full forward and RIPT's cached forward must be recorded; they do not by themselves establish failure or equivalent performance.

This conclusion applies to Goal. Other suites still need their own checkpoint/runtime evidence. RL training remains a separate blocked readiness category even if both new checks succeed.
