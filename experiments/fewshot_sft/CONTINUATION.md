# Goal 5-shot SFT continuation after the 200-step pilot

## Evidence and interpretation

The server logs supplied by the user report a successful 200-update run and 34 initial
nominal failures. Those are evidence of poor early rollout performance, not proof that
more steps alone will fix it. This checkout does not contain the server's actual pilot
optimizer or dataset, and this Mac has no CUDA, PyTorch or TensorFlow. The added zero-update
resume audit and GPU reload/rollout checks must pass on server4 before claiming that the
actual checkpoint resumes correctly. No long training or evaluation was launched by this change.

Source: local `2025_SimpleVLA-RL.pdf`, arXiv:2509.09674v1, §4.1 (pp. 9–10),
§5.1 (pp. 12–13), §6.2 (p. 16). The relevant pages were text-extracted and visually inspected.

- §4.1 explicitly changes the backbone interface to one view and discrete action-token
  cross-entropy; LIBERO omits proprio. It says SFT follows official datasets/hyperparameters,
  but does not specify the exact One-Trajectory SFT optimizer budget or checkpoint selection.
- §5.1 uses one complete expert demonstration per task (ten per suite), versus 500
  demonstrations per suite in Full-Trajectory SFT. Its Goal 59.6% One-Trajectory result
  belongs to that different architecture/protocol; it is not a promised target for this code.
- §6.2 studies RoboTwin2.0: zero initial task ability can leave outcome-reward RL with no
  learning signal. This motivates establishing an SFT policy and evaluator control before RL;
  it does not establish that 34 Goal failures prove a universal RL impossibility.

No RL learning rate, GRPO setting, rollout budget or RL schedule is used for SFT. The requested
1,000 / 3,000 / 6,000 cumulative update targets are an engineering diagnostic schedule, not
numbers from the paper. The continuous-action OFT model remains dual-camera + proprio + L1
MLP action head, LoRA rank 32, action chunk 8×7. The original AdamW learning rate 5e-4 and
MultiStepLR milestone 100,000 / gamma 0.1 / no warmup are retained for continuity; this is
not a claim that they are optimal for five-shot data. The launcher rejects totals above 6,000.

## Resume contract and limitations

`--resume` takes the raw training checkpoint, not an exported adapter (exports omit Adam).
LoRA is constructed using the original `get_peft_model` path before loading weights,
preserving the original optimizer parameter order. Heads and LoRA must match saved tensors
before any update; normalization must equal the original train-only statistics.
Adam moments, step counters, parameter shapes, restored state dictionaries, LR and scheduler
state are checked. New checkpoints additionally save the parameter-name order and a file
inventory. Legacy step-200 checkpoints lack parameter names: the original construction/order,
unchanged OFT source, count and shapes are verified, but no historical name list is invented.

`--verify-resume` builds and checks the real model/optimizer, writes `resume_verification.json`,
and exits with **zero** updates and runtime status `resume_verified`. It is separate from a
new one-step smoke and does not replace the existing successful `goal5-smoke-recovery` evidence.

TensorFlow iterator position, shuffle buffer, prefetch and RNG state were not saved in the
legacy checkpoint. TF and random seeds restart; continuation is not bitwise equivalent to an
uninterrupted run. New checkpoints still do not serialize the TF stream. `continuity` records
this explicitly. The worker asserts cumulative scheduler steps after every update, and saves
only at complete gradient-accumulation boundaries. The CPU regression test compares an
uninterrupted AdamW update at 201 against save/reload at 200, including a scheduler boundary;
it uses synthetic tensors and does not establish real TF/GPU equivalence.

## Diagnostics and validation

`validation.py` selects one additional complete expert per Goal task by default, seed 43,
from the original pinned RLDS. Both source episode IDs and full payload hashes must be
disjoint from the existing 50 train episodes. Duplicate held-out payloads are also rejected.
The original train manifest and normalization stay intact. The held-out dataset contains
only a `val` split and has its own selection/content audit. It supplies no statistics to training.

- `diagnostics.jsonl`: per-update normalized action-chunk L1, moving mean and EMA (window 100),
  LR, cumulative per-task sample draws, and microbatch action mean/std/min/max, per-dimension
  L1 and fraction outside [-1,1]. These are normalized training predictions, not executed actions.
- Exposure counts are sample draws, not unique frames or demonstrations. Legacy counts before
  step 200 are unknown; `exposure_known_since_step=200`. Subsequent checkpoints persist counts
  and smoothing state in `diagnostics_state.json`.
- `validation.jsonl`: a full finite traversal at the resumed starting step, every validation
  interval, requested checkpoint steps and the final step. It records frame-weighted L1,
  per-task and per-trajectory losses and frame counts. No random augmentation, shuffle,
  time-limit truncation or repeated data stream is used. Official terminal action-chunk padding
  remains included in the L1 metric. This is demonstration validation, separate from nominal
  simulator trials. It costs additional inference time beyond the optimizer steps.
- `component_verification.json` and `checkpoint_files.json`: raw-checkpoint tensor and file checks.
- `adapter_verification.json`: loaded LoRA/head/proprio file identities, exact tensor checks
  after the official inference dtype casts (FP32 trained proprio becomes BF16), and fixed-observation
  actions before export versus disk reload, `rtol=0`, `atol<=1e-6` for unmerged exports.
- `loaded_components.json`: verifies that the evaluator actually loaded those export tensors.

Loss and validation curves are included by `python -m experiments.fewshot_sft.report --write`.
Loss is a diagnostic, not a substitute for rollout success. Select a checkpoint using the
predeclared schedule, validation and bounded nominal diagnostics, and record every choice.

## Server commands (run deliberately, one stage at a time)

All paths below are relative to the repository root. The current server source path has two
`data` components because of the original download directory; no recursive NFS search is needed.

```bash
cd /nfs4/jhkim/repos/vla-rlvr
export HF_HUB_CACHE="$PWD/data/hf-cache"

# CPU regression tests, including real PyTorch AdamW continuity on this server.
bash experiments/server_env.sh original python -m experiments.fewshot_sft.cpu_tests

# Real step-200 resume audit: GPU load, zero optimizer updates; no new smoke training.
bash experiments/server_env.sh original python -m experiments.fewshot_sft.train \
  --run-id goal5-resume-audit --max-steps 1000 --save-freq 1000 \
  --resume outputs/fewshot_sft/runs/goal5-pilot--200_chkpt \
  --verify-resume --execute

# Select/materialize disjoint experts. Source revision/content comes from the train manifest.
bash experiments/server_env.sh original python -m experiments.fewshot_sft.validation \
  --source data/data/modified_libero_rlds/libero_goal_no_noops/1.0.0 \
  --train-dataset outputs/fewshot_sft/datasets/goal_5shot_seed42 \
  --output outputs/fewshot_sft/datasets/goal_validation_seed43 --execute
```

Inspect the audit runtime (`resume_verified`, actual_optimizer_steps 0), resume verification,
and validation audit before starting continued training. All new run/export/evaluation paths
refuse overwrite. Use a new run ID or label after a failed attempt, preserving its evidence.

First bound: 200 → 1,000 means **800 additional optimizer updates**.

```bash
bash experiments/server_env.sh original python -m experiments.fewshot_sft.train \
  --run-id goal5-sft-1000 --max-steps 1000 --save-freq 1000 \
  --checkpoint-steps 1000 --resume outputs/fewshot_sft/runs/goal5-pilot--200_chkpt \
  --validation outputs/fewshot_sft/datasets/goal_validation_seed43 \
  --smoke-evidence outputs/fewshot_sft/runs/goal5-smoke-recovery/runtime.json --execute
```

After reviewing loss/validation and a bounded rollout, these are the separate next bounds.
They are not launched automatically or chained to the first command.

```bash
bash experiments/server_env.sh original python -m experiments.fewshot_sft.train \
  --run-id goal5-sft-3000 --max-steps 3000 --save-freq 3000 \
  --checkpoint-steps 3000 --resume outputs/fewshot_sft/runs/goal5-sft-1000--1000_chkpt \
  --validation outputs/fewshot_sft/datasets/goal_validation_seed43 \
  --smoke-evidence outputs/fewshot_sft/runs/goal5-smoke-recovery/runtime.json --execute

bash experiments/server_env.sh original python -m experiments.fewshot_sft.train \
  --run-id goal5-sft-6000 --max-steps 6000 --save-freq 6000 \
  --checkpoint-steps 6000 --resume outputs/fewshot_sft/runs/goal5-sft-3000--3000_chkpt \
  --validation outputs/fewshot_sft/datasets/goal_validation_seed43 \
  --smoke-evidence outputs/fewshot_sft/runs/goal5-smoke-recovery/runtime.json --execute
```

A deliberately chosen single run can instead specify `--max-steps 6000 --save-freq 6000
--checkpoint-steps 1000 3000 6000` from step 200; it saves all three requested cumulative steps.
The staged commands above allow diagnosis before spending the remaining budget.

## Same-evaluator Full-SFT control and bounded checkpoint evaluation

The Full-SFT control uses `checkpoints/goal/oft` and the exact same nominal rollout function,
two cameras, proprio, action processing, seed, 300-step horizon and ten settling steps.
Only model initialization differs. Results go under `outputs/fewshot_sft/eval/full_sft_goal`,
not into historical Full-SFT outputs. Ten control episodes (one per task) are a functionality
check, not a reproduction of a published success rate. Run after the existing evaluation has
finished or after stopping only its known PID; do not launch competing jobs on the same GPU.

```bash
bash experiments/server_env.sh original python -m experiments.fewshot_sft.evaluate \
  --checkpoint checkpoints/goal/oft --full-sft --phase diagnostic --trials-per-task 1 \
  --run-label control --selection-note 'Released Full-SFT evaluator control; one trial per Goal task' --execute

bash experiments/server_env.sh original python -m experiments.fewshot_sft.checkpoint \
  --checkpoint outputs/fewshot_sft/runs/goal5-sft-1000--1000_chkpt \
  --output outputs/fewshot_sft/exports/goal5-step1000-adapter --export-unmerged --execute

bash experiments/server_env.sh original python -m experiments.fewshot_sft.evaluate \
  --checkpoint outputs/fewshot_sft/exports/goal5-step1000-adapter \
  --phase diagnostic --trials-per-task 2 \
  --selection-note 'Predeclared step-1000 diagnostic after step-200 failures; nominal model selection' --execute
```

For 3,000 and 6,000, export and diagnose their own checkpoint with fresh names the same way.
No assumption is made that a later checkpoint is better. A checkpoint with zero successes
cannot expand automatically: inspect expert losses, predictions, component proof and Full-SFT
control first. Action bounds stay identical to official pass-through; values are recorded,
not silently clipped. The old 34 failures and any running old evaluation are not modified.

Expansion requires complete, all-task, unchanged result files, a positive-success bounded
Full-SFT control and a positive-success diagnostic of the exact exported checkpoint. One
success is only a minimum functionality gate, not a statistical performance threshold.

```bash
bash experiments/server_env.sh original python -m experiments.fewshot_sft.evaluate \
  --checkpoint outputs/fewshot_sft/exports/goal5-step1000-adapter --phase pilot \
  --gate-evidence outputs/fewshot_sft/eval/goal_5shot_seed42/goal5-step1000-adapter/diagnostic/summary.json \
  --full-sft-evidence outputs/fewshot_sft/eval/full_sft_goal/oft/diagnostic-control/summary.json \
  --selection-note 'Expanded after step-1000 diagnostic and Full-SFT control; nominal used in selection' --execute
```

`--phase final` is a separate explicit command with the same evidence flags and 50 trials/task.
Diagnostic, pilot and final currently share initial-state indices; they are not independent
held-out model-selection sets. Report this when comparing success rates. No scale head or RIPT
RL is initialized or trained in these commands.

## Local validation of this change

The bundled Python (NumPy available) ran 33 tests: 32 passed, one PyTorch CPU continuity test
skipped because PyTorch is absent. Coverage includes disjoint ID/content selection, changed
normalization rejection, cumulative checkpoint/update boundaries with accumulation 1/2/4,
zero-success and stale-checkpoint expansion rejection, result tampering, existing export
integrity, and original data validation tests. Real TFDS preparation, actual step-200 optimizer
restore, GPU export equivalence and Full-SFT smoke remain server checks; no result is fabricated.

Changed files: `train.py`, `worker.py`, `resume_checks.py`, `prepare_dataset.py`, `validation.py`,
`diagnostics.py`, `checkpoint.py`, `evaluate.py`, `evaluation_checks.py`, `report.py`,
`cpu_tests.py`, `test_continuation.py`, `commands.py`, `README.md`, and this document.
