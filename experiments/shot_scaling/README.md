# Unified LIBERO shot scaling: CPU foundation and integration audit

This is a **partial implementation stopped at unvalidated integration gates**, not a working
end-to-end training pipeline. It adds an isolated orchestration package and preserves the
existing trainers, evaluators, historical sampler, checkpoints, and experiment results.
No training, model inference, benchmark rollout, data download, or RLDS extraction was run.
The local checkout is `/Users/jihyekim/Desktop/vla-rlvr`; the requested NFS checkout is not
mounted here. Paths are resolved from the repository, so the package can be copied to NFS.

## What works

- Hydra Compose/OmegaConf configuration, resolved before dispatch, with no cwd changes or
  persistent Hydra initialization. Unknown overrides are rejected by Hydra. Fixed model
  invariants, seed types, suite mappings, output location and positive bounds are validated.
- Nested episode selection from an **externally supplied complete-source inventory** for
  Goal/Spatial/Long × 1/2/5/10/20/Full. A SHA-256 order per task/seed is independent of K and
  input order. Full selects the actual listed count. Duplicate IDs, duplicate payloads,
  missing tasks, insufficient shots, wrong suite and changed checksums fail.
- CPU analysis of externally supplied raw action/proprio artifacts, with JSON and CSV output.
  Statistics separate transition weighting from equal episode weighting. Raw and z-scored
  actions, azimuth distributions (zero vectors excluded), magnitudes, temporal changes,
  gripper transitions, initial proprio statistics, within-task descriptor distances, and
  selected-to-complement descriptor coverage are saved. Z-scoring is descriptive and uses
  the complete source; it is **not** OFT BOUNDS_Q99 or a model-validation transform.
- Exclusive-create manifests with unique timestamp/UUID IDs, full resolved config, input
  hashes, Git revision/status, implementation hashes and runtime package versions. Missing
  GPU information is explicitly null. Completed CPU operations have a separate completion
  marker; a directory or manifest alone is not evidence of success.
- `prepare --resume` verifies an existing selection exactly, without rewriting it. This is
  manifest resumption only, not TFDS, training, or evaluation resumption.
- Episode-bundle aggregation rejects duplicate runs/episodes, unexpected protocol IDs,
  changed protocols and invalid outcomes. Incomplete runs are excluded from seed means.
  Evaluation-seed means are grouped within training seed; SEM is across training seeds.
  The Python `paired()` API requires exact cached D0/D1 lineage and matched protocols.
- Historical nominal/PLUS aggregation calls the **existing** read-only suite report builder
  in a subprocess (its legacy imports assume a script directory). Existing per-suite
  manifests/coverage validation stay in that builder. Results are explicitly labeled
  released-checkpoint comparisons, never new same-initialization D0/D1 results.
- Standalone artifact-only PNG/SVG plotting and Markdown index. Dataset plots are implemented.
  A documented generic metric-series input supports training, generalization and representation
  plots; automatic adapters from existing training/P3 artifacts remain planned. Missing points
  are labeled; plots do not connect/interpolate them. No model/dataset code is imported by plotting.

## Exact blockers and audit

| Component | Existing implementation / finding | Gate |
|---|---|---|
| Dataset | `fewshot_sft/prepare_dataset.py` validates TFDS payloads, cameras, flags, IDs and materialization; Goal only | New complete-source extractor and all-suite materializer need real TFDS validation |
| Historical sampler | `random.Random(seed).sample(candidates, K)` consumes RNG across tasks; only K=1/5 | Keep Goal5 seed42 verbatim; new SHA-order manifests are a different cohort |
| SFT | `fewshot_sft/train.py` and `worker.py` reuse official OFT forward/L1/AdamW/loader; source-guarded instrumentation | Goal 1/5 hard-coding in worker, diagnostics, checkpoint normalization and evaluator prevents a config-only extension |
| Initialization | Existing launcher resolves offline `openvla/openvla-7b`, not released LIBERO weights | No local pretrained snapshot/weight shards or real data for new conditions |
| Export | `checkpoint.py` supports verified safe merge or verified unmerged SFT adapter export | Merge BF16 tolerance 0.02 is not cached-RIPT equivalence; unmerged export is not a fresh RL base |
| Scale | RIPT README requests downloaded scale headers; interface loads `scale_trunk`/`log_b_head` | No NLL training entrypoint found; no exact-new-SFT scale checkpoint |
| RL adapters | `openvla_oft_interface.py:303–349` initializes base/heads then calls `get_peft_model`, or loads RL adapter + `openvla_headers.pt` | SFT LoRA must first be verified merged into the base (or explicitly preserved by a validated adapter design); never silently replace it |
| PPO probability | Runner samples un-clipped `exp(log_b)*scale_factor`; both stored old log-prob and optimizer recomputation use clipped log_b without factor | **Distribution mismatch blocks RL**, even though old/new log-prob routines agree with each other |
| Nominal/PLUS | Existing Goal evaluator, `suite_common.py`, and fewshot Goal export binding | Arbitrary same-SFT cached D0/D1 binding and resume validation not integrated |
| Runtime | Local test Python has no torch, TensorFlow or TFDS; macOS has no CUDA | Real GPU and RLDS smoke cannot be performed here |

Probability source locations: `runtime/ript-vla/ript/env_runner/openvla_oft_libero_runner.py`
lines 103–123 and 345–354; `ript/algos/rl_optimizers/rl_optimizer_openvla_oft.py` line 168.
Suite configs set `scale_factor: 5.0`, `log_scale_clip: [-2.0, 0.5]`. For mean=0,
log_b=0 and action=1, the sampling distribution log density is `-log(10)-1/5`, whereas
the reported log density is `-log(2)-1`. No PPO algorithm was changed. Resolving this needs
a separately reviewed distribution convention and a bounded numerical rollout/PPO test.

Declared source revisions are in `stage0/repos.lock.json`; runtime copies lack Git metadata.
The new `inspect` report hashes audited local source files rather than treating declared
upstream SHAs as verified local revisions. The pinned dataset revision is
`a7c9ae18499b6eea8a32f78a9302327b752b1b5f`. Supplied file hashes establish artifact identity;
they do not independently authenticate remote bytes or establish that an inventory really
covers all RLDS episodes. **No new selection is trainable until real extraction and
materialization verification pass.**

Existing PLUS protocols differ: Goal 2,591, Spatial 2,402, Long 2,519 conditions in the audited
evaluators. The new bundle aggregator uses explicit expected episode keys; it has no universal
condition count. Bounded completeness means complete for that bounded protocol, not the full suite.
Full-sequence SFT versus cached inference is never an isolated RL effect.

## Setup and CLI

NumPy 1.26.4 and Matplotlib 3.9.0 match the pinned RIPT requirements. If an earlier
version of this package installed Matplotlib 3.9.4, restore the shared environment:

```bash
python -m pip install "numpy==1.26.4" "matplotlib==3.9.0"
python -m pip check
```

```bash
python -m pip install -r experiments/shot_scaling/requirements.txt
python -m experiments.shot_scaling inspect
python -m experiments.shot_scaling inspect --commands
python -m unittest experiments.shot_scaling.tests.test_pipeline -v
python -m experiments.shot_scaling.smoke
```

All new output goes under `outputs/shot_scaling/<unique-id>/`. The CLI never schedules jobs.
`sft-train`, `sft-eval`, `scale-train`, `ript-train`, and `eval` always fail with exit 2 and
the exact blocker, with or without `--check`. There is deliberately no `--execute` shortcut.
They are not dummy successful stages. Other commands require their real inputs and reject
missing evidence. A missing completion marker denotes a failed/interrupted CPU operation.

Example Hydra configuration overrides:

```bash
python -m experiments.shot_scaling inspect suite=long shots=Full selection_seed=43 train_seed=8 eval_seed=2
python -m experiments.shot_scaling sft-train --check suite=goal shots=5 sft.max_steps=200 sft.save_freq=50 execution.gpu=0
python -m experiments.shot_scaling eval --check suite=spatial evaluation.dataset=plus evaluation.inference=d0_cached evaluation.max_tasks=2 evaluation.max_trials=1
```

Configs are `conf/config.yaml` plus `conf/suite/{goal,spatial,long}.yaml`. World size other
than 1 is rejected, not silently ignored. Scale/RL optimizer and dependency fields are
declarative contracts only until those stages are implemented and validated.

### Requested commands

The following variables name real input artifacts conforming to the contracts below.
No such real source inventory/trajectory artifacts or historical Goal5 selection manifest
were found in this local checkout. These commands therefore require server evidence.

```bash
# 1. Historical Goal 5-shot analysis: validates the existing selection, never reselects it.
python -m experiments.shot_scaling dataset-analyze --inventory "$GOAL_INVENTORY" --trajectories "$GOAL_TRAJECTORIES" --selection "$HISTORICAL_GOAL5_MANIFEST" suite=goal shots=5 selection_seed=42

# 2. New nested Goal manifests (MANIFEST_ONLY, not training datasets).
for shot in 1 2 5 10 20 Full; do
  python -m experiments.shot_scaling prepare --inventory "$GOAL_INVENTORY" suite=goal shots=$shot selection_seed=42
done

# 3. New bounded SFT/evaluation readiness (exit 2).
python -m experiments.shot_scaling sft-train --check suite=goal shots=5 sft.max_steps=1 sft.save_freq=1
python -m experiments.shot_scaling sft-eval --check suite=goal evaluation.max_tasks=1 evaluation.max_trials=1
# Existing historical pipeline's actual bounded dry-run flags, not the new nested cohort:
python -m experiments.fewshot_sft.train --suite goal --shots 5 --selection-seed 42 --train-seed 7 --batch-size 1 --max-steps 1 --save-freq 1 --run-id goal5-historical-smoke --smoke --dry-run
python -m experiments.fewshot_sft.evaluate --checkpoint "$VERIFIED_GOAL_SFT_EXPORT" --phase smoke --seed 1 --selection-note 'Bounded export smoke; historical Goal selection' --dry-run

# 4. Spatial/Long CPU action analysis.
python -m experiments.shot_scaling dataset-analyze --inventory "$SPATIAL_INVENTORY" --trajectories "$SPATIAL_TRAJECTORIES" suite=spatial
python -m experiments.shot_scaling dataset-analyze --inventory "$LONG_INVENTORY" --trajectories "$LONG_TRAJECTORIES" suite=long

# 5–6. Independent scale/RIPT readiness checks (exit 2).
python -m experiments.shot_scaling scale-train --check
python -m experiments.shot_scaling ript-train --check

# 7. Existing nominal/OOD report; new results require complete identity-bearing episode bundles.
python -m experiments.shot_scaling aggregate --historical
python -m experiments.shot_scaling aggregate --input "$EVALUATION_BUNDLES"

# 8. Dataset or metric-series figures, without extracting data or running inference.
python -m experiments.shot_scaling plot --input "$SAVED_ANALYSIS"
python -m experiments.shot_scaling.plotting --input "$SAVED_ANALYSIS" --output "$NEW_FIGURE_DIRECTORY"
```

## Artifact contracts

`tests/test_pipeline.py:fixture` is an executable **synthetic** example, not a data extractor.
`smoke.py` saves its complete inputs for inspection. JSON is supported; Parquet is not yet
implemented. Every inventory includes schema=1, suite, dataset_name, complete_source=true,
source_episode_count, source `{repo, revision, split, files}`, and episodes. `files` maps
relative source filenames to SHA-256 strings. Each episode contains exact episode_id/task,
positive steps, complete=true and SHA-256 of its full decoded source payload. `episodes_sha256`
is `core.digest(episodes)`. Duplicate payloads are rejected to prevent hidden held-out leakage.

Trajectory artifacts include inventory_sha256 and one row per source episode, preserving
episode_id/task/source_payload_sha256, raw actions shaped T×7 and initial_proprio shaped 8.
`trajectory_sha256 = core.digest([actions, initial_proprio])`. Artifact validation is not
proof of correct upstream extraction: this missing integration is an explicit gate.

Historical Goal selection manifests use the existing `fewshot_sft.validate_manifest` and
must match source file hashes and exact source episode records. They never pass through
the nested selector. Selection algorithm identity is part of every new manifest.

`tests/test_pipeline.py:bundle` documents the evaluation bundle schema. A JSON list contains
`manifest` and `episodes` per run. Expected keys are `[task_id, condition_id, trial_id]`,
bound into `protocol_sha256`; seed/checkpoint/inference identity stays in the manifest.
The `condition` object records common training settings/base/selection identity (excluding
independent train/evaluation seeds); `condition_sha256` prevents pooling different conditions.
`paired()` additionally checks checkpoint lineage; it is not invoked automatically by `aggregate`.
These are **future evaluator adapter contracts**; no new inference path exists here.

Plotting training/representation/generalization data uses this saved long-form schema:

```json
{"kind":"metric_series","synthetic":true,"series":[
  {"metric":"cka","label":"Synthetic example","x_label":"Layer","points":[
    {"x":0,"y":0.7,"status":"COMPLETE","sem":0.03,"independent_seeds":3},
    {"x":1,"y":null,"status":"INCOMPLETE"}
  ]}
]}
```

Allowed metrics: sft_loss, nominal_success, ript_success, mixed_success_group_rate,
shot_success, shift_delta_pp, cka, effective_rank, action_difference. Use separate series
for suite, inference mode and shift. Artifact producers must supply the correct x axis
(optimizer updates, observed environment interactions, shots, or layer) and independent
seed statistics. Missing exposure, interactions, RL checkpoints and model metrics remain
missing; configuration maxima are never reported as actual measured work.

## Validation and experiment matrix

15 new CPU tests cover nested subsets for all suites/seeds, order invariance, duplicate
payload leakage, suite rejection, historical sampler preservation, gripper conventions,
weighted statistics, full/complement coverage, Hydra lifecycle, immutable writes/resume,
checkpoint content changes, D0/D1 lineage, seed aggregation, bounded completeness, plotting,
and nonzero blocked-stage exits. The bounded smoke invokes actual supported CLI stages on
synthetic inputs plus the existing historical report. See `VALIDATION.md` for run evidence.

| Conditions / work | Status | Evidence or next gate |
|---|---|---|
| CPU contracts and synthetic artifact smoke | COMPLETE | Tests and smoke evidence; not experimental runs |
| Goal/Spatial/Long × 1/2/5/10/20/Full selection from supplied inventory | READY | Real full-source inventories required |
| Real all-suite TFDS extraction/materialization | BLOCKED | Local data/runtime missing; integration not implemented |
| All 18 new SFT conditions | BLOCKED | All-suite loader/export extension and original-base GPU smoke |
| Exact-SFT scale fitting and D0 snapshot | BLOCKED | NLL trainer, cached-action checks, complete checkpoint lineage |
| All 18 RIPT conditions | BLOCKED | Rollout/PPO distribution consistency plus preceding gates |
| New nominal/PLUS D0/D1 evaluation | BLOCKED | Shared-evaluator checkpoint binding and resume validation |
| Training/P3 metric adapters, Parquet, automatic matrix from new GPU runs | PLANNED | Implement after real-stage gates pass |
| Historical Goal/Long/Spatial, P3-A2/P3-B | PRESERVED | Existing builders/evidence; released checkpoint interpretation |
| Automatically started training/evaluation jobs | NONE RUNNING | No jobs launched |

New source files are confined to `experiments/shot_scaling/`: core/config, data contracts,
analysis, reporting, plotting, CLI, smoke, tests, requirements, and this documentation.
The regression test wrapper also refreshed `outputs/fewshot_sft/cpu_tests.json` (test metadata
only); no experimental episodes, selections, checkpoints or historical result files changed.
Use unittest directly for subsequent regressions to avoid refreshing that legacy test report.
