# P3-B: bounded representation diagnostics

This module is independent of the existing evaluators. It neither trains nor changes checkpoints, historical evaluation JSON, A1/A2 results, or candidate CSVs. A report can be generated without GPU results; absent evidence is explicitly marked NOT AVAILABLE.

## Audited inference and environment paths

`evaluate.py`, `p3_action_analysis.py`, the RIPT `openvla_oft_interface.py` loader and `openvla_oft_libero_runner.py` cached helper are the implementation references. Scale and RIPT use the same OFT base, proprio projector and normalization key. RIPT additionally loads its released LoRA and header weights; these policies are not assumed to have identical action heads. Both use the original prefix/suffix cached inference with Laplace sampling disabled. Full-sequence OFT inference is not substituted.

Each source/dataset capture and each model extraction runs in a separate process to isolate attention monkey-patching. Images follow the evaluator's 256-to-224 preparation and center crop. Actions retain the original eight chronological seven-component slots, with translation 0:3, rotation 3:6 and gripper 6. Controller source checks provide ordering evidence. Normalized means, unnormalized chunks and executed chunks are saved separately and checked.

Capture resolves physical BDDL and initial-state files, verifies task mappings and pinned repository revisions, seeds the environment with 0, sets initial-state index 0 and executes ten dummy steps. Balanced capture has a 300-step horizon. This is a diagnostic subset using singleton inference and reseeding per rollout, not a reproduction of all 50 nominal trials. Historical labels are selection metadata, not assertions that the new rollout reproduced the historical outcome.

Original LIBERO may carry the known `env_wrapper.py` reset fix that removes `finally: continue` and exits after a successful reset. Preflight accepts only those exact bytes relative to the pinned commit and records the working file hash in the manifest. Any other tracked Original LIBERO change fails. LIBERO-Plus must still have clean tracked source. An untracked `env_wrapper.py.bak` is untouched and is not imported.

Feature hooks collect the projected visual tokens (before proprio), decoder layers 0, 15 and 31 by default, and the action-head input after final normalization. Layer existence is checked at runtime. Decoder hooks require the cached prefix/suffix call pattern; pooling uses the first 56 suffix action tokens and excludes the two trailing tokens. Image and action tokens are mean-pooled to one vector per observation. Actual shapes, dtypes, feature definitions and implementation hashes are recorded. No feature is treated as an independent sample merely because it occupies another action slot.

## Bank and analysis

Smoke selects one nominal and one Camera condition, captures both source policies for at most 16 policy steps each, and retains one observation per rollout: four observations total. It tests plumbing, not CKA inference within two-frame groups.

Balanced selection has 10 nominal tasks and 8 conditions each for Camera, Sensor Noise and Lighting (two of each historical outcome transition where available). Both source policies contribute at most ten observations each: at most 680 observations over 68 rollouts. Sampling uses early/middle/late query-position thirds with quotas 4/3/3. Early termination and missing strata are recorded, never filled with duplicated observations. Selection diversifies base tasks; available A2 mixed outcomes help prioritize ambiguous conditions. Missing A2 evidence is reported explicitly.

Analysis includes centered linear CKA, effective rank from entropy of normalized singular values, feature norms, and normalized/raw/executed action differences. Results are grouped by stratum, source and base task. Bootstrap resamples base-task clusters, requires at least four distinct clusters and does not pretend frames are independent trials. Matched-N subsets, one-frame-per-rollout subsets, sample-size sensitivity and uniform-duplication checks are included. CKA is unavailable below three observations or for constant features.

These are exploratory geometric diagnostics. Action differences, CKA and effective rank do not establish representation information loss or a causal explanation of OOD failure. Outcome-selected conditions do not estimate the population OOD effect.

## Exact server commands

Run from `/nfs4/jhkim/repos/vla-rlvr`. Choose unused run IDs initially. The balanced commands below are manual follow-up commands; no full bank is started by smoke.

1. CPU-only tests (no model loading):

```bash
bash experiments/server_env.sh plus python -m unittest discover -s experiments/libero_plus -p 'test_*.py'
```

2. CPU preflight:

```bash
bash experiments/server_env.sh plus python experiments/libero_plus/p3_representation_analysis.py dry-run --run-id p3b-smoke-01 --mode smoke --seed 1
```

3. Minimal GPU smoke (capture, isolated replay and analysis):

```bash
bash experiments/server_env.sh plus python -u experiments/libero_plus/p3_representation_analysis.py capture --run-id p3b-smoke-01 --mode smoke --seed 1
bash experiments/server_env.sh plus python -u experiments/libero_plus/p3_representation_analysis.py extract --run-id p3b-smoke-01
bash experiments/server_env.sh plus python experiments/libero_plus/p3_representation_analysis.py analyze --run-id p3b-smoke-01
```

4. Validate smoke artifacts:

```bash
bash experiments/server_env.sh plus python experiments/libero_plus/p3_representation_analysis.py validate --run-id p3b-smoke-01
```

5. Optional bounded balanced capture, only after smoke validation:

```bash
bash experiments/server_env.sh plus python experiments/libero_plus/p3_representation_analysis.py dry-run --run-id p3b-balanced-01 --mode balanced --seed 1
bash experiments/server_env.sh plus python -u experiments/libero_plus/p3_representation_analysis.py capture --run-id p3b-balanced-01 --mode balanced --seed 1
```

6. Extract both policies on the frozen bank without environment interaction:

```bash
bash experiments/server_env.sh plus python -u experiments/libero_plus/p3_representation_analysis.py extract --run-id p3b-balanced-01
```

7. Analyze and validate:

```bash
bash experiments/server_env.sh plus python experiments/libero_plus/p3_representation_analysis.py analyze --run-id p3b-balanced-01 --bootstrap 200
bash experiments/server_env.sh plus python experiments/libero_plus/p3_representation_analysis.py validate --run-id p3b-balanced-01
```

8. Generate or explicitly refresh the unified report from disk:

```bash
bash experiments/server_env.sh plus python experiments/libero_plus/research_report.py --all --refresh
```

## Outputs, failure and resumption

All experiment artifacts live under `outputs/rlvr_analysis/p3b/<run-id>/`: `manifest.json`, frozen `bank.json`, per-rollout `bank/` observations/actions and metadata, per-policy `features/` arrays and metadata, worker logs, and `analysis.json`. Temporary worker configuration/cache directories also remain within the run. Report outputs are `outputs/rlvr_analysis/research_summary.md` and `.json`.

Capture and extraction can be rerun with the same run ID after a failure: completed bundles are validated and skipped; incomplete bundles are not accepted as completed. Capture resume must retain mode, seed and layer arguments. Manifest/source/checkpoint changes fail validation. A directory without a valid manifest needs a new run ID. Completed analysis is never overwritten. `--refresh` only replaces derived unified reports; it cannot overwrite bank or feature evidence.

Dry-run readiness means CPU prerequisites only. Missing modules (including Wand), dataset assets, task mappings, checkpoints or normalization evidence must be resolved before GPU loading. A passed CUDA/EGL check alone is insufficient. Local CPU tests use synthetic arrays/hooks; they cannot establish cached-model numerical correctness, simulator reproducibility, GPU memory suitability or real representation results. Those require the server smoke.

The unified report uses historical JSON when available, otherwise explicitly identifies the checked-in paired CSV as derived evidence. Missing A2 CSV and missing validated P3-B runs remain NOT AVAILABLE. No pasted outcome numbers are inserted as measured evidence.
