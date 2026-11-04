# Unified research report and physical-path readiness

The current review is **partial**, not a completed EEF recovery. Read
`outputs/research_dashboard/eef-review-02/physical_convention_evidence.json`.
Reference code constructs HDF5 EEF position + axis-angle, then RLDS state + gripper,
with same-index pre-action samples. The dataset snapshot lacks the producer
converter commit, and no matching regenerated HDF5 is available locally. Recorded
robosuite runtime version is not proof of the dataset-generation environment.
The original REVIEWED gate remains unchanged. No unrelated initial states were
compared. No real physical path is claimed in the rebuilt Figure 2.

## Reproduce the dashboard and four consolidated figure pairs

From the repository root in the existing numpy/matplotlib environment:

```bash
python -m experiments.dataset_analysis.integration \
  --output outputs/research_dashboard/eef-review-03
```

Choose a new output path on each run. Open its `index.html` directly in a browser.
No HTTP server, CDN, GPU, TensorFlow or dataset download is needed. The input
analysis and RLVR trees are hashed before/after; original files and manifests
are not rewritten. Existing reports are linked. `p3_pose_inventory.json` records
216 NPZ file hashes and classifies sparse observation state versus features/actions.

## After missing coordinate provenance is established

Obtain a producer source record tied to the pinned RLDS snapshot (or equivalent
identity-matched regeneration evidence), and matching **regenerated no-noop** HDF5.
The saved RLDS `source_file_path` identifies the HDF5 task file but omits demo ID;
match full recorded action/state sequences uniquely, not episode index, nearest
initial position or a fresh default simulator reset. Document any unmatched or
ambiguous demonstration and all numerical errors. Do not compare against original
unfiltered HDF5 as though it were the replayed modified data.

After valid, source-bound convention records have passed review, restore from
saved states without rereading TFRecords or changing selections:

```bash
python -m experiments.dataset_analysis.restore \
  --audit outputs/dataset_analysis/audits/goal.json \
          outputs/dataset_analysis/audits/spatial.json \
          outputs/dataset_analysis/audits/long.json \
  --convention reviewed/goal.json reviewed/spatial.json reviewed/long.json \
  --input outputs/dataset_analysis/libero_expert_trajectories/analysis.json \
  --output outputs/dataset_analysis/verified_eef_rebuild
```

`reviewed/*.json` do not currently exist: these are prerequisites, not files to
fabricate. The generated BLOCKED evidence cannot be passed as a reviewed convention.
The restore command validates evidence hashes and source identity, recomputes
physical statistics, checks selections against original manifests, then emits
3D/XY/XZ/YZ plus Figures 1–4. Independent visual review remains required.

## Future per-step evaluator logging (not executed)

`--log-eef` is available in:

- `python -m experiments.nominal.evaluate --suite long|spatial ...`
- `python -m experiments.libero_plus.evaluate_suites --suite long|spatial ...`
- `python -m experiments.libero_plus.evaluate --model scale|ript|sft ...`

Use a fresh output path. Existing results are never backfilled by rerunning
completed conditions. CPU preflight example (does **not** launch a rollout):

```bash
python -m experiments.nominal.evaluate \
  --suite spatial --model scale --log-eef --dry-run \
  --output outputs/libero-nominal-eef-review
```

The optional proxy logs every actual `env.step`, including settling steps,
with pre/post raw named EEF XYZ and xyzw quaternion, exact passed action, done,
phase, initial-state hash, released model identity and original evaluator manifest.
It never integrates actions, requests an additional observation, reseeds, modifies
input actions, changes action queues or performs extra simulator steps. Missing
poses have validity flags and NaNs. Interrupted episodes are marked incomplete.
Writing logs adds CPU/I/O overhead; GPU/EGL behavior has not been tested here.
The legacy Goal nominal vector runner is not modified by this patch.

For later new SFT/D0/RIPT/D1 experiments, the reusable `PoseLogger` accepts explicit
metadata, but the released evaluators must not be relabeled as new models. Bind
shot manifest hashes, parent SFT/D0 checkpoint identities and inference mode via
`comparison.schema.json` before those future jobs. Expert and sparse P3 data do
not establish a matched initial-state policy experiment.

## CPU tests

```bash
python -m unittest experiments.dataset_analysis.tests.test_dataset \
  experiments.dataset_analysis.tests.test_integration -v
python -m unittest discover -s experiments/libero_plus -p 'test_suite_extension.py' -v
```

Tests exercise source expressions/index alignment, pinned replay order, evidence
hashes and refusal of BLOCKED evidence, logger action identity and pre/post timing,
missing-pose flags, condition labels, preserved selections, and saved-data plots.
They are not real simulator or HDF5 numerical comparisons.
