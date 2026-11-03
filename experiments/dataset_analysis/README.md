# LIBERO expert dataset analysis

This replaces the primary analysis, without deleting or changing historical results.
No command starts training or rollouts. Real-data completion is currently **BLOCKED**
in the local checkout: original RLDS and previous shot-scaling results are absent.
CPU fixtures test algorithms only; they are not research data.

## Server execution

Use the existing TF/TFDS environment. The plotting dependency is matplotlib 3.9.0
to match ript-vla; see `experiments/shot_scaling/requirements.txt`.
Run from `/nfs4/jhkim/repos/vla-rlvr` after copying this code to the server.
Commands refuse to overwrite existing destinations.

First archive the old analysis. The archive retains originals, records their hashes
and manifest identities, hashes protected checkpoints/results, verifies the copy,
and sets read-only permissions. This is hash-verifiable read-only storage, not WORM.
Use additional `--protected` paths for checkpoint provenance outside these locations.

```bash
cd /nfs4/jhkim/repos/vla-rlvr
python -m experiments.dataset_analysis archive \
  --source outputs/shot_scaling \
  --output outputs/archives/shot_scaling_before_dataset_rebuild \
  --protected outputs/fewshot_sft checkpoints outputs/rlvr_analysis outputs/libero outputs/goal
```

Every protected directory must exist; review actual locations before changing this
list. No archive is made from an empty or missing source.

If the previous analysis was already deleted, do not create an empty archive.
Record the loss explicitly instead (only accepted for an absent/empty source):

```bash
python -m experiments.dataset_analysis record-loss \
  --source outputs/shot_scaling \
  --output outputs/archives/shot_scaling_loss_record \
  --reason "User deleted previous shot_scaling results before rebuild" \
  --protected outputs/fewshot_sft checkpoints
```

Then use `--archive outputs/archives/shot_scaling_loss_record` in the build command.
The report and provenance record will state that the old results are lost, not
archived. Surviving protected files must remain unchanged. The original RLDS and
historical Goal5 selection manifest are still required; this option does not
reconstruct lost selections or checkpoint provenance.

Find actual original version directories; do not select a sampled dataset under outputs:

```bash
find data -type f -name dataset_info.json -print
```

Goal's path below comes from the earlier server run. Set Spatial and Long variables
to the discovered original version directories before executing their commands.
All splits are read; omitting `--max-episodes` is required for a full audit.

```bash
python -m experiments.dataset_analysis audit \
  --source data/data/modified_libero_rlds/libero_goal_no_noops/1.0.0 \
  --suite goal --source-revision a7c9ae18499b6eea8a32f78a9302327b752b1b5f \
  --output outputs/dataset_analysis/audits/goal.json

python -m experiments.dataset_analysis audit \
  --source "${SPATIAL_SOURCE:?Set actual original Spatial TFDS version directory}" \
  --suite spatial --source-revision a7c9ae18499b6eea8a32f78a9302327b752b1b5f \
  --output outputs/dataset_analysis/audits/spatial.json

python -m experiments.dataset_analysis audit \
  --source "${LONG_SOURCE:?Set actual original Long TFDS version directory}" \
  --suite long --source-revision a7c9ae18499b6eea8a32f78a9302327b752b1b5f \
  --output outputs/dataset_analysis/audits/long.json
```

These preserve all recorded states but mark physical interpretations UNAVAILABLE.
To enable measured XYZ, use `--convention /absolute/path/reviewed.json` on a fresh
audit output. The evidence record must contain `status: REVIEWED`, the pinned
`dataset_revision`, `source_files_sha256` from the audit, `reviewer`,
`state_position_indices: [0,1,2]`, `state_orientation_indices: [3,4,5]`,
`position_units: m`, `orientation_representation: axis_angle_radians`, a named
`coordinate_frame`, and `state_timing: pre_action` or `post_action`.
Its `evidence` array must cover claims `position_mapping`, `orientation_mapping`,
`coordinate_frame`, `units`, and `state_action_alignment`; each item requires
`claim`, absolute `path`, `sha256`, and a review `explanation`. Inspect pinned
conversion code and environment state logging, not just array shape. File-hash
verification does not itself verify the scientific claim. Unverified conventions
are never silently assumed. Do not fabricate a reviewed record to enable figures.

Build from three full saved audits and the unchanged historical manifest:

```bash
python -m experiments.dataset_analysis build \
  --audit outputs/dataset_analysis/audits/goal.json \
          outputs/dataset_analysis/audits/spatial.json \
          outputs/dataset_analysis/audits/long.json \
  --archive outputs/archives/shot_scaling_before_dataset_rebuild \
  --historical outputs/fewshot_sft/datasets/goal_5shot_seed42/selection_manifest.json \
  --selection-seeds 42 43 44 --display-seed 42 \
  --output outputs/dataset_analysis/libero_expert_trajectories
```

If the canonical output already contains the local BLOCKED report, preserve that
directory and choose a fresh `--output` version for the build. Never overwrite a
previous analysis. The build emits analysis.json, provenance.json, status.json,
report.md, 18 selections per seed across three suites, the separate historical
Goal5 manifest, and exactly four PNG/PDF figure pairs with figure_manifest.json.
Physical panels explicitly show UNAVAILABLE when evidence/state is missing.
Review real figures before declaring the analysis complete. No supplementary
figures are generated. Reproduce plots without rereading TFRecords:

```bash
python -m experiments.dataset_analysis.plots \
  --input outputs/dataset_analysis/libero_expert_trajectories/analysis.json \
  --output outputs/dataset_analysis/reproduced_figures
```

## Later dataset materialization

The new nested selections do not yet have a validated all-suite TFDS writer/reload
path. This check validates IDs/checksums and deliberately exits 2 with an explicit
materialization block; it writes no data:

```bash
python -m experiments.dataset_analysis materialize-check \
  --audit outputs/dataset_analysis/audits/goal.json \
  --selection outputs/dataset_analysis/libero_expert_trajectories/selections/goal/seed-42/5.json
```

The existing writer supports only the historical Goal 1/5-shot sampler. It rejects
new nested manifests; do not use it to materialize the new conditions. For later
verification of the existing historical Goal5 dataset only:

```bash
python -m experiments.fewshot_sft.prepare_dataset \
  --source data/data/modified_libero_rlds/libero_goal_no_noops/1.0.0 \
  --source-revision a7c9ae18499b6eea8a32f78a9302327b752b1b5f \
  --shots 5 --selection-seed 42 \
  --selection-manifest outputs/fewshot_sft/datasets/goal_5shot_seed42/selection_manifest.json \
  --output outputs/fewshot_sft/datasets/goal_5shot_seed42 --resume-existing
```

Do not run this as a new-shot-scaling training preparation command.

## Validation and future comparison

```bash
python -m unittest experiments.dataset_analysis.tests.test_dataset -v
```

Tests cover three-suite task mapping, state/action length alignment, missing and
nonfinite data, duplicate split payloads, archive preservation, convention refusal,
known physical geometry, nested seeds, historical identity, and saved-data plotting.
They do not validate real TFDS ingestion, physical conventions or real visual paths.

`comparison.schema.json` defines the common future evaluator export. Keep recorded
EEF poses, exact executed actions, frame/unit/timing metadata and checkpoint lineage.
Expert, released Full-SFT, released Scale-cached/RIPT-cached and new D0/D1 conditions
are separate. Existing evaluators are not modified and no rollouts are started.
