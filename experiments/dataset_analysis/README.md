# Stage 1 — original expert dataset analysis only

Active workflow: inventory → verified archive → approved cleanup → fresh source
audits → dataset-only build. No training, evaluation, scale fitting, shot selection,
or dashboard/P3 generation belongs to Stage 1. Existing evaluation code and
historical workflows remain unchanged; see `LEGACY.md` for historical references.
The previous mixed-stage `build` is explicitly named `legacy-build`.

## Local outcome and limitations

See `outputs/stage1_review/report.md`. The local data directory is empty. Cached
real-data audits can be checked and summarized, but cannot authenticate original
RLDS bytes or establish physical conventions. These builds remain PARTIAL.
Do not interpret the revision CLI label or an old REVIEWED flag as proof.

## Server cleanup (run manually on the server)

Run from `/nfs4/jhkim/repos/vla-rlvr`, with no jobs using the target directories.
Never reuse the local cleanup plan on the server: all paths, hashes and processes
must be inventoried on the host where cleanup will happen.

```bash
mkdir -p outputs/stage1_review
python -m experiments.dataset_analysis.cleanup plan-cleanup \
  --output outputs/stage1_review/cleanup_plan.json &&
python -m experiments.dataset_analysis.cleanup archive \
  --plan outputs/stage1_review/cleanup_plan.json \
  --archive outputs/archives/stage1_before_cleanup &&
python -m experiments.dataset_analysis.cleanup verify \
  --archive outputs/archives/stage1_before_cleanup
```

`plan-cleanup` inventories the repository without changing source artifacts. With
`--output`, it atomically saves a complete proposal outside inventoried trees and
refuses overwrite; an inventory failure leaves no empty output file. Without
`--output`, it emits JSON on stdout. Inventory includes every regular
file's size/hash and empty directories, plus checkpoint/data/vendor/runtime guards.
Deletion-target symlinks, protected directory symlinks, special files and path
escapes are rejected. Protected file symlinks (such as HF snapshot-to-blob links)
are accepted only when they resolve to a regular file inside the same protected
tree. Their link text, resolved path, size and content hash are recorded and
rechecked; external or dangling links remain blocked. The archive
contains all four target trees, clearly marked reference-only. It is a verified
copy, not immutable storage. Never remove or edit it.

Before approval, inspect the exact `targets` entries and confirm historical Goal
5-shot selection, step-200 checkpoint and its provenance, completed released-model
nominal/OOD and P3 evidence. A missing local checkpoint is not proof it never existed.
Close readers and stop writers first; permission errors or active processes block
apply. Server activity cannot be checked by a local process scan. An exclusive
maintenance window is required; users must not start jobs during deletion.

Only after the exact proposal is explicitly approved, compute its `digest(plan)`
using `experiments.shot_scaling.core.digest` and pass that value:

```bash
python -m experiments.dataset_analysis.cleanup apply-cleanup \
  --plan outputs/stage1_review/cleanup_plan.json \
  --archive outputs/archives/stage1_before_cleanup \
  --approved-plan-sha256 APPROVED_COMPLETE_PLAN_DIGEST
```

The digest binds the complete proposal, not only the target names. Apply rechecks
archive, source/protected hashes and current processes; it unlinks only listed
files and removes empty directories. No `rm -rf` is used. Approval is a human
review step; possession of a digest is not evidence of consent. No apply was run
in the local Stage 1 work.

## Fresh CPU-only extraction and dataset-only build

Use the existing server TensorFlow/TFDS environment. After approved cleanup:

```bash
CUDA_VISIBLE_DEVICES='' python -m experiments.dataset_analysis audit \
  --source data/data/modified_libero_rlds/libero_goal_no_noops/1.0.0 \
  --suite goal --source-revision a7c9ae18499b6eea8a32f78a9302327b752b1b5f \
  --output outputs/stage1_review/fresh_audits/goal.json
CUDA_VISIBLE_DEVICES='' python -m experiments.dataset_analysis audit \
  --source data/data/modified_libero_rlds/libero_spatial_no_noops/1.0.0 \
  --suite spatial --source-revision a7c9ae18499b6eea8a32f78a9302327b752b1b5f \
  --output outputs/stage1_review/fresh_audits/spatial.json
CUDA_VISIBLE_DEVICES='' python -m experiments.dataset_analysis audit \
  --source data/data/modified_libero_rlds/libero_10_no_noops/1.0.0 \
  --suite long --source-revision a7c9ae18499b6eea8a32f78a9302327b752b1b5f \
  --output outputs/stage1_review/fresh_audits/long.json
python -m experiments.dataset_analysis build \
  --audit outputs/stage1_review/fresh_audits/goal.json \
          outputs/stage1_review/fresh_audits/spatial.json \
          outputs/stage1_review/fresh_audits/long.json \
  --output outputs/dataset_analysis
```

Source locations are those recorded in the existing server audits; adjust if the
actual original snapshot moved. Authenticate shard hashes against the pinned HF
snapshot independently; `--source-revision` alone does not do that. Existing
`audit.extract`, `analysis.analyze(seeds=())` and convention validation are reused.
No new loader, action transformation or sampling implementation was introduced.

Build saves task inventory CSV, episode features CSV, complete numerical
state/action sequences, action statistics, provenance and validation. Plotting reads
`analysis.json` only. It produces at most four PNGs and one figure index. Without
physical verification it creates three useful figures, omits Figure 2 completely,
and labels raw state/command values as physically unverified. Raw command threshold
0.5 crossings describe gripper events, not independently verified open/close states.

To reproduce figures without decoding any TFRecords:

```bash
python -c "from pathlib import Path; from experiments.dataset_analysis.stage1 import plot; plot(Path('outputs/dataset_analysis/analysis.json'), Path('outputs/stage1_review/reproduced_figures'))"
```

Use a fresh figure destination. `--convention FILE...` accepts source-bound reviewed
records validated by `audit.convention`; it does not accept a cached status flag.
Evidence must establish actual producer linkage, state mapping, units/frame,
axis-angle radians and temporal alignment. Compare identity-matched regenerated
HDF5 demonstrations when available. Original simulator `states[0]` is not the same
as the settled, recorded EEF state. Until these checks and original-byte provenance
are resolved, keep Stage 1 PARTIAL; this conservative builder never self-certifies PASS.

## Validation

```bash
python -m unittest experiments.dataset_analysis.tests.test_stage1 \
  experiments.dataset_analysis.tests.test_dataset \
  experiments.dataset_analysis.tests.test_integration -v
```

Fixture tests are safety/algorithm tests, not real RLDS evidence. Saved real audit
checks validate checksums, IDs, task coverage, counts, finite values and array
alignment; they cannot recompute image payload hashes without source images.

## Stage 2 later (not executed)

Only after Stage 1 review resolves its blockers, the existing Goal-only writer can
begin historical Goal 5-shot preparation with the archived original selection:

```bash
CUDA_VISIBLE_DEVICES='' python -m experiments.fewshot_sft.prepare_dataset \
  --source data/data/modified_libero_rlds/libero_goal_no_noops/1.0.0 \
  --source-revision a7c9ae18499b6eea8a32f78a9302327b752b1b5f \
  --shots 5 --selection-seed 42 \
  --selection-manifest outputs/archives/stage1_before_cleanup/artifacts/dataset_analysis/libero_expert_trajectories/selections/historical_goal5_seed42.json \
  --output outputs/fewshot_sft/datasets/goal_5shot_seed42
```

This exact command begins existing Goal-only preparation; it does not implement
all-suite Stage 2 or training. Confirm the historical manifest exists in the server
archive. A new all-suite writer remains future work.
