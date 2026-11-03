# Phase 2: expert command-trajectory variation

This addition reads the completed Goal artifacts and existing selection manifests. It does
not change selection, train a model, collect rollouts, or reconstruct physical paths from
commands. `variation.py` and `variation_plot.py` are separate CPU analysis/plotting modules.
The Phase 1 mean/std descriptor coverage remains intact and is not relabeled trajectory diversity.

## Availability audit

Six small JSON metadata files were fetched from the pinned public
`openvla/modified_libero_rlds` revision
`a7c9ae18499b6eea8a32f78a9302327b752b1b5f`. No TFRecord shards were downloaded.
Each file URL has this form:

```text
https://huggingface.co/datasets/openvla/modified_libero_rlds/resolve/a7c9ae18499b6eea8a32f78a9302327b752b1b5f/{dataset}/1.0.0/{filename}
```

Datasets are `libero_goal_no_noops`, `libero_spatial_no_noops`, `libero_10_no_noops`;
filenames are `features.json` and `dataset_info.json`. The saved names prefix the suite,
e.g. `goal_features.json`. Goal metadata hashes match the existing original-source inventory.
Spatial/Long schema inspection is not a validation of their actual episode contents.

All schemas declare observation `state[8]` (6D EEF pose + 2D gripper), `joint_state[7]`,
two RGB cameras, and action[7]. There are no serialized object-position fields. The schemas
also declare first/last/terminal flags, reward, discount, language and original file path.
The Goal trajectory artifact retains only initial_proprio and action sequences. It therefore
cannot support final EEF pose, actual path length/displacement, 3D EEF overlays or physical
path distances. Missing values stay null and figures are explicitly listed as unavailable.

OFT regeneration code records xyz and quaternion-to-axis-angle rotation before the action;
the OFT transform maps state[:6] to EEF and state[-2:] to gripper. The pinned metadata does
not establish physical frame/units, so plots label source coordinates rather than asserting
metres, a world frame or radian-valued dataset fields. Command directions are directions in
command-coordinate space. Terminal/reward conventions do not establish independent success.

## Methods and limits

- Episodes are joined by exact source IDs and payload hashes. Historical membership comes
  from the preserved prior analysis; the original historical selection manifest is absent
  locally and is not claimed independently revalidated. The historical/nested overlap is 3.
- Temporal action distance is RMS between 32 phase-aligned samples of the first six command
  dimensions. This is not DTW, physical trajectory distance, or elapsed-time alignment.
- Descriptors add raw length, command magnitudes/directions, successive cosines/reversals,
  zero-command fractions and gripper events/occupancy. Zero vectors are excluded from direction
  and successive-cosine denominators. Reversal means cosine < 0; no angular velocity is inferred.
- Gripper pattern distance is command-state disagreement over 32 phase bins. Original event
  phase and counts are retained separately. Raw -1/open and +1/close use the existing OFT
  threshold convention; command state is not measured gripper aperture.
- Initial-state distance is standardized Euclidean distance in the eight initial proprio
  values, not a physical pose metric or SO(3) geodesic.
- Normalization is task-specific and fitted only to the 20-shot candidate pool. Constant axes
  are excluded. Both episode-weighted and transition-weighted raw action statistics are saved.
- Goal's 33–50 episodes/task permit all 9,116 within-task pairs. A hard 64-episode/task budget
  (at most 128 if explicitly revised in the API) and fixed 32-point grid bound the work.
  Larger tasks fail instead of silently inventing an unreported sample. No cross-task distances.
- Main coverage uses the original fixed 228 reference IDs at every shot level. 64 analysis-only
  random nested permutations sample the remaining 20 episodes/task. This is a conditional
  baseline, not a full-dataset seed audit. All 20-shot selections coincide; their percentile is
  non-informative. Random p05–p95 bands are not confidence intervals.
- Historical comparison uses the same secondary 197-episode pool for both historical and
  nested5: the fixed 228 pool minus historical IDs. Main coverage still uses the original 228.
- No independent selection manifests or training datasets are written for random baselines.

## Commands actually used

Interpreter: `/tmp/libero-pipeline-venv/bin/python` (Python 3.12.14, NumPy 1.26.4,
Matplotlib 3.9.0). Run from `/Users/jihyekim/Desktop/vla-rlvr`.

```bash
/tmp/libero-pipeline-venv/bin/python -m unittest \
  experiments.shot_scaling.tests.test_pipeline \
  experiments.shot_scaling.tests.test_extract \
  experiments.shot_scaling.tests.test_variation -q

/tmp/libero-pipeline-venv/bin/python -m experiments.shot_scaling.variation \
  --run-root outputs/shot_scaling \
  --metadata-dir outputs/trajectory_variation/phase2-v2/metadata \
  --output outputs/trajectory_variation/phase2-v2/goal \
  --random-seeds 64

/tmp/libero-pipeline-venv/bin/python -m experiments.shot_scaling.variation_plot \
  --input outputs/trajectory_variation/phase2-v2/goal/variation.json \
  --output outputs/trajectory_variation/phase2-v2/goal/figures
```

Output paths are exclusive-create. Choose a new output directory when repeating analysis or
plotting; do not overwrite previous versions. The first development output is under phase2-v1;
the final report is **phase2-v2/goal/report.md**. Only the v2 results should be cited. The v2
roundoff fix treats numerically tied random metrics as ties instead of extreme percentiles.

## Validation and artifacts

29 CPU tests passed (19 existing + 10 variation tests). New checks cover zero-vector handling,
reversals, gripper-event semantics, temporal-order differences hidden by mean/std, alignment
endpoints, reference leakage, singleton diversity, candidate-only normalization, deterministic
nested random sampling, fixed references, 20-shot degeneracy and pairwise budgets. Test fixtures
are never included in user-facing research outputs.

Analysis writes `episode_features.csv`, `task_variation.json`, `shot_coverage.csv`,
`pairwise_comparisons.csv`, `variation.json`, `state_audit.json`, `report.md`, provenance and
completion markers. Standalone plotting writes 60 task-specific figures in both PNG and SVG,
plus an index and source-ID mapping for bounded gripper displays. No metric extraction occurs
inside the plotter. The full plots read only saved Phase 2 analysis.

The future comparison contract is `trajectory.schema.json`. It records source, checkpoint,
inference mode, initial state ID, observations, executed actions, measured EEF pose, gripper,
terminal/success, units and provenance. Unavailable fields remain null. Future integration must
reuse existing evaluators and record the exact action passed to env.step; no GPU rollout
collector was added.
