# Stage 1 — Full expert trajectory analysis

Analyze the existing pinned Goal, Spatial and Long RLDS demonstrations before
few-shot preparation. Use extracted sequences; do not re-decode TFRecords, regenerate
HDF5, run policies or train models. No backup, archive or cleanup workflow is needed
for regenerating these derived outputs.

## Update the existing server result

Copy the modified dataset_analysis source files to the server, then run in the
existing CPU-capable environment:

```bash
cd /nfs4/jhkim/repos/vla-rlvr
export CUDA_VISIBLE_DEVICES=''
export MPLBACKEND=Agg
python -u -m experiments.dataset_analysis build \
  --audit outputs/dataset_analysis/trajectory_data/goal.json \
          outputs/dataset_analysis/trajectory_data/spatial.json \
          outputs/dataset_analysis/trajectory_data/long.json \
  --output outputs/dataset_analysis --refresh
cat outputs/dataset_analysis/validation.json
```

`--refresh` replaces derived summaries and figures in place. The three input
trajectory files are loaded before any output writes and left untouched when they
already occupy the output trajectory_data directory. No archive, move, source-file
hash scan, HDF5 download or future-stage output is performed.

## Coordinate interpretation

`semantics.py` records `SOURCE_DOCUMENTED`, not `REVIEWED` or bitwise verified:

- state[:3]: EEF world XYZ position, meters under documented simulator conventions.
- state[3:6]: axis-angle orientation (rotation vector), radians.
- state[6:8]: gripper state.
- state[t]: observation before action[t] in the documented regeneration pipeline.

Official OpenVLA documentation links the regeneration and RLDS conversion code;
local reference source paths and URLs are recorded in manifest.json. Exact producer
commit, original environment and HDF5 numerical equivalence remain unresolved,
nonblocking provenance limitations. They are not prerequisites for descriptive
plots. `--convention` is unnecessary for this builder; legacy reviewed-convention
paths remain available to historical callers without changing their behavior.

Numerical checks require finite full state/action arrays, matching lengths, valid
suite/task/episode identities and the pinned revision label. Position min/max and
quantiles, adjacent recorded displacement quantiles, stationary sequences and
large jumps are reported. +/-5 m coordinate and 0.25 m adjacent displacement
thresholds are broad screening heuristics, not calibrated workspace limits.
Nothing is clipped, silently removed or integrated from actions. Flagged data or
unexpected counts produces PARTIAL; valid descriptive analysis produces PASS.
PASS refers only to this research scope, not exact producer authentication.

## Outputs

- analysis.json: all episode paths, action statistics, all within-task pair metrics.
- episode_features.csv: lengths, path lengths, displacement, action/gripper summaries.
- inventory.csv: suite/task/split episode and transition counts.
- task_variation.csv: path-length mean/std, initial XYZ std, pairwise RMS mean/median.
- trajectory_data/: original extracted numerical sequences retained.
- manifest.json / validation.json / report.md: interpretation, numerical checks and limitations.
- figures/index.md: single index for four multi-panel PNGs.

Figures: (1) counts and episode lengths; (2) 3D/XY/XZ overlays for all episodes of
the alphabetically first train task in each suite; (3) path lengths, pairwise path
RMS and command variation for all 30 tasks; (4) initial XY/XZ and XYZ distributions
for all episodes, colored by task. The figure manifest maps indices to full task
names and records every representative episode ID. No per-task PNG collection or
shot selection is generated.

Within each task/split, all episode pairs are compared after linear interpolation
to 32 normalized-index points. RMS is measured in the documented world frame without
centering; it includes initial-position differences. Pair values are correlated,
not independent trials. This is not DTW, velocity, success or policy robustness.
Path length ends at the final recorded pre-action state, not an invented endpoint.

To redraw figures without recomputing statistics:

```bash
python -c "from pathlib import Path; from experiments.dataset_analysis.stage1 import plot; plot(Path('outputs/dataset_analysis/analysis.json'), Path('outputs/dataset_analysis/figures'), refresh=True)"
```

## CPU regression checks

```bash
python -m unittest experiments.dataset_analysis.tests.test_descriptive \
  experiments.dataset_analysis.tests.test_dataset -v
```

Fixtures test numerical behavior only. Real-data completion is recorded separately
by the builder. No Stage 2–7 workflow is executed. Historical mixed-stage commands
and documentation in LEGACY.md are not the current Stage 1 workflow.
