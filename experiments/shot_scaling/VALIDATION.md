# Validation evidence — 2026-11-03

Execution host: local macOS checkout, Python 3.12.14 at
`/tmp/libero-pipeline-venv/bin/python`. Hydra 1.3.2, OmegaConf 2.3.1, NumPy 2.3.5,
Matplotlib 3.9.4. TensorFlow, TFDS and PyTorch are absent. Dependencies were installed
in a temporary virtual environment; no repository runtime environment was replaced.

| Check | Actual result |
|---|---|
| `python -m unittest experiments.shot_scaling.tests.test_pipeline -q` | 15 passed; 0 skipped/failures, final run 1.259 seconds |
| Existing fewshot CPU regression suite | 33 total: 32 passed, 1 skipped (PyTorch optimizer continuity unavailable) |
| `python -m experiments.shot_scaling.smoke` | 18 CLI invocations returned their expected codes; synthetic data operations succeeded; all 5 GPU/integration stages returned BLOCKED/2 |
| Historical report | Existing read-only suite report successfully rebuilt in isolated subprocess |
| `python -m compileall -q experiments/shot_scaling` | Passed |
| Visual inspection | Synthetic coverage PNG inspected; task labels fit and SYNTHETIC is visible |
| Real RLDS, training, export, scale fit, D0/D1 inference, evaluation | Not run; BLOCKED |

Final smoke evidence:

- [Machine-readable stages and exit codes](../../outputs/shot_scaling/20261103T073732Z-8049b17aa523/completion.json)
- [Immutable smoke manifest and runtime versions](../../outputs/shot_scaling/20261103T073732Z-8049b17aa523/manifest.json)
- [Synthetic inventory](../../outputs/shot_scaling/20261103T073732Z-8049b17aa523/inventory.json)
- [Synthetic trajectories](../../outputs/shot_scaling/20261103T073732Z-8049b17aa523/trajectories.json)
- [Five PNG/SVG example figures](../../outputs/shot_scaling/20261103T073734Z-890bbd026e57/figures/index.md)
- [Synthetic action translation directions](../../outputs/shot_scaling/20261103T073734Z-890bbd026e57/figures/translation_directions.png)
- [Synthetic shot coverage](../../outputs/shot_scaling/20261103T073734Z-890bbd026e57/figures/shot_coverage.png)

These outputs are under the repository's existing ignored `outputs/` directory. They are
local artifacts, not tracked experiment evidence. The smoke's `completion.json` records
all child output locations, including nested manifests, both Full and 5-shot analyses,
new-bundle aggregation, historical aggregation and inspection.

The legacy `experiments.fewshot_sft.cpu_tests` wrapper refreshes its own
`outputs/fewshot_sft/cpu_tests.json`; it was invoked once and updated that test-metadata
file. No historical selections, checkpoints, benchmark episode results, P3 results,
or existing summary result files were rewritten. For future read-only regressions use:

```bash
python -m unittest experiments.fewshot_sft.test_pipeline experiments.fewshot_sft.test_continuation -v
```

Only new source directory `experiments/shot_scaling/` appears in Git status. Actual new files:

- `__init__.py`, `__main__.py`: package and unified CLI.
- `core.py`: configuration, identity, manifests and runtime provenance.
- `data.py`: complete-source inventory validation, nested manifests and historical reuse.
- `analysis.py`: CPU raw/standardized action analysis and JSON/CSV output.
- `reporting.py`: protocol completeness, seed aggregation and strict paired-comparison API.
- `plotting.py`: standalone saved-artifact PNG/SVG figures.
- `smoke.py`: bounded synthetic CLI smoke, with no GPU work.
- `conf/config.yaml`, `conf/suite/goal.yaml`, `conf/suite/spatial.yaml`, `conf/suite/long.yaml`.
- `tests/__init__.py`, `tests/test_pipeline.py`, `requirements.txt`, `README.md`, `VALIDATION.md`.

No algorithm changes, commits, pushes, training jobs or full evaluations were performed.
The architecture audit, working command list, input contracts and final 18-condition
matrix are in [README.md](README.md). The complete experiment remains blocked at real
data materialization and GPU validation, with additional scale-training and PPO
distribution blockers downstream.
