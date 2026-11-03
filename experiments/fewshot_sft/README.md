# Few-shot continuous-action OpenVLA-OFT SFT

This isolated pipeline selects original LIBERO-Goal episodes, runs the official OFT trainer,
exports its learned LoRA/action/proprio components, and reuses the original nominal rollout
loop. It never trains a scale head or RL policy. Existing runtime sources, checkpoints and
Goal/Long/Spatial results are not edited.

Current local verification is limited to CPU fixtures/source contracts. This Mac has no
RLDS dataset, cached original pretrained OpenVLA snapshot, PyTorch, TensorFlow or CUDA.
Real TFDS materialization, training, merging, GPU inference and rollouts remain unverified.
No training, dataset download or evaluation has been launched. Do not interpret the code's
presence as a successful experiment.

All 24 CPU tests passed using the app's bundled Python (which includes NumPy):

```bash
/Users/jihyekim/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m experiments.fewshot_sft.cpu_tests
```

The system Python skips six array tests because it lacks NumPy. The recorded test evidence
identifies the interpreter actually used; these fixtures do not replace real TFDS tests.

## Commands

From the repository root, print the complete nine-stage commands:

```bash
python3 -m experiments.fewshot_sft.commands
```

The generator only prints text. It includes actual implemented flags, single-GPU selection,
Slurm `srun` guidance, and explicit `--execute` for every GPU stage. `acquire` checks only
exact known locations. Its `--execute` option fetches just the official 1.0.0 Goal dataset
(16 TFRecord shards, about 1.84 GB) from pinned commit
`a7c9ae18499b6eea8a32f78a9302327b752b1b5f`, after checking remote metadata and free
storage. The command generator sets the resulting source path and revision explicitly.
The training launcher is offline and resolves only
`openvla/openvla-7b` from the HF cache; it cannot start from a released LIBERO checkpoint.

```bash
python3 -m experiments.fewshot_sft.audit --save
python3 -m experiments.fewshot_sft.cpu_tests
python3 -m experiments.fewshot_sft.train --suite goal --shots 5 --selection-seed 42 --train-seed 7 --batch-size 1 --learning-rate 5e-4 --max-steps 200 --save-freq 50 --run-id goal5-pilot --dry-run
python3 -m experiments.fewshot_sft.report --write
```

The one-step smoke is a sequence: `train --smoke`, export the step-1 checkpoint, then
`evaluate --phase smoke`. Only its training phase provides `runs/goal5-smoke/runtime.json`.
It records actual optimizer steps, elapsed time, peak allocated and reserved CUDA memory.
Inference/rollout are separately evidenced by export verification and episode JSON.
Bounded training requires successful matching-dataset, matching-GPU, batch-1 smoke evidence.
No larger-batch memory claim is made. Smoke and export memory are separate measurements.

## Dataset contract

- Original source: `openvla/modified_libero_rlds`, `libero_goal_no_noops`, train split only.
- Every trajectory must contain one original Goal language instruction, correctly placed
  first/last flags, float32 `(7,)` actions, float32 `(8,)` state and uint8 RGB camera pairs.
- Select exactly K episodes per each of ten tasks, sorted by original TFDS ID before
  seeded sampling. Store original IDs, complete decoded-payload hashes, source file hashes,
  declared source revision, selection seed and per-task counts. No frame subsampling.
- `--selection-manifest` validates and reuses the recorded selection; any changed source,
  selection, seed or shot count fails. Existing destination directories are never replaced.
- Materialize only the selected episodes using the source FeaturesDict. Upstream
  `dataset_info.json` names the builder `libero_goal` despite its official
  `libero_goal_no_noops` directory. The tool checks both identities and preserves
  original encoded JPEG bytes to avoid lossy re-encoding. This is necessary
  because the existing loader does not expose an episode-ID selector before normalization
  and flattening. All step/episode metadata and temporal ordering are retained. The source
  `_no_noops` filtering is not reapplied and discarded transitions are not reconstructed.
- Re-read and compare every materialized episode hash before marking the subset verified.
- If execution stops after TFDS writes the subset, `prepare_dataset --resume-existing` verifies
  the saved selection, rereads the 50 episodes, recomputes normalization, and writes only
  missing final audit files. Existing evidence with different content is rejected.
- `normalization.json` uses the official gripper transform and mean/std/min/max/q01/q99
  over selected episodes. A process-local statistics provider avoids the upstream shared
  statistics-cache collision. The official loader still applies the original normalization
  mask, BOUNDS_Q99, augmentation, temporal chunking and boundary padding.
- Raw source file checksums establish local content identity. A supplied HF revision is a
  provenance label, not independent proof that those bytes match the remote revision.

## Official training integration and checkpoint semantics

`worker.py` loads `runtime/openvla-oft/vla-scripts/finetune.py` into a separate module.
The official forward pass, L1 loss, AdamW, scheduler, image augmentation and data loader
remain in use. The local OFT continuous-action class loads the original pretrained weights
directly; the pristine base directory is never rewritten by upstream auto-map/sync helpers.
Training uses two images, proprio, LoRA rank 32, an 8 × 7 action chunk and no FiLM/diffusion.

Five guarded source substitutions instrument finite loss/gradients and optimizer setup,
and correct upstream zero-based checkpoint/stop bookkeeping. Thus `--max-steps 1` means
one optimizer update, and accumulation cannot save a partial update. The shim fails when
an expected source anchor changes. Source hashes are recorded in each training config.

Checkpoints preserve the SFT adapter, DDP-format action head and proprio projector,
normalization, processor, optimizer/scheduler state and complete provenance. `--resume`
requires our unmerged checkpoint, a new run ID, the same dataset/base/training settings,
and a higher total `--max-steps`. It restores learned weights and optimizer/scheduler;
the seeded TF shuffle restarts. This is **not bitwise continuation of the data stream**.

Export uses PEFT `merge_and_unload(safe_merge=True)`, as in the official merge path, into
a fresh directory. It verifies deterministic `(8,7)` actions before/after merging and
after disk reload on fixed synthetic RGB/proprio inputs; `atol=0.02`, `rtol=0`, with raw
outputs and errors recorded. The absolute BF16 tolerance is explicit and can be tightened.
If merge comparison fails, `checkpoint --checkpoint PATH --diagnose-merge --execute` prints
repeatability, maximum and per-dimension action differences without writing an export.
The export destination is created only after the in-memory comparison passes; a failed
older export directory remains invalid and must not be evaluated.
This numerical smoke does not establish equivalence on all observations. Exports without
passing evidence or with changed file hashes cannot be evaluated.

## Nominal evaluation and model selection

The new entrypoint adds Goal and the explicit checkpoint **only within its own process**,
then calls the existing `suite_common.run(..., 'nominal')`. It retains the original task
ordering, 300-step Goal horizon, ten settling actions, full-sequence official inference,
center crop, deterministic mean action, image preprocessing, 8-action queue and initial
states. `pilot` bounds each task at 20 trials; `final` explicitly opts into 50. The reused
inner manifest records the original protocol's 50-trial schedule; `selection.json` records
the actual phase cap and the summary counts actual episode rows. Pilot and final share
the first 20 initial states: they are not independent evaluation samples.

Each phase requires a checkpoint-selection note. All evaluated checkpoints/notes are
retained by the report, so nominal-driven checkpoint tuning is visible. Per-task success,
trials, episode lengths, seed, checkpoint step and evaluator hash are preserved. Failed
phases retain partial evidence and cannot be silently overwritten; they require explicit
investigation before retrying. No full 500-episode evaluation runs automatically.

Training loss observations are saved in `metrics.jsonl` and aggregated as loss curves in
the JSON report. Held-out demonstration validation is explicitly unavailable; no training
episode is mislabelled held-out. Real TFDS/data-loader integration remains a server check.

## Compatibility findings and remaining gates

The RIPT interface imports the same OFT action-head/proprio loaders and adds a cached
attention patch. Its complete policy additionally needs a pretrained scale header;
RL `openvla_headers.pt` contains `action_header` and `scale_header`. SFT must not fabricate
these RL artifacts. The export reports SFT dimensions/merge/load evidence separately from
the **unexecuted cached RIPT inference**. No compatibility claim is made for a full RIPT
policy without that execution evidence.

Declared repository revisions are in `stage0/repos.lock.json`; local runtime copies lack
`.git`, so their SHAs are not independently verified. File hashes identify audited content.
The [official current trainer](https://github.com/moojink/openvla-oft/blob/main/vla-scripts/finetune.py)
and local code expose the same continuous-action options; a live upstream commit SHA was
not established. TFDS materialization follows its
[GeneratorBasedBuilder](https://www.tensorflow.org/datasets/api_docs/python/tfds/core/GeneratorBasedBuilder)
and [nested Dataset feature](https://www.tensorflow.org/datasets/api_docs/python/tfds/features/Dataset) APIs.

Read-only report: `python -m experiments.fewshot_sft.report`. Add `--write` to refresh only
`outputs/fewshot_sft/fewshot_sft_summary.json` and `.md`. The report does not touch any
`research_summary.json`, full-data results, checkpoints or experiment evidence.
