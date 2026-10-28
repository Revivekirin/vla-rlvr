# LIBERO-Long and LIBERO-Spatial evaluation extension

This extension adds Long and Spatial entrypoints while leaving the validated Goal evaluator and all Goal results untouched. It reuses the Goal evaluator's atomic result writer and configuration, and the same OpenVLA-OFT SFT and RIPT cached action functions. No training path is called. At audit time the working repository HEAD was `2fd6809e8796ba28c4471396e4756a3fd50e79d0`; the latest commit affecting Goal's `evaluate.py` was `2f160ae8d69ff047fe5a271afadea956ce3d16de`. The Goal evaluator was not changed in this extension.

## Suite and checkpoint audit

| Label | Benchmark key | OFT model repository | RIPT scale and adapter prefix | Local normalization key | Nominal horizon | Official Plus variants at pinned revision |
|---|---|---|---|---|---:|---:|
| Long | `libero_10` | `moojink/openvla-7b-oft-finetuned-libero-10` | `LIBERO_LONG` | `libero_10_no_noops` | 520 | 2,519 |
| Spatial | `libero_spatial` | `moojink/openvla-7b-oft-finetuned-libero-spatial` | `LIBERO_SPATIAL` | `libero_spatial_no_noops` | 220 | 2,402 |

The suggested `moojink/openvla-7b-oft-finetuned-libero-long` is not the released Long repository. Its official name ends in `libero-10`; the released RIPT Long adapter config also names that base. Sources: [OFT Long](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-10/tree/main), [OFT Spatial](https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-spatial/tree/main), [RIPT model card](https://huggingface.co/tanshh97/RIPT_VLA), [LIBERO-Plus pinned task map](https://raw.githubusercontent.com/sylvestf/LIBERO-plus/4976dc30028e805ff8094b55501d532c48fec182/libero/libero/benchmark/libero_suite_task_map.py) and [classification](https://raw.githubusercontent.com/sylvestf/LIBERO-plus/4976dc30028e805ff8094b55501d532c48fec182/libero/libero/benchmark/task_classification.json).

Local checkpoint directories currently contain OFT metadata and adapter configuration, but not Long/Spatial base shards, action/proprio heads, scale heads or trained adapters. The server inventory must be checked separately; this local inventory cannot establish server readiness. The downloader inspects the official Hugging Face file inventory and exact sizes before any download, selects only the two requested suites, checks free space, uses the remote repository commit SHA for each file, refuses to replace existing paths and saves file SHA256 provenance. It requires server network access and `huggingface_hub`.

The model configuration identifies `OpenVLAForActionPrediction`, a Llama 2 7B backbone and two vision encoders. `dataset_statistics.json` supplies the continuous-action normalization key and 7D action/8D proprio statistics. The adapter config specifies rank 32, alpha 16 and zero dropout. The runtime uses an 8×7 continuous L1 action head, two cameras, proprioception and center crop. GPU model loading and header tensor compatibility remain unverified until server smoke. Scale's LoRA B matrices must be zero; RIPT's must contain nonzero trained weights. A mismatch aborts model loading.

The pinned LIBERO-Plus map and classification define the Long and Spatial OOD suites. The runtime preflight checks the pinned checkout, task-map/classification alignment, asset directories and expected counts. `--check-import` compares the imported benchmark order with that mapping. Local LIBERO-Plus checkout is absent, so local runtime mapping and physical BDDL/init-state checks have not completed. No Goal conditions are reused as Long or Spatial conditions.

## Server sequence

Run from the repository root on a GPU node. `server_env.sh plus` is required for Plus because its environment imports Wand/ImageMagick. Use one job per output suite/model directory; avoid concurrent jobs that write the shared Plus runtime config.

1. Inspect official inventories and storage, without downloading:

```bash
bash experiments/server_env.sh original python experiments/nominal/checkpoint_sync.py --suite long
bash experiments/server_env.sh original python experiments/nominal/checkpoint_sync.py --suite spatial
df -h .
```

2. Download only matching checkpoint files. These commands can transfer roughly 16 GB of OFT files plus the requested RIPT heads/adapters per suite. Do not launch them while storage is insufficient:

```bash
bash experiments/server_env.sh original python -u experiments/nominal/checkpoint_sync.py --suite long --execute
bash experiments/server_env.sh original python -u experiments/nominal/checkpoint_sync.py --suite spatial --execute
```

3. CPU component check, then preflight and benchmark import check (repeat `--model sft|scale|ript` as needed):

```bash
bash experiments/server_env.sh original python experiments/nominal/verify_suite_checkpoints.py --suite long
bash experiments/server_env.sh original python experiments/nominal/verify_suite_checkpoints.py --suite spatial
bash experiments/server_env.sh original python experiments/nominal/evaluate.py --suite long --model scale
bash experiments/server_env.sh original python experiments/nominal/evaluate.py --suite spatial --model ript
bash experiments/server_env.sh plus python experiments/libero_plus/evaluate_suites.py --suite long --model scale --check-import
bash experiments/server_env.sh plus python experiments/libero_plus/evaluate_suites.py --suite spatial --model ript --check-import
```

4. Bounded nominal smoke: one task, one trial per model/suite. These result files belong to the same manifest as a later full run, and resume will skip them:

```bash
for suite in long spatial; do
  for model in sft scale ript; do
    bash experiments/server_env.sh original python -u experiments/nominal/evaluate.py --suite "$suite" --model "$model" --seed 1 --task-ids 1 --max-trials 1 --execute
  done
done
```

5. One identical-observation Scale/RIPT action comparison per suite, in isolated processes:

```bash
bash experiments/server_env.sh original python -u experiments/nominal/paired_smoke.py --suite long --seed 1 --task-id 1
bash experiments/server_env.sh original python -u experiments/nominal/paired_smoke.py --suite spatial --seed 1 --task-id 1
```

6. Bounded Plus smoke: one officially registered variant per model/suite:

```bash
for suite in long spatial; do
  for model in sft scale ript; do
    bash experiments/server_env.sh plus python -u experiments/libero_plus/evaluate_suites.py --suite "$suite" --model "$model" --seed 1 --task-ids 1 --execute
  done
done
```

7. Full nominal evaluation, **only after smoke is validated**:

```bash
for suite in long spatial; do
  for model in sft scale ript; do
    bash experiments/server_env.sh original python -u experiments/nominal/evaluate.py --suite "$suite" --model "$model" --seed 1 --execute
  done
done
```

8. Full Plus evaluation, **only after mapping and smoke are validated**:

```bash
for suite in long spatial; do
  for model in sft scale ript; do
    bash experiments/server_env.sh plus python -u experiments/libero_plus/evaluate_suites.py --suite "$suite" --model "$model" --seed 1 --execute
  done
done
```

9. Regenerate Goal's original research report and a separate consolidated Goal/Long/Spatial report:

```bash
bash experiments/server_env.sh original python experiments/libero_plus/research_report.py --all --suites --refresh
```

For a Slurm allocation, select a GPU through Slurm (for example `srun --gres=gpu:1 ...` using the site's partition/account flags), then run the same commands. The wrapper respects `CUDA_VISIBLE_DEVICES`; no physical GPU ID is hardcoded. Long/Spatial full runs are 10 tasks × 50 trials × 3 policies = 1,500 nominal episodes per suite, 3,000 for both. Plus has 2,519 + 2,402 official conditions × 3 policies = 14,763 OOD episodes. The six model/suite combinations are separate runs per benchmark type. Smoke rates are not population estimates.

Outputs: `outputs/libero-nominal/{long,spatial}/{sft,scale,ript}/seed-1/`, `outputs/libero-plus/{long,spatial}/{sft,scale,ript}/seed-1/`, `outputs/libero-paired-smoke/`, `outputs/checkpoint_provenance/`, and `outputs/rlvr_analysis/research_summary_all.{md,json}`. Each evaluator writes a manifest and per-episode JSON; completed records are checked and skipped. The report marks incomplete suites as `INCOMPLETE` and keeps P3-A2/P3-B Goal-specific. Paper percentages appear in a separate labeled table and never count as local measurements.
