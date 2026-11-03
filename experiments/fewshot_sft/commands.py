"""Print commands only. Never launches training or evaluation."""
COMMANDS = r'''# Run from the repository root. On the server:
# For the existing completed step-200 pilot, use CONTINUATION.md instead of rerunning preparation/smoke.
# cd /nfs4/jhkim/repos/vla-rlvr
# Local exact-path check is fast; optional fetch is pinned to the official Goal dataset only.
python -m experiments.fewshot_sft.acquire
# If absent and storage is sufficient, explicitly fetch just Goal (~1.84 GB):
# python -m experiments.fewshot_sft.acquire --execute
# The explicit Goal-only fetch writes this exact path; change it only if a different
# verified local path was printed above. No recursive NFS search is needed.
RLDS_SOURCE=data/modified_libero_rlds/libero_goal_no_noops/1.0.0
DATASET_REVISION=a7c9ae18499b6eea8a32f78a9302327b752b1b5f

# 1. Local audit, then read-only source dataset audit
python -m experiments.fewshot_sft.audit --save
bash experiments/server_env.sh original python -m experiments.fewshot_sft.prepare_dataset --source "${RLDS_SOURCE:?Set verified local TFDS version path}" --source-revision "${DATASET_REVISION:?Set dataset commit}" --shots 5 --selection-seed 42 --dry-run

# 2. Prepare exactly 50 complete demonstrations, once
bash experiments/server_env.sh original python -m experiments.fewshot_sft.prepare_dataset --source "${RLDS_SOURCE:?}" --source-revision "${DATASET_REVISION:?}" --shots 5 --selection-seed 42

# 3. CPU tests; record skips as well as passes
python -m experiments.fewshot_sft.cpu_tests

# GPU selection outside Slurm only: export CUDA_VISIBLE_DEVICES=0
# Under Slurm, preserve its CUDA_VISIBLE_DEVICES. Prefix GPU commands with:
# srun --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=8
# Request the site's diagnostic GPU allocation for smoke; no VRAM estimate exists yet.

# 4. One optimizer step, checkpoint tensor reload, then merge/inference and one rollout
bash experiments/server_env.sh original python -m experiments.fewshot_sft.train --suite goal --shots 5 --selection-seed 42 --train-seed 7 --batch-size 1 --learning-rate 5e-4 --max-steps 1 --save-freq 1 --run-id goal5-smoke --smoke --execute
bash experiments/server_env.sh original python -m experiments.fewshot_sft.checkpoint --checkpoint outputs/fewshot_sft/runs/goal5-smoke--1_chkpt --output outputs/fewshot_sft/exports/goal5-smoke-step1 --execute
bash experiments/server_env.sh original python -m experiments.fewshot_sft.evaluate --checkpoint outputs/fewshot_sft/exports/goal5-smoke-step1 --phase smoke --selection-note 'One-step pipeline validation; not checkpoint selection' --execute
# If BF16 merging fails its numerical check, use an unmerged adapter export at a NEW path:
# bash experiments/server_env.sh original python -m experiments.fewshot_sft.checkpoint --checkpoint outputs/fewshot_sft/runs/goal5-smoke--1_chkpt --output outputs/fewshot_sft/exports/goal5-smoke-step1-adapter --export-unmerged --execute
# bash experiments/server_env.sh original python -m experiments.fewshot_sft.evaluate --checkpoint outputs/fewshot_sft/exports/goal5-smoke-step1-adapter --phase smoke --selection-note 'One-step adapter-backed pipeline validation; BF16 merge rejected' --execute

# 5. Bounded SFT: 200 actual optimizer steps, checkpoints at 50/100/150/200
bash experiments/server_env.sh original python -m experiments.fewshot_sft.train --suite goal --shots 5 --selection-seed 42 --train-seed 7 --batch-size 1 --learning-rate 5e-4 --max-steps 200 --save-freq 50 --run-id goal5-pilot --smoke-evidence outputs/fewshot_sft/runs/goal5-smoke/runtime.json --execute

# 6. Export a predeclared checkpoint with verified unmerged adapter reload
bash experiments/server_env.sh original python -m experiments.fewshot_sft.checkpoint --checkpoint outputs/fewshot_sft/runs/goal5-pilot--200_chkpt --output outputs/fewshot_sft/exports/goal5-pilot-step200-adapter --export-unmerged --execute

# 7. Small diagnostic: 10 tasks x 2 trials, plus released Full-SFT control (10 x 1)
bash experiments/server_env.sh original python -m experiments.fewshot_sft.evaluate --checkpoint outputs/fewshot_sft/exports/goal5-pilot-step200-adapter --phase diagnostic --selection-note 'Predeclared step 200 diagnostic; nominal model selection' --execute
bash experiments/server_env.sh original python -m experiments.fewshot_sft.evaluate --checkpoint checkpoints/goal/oft --full-sft --phase diagnostic --trials-per-task 1 --run-label control --selection-note 'Full-SFT same-evaluator control' --execute

# 8. Expand only with positive, complete small-evaluation evidence for this exact model.
# Zero-success diagnostics are blocked. See CONTINUATION.md for the 1000/3000/6000 schedule.
bash experiments/server_env.sh original python -m experiments.fewshot_sft.evaluate --checkpoint outputs/fewshot_sft/exports/goal5-pilot-step200-adapter --phase pilot --gate-evidence outputs/fewshot_sft/eval/goal_5shot_seed42/goal5-pilot-step200-adapter/diagnostic/summary.json --full-sft-evidence outputs/fewshot_sft/eval/full_sft_goal/oft/diagnostic-control/summary.json --selection-note 'Expansion after nominal diagnostic and Full-SFT control; nominal used in selection' --execute

# 9. Read-only aggregation, or refresh the two derived summaries
python -m experiments.fewshot_sft.report
python -m experiments.fewshot_sft.report --write
'''

if __name__ == '__main__':
    print(COMMANDS)
