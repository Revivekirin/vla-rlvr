"""Print commands only. Never launches training or evaluation."""
COMMANDS = r'''# Run from the repository root. On the server:
# cd /nfs4/jhkim/repos/vla-rlvr
# RLDS_SOURCE must be an audited LOCAL TFDS version directory containing dataset_info.json.
# DATASET_REVISION must be the actual source HF dataset commit (no automatic download).
# These two values cannot be inferred on this Mac because the source data are absent.

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

# 5. Bounded SFT: 200 actual optimizer steps, checkpoints at 50/100/150/200
bash experiments/server_env.sh original python -m experiments.fewshot_sft.train --suite goal --shots 5 --selection-seed 42 --train-seed 7 --batch-size 1 --learning-rate 5e-4 --max-steps 200 --save-freq 50 --run-id goal5-pilot --smoke-evidence outputs/fewshot_sft/runs/goal5-smoke/runtime.json --execute

# 6. Export a predeclared checkpoint; LoRA merge plus fixed-input numerical verification
bash experiments/server_env.sh original python -m experiments.fewshot_sft.checkpoint --checkpoint outputs/fewshot_sft/runs/goal5-pilot--200_chkpt --output outputs/fewshot_sft/exports/goal5-pilot-step200 --execute

# 7. Pilot: 10 tasks x 20 trials
bash experiments/server_env.sh original python -m experiments.fewshot_sft.evaluate --checkpoint outputs/fewshot_sft/exports/goal5-pilot-step200 --phase pilot --selection-note 'Predeclared step 200; no prior nominal checkpoint tuning' --execute

# 8. Final: 10 tasks x 50 trials, explicit opt-in only
# Amend this note if checkpoint choice used pilot results; all notes remain in the report.
bash experiments/server_env.sh original python -m experiments.fewshot_sft.evaluate --checkpoint outputs/fewshot_sft/exports/goal5-pilot-step200 --phase final --selection-note 'Step 200 selected before pilot; pilot reviewed, no checkpoint retuning' --execute

# 9. Read-only aggregation, or refresh the two derived summaries
python -m experiments.fewshot_sft.report
python -m experiments.fewshot_sft.report --write
'''

if __name__ == '__main__':
    print(COMMANDS)
