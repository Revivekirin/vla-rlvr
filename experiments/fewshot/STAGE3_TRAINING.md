# Stage 3 Goal SFT schedules

`python -m experiments.fewshot.train` accepts Goal `Full`, `20`, `10`, `5`, `2`,
and `1` as independent conditions. Every new revised-profile run initializes from
the same pinned pretrained OpenVLA-7B snapshot. The `--seed` selects the Stage 2
nested episode subset; `--train-seed` controls training randomness. A distinct
`--run-id` is required for each run. The historical 5-shot dataset and the old
batch-one 1000-step run are not used as initializations.
For numeric shots, the default profile is `single_gpu_batch1_pilot`; for Full,
the historical `legacy_batch1` default is preserved. Specify `--profile`
explicitly when comparing conditions so a Full run also uses the fresh pilot.

The one-GPU batch-one pilot has effective batch 1. Its default checkpoints are
1000, 6000, 12000, 24000, and 52000 optimizer steps, truncated at `--max-steps`
with the final step always included. `--checkpoint-steps` overrides this list and
must include the final step. These are diagnostic milestones, not a paper-derived
or proven optimal schedule. The CLI prints frame draws and nominal dataset passes
for each checkpoint before any training starts. Omitting `--execute` is a dry run.

Example, for planning only:

```bash
python -m experiments.fewshot.train \
  --suite goal --seed 42 --train-seed 7 --shots 20 \
  --profile single_gpu_batch1_pilot --max-steps 25218 \
  --checkpoint-steps 1000 6000 12000 25218 \
  --run-id goal-20shot-seed42-b1-example
```

The `single_gpu_global64` profile uses one GPU, microbatch 1, accumulation 64,
and requires a successful one-update smoke result for the exact shot condition.
It matches the official example's sample count per optimizer update, but does
not reproduce eight-GPU DDP numerics or elapsed time. On the available A5000,
the smoke took 55 seconds for one update, so a 50k-step run is a multi-week job.

The official OpenVLA-OFT LIBERO recipe (`runtime/openvla-oft/LIBERO.md`) uses
8 GPUs x batch 8, learning rate 5e-4, 150005 maximum steps, and reports Goal's
50k checkpoint as its best. The RIPT-VLA OpenVLA-OFT README uses the released
official OFT SFT model rather than defining a few-shot OFT SFT optimizer budget.
Its QueST SFT script is a different architecture: `train_prior_goal.yaml` sets
200 epochs, `train_base.yaml` sets batch 128, accumulation 1 and a 10-epoch
save interval. Its RL/PPO accumulation 4 is not an SFT hyperparameter.

SimpleVLA-RL (arXiv:2509.09674v1, Sections 4.1 and 5.1) compares one complete
demonstration per task with its full-data setting. Its LIBERO SFT model changes
the image/proprio inputs and uses discrete action tokens with cross-entropy,
whereas this repo retains dual-camera continuous-action OFT with L1 loss. The
paper says its SFT follows official data/hyperparameters but does not specify
a separate one-trajectory optimizer-step or checkpoint schedule. Its reported
batch 64 and learning rate 5e-6 in Section 4.1 describe RL, not this SFT run.

For a fixed batch-one 52000-step schedule, Goal Full (52042 transitions) gets
approximately one dataset-sized volume of frame draws, while 1-shot seed 42
(1265 transitions) gets about 41.1 such volumes. These are not unique-frame
coverage or guaranteed success. The 2026-11-05 Goal 20-shot pilot was started
first to allow same-day bounded nominal evaluation; its recorded configuration
is in `GOAL_20SHOT_RUN_2026-11-05.md`. No other condition is launched automatically.

If a same-condition batch-one pilot is interrupted after a complete checkpoint,
`--resume` can start a **new** run ID from that checkpoint, restoring its model,
optimizer and scheduler. The checkpoint and dataset identities are verified first.
The RLDS shuffle/prefetch iterator is not restored, so this is not bitwise
equivalent to an uninterrupted run; the new run records that limitation. Never
use a checkpoint from a different shot, seed, profile, batch or base model.

Set `FEWSHOT_SFT_OUTPUT_ROOT` to an absolute directory to place new Stage 3
runs, exports and evaluation under that directory. Keep the same value for
training, export and evaluation commands. This leaves the Stage 2 RLDS and any
old checkpoint at their original paths; `--resume` may read an old checkpoint
from the repository while the new run writes elsewhere.

The first 2026-11-05 resume attempt exposed a PyTorch 2.2 serialization
compatibility issue: its weights-only loader cannot read the old
`MultiStepLR.milestones` `collections.Counter`. The worker now checks the
checkpoint hash first and permits a compatibility load only for that known
legacy error. New optimizer checkpoints serialize the milestones as a plain
dictionary, so their optimizer state remains weights-only loadable.
