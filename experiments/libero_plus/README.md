# Goal checkpoints on LIBERO-Plus

This integration evaluates the three existing **Goal** checkpoints on the official
2,591-task Goal expansion, not the entire four-suite 10,030-task benchmark.
Original LIBERO and its launchers/configuration are unchanged. No model is trained.

- Code pin: `4976dc30028e805ff8094b55501d532c48fec182`.
- Official protocol: one rollout per task, first official initial state, canonical
  task order. Classification IDs are **one-based**; suite indices are zero-based.
- Shift counts: Background 281, Robot 409, Camera 408, Language 410,
  Sensor Noise 379, Layout 425, Light 279.
- A process-local import path selects `vendor/LIBERO-plus`; its configuration is
  written only to `runtime/libero-plus-config`. No `pip install -e` or uninstall of
  Original LIBERO is necessary.
- All modes use environment seed 0, policy seed 1 (reset before each task for
  reproducible resume), 256px render, the existing 180-degree image rotation,
  224px resize and center crop, wrist image and proprioception, 10 settling steps,
  eight-action chunks, 300 policy steps, and `done` as success.
- Existing released checkpoint paths are fixed by model selection; there is no
  mix-SFT checkpoint argument. RIPT loads its adapter plus `openvla_headers.pt`.
  Modules are in eval mode with `requires_grad=False` and inference mode enabled.
  Existing loader config/code synchronization is disabled in this process so the
  prepared checkpoint directory is not rewritten. No optimizer is instantiated.
- Each completed task has an atomic, flushed JSON result recording official ID,
  name, category, difficulty, model, seed, success and failure. An interrupted task
  is retried; completed tasks are skipped. Run the same command to resume. A lock
  prevents simultaneous writers. A manifest rejects changed evaluation settings
  or wrapper code. Keep runtime source, checkpoints and assets unchanged during a
  run/resume; the manifest does not hash multi-GB model/asset contents.
- Errors abort and remain pending rather than entering the denominator as failures.
  A completed horizon without success is a failure. No videos are generated.
- Aggregation uses actual episode counts, reports partial coverage, withholds drop
  until full completion, and only compares OOD models with identical manifest
  conditions and task ID sets. Nominal/OOD tasks are not paired.

Sources: [official pinned README/protocol](https://github.com/sylvestf/LIBERO-plus/blob/4976dc30028e805ff8094b55501d532c48fec182/README.md),
[official classification](https://github.com/sylvestf/LIBERO-plus/blob/4976dc30028e805ff8094b55501d532c48fec182/libero/libero/benchmark/task_classification.json),
[official assets](https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/dd2bd61b7d9a6fef1abc52d606e983b41886a149).

## Nominal evidence and comparison limits

`../nominal/goal.csv` is the single nominal table, with checkpoint paths, seed,
counts, per-task counts/rates, and raw log paths. SFT: 486/500 (97.2%);
Scale: 488/500 (97.6%); RIPT: 488/500 (97.6%). All ten task denominators are 50.
SFT seed 1 comes from `stage0/scripts/run.py`; its raw log does not print the seed.
Scale/RIPT logs explicitly print model seed 1. Raw evidence is unchanged.

The inspected nominal paths agree on task order, first 50 initial states, env
seed 0, preprocessing, chunk size 8, 300-step Goal horizon plus 10 settling steps,
and success via `done`. They differ in inference: SFT uses OFT's single-sample
`predict_action`; Scale/RIPT use RIPT's cached two-pass batch forward and SDPA
patch. OOD keeps these inference paths, but shares the environment loop and
aggregation. Numerical equivalence has not been tested. Nominal SFT also catches
rollout exceptions and counts them as failures; RIPT propagates exceptions. The
new wrapper propagates exceptions consistently for all modes.

Nominal RIPT `eval_only` explicitly disables Laplace sampling. The new wrapper
also disables it. Thus the scale head is loaded/frozen, but its predicted scale
has no effect on executed mean actions. The nominal SFT→Scale +0.4 pp must **not**
be interpreted as evidence that the scale head improved the deterministic policy.
OOD drop remains descriptive because nominal and OOD task populations differ.

## Minimal local validation

Completed: Python syntax compilation; two focused tests on actual nominal-log
counts and aggregation/matched coverage (including duplicate and category
rejection); all three CLI dry-runs; simulator-free matching of all 2,591 task
names/IDs against the two downloaded official files at the pinned revision.

No GPU model load, simulator initialization, rollout, training, original benchmark
rerun, large asset download, or server connection was performed. Full benchmark
imports cannot be validated in this local environment without the server runtime
and Plus assets. Run the import command below on server4. Actual OOD rates and
cross-inference numerical equivalence remain unverified.

## Exact deployment and server4 commands

Run on the local Mac (only the new `experiments` files are transferred):

```bash
cd /Users/jihyekim/Desktop/vla-rlvr
rsync -av --exclude='__pycache__/' -e 'ssh -p 52675' experiments/ jhkim@10.20.22.83:/nfs4/jhkim/repos/vla-rlvr/experiments/
ssh -p 52675 jhkim@10.20.22.83
```

Run in that server shell. This setup downloads official **benchmark assets**
(~6.4 GB), not any training data or model weights. It is provided for execution
by the user and was not run by this task.

```bash
set -euo pipefail
source /home/jhkim/miniconda3/etc/profile.d/conda.sh
conda activate ript_vla_openvla_oft_stage0
cd /nfs4/jhkim/repos/vla-rlvr

if [ ! -d vendor/LIBERO-plus ]; then
  git init vendor/LIBERO-plus
  git -C vendor/LIBERO-plus remote add origin https://github.com/sylvestf/LIBERO-plus.git
  git -C vendor/LIBERO-plus fetch --depth 1 origin 4976dc30028e805ff8094b55501d532c48fec182
  git -C vendor/LIBERO-plus checkout --detach FETCH_HEAD
fi
test "$(git -C vendor/LIBERO-plus rev-parse HEAD)" = 4976dc30028e805ff8094b55501d532c48fec182

python -m pip install -r vendor/LIBERO-plus/extra_requirements.txt
python -c 'from wand.image import Image; import skimage'
mkdir -p data/libero-plus
curl -L --fail --retry 3 -C - https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/dd2bd61b7d9a6fef1abc52d606e983b41886a149/assets.zip -o data/libero-plus/assets.zip
unzip -n -q data/libero-plus/assets.zip -d vendor/LIBERO-plus/libero/libero

# Imports the selected benchmark and verifies every task ID/name/category.
# Does not instantiate a simulator or load a checkpoint.
python experiments/libero_plus/evaluate.py --check-import
```

If the `wand.image` import reports a missing ImageMagick shared library, install
that official system prerequisite and repeat the import/setup commands:

```bash
sudo apt-get update
sudo apt-get install -y libmagickwand-dev
```

Three zero-shot evaluations (one GPU, sequential, 2,591 episodes each):

```bash
export CUDA_VISIBLE_DEVICES=0
python -u experiments/libero_plus/evaluate.py --model sft --seed 1 --execute
python -u experiments/libero_plus/evaluate.py --model scale --seed 1 --execute
python -u experiments/libero_plus/evaluate.py --model ript --seed 1 --execute
python experiments/libero_plus/report.py --seed 1
```

Repeat an interrupted evaluation command verbatim to resume. Results live under
`outputs/libero-plus/{sft,scale,ript}/seed-1/`. Aggregation writes
`experiments/nominal/goal.csv` and `experiments/libero_plus/comparison.md` and can
also be run during an incomplete evaluation. Raw logs and task results are never
rewritten by aggregation.
