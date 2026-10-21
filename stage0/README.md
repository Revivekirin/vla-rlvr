# Stage 0 — official OpenVLA-OFT / RIPT-VLA integration

**Status: PARTIAL.** Audited 2026-10-21. The official implementation and all released checkpoint components are identified and pinned. Genuine Goal action/scale heads load strictly and run bounded CPU forward passes. Full 7B loading, adapter application, Linux dependency installation, CUDA inference, and simulator execution remain unverified. No full LIBERO evaluation or RL training was run.

## What is present

- `repos.lock.json`: immutable source revisions; `models.lock.json`: immutable Hub revisions, filenames, sizes, Git blob IDs and LFS SHA256s.
- `evidence/`: original Hub responses, host probe, component results, artifact integrity checks, and static checks.
- `scripts/prepare.py`: validates clean pinned checkouts, makes separate runtime copies, applies explicit patches, and writes a workspace-local LIBERO configuration.
- `scripts/download.py`: revision-pinned downloads with size and hash checks; metadata, components, or full model selection.
- `scripts/install_linux.sh` and `constraints.txt`: candidate Linux/CUDA environment. This is **not a validated, complete transitive dependency lock**. A successful install must pass `pip check` and save the resolved `linux-pip-freeze.txt` before promotion.
- `scripts/run.py`: dry-run by default; bounded GPU smoke or explicit Stage 1 evaluation launch. No training launch is supplied.

Upstream source remains untouched under `../vendor/`; patched copies are under `../runtime/`. Genuine downloaded files are under `../checkpoints/`. The root `.gitignore` excludes these large/generated directories. Preserve `stage0/` to reproduce them.

## Source pins

| Repository | Revision |
|---|---|
| [RIPT-VLA](https://github.com/Ariostgx/ript-vla) | `440990e8864e12e4578b490ff6359e4f2c49ae3e` |
| [OpenVLA-OFT](https://github.com/moojink/openvla-oft) | `e4287e94541f459edc4feabc4e181f537cd569a8` |
| [Original LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| [LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) | `4976dc30028e805ff8094b55501d532c48fec182` — reference only, not installed or substituted |
| [OFT Transformers fork](https://github.com/moojink/transformers-openvla-oft) | `bc339d9ad707454c0c115970db43c260067c61ab` |
| [OFT dlimp fork](https://github.com/moojink/dlimp_openvla) | `040105d256bd28866cc6620621a3d5f7b6b91b46` |

These are audit-time source pins, not a claim that the authors trained at these Git commits. The two dependency forks are pinned in the runtime `pyproject.toml`. LIBERO assets and 130 `.pruned_init` files are present in its checkout; runtime asset/rendering validation is pending.

## Exact official checkpoint mapping

The [RIPT README](https://github.com/Ariostgx/ript-vla/blob/440990e8864e12e4578b490ff6359e4f2c49ae3e/README.md) explicitly directs users to the suite-specific full models in [OFT LIBERO.md](https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/LIBERO.md). These are the individually trained models, not the combined four-suite policy.

| Suite | Official full-model Hub repository | Pinned revision | OFT component step / normalization key |
|---|---|---|---|
| Goal | `moojink/openvla-7b-oft-finetuned-libero-goal` | `c2d0f9fbbd82674683b397ff923168a12f6a307b` | 50000 / `libero_goal_no_noops` |
| Long | `moojink/openvla-7b-oft-finetuned-libero-10` | `db188c5eda473a9b81a603759b3444364a887b98` | 150000 / `libero_10_no_noops` |
| Object | `moojink/openvla-7b-oft-finetuned-libero-object` | `4c89574e1c538b6c102f43f0526d60a9d3650148` | 150000 / `libero_object_no_noops` |
| Spatial | `moojink/openvla-7b-oft-finetuned-libero-spatial` | `6d0231af0e48c5985f1ff86908f4674b84bc049b` | 150000 / `libero_spatial_no_noops` |

Long's released adapter names the `db188c...` snapshot explicitly. The audit-time Long main revision `95220f9a3421a7ff12d4218e73d09ade830fa9a3` differs only by an added README; all shared file metadata/hashes are identical. The historical snapshot is selected for execution. The other adapter metadata identifies the correct base repository through the author's local cache path but does not encode an exact historical base commit. Their selected immutable release revisions are recorded above; historical training provenance beyond this is not asserted.

Each OFT directory needs all four `model-0000*-of-00004.safetensors` shards and the index, `action_head--STEP_checkpoint.pt`, `proprio_projector--STEP_checkpoint.pt`, config, processor/tokenizer files, and `dataset_statistics.json`. The original OFT SFT LoRA under `lora_adapter/` is already merged into the full model and is deliberately excluded from downloads to avoid double application.

All RIPT artifacts come from [tanshh97/RIPT_VLA](https://huggingface.co/tanshh97/RIPT_VLA/tree/57532f4abbf81b89a8ff6a642a996fc54b6a6a10), pinned at `57532f4abbf81b89a8ff6a642a996fc54b6a6a10`:

| Suite | SFT scale head | RIPT adapter directory |
|---|---|---|
| Goal | `openvla_oft/scale_header/LIBERO_GOAL_scale_header.pth` | `openvla_oft/ript_adaptors/LIBERO_GOAL_lora/` |
| Long | `openvla_oft/scale_header/LIBERO_LONG_scale_header.pth` | `openvla_oft/ript_adaptors/LIBERO_LONG_lora/` |
| Object | `openvla_oft/scale_header/LIBERO_OBJECT_scale_header.pth` | `openvla_oft/ript_adaptors/LIBERO_OBJECT_lora/` |
| Spatial | `openvla_oft/scale_header/LIBERO_SPATIAL_scale_header.pth` | `openvla_oft/ript_adaptors/LIBERO_SPATIAL_lora/` |

Each adapter directory must include **`adapter_config.json`, `adapter_model.safetensors`, and `openvla_headers.pt`**. The last file contains `action_header` and `scale_header` and is essential. All four released configs specify rank 32, alpha 16, dropout 0.0. Author-specific absolute base paths are metadata; the official policy constructor receives the actual local OFT full-model path explicitly.

## Three configurations and loading semantics

| Mode | Execution path | Components / current verification |
|---|---|---|
| `sft` | OFT `experiments/robot/libero/run_libero_eval.py` | Full model + original OFT action head + proprio projector + stats/processor. No RIPT scale head or RL adapter. Goal action head CPU-tested; full policy unverified. |
| `scale` | RIPT eval entry point, `lora_adaptor_ckpt=null` | Same OFT components + matching SFT scale head. RIPT internally adds a fresh Gaussian-initialized PEFT wrapper with zero initial LoRA update; no released RL weights are loaded. Goal heads CPU-tested; full policy unverified. |
| `ript` | RIPT eval entry point, adapter directory supplied | Loads OFT and SFT scale head, then applies adapter and **overwrites action and scale heads from `openvla_headers.pt`**. Goal replacement heads CPU-tested and adapter header inspected; 7B adapter attachment unverified. |

`header_checkpoint=null` causes random scale-head initialization; it is not the intended released SFT-scale configuration. `fix_scale_head=True` excludes its parameters from the header optimizer but does not call `requires_grad_(False)`.

Source: `ript/algos/rl_optimizers/openvla_oft_interface.py:281–383`; OFT `experiments/robot/openvla_utils.py:253–498`. The full model loads in BF16; action/proprio heads are loaded separately. Local OFT loading rewrites `config.json` auto-mapping and may synchronize modeling code from the checkout. Therefore commands run from the pinned OFT root. Downloads are integrity-checked before loading; local code/config changes during loading are expected upstream behavior and should be retained in run records. This is not byte-for-byte execution of arbitrary Hub modeling code.

## Prediction, normalization, reward, and optimization audit

**Action prediction.** Two images (front and wrist), 224-pixel preprocessing with center crop, and eight-dimensional proprioception feed parallel prediction of eight seven-dimensional actions. RIPT uses a prompt/vision forward pass without gradients and cached keys/values, then an action-token pass with gradients. It patches `LlamaSdpaAttention` for the longer attention mask. This requires the pinned bidirectional OFT Transformers fork. The action head predicts continuous normalized means; the scale head predicts log scales. Evaluation explicitly forces `use_laplace_sampling=False` and executes the mean with eight-step open-loop action queues. SFT-only and SFT+scale are expected to share the underlying mean weights, but their different forward implementations have **not** been shown numerically equivalent here.

**Distribution.** Training rollouts sample independent Laplace actions with `b = exp(log_b) * scale_factor`. `laplace_log_prob` computes `-log_b - |a-mu|/exp(log_b) - log(2)`, after optional log-scale clipping. Critically, sampling uses unclipped scales, while stored/recomputed log probabilities clip and omit `scale_factor`. The released defaults use clipping `[-2, 0.5]` and exploration scale factors above one. Thus these log probabilities do not describe the actual sampling distribution. The CPU check verifies the formula only in the unclipped, unit-scale case; it does not validate PPO correctness for released exploration settings. No algorithm change was made.

**Normalization.** All four downloaded statistics files have the expected suite key, seven action dimensions and eight proprio dimensions. Proprio uses masked q01/q99 scaling and clipping to [-1,1]. Actions use masked q01/q99 inverse scaling; this inverse does not itself clamp sampled actions. The gripper is binarized, mapped from [0,1] to [-1,1], then sign-inverted for LIBERO. Long maps to `libero_10`, not `libero_long` in the benchmark/statistics API.

**Environment and rewards.** Original LIBERO fixed `.pruned_init` states are used for evaluation, with environment seed 0 and ten settling steps. Suite action horizons are Spatial 220, Object 280, Goal 300, Long 520. The runner accumulates success from `done`; Original LIBERO's `bddl_base_domain.py:800–808` sets `done` from `_check_success()`. `SuccessReward` returns 1 for success and 0 otherwise. The OFT runner leaves its `total_reward` accumulator at zero, so its reported average environment reward is not a useful return metric; success rate is the relevant metric.

**RLOO/PPO.** The suite configs use K=8 rollouts, dynamic sampling, one PPO epoch, 0.1 clipping, action-dimension log-prob summation, gradient accumulation 4, and chunk batch size 1. RLOO computes each rollout's baseline from the other K−1 rewards of that initialization; advantage is reward minus baseline. PPO exponentiates new-minus-old action log probabilities, applies the clipped surrogate, normalizes over chunks/actions and accumulation, and updates AdamW groups. No value model or explicit reference-policy KL penalty appears in this OFT optimizer. Dynamic sampling filters uniform-success groups. Further unresolved training issues: `valid_mask` is used for metrics but not to exclude padded episodes from the loss; the zero-accepted-rollout fallback uses QueST-shaped episode fields; non-divisible accumulation has no final flush; increasing `max_step_batch_size` can skip the last chunk because of the exclusive end index. These paths were inspected, not trained.

**Trainable scope.** PEFT targets all linear layers, rank 32/alpha 16, with the backbone's non-adapter parameters frozen by PEFT. The optimizer receives `requires_grad` model parameters and the action head; the scale head joins only when not fixed. The proprio projector is excluded from optimizers. Vision/proprio computations and prompt caching run under `no_grad`, so some nominally trainable adapters do not receive gradients through those branches. Actual adapter trainability after `load_adapter`, missing/unexpected key handling, and per-parameter gradients must still be checked on the target runtime. The Goal safetensors header contains 879 BF16 tensors, including an `lm_head.base_layer.weight` as well as LoRA tensors: the artifact is not exclusively low-rank matrices.

Sources: `ript/env_runner/openvla_oft_libero_runner.py`, `ript/algos/rl_optimizers/rl_optimizer_openvla_oft.py`, `rollout_generator.py`, `ript/algos/reward_functions/libero.py`, OFT `prismatic/extern/hf/modeling_prismatic.py` and `experiments/robot/openvla_utils.py` at the pinned commits.

## Official scripts inspected and local patches

All four scripts under each of `scripts/openvla_oft/eval/` and `scripts/openvla_oft/stage_3_ript/` were read and passed `bash -n`.

- Eval scripts call `torchrun train_ript_openvla_oft.py` with the suite config, `algo.eval_only=True`, and three checkpoint placeholders. The eval runner disables sampling regardless of training config. Its default is 50 trials per task.
- Training scripts set 12 steps, batch size 24, two parallel environments, and exploration scale 5 (Spatial 2). Long/Object restrict training to four named tasks; Goal/Spatial use all suite tasks. They contain duplicate `algo.model_seed` overrides. No training script was executed.
- **Eval blocker:** after eval, upstream constructs `iter(train_dataloader)` although the dataloader is created only in the training branch. `patches/eval-only-exit.patch` adds synchronized cleanup and return. A regression check executes the real patched control-flow suffix with a stub runner and verifies evaluation, aggregation, cleanup, and exit without touching training state.
- **Training blocker, unpatched:** `train_ript_openvla_oft.py:189` passes `enable_rollout_stats_tracking=True` to `RLOptimizerOpenVLAOFT`, whose constructor does not accept it. Training is not ready to launch without a separate fix/review.
- **Dependency conflict:** RIPT pins `diffusers==0.28.0`, OFT pins `0.30.3`. The runtime copy reconciles to OFT's `0.30.3`; this is an explicit candidate choice for L1 evaluation, not a validated statement about every RIPT/QueST path. The patch is saved.
- Two more patches pin the previously floating Transformers/dlimp Git dependencies. No reward, sampling, head-loading, or PPO logic was patched.

## Hardware and installation feasibility

The observed host is macOS arm64, with no `nvidia-smi`, no CUDA devices, and no CUDA-capable PyTorch. An existing Python 3.11 / Torch 2.14 environment was used read-only for isolated CPU component checks; it is **not** the official execution environment. The released trainer initializes NCCL unconditionally and uses CUDA device IDs. CPU/MPS execution is not a verified substitute.

The candidate target uses Python 3.10, Torch 2.2.0 / torchvision 0.17.0 / torchaudio 2.2.0 with CUDA 12.1, the custom Transformers fork, PEFT 0.11.1, Accelerate 1.6.0, and RIPT's requested FlashAttention 2.7.4.post1. OFT's setup instead lists FlashAttention 2.5.5; its current model loader uses SDPA rather than explicitly requesting FlashAttention. This mismatch and CUDA extension build compatibility remain untested. Keep the custom SDPA implementation; do not silently replace it with upstream Transformers.

Each suite's full OFT download is approximately **14.39 GiB**, excluding the redundant SFT LoRA. The RIPT scale head is ~0.313 GiB, adapter ~0.451 GiB, and replacement headers ~0.594 GiB. Four complete suite packages require roughly 63 GiB before extra caches/runtime files. Current local downloads contain metadata for all four and genuine Goal components, but **no full-model shards**.

BF16 base weights alone occupy roughly 14 GiB before extra heads, adapters, activations, caches, and simulator memory. A 16 GB GPU is not a credible budget for this full released path. A 24 GB device might fit batch-one inference, but this is unmeasured; a 40/80 GB A100 provides more margin. OFT reports ~25 GB for supervised batch-one training and ~62 GB for batch-eight; these are not measurements of RIPT training. RIPT recommends four GPUs for training. Stage 1 commands here use one GPU to avoid distributed device-remapping and cross-rank evaluation confounds. No VRAM-fit or throughput claim has been experimentally established.

## Reproduction and Stage 1 commands

Run from the project root. For a different Linux machine, transfer `stage0/` and optionally `checkpoints/` into a new project directory; **do not transfer the generated runtime config with its old absolute paths**. `prepare.py` will clone exact source revisions and create new local paths. It refuses to overwrite an existing runtime.

```bash
python3 stage0/scripts/prepare.py  # Already completed on this Mac; run once in a fresh directory.
conda create -n ript_vla_openvla_oft_stage0 python=3.10.14 -y
conda activate ript_vla_openvla_oft_stage0
bash stage0/scripts/install_linux.sh
```

Prerequisites: Linux NVIDIA driver, BF16-capable GPU, CUDA toolkit with `nvcc`, compiler/build tools, and working EGL/OpenGL system libraries. Installation may still require platform-specific adjustments; capture any changes and the successful resolved package freeze. Do not claim PASS merely because installation finishes.

Download only the selected suite first:

```bash
python stage0/scripts/download.py --suite goal --level full
python stage0/scripts/run.py smoke --suite goal --mode sft --execute --timeout 1800
python stage0/scripts/run.py smoke --suite goal --mode scale --execute --timeout 1800
python stage0/scripts/run.py smoke --suite goal --mode ript --execute --timeout 1800
```

Smoke performs one synthetic-observation action query for SFT and two for RIPT modes (mean and sampled), validates shape/finite outputs, and records peak GPU memory. It does not reset or step a simulator or optimize parameters. The launcher stops its process group at the timeout. Repeat these three checks for `long`, `object`, and `spatial` after downloading each full suite. End-to-end SFT-vs-scale numerical parity, strict PEFT key reconciliation, and a bounded real-environment rendering/step smoke remain follow-up checks before a robust evaluation claim.

**Stage 1 only — not executed during Stage 0:** each command below launches 10 tasks × 50 trials on Original LIBERO, seed 1. The launcher always defaults to printing its command; `--execute` explicitly runs it. Preserve logs in `stage0/evidence/` and official output files under `outputs/`. A timeout/incomplete run must not be reported as a completed suite.

```bash
for suite in goal long object spatial; do
  python stage0/scripts/download.py --suite "$suite" --level full
  for mode in sft scale ript; do
    python stage0/scripts/run.py eval --suite "$suite" --mode "$mode" --execute --timeout 86400
  done
done
```

The commands force the correct `PYTHONPATH`, local LIBERO configuration, EGL, offline model loading, disabled WandB, a pinned local checkpoint path, and `eval_only=True` for both RIPT modes. Evaluation does not need demonstration HDF5/RLDS datasets; training would. No LIBERO-plus code is imported. For paper-level comparison, repeat with a documented seed protocol and common hardware; the one-seed runs above are a reproducible first evaluation, not the authors' three-seed aggregate.

## Executed evidence and remaining gates

| Check | Result |
|---|---|
| Source revision inventory / clean upstream checkouts | Verified |
| All four official base/scale/adapter mappings and immutable inventories | Verified |
| All four normalization keys and action/proprio dimensions | Verified |
| 72 downloaded artifact hashes | Verified (`evidence/local_audit.json`) |
| Goal SFT action + scale weights, RIPT replacement action + scale weights | Strict CPU state-dict loads passed |
| Bounded head forward on synthetic hidden states | Finite `[1,8,7]` for both head pairs |
| Unit-scale unclipped Laplace log-prob formula | Matches `torch.distributions.Laplace` |
| Released Goal LoRA file | SHA256 verified, 879 tensor headers inspected; not attached to 7B model |
| Script syntax / patched eval exit | Passed static and stubbed regression checks |
| Linux package resolution / `pip check` / FlashAttention build | Not run |
| Full-model loading and strict adapter reconciliation | Not run |
| CUDA action inference / measured VRAM / simulator rendering | Not run |
| Full evaluation / RL training | Intentionally not run |

To reproduce the checks already run:

```bash
python3 stage0/scripts/audit_local.py
# Use a CPU PyTorch environment; source class extraction avoids all simulator imports.
python stage0/scripts/check_components.py
```

Stage 0 remains **PARTIAL** until target dependencies, complete checkpoint loading, and bounded genuine-policy GPU inference are verified. Additional training defects are recorded as a separate unresolved boundary; Stage 0 does not silently change the released algorithm to make training appear validated.
