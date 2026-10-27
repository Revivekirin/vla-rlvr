# SFT versus Scale deterministic-action investigation

## Confirmed from the checked-in source

The evaluator selects identical **paths and loading functions** for the base model,
action head, proprio projector, processor, and normalization statistics:

| Component | SFT and Scale source |
|---|---|
| Base | `checkpoints/goal/oft` via `get_model` / `get_vla` |
| Action head | `action_head--50000_checkpoint.pt` via `get_action_head` |
| Proprio projector | `proprio_projector--50000_checkpoint.pt` via `get_proprio_projector` |
| Statistics | `dataset_statistics.json`; same `libero_goal` / `_no_noops` fallback |
| Precision / images | BF16, no quantization, two images, center crop, proprio enabled |

See `evaluate.py:123–149`, runtime OFT `run_libero_eval.initialize_model`,
`openvla_utils.get_vla/get_action_head/get_proprio_projector`, and RIPT
`openvla_oft_interface.initialize_model`. Both loaders disable checkpoint code/config
sync in the evaluator. Only RIPT-RL supplies the released adapter and replaces the
action head with `openvla_headers.pt`; Scale does neither. Equality of actual loaded
parameter bytes on the evaluation server is **not yet measured**.

Scale wraps the base using PEFT (`r=32`, alpha=16, dropout=0,
`init_lora_weights="gaussian"`, all linear modules). This requests PEFT's zero-B
initialization, hence zero mathematical LoRA update, not a trained RL adapter.
The installed PEFT implementation and actual B tensors are unavailable locally;
zero contribution is expected, not claimed as a completed tensor measurement.
The harness asserts every B tensor is exactly zero and all LoRA tensors are finite.
A zero update alone does not establish whole-path numerical equivalence.

The deterministic branch explicitly assigns `normalized_actions =
normalized_actions_mean`. The action head computes that mean before the separate
scale head runs; log-scale is used for the log probability, not the action.
Both paths use model `_unnormalize_actions` and the evaluator's `process_action`
(gripper binarization and sign inversion). No scale-output dependency exists in
this deterministic action computation.

**The model computations differ structurally.** SFT's
`modeling_prismatic._regression_or_discrete_prediction` performs one full-sequence
language-model forward. RIPT's `_regression_or_discrete_prediction_batch` first
runs only the prompt/vision/proprio prefix with a KV cache, then runs the remaining
action/stop tokens against that cache. Importing `openvla_oft_interface` also
replaces `LlamaSdpaAttention.forward` globally. Its mask repeats the last mask row
and sets `is_causal=False`. Stage 0 pins the bidirectional OFT Transformers fork.

Bidirectional full-sequence inference is generally **not equivalent** to this
prefix-cache split. In a full pass, prefix tokens can attend to action/stop tokens;
in the cached prefix pass those tokens do not exist. Prefix outputs can differ in
the first decoder layer, then action outputs can differ in subsequent layers
because their prefix keys/values differ. This is a semantic difference, not merely
an argument about BF16 rounding. The first structural divergence is at the
language-model call after multimodal assembly; the first numerical divergence on
these checkpoints remains unmeasured. Actual preprocessing equivalence must also
be checked before assigning the observed difference to attention.

Both paths form the same lowercase instruction prompt and use shared image and
proprio normalization helpers. One is a scalar processor call and one a singleton
batch call. SFT also mutates `observation['state']` to its normalized version;
Scale copies it. A naive sequential test reusing the same dictionary would
normalize twice and give a false discrepancy. The harness supplies fresh copies.

## Hypotheses, not established outcome explanations

The bidirectional cache split is a concrete candidate for different action means,
and therefore for some of the 278 reported changed outcomes. Its contribution to
the Language loss (48 lost, 9 gained) cannot be inferred from aggregate rates.
The checked-in report confirms Language successes fall from 251 to 212, but the
per-task outcome JSONs needed to independently recount losses/gains are absent.
Instruction-dependent sensitivity to the changed attention graph is plausible,
but untested. No claim is made that RNG or task initialization caused these losses.

Since action equivalence has not been demonstrated, no environment-cause audit
or rollouts were performed. If native actions match on the representative inputs,
the next bounded investigation should examine only repeated reset/init-state
observations, environment RNG (including sensor-noise draws), the eight-action
queue, and task initialization. Agreement on three inputs would not prove
agreement over every trajectory.

## Bounded reproduction

`compare_actions.py` leaves the evaluator and checkpoints untouched. Use the
existing evaluation environment on the machine with the model shards and Plus
assets. It never trains or executes policy rollouts.

```bash
python experiments/libero_plus/compare_actions.py preflight
# Only if shared observations do not already exist: 3 resets, 10 dummy steps each.
bash experiments/server_env.sh plus python experiments/libero_plus/compare_actions.py capture
bash experiments/server_env.sh plus python experiments/libero_plus/compare_actions.py compare
```

Capture picks one task each from Language Instructions, Light Conditions, and
Sensor Noise, preferring the first SFT-success/Scale-failure condition if original
results are present, otherwise the first condition in the category. Exact task
metadata and language are saved. No task IDs or observations were invented here.
Alternatively pass `--directory PATH` containing one to three shared NPZ files
with `full_image`, `wrist_image`, `state`, and scalar string `instruction` (optional
scalar JSON string `metadata`). Images/state must be the raw **prepared**
observation returned by `oft.prepare_observation(raw, 224)`, before normalization.

Compare loads only one policy at a time in fresh subprocesses, preventing the
Scale import's global patch from contaminating SFT. It hashes canonical loaded
base/head/projector state tensors, records norm stats, package versions, attention
implementation, and LoRA nonzero counts. It runs at most twelve forwards:

- Three native SFT forwards.
- Three native Scale forwards.
- Three full-sequence forwards using the Scale-loaded model and OFT action path.
- Three native Scale forwards replacing log-scale with a constant.

Each input is copied for each forward. Traces include processed token IDs, masks,
pixels, normalized proprio, assembled multimodal embeddings, first two decoder
layer outputs, final action hidden states, normalized means, unnormalized actions,
and all eight final actions exactly as passed to `env.step` (without stepping).
Prefix/action layer outputs are concatenated to align with SFT's sequence.
`report.json` reports the first recorded difference and max absolute and symmetric
relative differences (`abs(a-b)/max(abs(a),abs(b),1e-8)`) for each shared stage.
Relative errors near zero should be read alongside absolute errors.

If native differs but full-sequence control agrees, that isolates the cached
forward path from the loaded weights/LoRA wrapper. If the full control also
differs, inspect the recorded preprocessing, weights and attention implementation
before concluding cache splitting is the sole cause. The scale-head perturbation
control should preserve means and final actions exactly.

## Local execution status (2026-10-27)

Preflight ran and found all four base-model safetensors shards missing, no
`vendor/LIBERO-plus`, and no NumPy, PyTorch, Transformers or PEFT in local Python.
No saved observation NPZs or original Plus per-task JSONs are present in this
checkout. Therefore no representative checkpoint forward could run here.

| Measurement | Result |
|---|---|
| Normalized mean max absolute / relative difference | **Not measured** |
| Final executed action max absolute / relative difference | **Not measured** |
| Loaded weight fingerprints equal | **Not measured** |
| LoRA B tensors exactly zero | **Not measured** (configured zero initialization) |
| Scale output influences deterministic action | **No, by source dataflow** |

Validation here covers Python compilation, CLI help, preflight, and refusal to
start inference with missing prerequisites. The GPU harness itself remains
runtime-unvalidated. No checkpoints, evaluation manifests, or evaluation source
were modified; no training or full evaluations were run.
