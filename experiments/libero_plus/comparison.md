# Goal nominal / LIBERO-Plus comparison

Goal checkpoints only; episode-weighted rates. Missing OOD data remain unavailable.
Drop = nominal SR − OOD SR (percentage points); shown only for complete OOD runs.

| Model | Nominal successes / episodes | Nominal SR | OOD successes / completed (expected) | OOD SR | Drop (pp) |
|---|---:|---:|---:|---:|---:|
| sft | 486/500 | 97.2% | 1675/2591 (2591) | 64.65% | +32.55 |
| scale | 488/500 | 97.6% | 1667/2591 (2591) | 64.34% | +33.26 |
| ript | 488/500 | 97.6% | 1620/2591 (2591) | 62.52% | +35.08 |

| Model | Shift | Successes / completed (expected) | SR |
|---|---|---:|---:|
| sft | Background Textures | 272/281 (281) | 96.80% |
| sft | Camera Viewpoints | 251/408 (408) | 61.52% |
| sft | Language Instructions | 251/410 (410) | 61.22% |
| sft | Light Conditions | 254/279 (279) | 91.04% |
| sft | Objects Layout | 277/425 (425) | 65.18% |
| sft | Robot Initial States | 92/409 (409) | 22.49% |
| sft | Sensor Noise | 278/379 (379) | 73.35% |
| scale | Background Textures | 272/281 (281) | 96.80% |
| scale | Camera Viewpoints | 253/408 (408) | 62.01% |
| scale | Language Instructions | 212/410 (410) | 51.71% |
| scale | Light Conditions | 270/279 (279) | 96.77% |
| scale | Objects Layout | 274/425 (425) | 64.47% |
| scale | Robot Initial States | 90/409 (409) | 22.00% |
| scale | Sensor Noise | 296/379 (379) | 78.10% |
| ript | Background Textures | 271/281 (281) | 96.44% |
| ript | Camera Viewpoints | 235/408 (408) | 57.60% |
| ript | Language Instructions | 210/410 (410) | 51.22% |
| ript | Light Conditions | 263/279 (279) | 94.27% |
| ript | Objects Layout | 273/425 (425) | 64.24% |
| ript | Robot Initial States | 85/409 (409) | 20.78% |
| ript | Sensor Noise | 283/379 (379) | 74.67% |

| Comparison (second − first) | Nominal difference (pp) | OOD difference (pp) |
|---|---:|---:|
| sft → scale | +0.40 | -0.31 |
| scale → ript | +0.00 | -1.81 |
| sft → ript | +0.40 | -2.12 |

Nominal source: `experiments/nominal/goal.csv` (per-task counts and evidence paths).
No nominal/OOD task pairing is assumed. Nominal and OOD use different task populations.
Nominal SFT uses OFT single-sample inference; Scale/RIPT use RIPT cached batch inference.
OOD shares one environment loop but retains those model inference paths; numerical equivalence is unverified.
The scale head is loaded but does not change mean actions when sampling is disabled.
