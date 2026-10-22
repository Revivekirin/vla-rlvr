# Goal nominal / LIBERO-Plus comparison

Goal checkpoints only; episode-weighted rates. Missing OOD data remain unavailable.
Drop = nominal SR − OOD SR (percentage points); shown only for complete OOD runs.

| Model | Nominal successes / episodes | Nominal SR | OOD successes / completed (expected) | OOD SR | Drop (pp) |
|---|---:|---:|---:|---:|---:|
| sft | 486/500 | 97.2% | unavailable | unavailable | unavailable |
| scale | 488/500 | 97.6% | unavailable | unavailable | unavailable |
| ript | 488/500 | 97.6% | unavailable | unavailable | unavailable |

| Model | Shift | Successes / completed (expected) | SR |
|---|---|---:|---:|
| — | All seven shifts | unavailable | unavailable |

| Comparison (second − first) | Nominal difference (pp) | OOD difference (pp) |
|---|---:|---:|
| sft → scale | +0.40 | unavailable / unmatched coverage |
| scale → ript | +0.00 | unavailable / unmatched coverage |
| sft → ript | +0.40 | unavailable / unmatched coverage |

Nominal source: `experiments/nominal/goal.csv` (per-task counts and evidence paths).
No nominal/OOD task pairing is assumed. Nominal and OOD use different task populations.
Nominal SFT uses OFT single-sample inference; Scale/RIPT use RIPT cached batch inference.
OOD shares one environment loop but retains those model inference paths; numerical equivalence is unverified.
The scale head is loaded but does not change mean actions when sampling is disabled.
