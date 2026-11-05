# Goal 20-shot SFT 실행 기록 — 2026-11-05

이 문서는 `goal-20shot-seed42-b1-25218` 실험의 **확인된 설정**을 기록한다. 기준 시각은 2026-11-05 KST이다. 최초 실행은 6,000-step checkpoint 직후 `/nfs4` 용량 부족(`OSError: [Errno 28] No space left on device`)으로 종료되었다. checkpoint 폴더에 검증 파일은 있으나 내용 검사는 아직 수행하지 않았다. 이후 학습, export 및 LIBERO 성공률은 아직 확인되지 않았다. 아래 LIBERO evaluation 칸은 실행 후 채우기 위해 비워 두었다.

## 실험 목적과 범위

- 목표: Stage 2에서 구성한 LIBERO-Goal 20-shot expert RLDS로 연속 action OpenVLA-OFT SFT를 수행하고, 이후 별도 LIBERO nominal evaluation을 준비한다.
- 이 실험은 시간 제약에 따른 **single-GPU batch-1 pilot**이다. 공식 8-GPU/global-batch-64 OFT 재현 실험이 아니다. 성공률은 학습 loss나 명목 데이터 노출량만으로 보장되지 않는다.
- 이전 Goal Full batch-1 1,000-step 실험 및 공개 OFT SFT 체크포인트에서 이어 학습하지 않는다. 이 실행은 pinned pretrained OpenVLA-7B에서 새로 시작한다.
- 학습 중 scale head, RIPT, RLVR, SFT 정책 rollout은 실행하지 않는다.

## 코드 및 데이터 식별

| 항목 | 기록 |
| --- | --- |
| 서버 작업 디렉터리 | `/nfs4/jhkim/repos/vla-rlvr` |
| 로컬에서 푸시한 CLI 구현 커밋 | `0f17a9a6c85fe6a4a75fe9affd1d8bd03c6964e0` |
| 서버의 실제 HEAD | `training_config.json` 및 서버 `git rev-parse HEAD`로 추후 확인 |
| 학습 엔트리포인트 | `python -m experiments.fewshot.train` → `experiments.fewshot_sft.worker` |
| 기반 trainer | `runtime/openvla-oft/vla-scripts/finetune.py`; SHA-256은 `training_config.json`의 `official_finetune_sha256`에 기록 |
| 기반 모델 | `openvla/openvla-7b` pretrained snapshot `47a0ec7fc4ec123775a391911046cf33cf9ed83f` |
| 기반 모델 캐시 | `/nfs4/jhkim/repos/vla-rlvr/data/hf-cache/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f` |
| Stage 2 데이터 | `/nfs4/jhkim/repos/vla-rlvr/outputs/fewshot/datasets/goal/seed42/20shot` |
| RLDS 루트 | 위 데이터 디렉터리의 `rlds/` |
| RLDS 이름 | `libero_goal_no_noops` |
| episode 선택 | Goal 10개 task별 expert episode 20개, 총 200개; 선택 seed `42` |
| transition 수 | `25,218` |
| 선택 ID·원본 파일 해시·source revision | 데이터 디렉터리의 `selection_manifest.json`, `validation.json` 참고. 이 문서에 확인되지 않은 해시를 전사하지 않음 |
| 학습 정규화 | 해당 20-shot 데이터의 `normalization.json` 사용; `validation.json`의 `normalization_sha256`과 일치해야 함 |

Stage 2는 원본 RLDS의 **전체 expert episode**를 선택·물질화했다. 학습 loader는 episode를 frame sample로 전개하고 shuffle/augmentation을 적용한다. episode 선택 seed `42`와 학습 난수 seed `7`은 다른 값이다. 출처와 코드 파일의 실제 SHA-256은 서버 실행 시 생성된 `training_config.json`에도 저장된다.

## 서버 환경

| 항목 | 확인된 값 |
| --- | --- |
| Host | `server4` |
| Conda 환경 | `ript_vla_openvla_oft_stage0` |
| GPU | NVIDIA RTX A5000 1개, 표시 메모리 `24,564 MiB` |
| `nvidia-smi` 드라이버 / 표시 CUDA | `535.230.02` / `12.2` |
| 학습 실행 조건 | CUDA와 BF16 지원을 CLI가 사전 확인, `WORLD_SIZE=1` |
| 모델 캐시 환경 변수 | `HF_HUB_CACHE="$PWD/data/hf-cache"` |
| worker 실행 환경 | `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `WANDB_MODE=disabled`, `PYTHONHASHSEED=7` |
| 정확한 실행 시 패키지 버전·최대 GPU 메모리 | 완료 시 `runtime.json`에 기록 예정; 아직 확인되지 않음 |

`nvidia-smi`의 CUDA 12.2는 드라이버가 표시하는 버전이다. 설치된 PyTorch CUDA 빌드 버전과 동일하다고 단정하지 않는다.

## 모델·학습 설정

| 항목 | 설정 |
| --- | --- |
| Suite / shots | Goal / task당 20 episode |
| Run ID / profile | `goal-20shot-seed42-b1-25218` / `single_gpu_batch1_pilot` |
| 초기화 / resume | pinned pretrained OpenVLA-7B / `resume=None` |
| 입력 | 제3자 시점 이미지 + wrist 이미지, proprio 8차원, language instruction |
| 출력 | 연속 action 7차원 × 8-step chunk; dataset 끝의 action chunk는 공식 loader의 terminal padding 사용 |
| 학습 대상 | LoRA adapter(rank 32, dropout 0), action head, proprio projector |
| 목적 함수 | 정규화된 action chunk의 L1 regression; diffusion·FiLM 미사용 |
| Action/proprio 정규화 | LIBERO `bounds_q99`, 현재 20-shot 데이터의 통계 사용 |
| 이미지 augmentation | 켜짐 |
| Optimizer | 공식 OFT trainer의 AdamW |
| 학습률 | `5e-4`; warmup 0 step, MultiStepLR decay milestone 100,000 step, gamma 0.1. 이 25,218-step 실행 안에서는 decay에 도달하지 않음 |
| GPU당 microbatch / gradient accumulation / effective batch | `1 / 1 / 1` |
| 총 optimizer update | `25,218` step 목표 |
| Data loader shuffle buffer | `1,000` |
| 학습 중 별도 expert validation | 없음 (`validation=None`); 기록되는 L1은 train loss |
| W&B | 비활성화; 로컬 `metrics.jsonl`, `diagnostics.jsonl` 기록 |
| checkpoint 저장 | 6,000 / 12,000 / 25,218 optimizer step; 각 시점에 LoRA, action head, proprio projector, optimizer·scheduler, 데이터 통계 및 검증 메타데이터 저장 |

공식 OFT trainer를 사용하되 로컬 worker가 checkpoint 경계, optimizer-step 집계, 유한 gradient 확인, 학습 대상 모듈 확인 및 저장 후 tensor reload 검증을 계측한다. 상세 구현은 [train.py](train.py), [worker.py](../fewshot_sft/worker.py), [diagnostics.py](../fewshot_sft/diagnostics.py)를 참고한다.

## 데이터 노출량과 체크포인트

| 목표 optimizer step | batch-1 frame draw | 25,218 transition 대비 명목 노출량 | 예상 checkpoint 경로 |
| ---: | ---: | ---: | --- |
| 6,000 | 6,000 | 0.238회 | `outputs/fewshot_sft/runs/goal-20shot-seed42-b1-25218--6000_chkpt` |
| 12,000 | 12,000 | 0.476회 | `outputs/fewshot_sft/runs/goal-20shot-seed42-b1-25218--12000_chkpt` |
| 25,218 | 25,218 | 1.000회 | `outputs/fewshot_sft/runs/goal-20shot-seed42-b1-25218--25218_chkpt` |

`명목 노출량 = optimizer step × effective batch / 25,218`이다. Shuffle과 반복 샘플 추출이 있으므로 모든 transition을 정확히 한 번 봤다는 뜻은 아니다. 체크포인트 step은 미리 정한 진단 지점이며 성능 최적점으로 검증된 값이 아니다.

## 실행 명령과 로그

```bash
cd /nfs4/jhkim/repos/vla-rlvr
export HF_HUB_CACHE="$PWD/data/hf-cache"
python -m experiments.fewshot.train \
  --suite goal --seed 42 --train-seed 7 --shots 20 \
  --profile single_gpu_batch1_pilot --max-steps 25218 \
  --checkpoint-steps 6000 12000 25218 \
  --run-id goal-20shot-seed42-b1-25218 --execute
```

사용자는 이를 `nohup bash`와 `tee`로 백그라운드 실행했다. `tee`는 launcher의 설정·최종 메시지를 기록하고, 실제 trainer 출력은 `train.py`가 별도의 `train.log`에 기록한다.

| 파일 | 용도 |
| --- | --- |
| `outputs/fewshot_sft/runs/goal-20shot-seed42-b1-25218/training_config.json` | 실행 설정, 데이터·기반 모델·코드 식별 해시 |
| `outputs/fewshot_sft/runs/goal-20shot-seed42-b1-25218/train.log` | 실제 trainer stdout/stderr 및 progress |
| `outputs/fewshot_sft/runs/goal-20shot-seed42-b1-25218/diagnostics.jsonl` | step별 train L1, 이동평균, LR, task별 frame draw |
| `outputs/fewshot_sft/runs/goal-20shot-seed42-b1-25218/metrics.jsonl` | trainer metric |
| `outputs/fewshot_sft/runs/goal-20shot-seed42-b1-25218/runtime.json` | 종료 상태, 실제 optimizer step, 소요 시간, GPU 메모리, 패키지 버전 |
| `outputs/fewshot_sft/runs/goal-20shot-seed42-b1-25218/loss_trend.json` | 모든 step 기록과 체크포인트 확인 후 생성되는 train loss 요약 |
| `outputs/fewshot_sft/logs/goal-20shot-seed42-b1-25218.launch.log` | `tee`로 수집한 launcher 출력 |
| `outputs/fewshot_sft/logs/goal-20shot-seed42-b1-25218.exitcode` | launcher의 최종 반환 코드 |

백그라운드 실행 중 `243/25218` 진행률이 보였고 6,000 step에서 checkpoint를 저장했다. 그 후 `/nfs4` 용량 부족으로 worker가 종료되었다. 최종 결과 판정은 progress bar만으로 하지 않고 `runtime.json`의 `status=passed`, 누적 optimizer step `25218`, launcher `PASS`, 지정 checkpoint의 검증 파일을 확인한 뒤 기록한다.

## 6,000-step 용량 부족 사건

사용자 제공 `df` 출력에서 `/nfs4`는 총 3.7 TiB 중 1.5 GiB만 남아 사용률 100%였고, inode 사용률은 74%였다. 최초 실행의 6,000-step checkpoint 폴더는 약 1.9 GiB이며, `optimizer.pt` 약 1.1 GiB, `action_head` 약 289 MiB, `proprio_projector` 약 65 MiB와 LoRA adapter를 포함한다. `component_verification.json` 및 `checkpoint_files.json`이 존재하므로 checkpoint 저장 절차가 끝난 흔적은 있으나, 저장 내용을 재검사하기 전까지 재개 가능하다고 단정하지 않는다. 20-shot RLDS 약 848 MiB는 현재 학습 입력이므로 공간 확보 대상으로 취급하지 않는다.

남은 12,000·25,218-step checkpoint에도 각각 약 1.9 GiB가 필요할 수 있다. 충분한 저장 공간을 확보하기 전에는 재시작하지 않는다. 재개한다면 동일한 Goal 20-shot, 선택·학습 seed, batch·optimizer 설정, 기반 모델 및 검증된 6,000-step checkpoint의 optimizer/scheduler 상태를 유지하고 **새 run ID**를 사용한다. RLDS shuffle/prefetch iterator는 복원되지 않으므로 재개 실행은 중단 없는 단일 실행과 bitwise 동등하지 않음을 기록한다.

서버의 `/home`은 같은 시점에 686 GiB 여유가 확인되었다. 기존 RLDS와 6,000-step checkpoint는 `/nfs4`에 그대로 두고, 후속 실행의 새 로그·checkpoint·export·evaluation만 `/home/jhkim/vla-rlvr-stage3-sft`에 기록할 수 있다. 이를 위해 모든 후속 명령에서 `FEWSHOT_SFT_OUTPUT_ROOT=/home/jhkim/vla-rlvr-stage3-sft`를 동일하게 설정한다. 이 환경 변수는 Stage 2 데이터의 위치를 바꾸지 않는다. 실제 재개 명령과 결과는 실행 후 기록한다.

## 학습·체크포인트 검증 기록

| 검증 항목 | 현재 기록 / 완료 후 확인 위치 |
| --- | --- |
| Stage 2 subset identity·episode/transition count·정규화 해시 | CLI preflight 통과 후 설정이 출력됨; 상세 원본은 `validation.json` 및 `training_config.json` |
| CUDA BF16·기반 pretrained snapshot | CLI preflight 통과 후 설정이 출력됨 |
| 실제 optimizer step·종료 상태 | 최초 실행은 용량 부족으로 실패; 6,000-step checkpoint 폴더 존재, 완료 여부 미검증 |
| train loss 추세 | 아직 미확인; `diagnostics.jsonl`, `loss_trend.json` |
| LoRA/action head/proprio 저장·reload 검증 | 아직 미확인; 각 checkpoint의 `component_verification.json`, `checkpoint_files.json` |
| action/proprio 정규화 일치 | 실행 시 worker가 검사; 최종 checkpoint 검증은 아직 미확인 |
| GPU 최대 메모리·정확한 소요 시간 | 아직 미확인; `runtime.json` |

`loss_trend.json`이나 낮은 train L1은 LIBERO 성공률 증거가 아니다.

## LIBERO evaluation — 실행 후 기입

| 항목 | 값 |
| --- | --- |
| 평가 대상 checkpoint |  |
| export 경로·형식 |  |
| export/reload 검증 결과 |  |
| 평가 코드 버전·해시 |  |
| 평가 환경·GPU·LIBERO 버전 |  |
| evaluation phase / seed / task 범위 / task별 trial 수 |  |
| episode별 결과 경로 |  |
| 총 성공 수 / 총 trial 수 / 성공률 |  |
| task별 성공률 |  |
| 실패 유형·action 이상 여부 |  |
| 평가 소요 시간·제한 사항 |  |
