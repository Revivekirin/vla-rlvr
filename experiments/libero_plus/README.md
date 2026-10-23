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
rerun, large asset download, or server connection was performed. User-provided server4 logs subsequently confirmed the benchmark import and
SFT tasks 1–10 completing successfully. The next environment creation failed
with a mesh-read error followed by `Too many open files`; the file-handle limit
fix has not yet been confirmed by a full run. Full OOD rates and cross-inference
numerical equivalence remain unverified.

## 서버 실행 요약: 매번 export/ulimit하지 않기

서버별 Conda 설치는 독립적이다. **Slurm으로 할당받은 server4 안에서**
설치와 평가를 실행한다. slurmmaster에서 환경을 설치하지 않는다.
아래 공통 래퍼가 호출할 때마다 Conda 활성화, 환경변수, 파일 핸들 한도를
자동 적용한다. `.bashrc`나 기존 Conda 환경의 activation hook 수정은 필요 없다.

```bash
cd /nfs4/jhkim/repos/vla-rlvr
bash experiments/server_env.sh plus python -u experiments/libero_plus/evaluate.py --model sft --seed 1 --execute
```

`server_env.sh`는 **bash로 실행하며 source하지 않는다**. 오류는 자식 프로세스만
종료시킨다. 대화형 Slurm 셸에는 `set -euo pipefail`을 실행하지 않는다.
이미 켰다면 `set +e`로 자동 종료를 끈다. 래퍼는 실행 중인 작업의 환경을
바꾸지 않으며, 재실행한 프로세스부터 적용된다.

## 1. server4 최초 1회 설치 (sudo 불필요)

이미 완료한 설치는 반복하지 않는다. 기존 모델 환경과 Original LIBERO는
재설치하지 않는다. 먼저 server4의 Conda를 활성화한다.

```bash
set +e
hostname
source /path/profile.d/conda.sh
conda activate ript_vla_openvla_oft_stage0
```

### 별도 LIBERO-Plus checkout

이미 checkout이 있다면 revision 확인만 실행한다. 원본 `vendor/LIBERO`를
덮어쓰거나 `pip uninstall libero` / `pip install -e`를 실행하지 않는다.

```bash
if [ ! -d vendor/LIBERO-plus ]; then
  git init vendor/LIBERO-plus &&
  git -C vendor/LIBERO-plus remote add origin https://github.com/sylvestf/LIBERO-plus.git &&
  git -C vendor/LIBERO-plus fetch --depth 1 origin 4976dc30028e805ff8094b55501d532c48fec182 &&
  git -C vendor/LIBERO-plus checkout --detach FETCH_HEAD
fi
test "$(git -C vendor/LIBERO-plus rev-parse HEAD)" = 4976dc30028e805ff8094b55501d532c48fec182 && echo 'Revision OK'
```

### Python 추가 패키지와 ImageMagick 공유 라이브러리

`wand` Python 패키지만으로는 부족하다. ImageMagick은 **server4의 별도 Conda
prefix**에 설치해 기존 모델 환경의 패키지를 변경하지 않는다.

```bash
python -m pip install -r vendor/LIBERO-plus/extra_requirements.txt

# 이 prefix에 이미 설치했다면 conda create는 생략
conda create -y --prefix /home/jhkim/miniconda3/envs/liberoplus_imagemagick --override-channels -c conda-forge imagemagick

bash experiments/server_env.sh plus python -c 'from wand.image import Image; from wand.version import MAGICK_VERSION; import skimage; print(MAGICK_VERSION)'
```

마지막 명령이 버전을 출력해야 한다. 다른 서버에서는 그 서버의 Conda로
동일하게 준비한다. 기본 경로가 다른 경우 래퍼 호출 앞에
`VLA_CONDA_ROOT=/다른/miniconda3`를 지정할 수 있다.
`VLA_CONDA_ENV`, `VLA_MAGICK_HOME`도 필요한 경우에만 override한다.

### 공식 assets 다운로드와 경로 정리

완료된 `data/libero-plus/assets.zip`이 있으면 재다운로드하지 않는다.
처음 다운로드하거나 중단된 다운로드를 이어받을 때만 실행한다 (~6.4 GB).
완료된 ZIP에 `curl -C -`를 다시 실행하면 HTTP 416이 날 수 있다.

```bash
mkdir -p data/libero-plus
curl -L --fail --retry 3 -C - https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/dd2bd61b7d9a6fef1abc52d606e983b41886a149/assets.zip -o data/libero-plus/assets.zip
```

실제 ZIP은 `inspire/.../LIBERO-plus-0/assets/` 경로를 포함한다.
단순 `unzip -d` 대신 아래 명령으로 `assets/` 앞부분을 제거한다.
Plus assets 파일만 추출/덮어쓰며 Original LIBERO, checkpoint, 평가 결과는
변경하지 않는다. 평가 실행 중에는 재추출하지 않는다.

```bash
python - <<'PYASSETS'
from pathlib import Path, PurePosixPath
import shutil
import zipfile

dest = Path('vendor/LIBERO-plus/libero/libero/assets')
count = 0
with zipfile.ZipFile('data/libero-plus/assets.zip') as archive:
    for entry in archive.infolist():
        parts = PurePosixPath(entry.filename).parts
        if entry.is_dir() or 'assets' not in parts:
            continue
        relative = parts[parts.index('assets') + 1:]
        if not relative or '..' in relative:
            raise ValueError(entry.filename)
        target = dest.joinpath(*relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(entry) as source, target.open('wb') as output:
            shutil.copyfileobj(source, output)
        count += 1
print('Extracted asset files:', count)
required = dest / 'scenes/tabletop250/tabletop_table_FabricCarpet001_NRM_3K.xml'
assert required.is_file(), required
print('Verified:', required)
PYASSETS
```

## 2. 환경변수와 파일 핸들 한도

래퍼가 매 실행 시 적용하는 설정:

| 설정 | Original / Plus | 목적 |
|---|---|---|
| Conda `ript_vla_openvla_oft_stage0` | 공통 | 서버의 기존 모델 실행 환경 |
| `ulimit -Sn "$(ulimit -Hn)"` | 공통 | soft 파일 핸들 한도를 현재 hard 한도까지 올림 |
| `MUJOCO_GL=egl` | 공통 | headless 렌더링 |
| `TOKENIZERS_PARALLELISM=false` | 공통 | tokenizer 병렬 처리 비활성화 |
| `WANDB_MODE=disabled` | 공통 | 현재 평가의 W&B 비활성화 |
| `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` | 공통 | 준비된 로컬 checkpoint 사용 |
| `MAGICK_HOME` | Plus | 별도 ImageMagick prefix |
| `LD_LIBRARY_PATH`에 `$MAGICK_HOME/lib` 추가 | Plus | MagickWand 공유 라이브러리 검색 |
| `CUDA_VISIBLE_DEVICES` | 수정하지 않음 | Slurm GPU 할당 유지 |

`PYTHONPATH` / `LIBERO_CONFIG_PATH`는 공통으로 강제하지 않는다.
Original은 `stage0/scripts/run.py`가 `vendor/LIBERO`와 `runtime/libero-config`를
선택하고, Plus는 `evaluate.py`가 `vendor/LIBERO-plus`와
`runtime/libero-plus-config`를 선택한다. 모델 경로 및 평가 조건은 변경하지 않는다.

**파일 핸들 한도는 LIBERO-Plus 전용이 아니다.** Original LIBERO도 동일한
OS 제한을 받으므로 두 평가 모두 공통 래퍼로 실행한다. 다만 기존 Original
평가가 이미 한도 증가를 사용했는지는 당시 셸 기록 없이는 확정할 수 없다.

soft limit은 새 작업에 상속되지만 hard limit을 초과할 수 없다. 래퍼는
soft/hard 값을 시작 시 출력한다. hard 값 자체가 낮다면 sudo 없이 더 높일 수
없으며 클러스터 관리자/Slurm 설정 변경이 필요하다. 큰 한도에서도
`Too many open files`가 반복되면 환경 종료 후 핸들 누수를 조사해야 한다.
한도 증가는 누수 수정을 대신하지 않는다. 실행 중 프로세스에는 새 한도가
소급 적용되지 않으므로 저장된 결과를 유지하고 재개한다.

설정만 확인하기 (모델/시뮬레이터 초기화 없음):

```bash
bash experiments/server_env.sh plus bash -c 'printf "Conda: %s\nMagick: %s\nGPU: %s\n" "$CONDA_PREFIX" "$MAGICK_HOME" "${CUDA_VISIBLE_DEVICES:-unset}"; ulimit -Sn; ulimit -Hn'
```

## 3. 검사, 실행, 재개, 집계

Plus 최초 준비 완료 후 import/task mapping 확인 (GPU 모델 로딩/rollout 없음):

```bash
bash experiments/server_env.sh plus python experiments/libero_plus/evaluate.py --check-import
```

세 모델 순차 실행. 하나가 실패하면 다음 모델을 실행하지 않는다.
셸에는 `set +e`를 유지한다. 총 7,773 episode이며 현재 영상/W&B 기록은 없다.

```bash
bash experiments/server_env.sh plus python -u experiments/libero_plus/evaluate.py --model sft --seed 1 --execute &&
bash experiments/server_env.sh plus python -u experiments/libero_plus/evaluate.py --model scale --seed 1 --execute &&
bash experiments/server_env.sh plus python -u experiments/libero_plus/evaluate.py --model ript --seed 1 --execute &&
bash experiments/server_env.sh plus python experiments/libero_plus/report.py --seed 1
```

중단된 모델만 같은 명령으로 재실행하면 완료된 task를 건너뛴다.
`manifest.json`이나 `task-*.json`을 삭제/수정하지 않는다. 결과가 10개
저장돼 있다면 11번째부터 재개한다. 집계는 실행 중에도 별도 셸에서 가능하다.

Original LIBERO도 동일한 환경 래퍼를 사용한다. 아래는 **dry-run**이며 기존에
완료된 nominal 평가를 재실행할 필요는 없다.

```bash
bash experiments/server_env.sh original python stage0/scripts/run.py eval --suite goal --mode sft --timeout 86400
```

새 Original 평가를 의도적으로 실행할 때만 `--execute`를 추가한다.
기존 Original 런처는 `stage0/evidence/eval-goal-sft.log`를 쓰기 모드로 열기
때문에 같은 suite/mode의 원본 로그를 먼저 별도로 보존해야 한다.
Original 런처에는 Plus의 task별 resume 기능이 없다. 기본 timeout은
1,800초이므로 긴 실행에는 위처럼 명시한다.

## 4. 파일 구조

```text
/home/jhkim/miniconda3/                 # server4 전용 Conda
└── envs/
    ├── ript_vla_openvla_oft_stage0/    # 기존 모델/시뮬레이터 Python 환경
    └── liberoplus_imagemagick/         # MagickWand 공유 라이브러리 별도 설치

/nfs4/jhkim/repos/vla-rlvr/
├── experiments/
│   ├── server_env.sh                  # Original/Plus 공통 실행 환경 래퍼
│   ├── nominal/goal.csv               # nominal 3모델 결과와 evidence 경로
│   └── libero_plus/
│       ├── README.md                  # 이 문서
│       ├── evaluate.py                # Plus 평가, 저장/재개
│       ├── resume_bounded.py          # 100 task마다 평가 프로세스 교체
│       ├── report.py                  # 실제 counts로 비교 보고서 생성
│       ├── test_report.py             # 가벼운 parsing/aggregation 테스트
│       └── comparison.md              # nominal / OOD 비교 보고서
├── vendor/
│   ├── LIBERO/                        # Original 유지
│   └── LIBERO-plus/libero/libero/
│       ├── benchmark/                 # 공식 task map/classification
│       ├── bddl_files/                # task 정의
│       ├── init_files/                # 공식 초기 상태
│       └── assets/                    # ZIP의 assets/ 내용이 이곳에 직접 위치
│           ├── scenes/tabletop250/
│           ├── textures/
│           └── ...
├── runtime/
│   ├── openvla-oft/                   # 기존 OFT runtime
│   ├── ript-vla/                      # 기존 RIPT runtime
│   ├── libero-config/                 # Original config
│   └── libero-plus-config/config.yaml # Plus가 생성하는 별도 config
├── checkpoints/
│   ├── goal/oft/                      # 기존 SFT base checkpoint
│   └── ript/openvla_oft/
│       ├── scale_header/LIBERO_GOAL_scale_header.pth
│       └── ript_adaptors/LIBERO_GOAL_lora/
│           └── openvla_headers.pt     # adapter와 함께 사용하는 RL heads
├── data/libero-plus/assets.zip        # 공식 assets 원본 ZIP
├── stage0/evidence/eval-goal-*.log     # 기존 nominal 원본 로그: 보존
└── outputs/libero-plus/
    └── {sft,scale,ript}/seed-1/
        ├── manifest.json             # 평가 조건/코드 hash
        ├── .lock                     # 동시 쓰기 방지
        └── task-0001.json ...         # task별 success/failure, ID, shift, steps
```

## 5. 오류별 조치

| 증상 | 조치 |
|---|---|
| `MagickWand shared library not found` | server4의 별도 ImageMagick 설치 확인 후 `server_env.sh plus`로 실행 |
| `assets/scenes/...xml` 없음 | ZIP의 `inspire/.../assets/` prefix를 제거해 추출했는지 확인 |
| mesh 읽기 오류 뒤 `Too many open files` | 파일 존재 여부와 래퍼가 출력한 soft/hard 한도 확인; 재발하면 핸들 누수 조사 |
| 오류와 함께 slurmmaster로 복귀 | 대화형 셸의 `set -e` 여부 확인; 새 할당 후 `set +e`, 자식 래퍼로 실행 |
| `Resume manifest mismatch` | 코드/조건 변경 여부 확인; 기존 결과를 삭제해 우회하지 않음 |

이 래퍼는 공통 환경 설정만 추가한다. 현재 평가 Python 코드와 manifest hash는
변경하지 않으므로 기존 결과를 그대로 재개할 수 있다.

## 7. mesh 파일 FD 누적에 따른 프로세스 교체

server4에서 SFT 약 340 task 진행 시 33,697개 FD가 관측됐다. 동일한 Panda
`.obj` 파일이 각각 673번 열려 있었다. MuJoCo 3.1.6 / robosuite 1.4.1의
환경 종료 경로까지 확인했지만, 라이브러리 내 정확한 누수 지점은 미확정이다.
soft limit 증가는 FD 누적을 해결하지 않는다.

`resume_bounded.py`는 **기존 evaluate.py를 수정하지 않고**, 기본 100개 task를
완료할 때마다 해당 평가 프로세스를 종료한 다음 동일 명령을 새 프로세스로
재시작한다. 완료된 JSON만 진행 상황으로 계산한다. 중단 시 진행 중이던
1개 task는 저장되지 않을 수 있으며 새 프로세스가 다시 실행한다.
기존 manifest의 evaluator hash, 체크포인트, 완료 결과는 유지된다.
매 재시작마다 모델을 다시 로드하므로 실행 시간은 조금 늘어난다.

현재 동일 모델 evaluator가 실행 중이라면 먼저 그 터미널에서 Ctrl+C로
종료하고, `ps`로 해당 evaluator가 종료됐는지 확인한다. 동시 실행하지 않는다.
그 후 다음 명령으로 SFT를 재개한다:

```bash
cd /nfs4/jhkim/repos/vla-rlvr
mkdir -p outputs/libero-plus/logs
set -o pipefail
bash experiments/server_env.sh plus python -u experiments/libero_plus/resume_bounded.py --model sft --seed 1 --tasks-per-process 100 2>&1 | tee -a outputs/libero-plus/logs/sft-seed1.log
```

Scale/RIPT도 같은 감독 방식을 적용한다:

```bash
bash experiments/server_env.sh plus python -u experiments/libero_plus/resume_bounded.py --model scale --seed 1 --tasks-per-process 100 2>&1 | tee -a outputs/libero-plus/logs/scale-seed1.log
bash experiments/server_env.sh plus python -u experiments/libero_plus/resume_bounded.py --model ript --seed 1 --tasks-per-process 100 2>&1 | tee -a outputs/libero-plus/logs/ript-seed1.log
```

Scale/RIPT 첫 실행에서는 감독 스크립트가 기존 평가기를 시작하고 각 모델의
manifest를 만든다. 기존 manifest가 있으면 설정을 확인한 뒤 완료된 결과부터
이어간다.
