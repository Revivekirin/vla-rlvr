# LIBERO-Plus Goal 평가 운영 가이드

이 저장소의 기존 Goal 체크포인트 3개(`sft`, `scale`, `ript`)를 공식
LIBERO-Plus Goal 확장 2,591개 task에서 각각 1회 평가한다. 학습은 하지 않는다.
Original LIBERO와 LIBERO-Plus는 별도 checkout/config를 사용한다.

- LIBERO-Plus 코드 revision: `4976dc30028e805ff8094b55501d532c48fec182`
- 공식 task ID는 1부터, benchmark 인덱스는 0부터 시작한다.
- 완료 결과는 task별 JSON으로 즉시 저장된다. 같은 명령을 다시 실행하면 완료한
  task를 건너뛴다. `manifest.json`과 `task-*.json`은 삭제하거나 수정하지 않는다.
- 실행 코드인 `evaluate.py`가 바뀌면 기존 manifest의 코드 hash와 달라질 수 있다.
  평가 중 이 파일은 교체하지 않는다.

## 저장 위치

아래 경로는 **저장소 루트 기준 상대경로**다. 특정 서버 계정, 호스트,
IP, 개인 디렉터리 경로는 이 문서에 적지 않는다.

| 항목 | 경로 |
|---|---|
| Original LIBERO | `vendor/LIBERO/` |
| LIBERO-Plus | `vendor/LIBERO-plus/` |
| Plus task 분류 | `vendor/LIBERO-plus/libero/libero/benchmark/task_classification.json` |
| Plus 압축 해제 assets | `vendor/LIBERO-plus/libero/libero/assets/` |
| 공식 assets ZIP | `data/libero-plus/assets.zip` |
| SFT base | `checkpoints/goal/oft/` |
| Scale head | `checkpoints/ript/openvla_oft/scale_header/LIBERO_GOAL_scale_header.pth` |
| RIPT adapter/heads | `checkpoints/ript/openvla_oft/ript_adaptors/LIBERO_GOAL_lora/` |
| Plus 평가 결과 | `outputs/libero-plus/{sft,scale,ript}/seed-1/` |
| Plus 실행 로그 | `outputs/libero-plus/logs/` |
| Nominal 결과 | `experiments/nominal/goal.csv` |
| 비교 보고서 | `experiments/libero_plus/comparison.md` |

Nominal Goal 결과는 SFT 486/500(97.2%), Scale 488/500(97.6%),
RIPT 488/500(97.6%)이다. 각 task의 성공 건수와 원본 evidence 경로는 CSV에
기록돼 있다. 원본 evidence는 보존한다.

## 최초 설치와 import 확인

GPU가 할당된 계산 노드에서 실행한다. 노드마다 Conda 환경이 독립적이라면
**실제 평가 노드의 환경**에 설치한다. `server_env.sh`는 Conda 활성화,
EGL/offline 환경변수, MagickWand 경로, 파일 핸들 soft limit 설정을 담당한다.
Conda 경로가 기본 설정과 다르면 `VLA_CONDA_ROOT`로 지정한다.

공식 LIBERO-Plus를 별도 경로에 checkout한다. Original LIBERO 설치를
제거하거나 덮어쓰지 않는다.

```bash
cd /path/to/vla-rlvr

git clone https://github.com/sylvestf/LIBERO-plus.git vendor/LIBERO-plus
git -C vendor/LIBERO-plus checkout --detach 4976dc30028e805ff8094b55501d532c48fec182
```

이미 checkout이 있으면 clone을 반복하지 않고 revision만 확인한다.

```bash
test "$(git -C vendor/LIBERO-plus rev-parse HEAD)" = 4976dc30028e805ff8094b55501d532c48fec182
```

Plus의 추가 Python 패키지 `wand`, `scikit-image`를 기존 평가 Conda 환경에
설치한다. ImageMagick 공유 라이브러리는 sudo 없이 **평가 노드의 별도 Conda
prefix**에 설치할 수 있다. 이미 설치돼 있으면 생략한다.

```bash
python -m pip install -r vendor/LIBERO-plus/extra_requirements.txt
conda create -y --prefix /path/to/liberoplus_imagemagick --override-channels -c conda-forge imagemagick
```

별도 prefix를 사용한다면 평가 실행 환경에 `VLA_MAGICK_HOME`을 해당 prefix로
지정한다. 공식 assets ZIP은 약 6.4 GB다. 실제 ZIP에는 긴 제작자 경로가
`assets/` 앞에 붙어 있으므로 단순 `unzip -d`는 올바른 위치에 풀리지 않는다.
압축 해제 명령과 구조 검사는 아래와 같다.

```bash
mkdir -p data/libero-plus
curl -L --fail --retry 3 -C - \
  https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/dd2bd61b7d9a6fef1abc52d606e983b41886a149/assets.zip \
  -o data/libero-plus/assets.zip

python - <<'PYASSETS'
from pathlib import Path, PurePosixPath
import shutil
import zipfile

dest = Path('vendor/LIBERO-plus/libero/libero/assets')
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
assert (dest / 'scenes/tabletop250/tabletop_table_FabricCarpet001_NRM_3K.xml').is_file()
PYASSETS
```

ZIP이 완전히 내려받아져 있으면 다시 다운로드하지 않는다. assets 추출은
평가 실행 중에는 하지 않는다. 설치 후 import/task mapping 확인:

```bash
bash experiments/server_env.sh plus \
  python experiments/libero_plus/evaluate.py --check-import
```

## 실행 전 FD 한도 확인

서버의 파일 핸들 한도는 **실제 평가 프로세스**에서 확인한다. Original
LIBERO도 같은 OS 제한을 받는다. `server_env.sh`가 새 프로세스의 soft 한도를
현재 hard 한도까지 올린다. Slurm이 정한 hard 한도는 올릴 수 없다.

최신 `resume_bounded.py`는 시작할 때 `FD soft=... hard=... restart=...`를
출력한다. FD 재시작 임계치는 다음과 같이 자동 결정된다.

```text
실제 재시작 임계치 = min(--max-open-files 값, 평가 프로세스 soft 한도의 절반)
```

예를 들어 soft 한도가 131,072이고 `--max-open-files 262144`라면 실제
재시작 임계치는 **65,536**이다. 사용자가 임의로 한도를 추측해 매번 옵션을
바꿀 필요가 없다. 그보다 먼저 `--tasks-per-process 100`에 도달하면
100 task 완료 후 교체한다. 이 두 조건 중 먼저 도달한 쪽이 작동한다.
`8192`처럼 낮은 FD 임계치는 몇 task마다 체크포인트를 재로딩할 수 있다.

평가 중 확인:

```bash
ps -u "$USER" -o pid,ppid,etime,stat,args \
  | grep -E '[r]esume_bounded.py|[e]xperiments/libero_plus/evaluate.py'

# 위 출력의 evaluate.py PID를 지정
EVAL_PID=12345  # 위 ps 출력의 실제 evaluate.py PID로 교체
grep 'Max open files' "/proc/$EVAL_PID/limits"
find "/proc/$EVAL_PID/fd" -mindepth 1 -maxdepth 1 -type l | wc -l
```

FD는 mesh 파일 로딩 과정에서 task마다 누적되는 것이 관측됐다. 감독
스크립트가 평가 프로세스를 교체하면 FD가 해제된다. `ulimit` 증가는 누수
자체의 수정은 아니다. FD가 한 task 안에서 급격히 증가하면 2초 간격의
감시 전에 오류가 날 수 있다.

## SFT → Scale → RIPT → 집계

GPU를 할당받은 셸에서, 저장소 루트로 이동한 뒤 아래 **한 블록**을 실행한다.
진행 중인 동일 모델 평가가 있다면 먼저 그 평가 터미널에서 Ctrl+C로 종료한다.
완료된 JSON과 manifest는 유지한다. `set -o pipefail`로 평가 실패 시 다음
모델을 시작하지 않는다. `tee -a`는 재개 로그를 이어 쓴다.

```bash
cd /path/to/vla-rlvr
mkdir -p outputs/libero-plus/logs
set +e
set -o pipefail

bash experiments/server_env.sh plus \
  python -u experiments/libero_plus/resume_bounded.py \
  --model sft --seed 1 --tasks-per-process 100 --max-open-files 262144 \
  2>&1 | tee -a outputs/libero-plus/logs/sft-seed1.log &&

bash experiments/server_env.sh plus \
  python -u experiments/libero_plus/resume_bounded.py \
  --model scale --seed 1 --tasks-per-process 100 --max-open-files 262144 \
  2>&1 | tee -a outputs/libero-plus/logs/scale-seed1.log &&

bash experiments/server_env.sh plus \
  python -u experiments/libero_plus/resume_bounded.py \
  --model ript --seed 1 --tasks-per-process 100 --max-open-files 262144 \
  2>&1 | tee -a outputs/libero-plus/logs/ript-seed1.log &&

bash experiments/server_env.sh plus \
  python experiments/libero_plus/report.py --seed 1
```

**이 자동 FD 조정은 최신 `resume_bounded.py`가 계산 노드에 배포돼 있어야
작동한다.** 기존 결과는 코드 변경 없이 재개된다. 모델당 2,591개 task,
총 7,773개 episode다. Slurm 작업 시간이 부족해 중단되면 같은 명령을
다시 실행한다. 중단 중 진행하던 task 하나는 재실행될 수 있다.

진행 확인과 로그 확인:

```bash
for model in sft scale ript; do
  printf '%s: ' "$model"
  find "outputs/libero-plus/$model/seed-1" -maxdepth 1 -name 'task-*.json' -type f 2>/dev/null | wc -l
done

tail -n 40 -F outputs/libero-plus/logs/sft-seed1.log
```

`Ctrl+C`로 tail만 종료해도 평가는 계속된다. 집계 결과는
`experiments/libero_plus/comparison.md`와 `experiments/nominal/goal.csv`에
저장된다. 실제 완료 건수와 분모를 사용하며 미완료 OOD drop은 표시하지 않는다.

## 비교 해석

세 모델은 Plus에서 같은 task 순서·초기 상태·전처리·action chunk·horizon·성공
기준으로 평가된다. SFT의 OFT 단일 샘플 추론과 Scale/RIPT의 cached batch
추론 경로는 다르며 수치적 동등성은 확인되지 않았다. Scale head는 현재
결정론적 평가에서 실행 행동을 바꾸지 않는다. Nominal/OOD task 모집단이
다르므로 task별 쌍을 가정하지 않는다.

공식 자료: [LIBERO-Plus 코드와 프로토콜](https://github.com/sylvestf/LIBERO-plus/tree/4976dc30028e805ff8094b55501d532c48fec182),
[공식 assets](https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/dd2bd61b7d9a6fef1abc52d606e983b41886a149).

최종 task JSON이 저장된 직후 Linux가 종료 중인 자식 프로세스의
`/proc/<pid>/fd` 접근을 거부할 수 있다. 감독 스크립트는 완료 개수를 먼저
확인하고 이 경우 FD 조회를 건너뛴다. `PermissionError`도 종료 중 상태로
처리한다. 해당 오류로 SFT가 이미 2,591/2,591 저장됐다면 재실행 시
`All tasks already complete`로 종료하며 다음 모델을 실행할 수 있다.
