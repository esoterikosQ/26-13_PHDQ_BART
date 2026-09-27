# Neuron에서 LLM(kanana-1.5-2.1b) 실행 안내

`plan2.md` §0-1의 인계 방식을 따른다. Neuron 작업(접속, 작업 제출, 결과 올리기)은 사용자가 하고, 코드와 작업 스크립트는 이 저장소에서 가져간다.

**파일 주고받기 규칙 (plan2 §0)**
- 코드·스크립트 수정은 GitHub로만 한다: Claude가 GitHub `main`에 올리고 → Neuron·itcerdo에서 `git pull`. Neuron에서 추적 파일을 직접 고치지 않는다(고칠 게 있으면 알려 주면 반영). Neuron 전용 값은 git에 올라가지 않는 `gec2/slurm/paths.local.sh`에 둔다.
- 결과는 `bash gec2/slurm/share_results.sh`로 push한다(`git add` → commit → push만 함). 무엇이 올라갈지는 `.gitignore`가 정한다: `logs/slurm/`과 `neuron_outputs/`의 로그·설정·점수·출력만 올라가고, 체크포인트와 데이터 원문 사본은 제외된다. itcerdo·Claude는 `git pull`로 받는다.
- 데이터셋·체크포인트처럼 GitHub에 올리지 않는 파일은 사용자가 직접 옮긴다.
- `git pull`은 작업이 대기 중이거나 끝났을 때 한다. 작업 스크립트 본문(`llm_*.sh`, `common.sh`, `paths.sh`)은 실행 시점에 읽히므로, 실행 중에 바꾸면 그 작업이 깨질 수 있다.

경로는 `gec2/slurm/paths.sh`에 있다. 다르면 `gec2/slurm/paths.local.sh`에 같은 변수를 적어 덮어쓴다.

| 항목 | 값 |
| --- | --- |
| 저장소 | `paths.sh`가 있는 clone 위치를 자동으로 씀 |
| 데이터 | `/scratch/r984a02/phdq_bart/data/Preprocessed/<data>/` (없으면 `data/<data>/`) |
| 모델 | `/scratch/r984a02/phdq_bart/models/kanana-1.5-2.1b-instruct-2505` |
| conda 환경 | 이름 `phdq_bart` (`CONDA_ENV`) |
| 출력 | 저장소 안 `neuron_outputs/kanana-1.5-2.1b/<data>/<run_id>/` (체크포인트 `ckpt/`는 git 제외) |
| slurm 로그 | 저장소 루트의 `logs/slurm/<작업이름>_<jobid>.out` (git에 올라감) |

## 1. 한 번만 하는 준비 (로그인 노드)

```bash
cd <저장소 루트> && git pull && mkdir -p logs/slurm
conda activate phdq_bart
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # 이미 설치했으면 생략
pip install -r gec2/requirements.txt
python -c "import torch, transformers, pytorch_lightning as pl; print(torch.__version__, torch.version.cuda, transformers.__version__, pl.__version__)"
#   → 2.11.0+cu128 12.8 4.57.6 2.6.6

hf download kakaocorp/kanana-1.5-2.1b-instruct-2505 \
   --local-dir /scratch/r984a02/phdq_bart/models/kanana-1.5-2.1b-instruct-2505 \
   --include "*.json" "model.safetensors" "LICENSE"
```

작업 스크립트는 비대화형 bash라 `~/.bashrc`의 conda 설정을 읽지 않는다. 그래서 `conda.sh`를 `$CONDA_EXE`나 `~/miniconda3` 같은 흔한 위치에서 찾아 불러온다. 못 찾으면 아래처럼 적어 둔다:

```bash
echo "CONDA_BASE=$(conda info --base)" >> gec2/slurm/paths.local.sh
```

**로그인 노드에서는 python 계산을 하지 않는다**(공유 노드 규칙). 로그인 노드에서 하는 일은 `git pull`, `sbatch`, `share_results.sh`, 모델 내려받기뿐이다. 계산 스크립트(`precheck.sh`, `llm_*.sh`)는 작업 밖에서 실행하면 바로 멈추게 되어 있다.

사전 점검(§4, 토크나이저 기반)은 CPU 작업으로 제출한다. 스모크 작업도 시작할 때 같은 점검을 한다.

```bash
sbatch gec2/slurm/precheck.sbatch   # cpu 파티션, 코어 4, 30분 → logs/slurm/gec2-precheck_<jobid>.out
```

itcerdo 결과와 같아야 한다: 데이터 줄 수(korean_learner 19,898 / 4,264 / 4,265, native 12,292 / 2,634 / 2,634, lang8 76,692 / 16,434 / 16,434, union 108,883 / 23,333 / 23,334), 원문 복원 100%, unk 0, 최대 입력 181·출력 161 토큰.

## 2. 스모크 (GPU 확인 + 메모리 측정 + 512줄 1 epoch)

자원 순서는 A100 × 2 → H200 × 1이다(V100은 쓰지 않음). 파티션·GPU·CPU·`--comment`는 스크립트에 들어 있다.

| 스크립트 | 파티션 | GPU | task × CPU | `--comment` |
| --- | --- | --- | --- | --- |
| `a100x2_*.sbatch` | `amd_a100nv_8` | 2 | 2 × 8 | `field=nlp;appl=pytorch-ddp` |
| `h200x1_*.sbatch` | `amd_h200nv_8` | 1 | 1 × 8 | `field=nlp;appl=pytorch` |

```bash
cd /scratch/r984a02/phdq_bart/26-13_PHDQ_BART      # 반드시 저장소 루트에서 제출
sbatch gec2/slurm/a100x2_smoke.sbatch               # A100이 막히면: sbatch gec2/slurm/h200x1_smoke.sbatch
```

끝나면 `bash gec2/slurm/share_results.sh`로 올린다(slurm 로그 `logs/slurm/gec2-a100x2_smoke_<jobid>.out`와 run 디렉토리 `neuron_outputs/kanana-1.5-2.1b/native/smoke_*`). 이 결과로 GPU 메모리(40/80GB), micro batch·생성 batch·gradient checkpointing 여부, 대표 epoch 작업의 `--time`을 정해 스크립트를 갱신한다.

## 3. 대표 epoch 측정 → 본 실행

```bash
git pull   # 스모크 결과로 갱신된 스크립트
sbatch gec2/slurm/a100x2_train.sbatch korean_learner 1e-5 0 --epochs_per_job 1
```

1 epoch(학습 + 전체 검증 생성 + 저장) 시간을 재고 멈춘다. `share_results.sh`로 올리면 epoch당 시간으로 `--time`을 계산해 스크립트에 적는다. 이어서 같은 run을 재개한다:

```bash
RESUME=1 sbatch gec2/slurm/a100x2_train.sbatch korean_learner 1e-5 0
```

- 재개는 `ckpt/last.ckpt`(모델·옵티마이저·스케줄러·step·조기 종료 상태)에서 이어 가고, 테스트는 `ckpt/best`(최고 검증 GLEU)로 한다.
- `train_state.json`의 `stop_reason`이 `early_stop` 또는 `max_epochs`면 학습과 테스트까지 끝난 것이다. 비어 있으면 작업 분할로 멈춘 것이므로 `RESUME=1`로 다시 제출한다.
- 같은 run은 처음 시작한 GPU 구성으로 이어 간다(A100 × 2로 시작했으면 A100 × 2로 재개).
- batch 기본값은 스모크 측정으로 정했다: GPU당 micro 16, 생성 batch 64, gradient checkpointing 끔(전역 batch 64 고정). 메모리가 부족하면 `GC=--gradient_checkpointing sbatch ...` 또는 `MICRO=8 GEN_BATCH=32 sbatch ...`.

## 3-1. 시나리오 2 (union → 개별)

같은 seed의 union run이 끝난 뒤, 그 최고 체크포인트(`ckpt/best`)에서 가중치만 가져와 korean_learner·native·lang8을 새로 학습한다. 시작 lr은 union lr(1e-5)의 1/3이다.

```bash
S2=1e-5 sbatch --time=02:00:00 gec2/slurm/a100x2_train.sbatch korean_learner 3.3333e-6 <seed>
S2=1e-5 sbatch --time=01:00:00 gec2/slurm/a100x2_train.sbatch native 3.3333e-6 <seed>
S2=1e-5 sbatch --time=05:00:00 gec2/slurm/a100x2_train.sbatch lang8 3.3333e-6 <seed>
```

run 디렉토리는 `neuron_outputs/kanana-1.5-2.1b/<data>/s2_lr1e-5_seed<N>/`이다. 학습·테스트가 끝난 run은 재개용 `ckpt/last.ckpt`를 자동으로 지우고 `ckpt/best`만 남긴다. union run의 `ckpt/best`는 시나리오 2의 시작점이므로 지우지 않는다.

## 4. 공유할 것 (run마다)

작업이 끝날 때마다 로그인 노드에서 `bash gec2/slurm/share_results.sh`를 실행한다. `.gitignore` 기준으로 run 디렉토리에서 `ckpt/`(재개용 약 25GB, 평가용 약 8GB)와 데이터 원문 사본(`source.txt`, `reference.txt`)을 뺀 나머지와 `logs/slurm/*.out`이 올라간다.

| 파일 | 내용 |
| --- | --- |
| `run_info*.json` | 커밋 해시, 실행 명령·인자, SLURM 변수, nvidia-smi 요약, 패키지 버전 |
| `train_config*.json`, `model_info.json` | 옵티마이저·lr·스케줄·warmup·wd·clip·dropout·전역 batch·정밀도, 파라미터 수 |
| `check_leak.json`, `check_train_mode.json` | 누설·학습 모드 검사 |
| `train_steps.jsonl`, `epochs.jsonl` | step별 loss·lr, epoch별 검증 loss·GLEU·학습/검증 시간·GPU별 최대 메모리·조기 종료 상태 |
| `val/epochNN/`, `test/` | 생성 결과와 점수 (`hypothesis.txt`, `gleu.txt`, `m2score.txt`, `scores.json`) |
| `slurm/` | 작업 스크립트, nvidia-smi |

사전 점검 결과(`neuron_outputs/precheck/`)도 같이 올라간다.
