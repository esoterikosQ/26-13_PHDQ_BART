# Neuron에서 LLM(kanana-1.5-2.1b) 실행 안내

`plan2.md` §0-1의 인계 방식을 따른다. Neuron 작업(접속, 작업 제출, 결과 올리기)은 사용자가 하고, 코드와 작업 스크립트는 이 저장소에서 가져간다.

**파일 주고받기 규칙 (plan2 §0)**
- 코드·스크립트 수정은 GitHub로만 한다: Claude가 GitHub `main`에 올리고 → Neuron·itcerdo에서 `git pull`. Neuron에서 추적 파일을 직접 고치지 않는다(고칠 게 있으면 알려 주면 반영). Neuron 전용 값은 git에 올라가지 않는 `gec2/slurm/paths.local.sh`에 둔다.
- 결과는 `bash gec2/slurm/share_results.sh`로 GitHub에 올린다(체크포인트·데이터 원문 사본 제외, `results/neuron/`). itcerdo·Claude는 `git pull`로 받는다.
- **데이터셋과 체크포인트는 GitHub로 옮기지 않는다.** 데이터는 신청서 동의·비상업 조건이고, 체크포인트는 GitHub 파일 한도(100MB)를 넘는다. 저장소가 공개 상태면 `share_results.sh`가 올리지 않고 멈춘다.
- `git pull`은 작업이 대기 중이거나 끝났을 때 한다. 작업 스크립트 본문(`llm_*.sh`, `common.sh`, `paths.sh`)은 실행 시점에 읽히므로, 실행 중에 바꾸면 그 작업이 깨질 수 있다.

경로는 `gec2/slurm/paths.sh`에 있다. 다르면 `gec2/slurm/paths.local.sh`에 같은 변수를 적어 덮어쓴다.

| 항목 | 값 |
| --- | --- |
| 저장소 | `paths.sh`가 있는 clone 위치를 자동으로 씀 |
| 데이터 | `/scratch/r984a02/phdq_bart/data/Preprocessed/<data>/` (없으면 `data/<data>/`) |
| 모델 | `/scratch/r984a02/phdq_bart/models/kanana-1.5-2.1b-instruct-2505` |
| conda 환경 | 이름 `phdq_bart` (`CONDA_ENV`) |
| 출력 | `/scratch/r984a02/phdq_bart/outputs2/kanana-1.5-2.1b/<data>/<run_id>/` |
| slurm 로그 | 저장소 루트의 `logs/slurm/<작업이름>_<jobid>.out` (git에는 안 올라감; 작업이 끝나면 run 디렉토리 `slurm/`에도 복사) |

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

사전 점검 (§4, 토크나이저 기반, GPU 불필요; conda·데이터·모델 경로도 함께 확인됨):

```bash
bash gec2/slurm/precheck.sh      # 결과: /scratch/r984a02/phdq_bart/outputs2/precheck/kanana-1.5-2.1b.json
```

itcerdo 결과와 같아야 한다: 원문 복원 100%, unk 0, 최대 입력+출력 342 토큰. 실패하면 화면에 나온 `오류:` 줄을 알려 준다.

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

끝나면 `bash gec2/slurm/share_results.sh`로 올린다(slurm 로그 `logs/slurm/gec2-a100x2_smoke_<jobid>.out`와 run 디렉토리 `outputs2/kanana-1.5-2.1b/native/smoke_*`가 함께 올라감). 이 결과로 GPU 메모리(40/80GB), micro batch·생성 batch·gradient checkpointing 여부, 대표 epoch 작업의 `--time`을 정해 스크립트를 갱신한다.

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
- batch 조정은 환경변수로: `MICRO=16 GEN_BATCH=64 GC= sbatch ...` (`GC=`는 gradient checkpointing 끄기). 전역 batch는 64로 고정된다.

## 4. 공유할 것 (run마다)

작업이 끝날 때마다 로그인 노드에서 `bash gec2/slurm/share_results.sh`를 실행한다. run 디렉토리에서 `ckpt/`(재개용 약 25GB, 평가용 약 8GB)와 데이터 원문 사본(`source.txt`, `reference.txt`)을 뺀 나머지가 `results/neuron/runs/`로, slurm 로그가 `results/neuron/slurm_logs/`로 올라간다.

| 파일 | 내용 |
| --- | --- |
| `run_info*.json` | 커밋 해시, 실행 명령·인자, SLURM 변수, nvidia-smi 요약, 패키지 버전 |
| `train_config*.json`, `model_info.json` | 옵티마이저·lr·스케줄·warmup·wd·clip·dropout·전역 batch·정밀도, 파라미터 수 |
| `check_leak.json`, `check_train_mode.json` | 누설·학습 모드 검사 |
| `train_steps.jsonl`, `epochs.jsonl` | step별 loss·lr, epoch별 검증 loss·GLEU·학습/검증 시간·GPU별 최대 메모리·조기 종료 상태 |
| `val/epochNN/`, `test/` | 생성 결과와 점수 (`hypothesis.txt`, `gleu.txt`, `m2score.txt`, `scores.json`) |
| `slurm/` | 작업 스크립트, slurm 로그, nvidia-smi |

사전 점검 결과(`outputs2/precheck/`)도 같은 스크립트로 올라간다.

GitHub에 올리려면 Neuron에서 이 저장소에 push할 수 있는 인증이 필요하다. 이 저장소에만 쓰기 권한이 있는 deploy key나 fine-grained token을 권한다.
