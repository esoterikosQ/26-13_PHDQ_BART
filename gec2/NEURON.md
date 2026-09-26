# Neuron에서 LLM(kanana-1.5-2.1b) 실행 안내

`plan2.md` §0-1의 인계 방식을 따른다. Neuron 작업(접속, 데이터 이전, 작업 제출)은 사용자가 하고, 코드와 작업 스크립트는 이 저장소에서 가져간다. 경로는 `gec2/slurm/paths.sh`에 모여 있다(작업 디렉토리 `/scratch/r984a02/phdq_bart`). 바꿀 값이 있으면 같은 폴더에 `paths.local.sh`를 만들어 덮어쓴다(git 제외).

| 항목 | 값 |
| --- | --- |
| 저장소 | `/scratch/r984a02/phdq_bart/26-13_PHDQ_BART` |
| 데이터 | `/scratch/r984a02/phdq_bart/data/Preprocessed/<data>/` (없으면 `data/<data>/`를 자동으로 씀) |
| 모델 | `/scratch/r984a02/phdq_bart/models/kanana-1.5-2.1b-instruct-2505` |
| conda 환경 | `/scratch/r984a02/phdq_bart/conda/gec2` (`conda create -p`, 홈 용량을 쓰지 않음) |
| 출력 | `/scratch/r984a02/phdq_bart/outputs2/kanana-1.5-2.1b/<data>/<run_id>/` |
| 캐시 | `/scratch/r984a02/phdq_bart/.cache/{huggingface,pip}` |

## 1. 한 번만 하는 준비 (로그인 노드)

```bash
cd /scratch/r984a02/phdq_bart
git clone https://github.com/esoterikosQ/26-13_PHDQ_BART.git
cd 26-13_PHDQ_BART && mkdir -p logs/slurm

# conda 환경 (itcerdo와 같은 패키지 버전; CUDA 12.8용 torch)
export PIP_CACHE_DIR=/scratch/r984a02/phdq_bart/.cache/pip
conda create -p /scratch/r984a02/phdq_bart/conda/gec2 python=3.12 -y
conda activate /scratch/r984a02/phdq_bart/conda/gec2
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r gec2/requirements.txt
python -c "import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__)"
#   → 2.11.0+cu128 12.8 4.57.6

# 모델을 로컬 디렉토리로 받기 (계산 노드는 오프라인으로 읽음)
export HF_HOME=/scratch/r984a02/phdq_bart/.cache/huggingface
hf download kakaocorp/kanana-1.5-2.1b-instruct-2505 \
   --local-dir /scratch/r984a02/phdq_bart/models/kanana-1.5-2.1b-instruct-2505 \
   --include "*.json" "model.safetensors" "LICENSE"
```

conda를 `module load`로 불러오는 환경이면, 그 줄을 `gec2/slurm/paths.sh`의 `activate_env()` 첫 줄(또는 `paths.local.sh`)에 넣는다. 계산 노드의 작업 스크립트는 `eval "$(conda shell.bash hook)"; conda activate <환경>`으로 활성화한다.

사전 점검 (§4, 토크나이저 기반, GPU 불필요; 데이터·모델 경로도 함께 확인됨):

```bash
bash gec2/slurm/precheck.sh      # 결과: outputs2/precheck/kanana-1.5-2.1b.json
```

itcerdo 결과와 같아야 한다: 원문 복원 100%, unk 0, 최대 입력+출력 342 토큰.

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

끝나면 `logs/slurm/gec2-a100x2_smoke_<jobid>.out`과 run 디렉토리(`outputs2/kanana-1.5-2.1b/native/smoke_*`)를 공유한다. 이 결과로 GPU 메모리(40/80GB), micro batch·생성 batch·gradient checkpointing 여부, 대표 epoch 작업의 `--time`을 정해 스크립트를 갱신한다.

## 3. 대표 epoch 측정 → 본 실행

```bash
git pull   # 스모크 결과로 갱신된 스크립트
sbatch gec2/slurm/a100x2_train.sbatch korean_learner 1e-5 0 --epochs_per_job 1
```

1 epoch(학습 + 전체 검증 생성 + 저장) 시간을 재고 멈춘다. 결과를 공유하면 epoch당 시간으로 `--time`을 계산해 스크립트에 적는다. 이어서 같은 run을 재개한다:

```bash
RESUME=1 sbatch gec2/slurm/a100x2_train.sbatch korean_learner 1e-5 0
```

- 재개는 `ckpt/last.ckpt`(모델·옵티마이저·스케줄러·step·조기 종료 상태)에서 이어 가고, 테스트는 `ckpt/best`(최고 검증 GLEU)로 한다.
- `train_state.json`의 `stop_reason`이 `early_stop` 또는 `max_epochs`면 학습과 테스트까지 끝난 것이다. 비어 있으면 작업 분할로 멈춘 것이므로 `RESUME=1`로 다시 제출한다.
- 같은 run은 처음 시작한 GPU 구성으로 이어 간다(A100 × 2로 시작했으면 A100 × 2로 재개).
- batch 조정은 환경변수로: `MICRO=16 GEN_BATCH=64 GC= sbatch ...` (`GC=`는 gradient checkpointing 끄기). 전역 batch는 64로 고정된다.

## 4. 공유할 것 (run마다)

run 디렉토리를 통째로 공유하되 `ckpt/`는 크므로(재개용 약 25GB, 평가용 약 8GB) 뺀다.

| 파일 | 내용 |
| --- | --- |
| `run_info*.json` | 커밋 해시, 실행 명령·인자, SLURM 변수, nvidia-smi 요약, 패키지 버전 |
| `train_config*.json`, `model_info.json` | 옵티마이저·lr·스케줄·warmup·wd·clip·dropout·전역 batch·정밀도, 파라미터 수 |
| `check_leak.json`, `check_train_mode.json` | 누설·학습 모드 검사 |
| `train_steps.jsonl`, `epochs.jsonl` | step별 loss·lr, epoch별 검증 loss·GLEU·학습/검증 시간·GPU별 최대 메모리·조기 종료 상태 |
| `val/epochNN/`, `test/` | 생성 결과와 점수 (`hypothesis.txt`, `gleu.txt`, `m2score.txt`, `scores.json`) |
| `slurm/` | 작업 스크립트, slurm 로그, nvidia-smi |

사전 점검 결과 `outputs2/precheck/kanana-1.5-2.1b.json`도 한 번 공유한다.
