# Neuron에서 LLM(kanana-1.5-2.1b) 실행 안내

`plan2.md` §0-1의 인계 방식을 따른다. Neuron 작업(접속, 데이터 이전, 작업 제출)은 사용자가 하고, 코드와 작업 스크립트는 이 저장소에서 가져간다.

> **아직 확정하지 않은 것**: 파티션, GPU 요청(`--gres`), `--comment`, CPU 수. 실제로 동작한 Neuron SLURM 스크립트 사례를 받기 전에는 스크립트에 고정하지 않고, 제출 명령의 인자로 넘긴다. 사례를 받으면 `#SBATCH` 지시문에 값으로 적어 넣는다.

## 1. 한 번만 하는 준비 (로그인 노드)

```bash
git clone https://github.com/esoterikosQ/26-13_PHDQ_BART.git && cd 26-13_PHDQ_BART
cp gec2/slurm/paths.example.sh gec2/slurm/paths.local.sh    # 경로 값을 채운다 (git에는 올라가지 않음)
mkdir -p logs/slurm

# 가상환경 (itcerdo와 같은 버전; CUDA 12.8용 torch)
python3 -m venv .venv-gec2 && source .venv-gec2/bin/activate     # python 3.10 이상. 모듈 시스템이면 먼저 module load
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r gec2/requirements.txt

# 모델을 로컬 디렉토리로 받기 (계산 노드는 오프라인으로 읽음)
source gec2/slurm/paths.local.sh
hf download kakaocorp/kanana-1.5-2.1b-instruct-2505 --local-dir "$MODEL_DIR" --include "*.json" "model.safetensors" "LICENSE"
```

데이터: itcerdo `~/projects/phdq_bart/data/Preprocessed/`의 4개 데이터셋 디렉토리(`korean_learner`, `native`, `lang8`, `union`)를 `DATA_ROOT`로 옮긴다. 각 디렉토리에 `<data>_{train,val,test}.txt`와 `<data>_test.m2`가 있어야 한다.

사전 점검(§4, 토크나이저 기반, GPU 불필요):

```bash
bash gec2/slurm/precheck.sh      # 결과: $OUT_ROOT/precheck/kanana-1.5-2.1b.json
```

itcerdo에서 같은 점검을 한 결과(원문 복원 100%, unk 0, 최대 입력+출력 342 토큰, 누설 검사 통과)와 같아야 한다.

## 2. 스모크 (GPU 확인 + 메모리 측정 + 512줄 1 epoch)

```bash
sbatch -p <partition> --gres=gpu:<N> --ntasks-per-node=<N> --cpus-per-task=<C> \
       --comment="field=<field>;appl=pytorch" gec2/slurm/llm_smoke.sbatch
```

- `--ntasks-per-node`는 GPU 수와 같게 한다(Lightning이 task마다 GPU 1개를 쓴다). 다르면 스크립트가 바로 멈춘다.
- 자원 순서: A100 × 2 → H200 × 1 (V100은 쓰지 않음).
- 끝나면 `logs/slurm/gec2-llm-smoke_<jobid>.out`과 run 디렉토리(`$OUT_ROOT/kanana-1.5-2.1b/native/smoke_*`)를 공유한다. 이 결과로 micro batch·생성 batch·gradient checkpointing 여부와 대표 epoch 작업의 `--time`을 정해 스크립트를 갱신한다.

## 3. 대표 epoch 측정 → 본 실행

```bash
git pull   # 스모크 결과로 갱신된 스크립트
sbatch <자원 인자> gec2/slurm/llm_train.sbatch korean_learner 1e-5 0 --epochs_per_job 1
```

1 epoch(학습 + 전체 검증 생성 + 저장) 시간을 재고 멈춘다. 결과를 공유하면 epoch당 시간으로 `--time`을 계산해 스크립트에 적는다. 이어서 같은 run을 재개한다:

```bash
RESUME=1 sbatch <자원 인자> gec2/slurm/llm_train.sbatch korean_learner 1e-5 0
```

- 재개는 `ckpt/last.ckpt`(모델·옵티마이저·스케줄러·step·조기 종료 상태)에서 이어 가고, 테스트는 `ckpt/best`(최고 검증 GLEU)로 한다.
- `train_state.json`의 `stop_reason`이 `early_stop` 또는 `max_epochs`면 학습이 끝나고 테스트까지 수행된 것이다. 비어 있으면 작업 분할로 멈춘 것이므로 `RESUME=1`로 다시 제출한다.
- 메모리·batch 조정은 환경변수로: `MICRO=16 GEN_BATCH=64 GC= sbatch ...` (`GC=`는 gradient checkpointing 끄기). 전역 batch는 64로 고정된다.

## 4. 공유할 것 (run마다)

run 디렉토리 `$OUT_ROOT/kanana-1.5-2.1b/<data>/<run_id>/`를 통째로 공유한다. `ckpt/`는 크므로(재개용 약 25GB, 평가용 약 8GB) 빼고 공유한다.

| 파일 | 내용 |
| --- | --- |
| `run_info*.json` | 커밋 해시, 실행 명령·인자, SLURM 변수, nvidia-smi 요약, 패키지 버전 |
| `train_config*.json`, `model_info.json` | 옵티마이저·lr·스케줄·warmup·wd·clip·dropout·전역 batch·정밀도, 파라미터 수 |
| `check_leak.json`, `check_train_mode.json` | 누설·학습 모드 검사 |
| `train_steps.jsonl`, `epochs.jsonl` | step별 loss·lr, epoch별 검증 loss·GLEU·학습/검증 시간·GPU별 최대 메모리·조기 종료 상태 |
| `val/epochNN/`, `test/` | 생성 결과와 점수 (`hypothesis.txt`, `gleu.txt`, `m2score.txt`, `scores.json`) |
| `slurm/` | 작업 스크립트, slurm 로그, nvidia-smi |

사전 점검 결과 `$OUT_ROOT/precheck/kanana-1.5-2.1b.json`도 한 번 공유한다.
