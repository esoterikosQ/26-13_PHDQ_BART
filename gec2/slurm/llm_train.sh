# LLM 학습 본문 (a100x2_train.sbatch / h200x1_train.sbatch가 호출). 인자: <data> <lr> <seed> [추가 인자...]
# 환경변수: RESUME=1 (ckpt/last.ckpt에서 재개), MICRO·GEN_BATCH (GPU당 batch), GC=--gradient_checkpointing (켜기)
source gec2/slurm/common.sh
DATA=$1; LR=$2; SEED=$3; shift 3
# 기본값 근거: 스모크 915975 (A100-SXM4-80GB, 최장 문장 batch, GPU 1개 측정)
#   학습 micro 16·GC 없음 57.5GB, micro 8·GC 없음 42.4GB, GC 켜면 micro 8–16 38.9GB / 생성 batch 64 51.8GB, 32 39.5GB (80GB 중)
#   → 속도를 위해 GC 없이 micro 16 (A100×2: accumulation 2, H200×1: 4), 생성 batch 64
MICRO=${MICRO:-16}; GLOBAL=64; GEN_BATCH=${GEN_BATCH:-64}; GC=${GC-}   # GC=--gradient_checkpointing 으로 켜기
RUN_ID=lr${LR}_seed${SEED}
RUN_DIR=$OUT_ROOT/kanana-1.5-2.1b/$DATA/$RUN_ID
[ $((GLOBAL % (MICRO * NGPU))) -eq 0 ] || { echo "global 64가 micro×GPU로 나눠지지 않음"; exit 1; }
EXTRA=("$@")
[ "${RESUME:-0}" = 1 ] && EXTRA+=(--resume)
"${SRUN[@]}" python3 -m gec2.train_llm --model "$MODEL_DIR" --model_tag kanana-1.5-2.1b --data $DATA --run_id $RUN_ID \
  --data_root "$DATA_ROOT" --out_root "$OUT_ROOT" --lr $LR --seed $SEED --devices $NGPU \
  --micro_batch $MICRO --global_batch $GLOBAL --gen_batch_size $GEN_BATCH $GC "${EXTRA[@]}"
save_job_files "$RUN_DIR"
echo "== 작업 종료: $RUN_DIR (train_state.json의 stop_reason이 비어 있으면 RESUME=1로 이어서 제출)"
