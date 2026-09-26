# LLM 학습 본문 (a100x2_train.sbatch / h200x1_train.sbatch가 호출). 인자: <data> <lr> <seed> [추가 인자...]
# 환경변수: RESUME=1 (ckpt/last.ckpt에서 재개), MICRO·GEN_BATCH (GPU당 batch), GC= (gradient checkpointing 끄기)
source gec2/slurm/common.sh
DATA=$1; LR=$2; SEED=$3; shift 3
MICRO=${MICRO:-8}; GLOBAL=64; GEN_BATCH=${GEN_BATCH:-32}; GC=${GC---gradient_checkpointing}   # 스모크 결과로 조정
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
