# plan2 §4 스모크 본문 (a100x2_smoke.sbatch / h200x1_smoke.sbatch가 호출).
# (0) 사전 점검 (1) 최장 문장 batch로 micro batch·생성 batch별 최대 메모리 측정 (2) native 512줄 1 epoch 학습·검증·테스트.
source gec2/slurm/common.sh
RUN_ID=smoke_${NGPU}gpu_${SLURM_JOB_ID}
RUN_DIR=$OUT_ROOT/kanana-1.5-2.1b/native/$RUN_ID

echo "== 0) 사전 점검 (§4, 토크나이저 기반)"
python3 -m gec2.precheck --model "$MODEL_DIR" --model_tag kanana-1.5-2.1b --kind llm --data_root "$DATA_ROOT" \
  --out "$OUT_ROOT/precheck" || { echo "오류: 사전 점검 실패"; exit 1; }

echo "== 1) 메모리 측정 (GPU 1개, 최장 문장 batch)"
for spec in "8 32" "16 64" "8 32 --gradient_checkpointing" "16 64 --gradient_checkpointing" "32 64 --gradient_checkpointing"; do
  set -- $spec
  CUDA_VISIBLE_DEVICES=0 python3 -m gec2.memprobe --model "$MODEL_DIR" --kind llm --data_root "$DATA_ROOT" \
    --micro_batch $1 --gen_batch_size $2 ${3:-} 2>&1 | grep -E "학습 최대|OutOfMemoryError" | cut -c1-200 || true
done

echo "== 2) native 512줄 1 epoch (GPU $NGPU개)"
MICRO=8; GLOBAL=64
[ $((GLOBAL % (MICRO * NGPU))) -eq 0 ] || { echo "global 64가 micro×GPU로 나눠지지 않음"; exit 1; }
"${SRUN[@]}" python3 -m gec2.train_llm --model "$MODEL_DIR" --model_tag kanana-1.5-2.1b --data native --run_id $RUN_ID \
  --data_root "$DATA_ROOT" --out_root "$OUT_ROOT" --lr 1e-5 --seed 0 --devices $NGPU \
  --micro_batch $MICRO --global_batch $GLOBAL --gen_batch_size 32 --gradient_checkpointing \
  --limit_train 512 --limit_val 512 --limit_test 256 --max_epochs 1 --log_every 1
save_job_files "$RUN_DIR"
echo "== 완료: $RUN_DIR 를 통째로 공유"
