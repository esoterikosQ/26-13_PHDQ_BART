#!/bin/bash
#SBATCH --job-name=gec2-llm-smoke
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%j.out
# --time 근거: 모델 로드 약 2분 + 메모리 측정 8회(각 1–2분) + native 512줄 1 epoch 학습·검증 512줄·테스트 256줄(약 5분 이하 예상) + 여유.
#
# plan2 §4 스모크: (1) 최장 문장 batch로 micro batch·생성 batch별 최대 메모리 측정 (2) native 512줄 1 epoch 학습·검증·테스트.
# 파티션·GPU·comment·task 수는 제출 명령에서 지정한다 (Neuron 실사례 확인 전에는 스크립트에 고정하지 않음, plan2 §0-1).
#   mkdir -p logs/slurm
#   sbatch -p <partition> --gres=gpu:<N> --ntasks-per-node=<N> --cpus-per-task=<C> \
#          --comment="field=<field>;appl=pytorch" gec2/slurm/llm_smoke.sbatch
source gec2/slurm/common.sh
RUN_ID=smoke_${NGPU}gpu_${SLURM_JOB_ID}
RUN_DIR=$OUT_ROOT/kanana-1.5-2.1b/native/$RUN_ID

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
