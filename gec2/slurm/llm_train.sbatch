#!/bin/bash
#SBATCH --job-name=gec2-llm
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x_%j.out
# --time 근거 (임시): 대표 epoch 측정용(--epochs_per_job 1). plan2 기준 1 epoch ≤ 6시간 + 모델 로드·누설 검사 약 10분 + 여유.
#   → 스모크·대표 epoch 실측 후 "epoch당 시간 × 최대 10 epoch + 준비·최종 테스트 + 20%"로 교체하고 계산식을 여기에 남긴다.
#   48시간(파티션 최대)을 넘으면 --epochs_per_job K로 나눠 제출하고 다음 작업은 RESUME=1로 이어간다.
#
# 사용 (저장소 루트에서):
#   sbatch -p <partition> --gres=gpu:<N> --ntasks-per-node=<N> --cpus-per-task=<C> --comment="field=<field>;appl=pytorch" \
#          gec2/slurm/llm_train.sbatch <data> <lr> <seed> [추가 인자...]
#   예) 대표 epoch 측정:   ... llm_train.sbatch korean_learner 1e-5 0 --epochs_per_job 1
#       이어서 학습(재개):  RESUME=1 sbatch ... llm_train.sbatch korean_learner 1e-5 0
#   (sbatch는 환경변수를 작업에 넘기므로 RESUME=1을 앞에 붙이면 됨)
source gec2/slurm/common.sh
DATA=$1; LR=$2; SEED=$3; shift 3
MICRO=${MICRO:-8}; GLOBAL=64; GEN_BATCH=${GEN_BATCH:-32}; GC=${GC---gradient_checkpointing}   # 스모크 결과로 조정 (GC= 로 끄기)
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
