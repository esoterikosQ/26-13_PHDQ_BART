#!/usr/bin/env bash
# 한 run 전체 실행: 특수 토큰 점검 → 학습 → 최고 검증 GLEU 체크포인트로 테스트 → M² 채점
# 사용: bash tools/run_one.sh <data> <run_id> <seed> <lr> <recipe:root|legacy> [init_ckpt(시나리오2 2단계)]
# 저장소 루트에서 실행 (데이터는 ../data/Preprocessed)
set -euo pipefail
DATA=$1; RUN_ID=$2; SEED=$3; LR=$4; RECIPE=$5; INIT_CKPT=${6:-}
cd "$(dirname "$0")/.."   # 저장소 루트
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
LOG=logs/${DATA}_${RUN_ID}
stamp() { TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(stamp)] START $DATA $RUN_ID seed=$SEED lr=$LR recipe=$RECIPE init=${INIT_CKPT:-none}" | tee -a logs/runs.log

python3 tools/check_special_tokens.py $RECIPE > ${LOG}_check.log 2>&1 \
  || { echo "[$(stamp)] FAIL special tokens ($DATA $RUN_ID) — ${LOG}_check.log" | tee -a logs/runs.log; exit 1; }

EXTRA=()
if [ -n "$INIT_CKPT" ]; then EXTRA=(--model_ckpt_path "$INIT_CKPT" --resume_finetune); fi
python3 run.py --recipe "$RECIPE" --data "$DATA" --run_id "$RUN_ID" --lr "$LR" --batch_size 64 --max_epochs 10 \
  --max_seq_len 128 --seed "$SEED" "${EXTRA[@]}" > ${LOG}_train.log 2>&1
CKPT=$(ls outputs/$DATA/$RUN_ID/model_ckpt/*.ckpt)
echo "[$(stamp)] TRAIN DONE $DATA $RUN_ID best_ckpt=$CKPT" | tee -a logs/runs.log

python3 run.py --recipe "$RECIPE" --eval_test --data "$DATA" --run_id "$RUN_ID" --seed "$SEED" \
  --model_ckpt_path "$CKPT" --batch_size 64 --max_seq_len 128 > ${LOG}_test.log 2>&1
EPOCH=$(python3 -c "import torch; print(torch.load('$CKPT', map_location='cpu', weights_only=False)['epoch'])")
G=outputs/generation/$DATA/$RUN_ID/epoch$EPOCH/test
echo "[$(stamp)] TEST DONE $DATA $RUN_ID $(head -1 $G/gleu.txt)" | tee -a logs/runs.log

python3 metric/m2scorer/scripts/m2scorer.py $G/hypothesis.txt ../data/Preprocessed/$DATA/${DATA}_test.m2 > $G/m2score.txt 2> ${LOG}_m2.log
echo "[$(stamp)] M2 DONE $DATA $RUN_ID $(tr '\n' ' ' < $G/m2score.txt)" | tee -a logs/runs.log
