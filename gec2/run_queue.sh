#!/usr/bin/env bash
# itcerdo seq2seq 대기열: 목록 파일의 run을 차례로 실행 (test/scores.json이 있는 run은 건너뜀).
# 목록 형식(공백 구분, #은 주석): <model> <model_tag> <data> <run_id> <seed> <lr> <micro_batch> <gen_batch_size> [추가 인자...]
# 사용 (저장소 루트, tmux 안에서): bash gec2/run_queue.sh gec2/queues/lr_sweep.txt
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv-gec2/bin/activate
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
QUEUE=$1; RLOG=logs/gec2_runs.log
stamp() { TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(stamp)] QUEUE START $QUEUE" | tee -a $RLOG
grep -vE '^\s*(#|$)' "$QUEUE" | while read -r MODEL TAG DATA RUN SEED LR MB GB EXTRA; do
  OUT=outputs2/$TAG/$DATA/$RUN
  if [ -f $OUT/test/scores.json ]; then echo "[$(stamp)] SKIP $TAG $DATA $RUN (완료)" | tee -a $RLOG; continue; fi
  LOG=logs/gec2_${TAG}_${DATA}_${RUN}.log
  echo "[$(stamp)] START $TAG $DATA $RUN seed=$SEED lr=$LR micro=$MB gen=$GB $EXTRA" | tee -a $RLOG
  python3 -m gec2.train_seq2seq --model $MODEL --model_tag $TAG --data $DATA --run_id $RUN --seed $SEED --lr $LR \
    --micro_batch $MB --gen_batch_size $GB $EXTRA > $LOG 2>&1 < /dev/null
  if [ $? -ne 0 ] || [ ! -f $OUT/test/scores.json ]; then echo "[$(stamp)] FAIL $TAG $DATA $RUN — $LOG" | tee -a $RLOG; continue; fi
  echo "[$(stamp)] DONE $TAG $DATA $RUN $(python3 -c "import json;s=json.load(open('$OUT/test/scores.json'));print(f\"best_epoch {s['best_epoch']} val {s['best_val_gleu']:.2f} test GLEU {s['gleu']:.2f} P {s['p']:.2f} R {s['r']:.2f} F0.5 {s['f05']:.2f}\")")" | tee -a $RLOG
  rm -f $OUT/ckpt/last.ckpt   # 완료 run은 재개용 체크포인트 불필요 (평가용 ckpt/best만 유지)
done
echo "[$(stamp)] QUEUE DONE $QUEUE" | tee -a $RLOG
