#!/usr/bin/env bash
# RTX5090 노드에서 queues/ 디렉토리에 있는 언어모델 파인튜닝의 설정에 따라 순차적으로 학습을 실행하는 스크립트
# 파인튜닝 설정에 포함된 주요 인자 목록 : <model> <model_tag> <data> <run_id> <seed> <lr> <micro_batch> <gen_batch_size> [추가 인자...]
# 설정값에 따라 train_seq2seq.py를 호출하는 방식으로 core.py 코드를 실행

# bash의 실행 변수 설정
# -u : 정의되지 않은 변수를 쓰면 종료
# -o pipefail : 파이프라인에서 오류가 발생하면 종료
set -uo pipefail

# 작업 경로 이동
cd "$(dirname "$0")/.."

# 가상환경 활성화(.venv-gec2/ 경로에 가상환경을 설치해야 함)
source .venv-gec2/bin/activate

# CUDA 설정 (RTX5090 사용)
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# QUEUE 변수는 쉘스크립트 실행시 입력으로 받음
# RLOG : 동일한 로그 파일에 전체 작업 과정을 쌓아가며 기록
QUEUE=$1; RLOG=logs/gec2_runs.log

# 현재 시간을 출력하는 사용자 정으 ㅣ함수
stamp() { TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S'; }

# QUEUE 시작 로그 출력 및 기록
echo "[$(stamp)] QUEUE START $QUEUE" | tee -a $RLOG

# 입력으로 받은 QUEUE 파일에서 전체 작업 수행에 필요한 모델을 읽어 반복적으로 파인튜닝 작업 수행
grep -vE '^\s*(#|$)' "$QUEUE" | while read -r MODEL TAG DATA RUN SEED LR MB GB EXTRA; do
  OUT=outputs2/$TAG/$DATA/$RUN

  # 전체 작업이 이미 완료되었는지 여부를 scores.json 파일 존재 여부를 기준으로 판단 / 파일이 존재하면 해당 인자 조합의 파인튜닝 작업을 건너뜀
  if [ -f $OUT/test/scores.json ]; then echo "[$(stamp)] SKIP $TAG $DATA $RUN (완료)" | tee -a $RLOG; continue; fi
  LOG=logs/gec2_${TAG}_${DATA}_${RUN}.log
  echo "[$(stamp)] START $TAG $DATA $RUN seed=$SEED lr=$LR micro=$MB gen=$GB $EXTRA" | tee -a $RLOG
  python3 -m gec2.train_seq2seq --model $MODEL --model_tag $TAG --data $DATA --run_id $RUN --seed $SEED --lr $LR \
    --micro_batch $MB --gen_batch_size $GB $EXTRA > $LOG 2>&1 < /dev/null
  
  # 파인튜닝 작업 종료 여부를 python 작업 수행 결과와 scores.json 파일 존재 여부를 기준으로 판단
  if [ $? -ne 0 ] || [ ! -f $OUT/test/scores.json ]; then echo "[$(stamp)] FAIL $TAG $DATA $RUN — $LOG" | tee -a $RLOG; continue; fi
  echo "[$(stamp)] DONE $TAG $DATA $RUN $(python3 -c "import json;s=json.load(open('$OUT/test/scores.json'));print(f\"best_epoch {s['best_epoch']} val {s['best_val_gleu']:.2f} test GLEU {s['gleu']:.2f} P {s['p']:.2f} R {s['r']:.2f} F0.5 {s['f05']:.2f}\")")" | tee -a $RLOG
  rm -f $OUT/ckpt/last.ckpt
done
echo "[$(stamp)] QUEUE DONE $QUEUE" | tee -a $RLOG
