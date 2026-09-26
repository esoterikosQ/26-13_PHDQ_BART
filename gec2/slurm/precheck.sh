#!/bin/bash
# plan2 §4 사전 점검을 로그인 노드에서 실행 (토크나이저 기반: 원문 복원·unk·길이). GPU 없이 가능.
# 누설 검사까지 하려면 GPU가 있는 곳에서 --leak을 붙인다 (학습 작업도 시작 전에 자동으로 수행함).
#   bash gec2/slurm/precheck.sh [--leak]        (저장소 안 어디서 실행해도 됨)
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"
[ -f "$REPO/gec2/slurm/paths.local.sh" ] && source "$REPO/gec2/slurm/paths.local.sh"
cd "$REPO"
activate_env
[ -f "$MODEL_DIR/config.json" ] || { echo "오류: 모델 없음 $MODEL_DIR"; exit 1; }
[ -f "$DATA_ROOT/korean_learner/korean_learner_train.txt" ] || { echo "오류: 데이터 없음 $DATA_ROOT/korean_learner/"; exit 1; }
echo "REPO=$REPO DATA_ROOT=$DATA_ROOT MODEL_DIR=$MODEL_DIR python=$(which python3)"
python3 -m gec2.precheck --model "$MODEL_DIR" --model_tag kanana-1.5-2.1b --kind llm --data_root "$DATA_ROOT" \
  --out "$OUT_ROOT/precheck" "$@"
echo "결과: $OUT_ROOT/precheck/kanana-1.5-2.1b.json"
