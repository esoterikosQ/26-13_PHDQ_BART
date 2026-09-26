#!/bin/bash
# plan2 §4 사전 점검을 로그인 노드에서 실행 (토크나이저 기반: 원문 복원·unk·길이). GPU 없이 가능.
# 누설 검사까지 하려면 GPU가 있는 곳에서 --leak을 붙인다 (학습 작업도 시작 전에 자동으로 수행함).
#   bash gec2/slurm/precheck.sh [--leak]
source gec2/slurm/paths.sh; [ -f gec2/slurm/paths.local.sh ] && source gec2/slurm/paths.local.sh
cd "$REPO" && activate_env
python3 -m gec2.precheck --model "$MODEL_DIR" --model_tag kanana-1.5-2.1b --kind llm --data_root "$DATA_ROOT" \
  --out "$OUT_ROOT/precheck" "$@"
