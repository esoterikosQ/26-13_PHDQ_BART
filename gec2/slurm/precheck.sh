#!/bin/bash
# plan2 §4 사전 점검 본문 (토크나이저 기반: 원문 복원·unk·길이, GPU 불필요). precheck.sbatch가 호출한다.
# 로그인 노드에서 직접 실행하지 않는다(공유 노드 규칙: 전체 데이터 토큰화는 계산 작업).
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"
[ -f "$REPO/gec2/slurm/paths.local.sh" ] && source "$REPO/gec2/slurm/paths.local.sh"
require_job "sbatch gec2/slurm/precheck.sbatch"
cd "$REPO"
set +eu; activate_env; rc=$?; set -eu
[ $rc -eq 0 ] || exit 1
limit_threads
[ -f "$MODEL_DIR/config.json" ] || { echo "오류: 모델 없음 $MODEL_DIR"; exit 1; }
for d in korean_learner native lang8 union; do
  for f in ${d}_train.txt ${d}_val.txt ${d}_test.txt ${d}_test.m2; do
    [ -f "$DATA_ROOT/$d/$f" ] || { echo "오류: 데이터 없음 $DATA_ROOT/$d/$f"; exit 1; }
  done
  echo "$d: $(grep -c '' "$DATA_ROOT/$d/${d}_train.txt") / $(grep -c '' "$DATA_ROOT/$d/${d}_val.txt") / $(grep -c '' "$DATA_ROOT/$d/${d}_test.txt") 줄 (train/val/test; itcerdo와 같아야 함)"
done
echo "REPO=$REPO DATA_ROOT=$DATA_ROOT MODEL_DIR=$MODEL_DIR python=$(which python3) threads=$OMP_NUM_THREADS"
python3 -m gec2.precheck --model "$MODEL_DIR" --model_tag kanana-1.5-2.1b --kind llm --data_root "$DATA_ROOT" \
  --out "$OUT_ROOT/precheck" "$@"
echo "결과: $OUT_ROOT/precheck/kanana-1.5-2.1b.json"
