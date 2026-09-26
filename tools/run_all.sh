#!/usr/bin/env bash
# 나머지 전체 실험 대기열 (recipe legacy, batch 64, 10 epoch). 한 run이 실패해도 기록 후 다음으로 진행한다.
# 사용: 저장소 루트에서 bash tools/run_all.sh <tag>   (run_id: <tag>_seedN, afterunion_<tag>_seedN / 기록: logs/runs.log, 끝나면 "QUEUE DONE")
set -uo pipefail
TAG=${1:?tag 필요}
cd "$(dirname "$0")/.."   # 저장소 루트
stamp() { TZ=Asia/Seoul date '+%Y-%m-%d %H:%M:%S'; }
run() {  # data run_id seed lr [init_ckpt]
  if [ -f "outputs/$1/$2/.done" ]; then echo "[$(stamp)] SKIP $1 $2 (이미 완료)" | tee -a logs/runs.log; return; fi
  if bash tools/run_one.sh "$1" "$2" "$3" "$4" legacy ${5:+"$5"}; then
    touch "outputs/$1/$2/.done"
  else
    echo "[$(stamp)] FAIL $1 $2 (run_one 종료 코드 $?) — 다음 run으로 진행" | tee -a logs/runs.log
  fi
}
echo "[$(stamp)] QUEUE START tag=$TAG" | tee -a logs/runs.log

# 시나리오 1: 개별 데이터셋 (lr 3e-5). union은 시나리오 2의 1단계를 겸한다.
for s in 0 1 2; do run korean_learner ${TAG}_seed$s $s 3e-5; done
for s in 0 1 2; do run native ${TAG}_seed$s $s 3e-5; done
for s in 0 1 2; do run union  ${TAG}_seed$s $s 3e-5; done

# 시나리오 2: union 체크포인트(같은 seed)에서 개별 데이터셋 추가 학습 (lr 1e-5)
stage2() {  # data seed
  local ck; ck=$(ls outputs/union/${TAG}_seed$2/model_ckpt/*.ckpt 2>/dev/null | head -1)
  if [ -z "$ck" ]; then echo "[$(stamp)] FAIL $1 afterunion_${TAG}_seed$2 (union seed$2 체크포인트 없음)" | tee -a logs/runs.log; return; fi
  run "$1" afterunion_${TAG}_seed$2 "$2" 1e-5 "$ck"
}
for s in 0 1 2; do stage2 korean_learner $s; done
for s in 0 1 2; do stage2 native $s; done

# lang8 (가장 오래 걸려 마지막)
for s in 0 1 2; do run lang8 ${TAG}_seed$s $s 3e-5; done
for s in 0 1 2; do stage2 lang8 $s; done

echo "[$(stamp)] QUEUE DONE tag=$TAG" | tee -a logs/runs.log
