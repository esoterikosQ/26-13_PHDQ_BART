#!/bin/bash
# Neuron 결과를 GitHub로 올린다 (plan2 §0-1 공유 단계). 로그인 노드에서 실행 (계산 노드는 인터넷이 없을 수 있음).
#   bash gec2/slurm/share_results.sh ["커밋 메시지"]
# 올리는 것: $OUT_ROOT의 run 디렉토리들(체크포인트 제외), 사전 점검 json, logs/slurm/*.out → results/neuron/{runs,slurm_logs}/
# 올리지 않는 것: ckpt/ (용량·GitHub 100MB 제한), 데이터 원문 사본(source.txt, reference.txt, gold_*.m2 — 데이터에서 다시 만들 수 있음)
# 데이터셋은 비상업·신청서 동의 조건이라 공개 저장소에는 모델 출력도 올리지 않는다: 저장소가 public이면 중단한다.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/paths.sh"
[ -f "$REPO/gec2/slurm/paths.local.sh" ] && source "$REPO/gec2/slurm/paths.local.sh"
cd "$REPO"
SLUG=$(git remote get-url origin | sed -E 's#(git@github.com:|https://github.com/)##; s#\.git$##')
VIS=$(curl -s -o /dev/null -w "%{http_code}" "https://api.github.com/repos/$SLUG")
if [ "$VIS" = 200 ] && curl -s "https://api.github.com/repos/$SLUG" | grep -q '"private": false'; then
  echo "중단: $SLUG 는 공개 저장소임. 데이터 조건 때문에 결과(모델 출력)를 올리지 않음 — 저장소를 private으로 바꾼 뒤 다시 실행"; exit 1
fi
DEST=results/neuron
mkdir -p "$DEST/slurm_logs"
rsync -a --prune-empty-dirs --exclude 'ckpt/' --exclude '*.ckpt' --exclude '*.safetensors' \
  --exclude 'source.txt' --exclude 'reference.txt' --exclude 'gold_*.m2' \
  "$OUT_ROOT/" "$DEST/runs/"   # outputs2/는 .gitignore 패턴에 걸리므로 runs/로
cp -p logs/slurm/*.out "$DEST/slurm_logs/" 2>/dev/null || true
BIG=$(find "$DEST" -type f -size +50M)
[ -z "$BIG" ] || { echo "중단: 50MB 넘는 파일이 있음 (GitHub 100MB 제한):"; echo "$BIG"; exit 1; }
git add "$DEST"
if git diff --cached --quiet; then echo "새 결과 없음"; exit 0; fi
git commit -q -m "${1:-Neuron results $(TZ=Asia/Seoul date '+%Y-%m-%d %H:%M')}"
git pull --rebase -q origin main
git push -q origin HEAD:main
echo "올림: $(git log --oneline -1)"
