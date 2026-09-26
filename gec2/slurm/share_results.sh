#!/bin/bash
# Neuron 결과를 GitHub에 push한다 (로그인 노드, 저장소 루트에서).
#   bash gec2/slurm/share_results.sh ["커밋 메시지"]
# 올라가는 것은 .gitignore가 걸러 준다: logs/slurm/*.out, neuron_outputs/ 의 로그·설정·점수·출력.
# 체크포인트(ckpt/, *.safetensors)와 데이터 원문 사본(source.txt, reference.txt, gold_*.m2)은 제외된다.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
git add logs/slurm neuron_outputs 2>/dev/null || git add logs/slurm
git commit -m "${1:-Neuron results $(date '+%Y-%m-%d %H:%M')}" || { echo "새 결과 없음"; exit 0; }
git pull --rebase origin main
git push origin HEAD:main
