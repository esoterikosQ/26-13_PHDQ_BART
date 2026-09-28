#!/bin/bash
#SBATCH --job-name=gec2-h200x1_train
#SBATCH --partition=amd_h200nv_8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
# 자원: ssh.md의 showque/showappl 기준 (amd_h200nv_8: GPU당 코어 8, --gres 필수, --comment 형식 field=nlp;appl=pytorch).
# --time 근거: H200×1 실측(2026-09-27–28, 단계 5): epoch당 korean_learner 약 3.8분, lang8 시나리오 2 약 12.6분, union 약 18.3분
#   → A100×2(korean_learner 5.1분, union 약 23.5분)보다 약 25% 빠름. A100×2의 --time(이미 20% 여유 포함)을 그대로 써도 충분:
#   korean_learner 02:00:00(기본), 제출 때 덮어쓰기: native --time=01:00:00, lang8 --time=05:00:00, union --time=07:00:00
# 같은 run의 재개(RESUME=1)는 처음 시작한 GPU 구성으로 한다 (전역 batch 64·lr·스케줄은 구성과 무관하게 같음, micro 16 × accumulation 4).
# 제출 (저장소 루트에서, 처음 한 번 mkdir -p logs/slurm): sbatch gec2/slurm/h200x1_train.sbatch <data> <lr> <seed> [추가 인자]   (재개: RESUME=1 sbatch ...)
bash gec2/slurm/llm_train.sh "$@"
