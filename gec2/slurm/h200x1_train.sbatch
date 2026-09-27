#!/bin/bash
#SBATCH --job-name=gec2-h200x1_train
#SBATCH --partition=amd_h200nv_8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/%x_%j.out
# 자원: ssh.md의 showque/showappl 기준 (amd_h200nv_8: GPU당 코어 8, --gres 필수, --comment 형식 field=nlp;appl=pytorch).
# --time 근거: H200×1은 아직 실측 전. A100×2 실측(a100x2_train.sbatch 주석: korean_learner epoch 5.1분, 데이터셋별 선형)에 1.5배 여유를 둔다.
#   H200 1장의 BF16 이론 성능(약 989 TFLOPS)은 A100 2장(624)보다 높지만, 첫 작업이라 짧게 잡지 않는다. 첫 run 실측 후 여기에 기록하고 줄인다.
#   korean_learner 03:00:00(기본), 제출 때 덮어쓰기: native --time=01:30:00, lang8 --time=07:30:00, union --time=10:30:00
# 같은 run의 재개(RESUME=1)는 처음 시작한 GPU 구성으로 한다 (전역 batch 64·lr·스케줄은 구성과 무관하게 같음, micro 16 × accumulation 4).
# 제출 (저장소 루트에서, 처음 한 번 mkdir -p logs/slurm): sbatch gec2/slurm/h200x1_train.sbatch <data> <lr> <seed> [추가 인자]   (재개: RESUME=1 sbatch ...)
bash gec2/slurm/llm_train.sh "$@"
