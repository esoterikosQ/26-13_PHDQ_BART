#!/bin/bash
#SBATCH --job-name=gec2-h200x1_smoke
#SBATCH --partition=amd_h200nv_8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x_%j.out
# 자원: ssh.md의 showque/showappl 기준 (amd_h200nv_8: GPU당 코어 8, --gres 필수, --comment 형식 field=nlp;appl=pytorch).
# --time 근거: 모델 로드 약 2분 + 메모리 측정 5회(각 1–2분) + native 512줄 1 epoch 학습·검증 512줄·테스트 256줄(5분 이하 예상) + 여유.
# 제출 (저장소 루트에서, 처음 한 번 mkdir -p logs/slurm): sbatch gec2/slurm/h200x1_smoke.sbatch 
bash gec2/slurm/llm_smoke.sh "$@"
