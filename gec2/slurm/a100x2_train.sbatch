#!/bin/bash
#SBATCH --job-name=gec2-a100x2_train
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --comment="field=nlp;appl=pytorch-ddp"
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%x_%j.out
# 자원: ssh.md의 showque/showappl 기준 (amd_a100nv_8: GPU당 코어 8, --gres 필수, --comment 형식 field=nlp;appl=pytorch-ddp).
# --time 근거: 임시 값. 대표 epoch 측정(--epochs_per_job 1)용: plan2 기준 1 epoch ≤ 6시간 + 모델 로드·누설 검사 약 10분 + 여유. 실측 후 'epoch당 시간 × 최대 10 epoch + 준비·최종 테스트 + 20%'로 바꾸고 계산식을 여기에 남긴다. 48시간을 넘으면 --epochs_per_job K로 나눠 제출하고 RESUME=1로 이어간다.
# 제출 (저장소 루트에서, 처음 한 번 mkdir -p logs/slurm): sbatch gec2/slurm/a100x2_train.sbatch <data> <lr> <seed> [추가 인자]   (재개: RESUME=1 sbatch ...)
bash gec2/slurm/llm_train.sh "$@"
