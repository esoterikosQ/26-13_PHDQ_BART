#!/bin/bash
#SBATCH --job-name=gec2-a100x2_train
#SBATCH --partition=amd_a100nv_8
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --comment="field=nlp;appl=pytorch-ddp"
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x_%j.out
# 자원: ssh.md의 showque/showappl 기준 (amd_a100nv_8: GPU당 코어 8, --gres 필수, --comment 형식 field=nlp;appl=pytorch-ddp).
# --time 근거: 대표 epoch 측정용(--epochs_per_job 1). 스모크 915975(A100×2, GC 켬): native 512줄 학습 9.2초, 검증 512줄 생성 11.8초·검증 전체 31.7초(최고 체크포인트 8GB 저장 포함)
#   → korean_learner 1 epoch(학습 19,898 + 검증 생성 4,264 + 저장) 약 15–20분 추정 + 모델 로드·누설 검사 약 5분, 첫 측정이라 여유를 크게 둬 2시간.
#   실측 후 'epoch당 시간 × 최대 10 epoch + 준비·최종 테스트 + 20%'로 바꾸고 계산식을 여기에 남긴다. 48시간을 넘으면 --epochs_per_job K로 나눠 제출하고 RESUME=1로 이어간다.
# 제출 (저장소 루트에서, 처음 한 번 mkdir -p logs/slurm): sbatch gec2/slurm/a100x2_train.sbatch <data> <lr> <seed> [추가 인자]   (재개: RESUME=1 sbatch ...)
bash gec2/slurm/llm_train.sh "$@"
