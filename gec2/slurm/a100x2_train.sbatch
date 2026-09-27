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
# --time 근거: 대표 epoch 실측 (작업 916172–916174, korean_learner, A100×2, micro 16 × accumulation 2, GC 없음, 생성 batch 64)
#   준비(모델 로드·인코딩·누설 검사) 1.5분 / epoch당 학습 3.6–3.7분 + 검증 2.1–2.8분(최고 체크포인트 저장 포함) + 재개용 체크포인트 저장 1.2–1.3분 ≈ 7.5분
#   korean_learner 전체: 1.5 + 10 × 7.5 + 최종 테스트 약 4분(생성 4,265문장 + 모델 로드) = 80.5분 × 1.2 = 97분 → 02:00:00
#   재개 작업 916205(korean_learner 9 epoch + 테스트 49분) → 이후 epoch는 약 5.1분(첫 epoch만 7.5분).
#   native 실측 (작업 916288): 10 epoch 32분(epoch당 3.2분 = korean_learner 5.1분 × 문장 수 비 0.62, 선형 확인) + 준비·테스트 2분 = 34분.
#   다른 데이터셋은 제출할 때 --time으로 덮어쓴다 (epoch당 = 5.1분 × 학습 문장 수 / 19,898):
#     native  실측 34분                                         → sbatch --time=01:00:00
#     lang8   19.7분 × 10 + 준비·테스트 약 8분 = 205분 × 1.2 = 4.1시간  → sbatch --time=05:00:00
#     union   27.9분 × 10 + 준비·테스트 약 12분 = 291분 × 1.2 = 5.8시간 → sbatch --time=07:00:00
#   48시간(파티션 최대)을 넘지 않으므로 작업 분할은 필요 없다.
# 제출 (저장소 루트에서, 처음 한 번 mkdir -p logs/slurm): sbatch gec2/slurm/a100x2_train.sbatch <data> <lr> <seed> [추가 인자]   (재개: RESUME=1 sbatch ...)
bash gec2/slurm/llm_train.sh "$@"
