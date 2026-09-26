#!/bin/bash
#SBATCH --job-name=gec2-precheck
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --comment="field=nlp;appl=pytorch"
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x_%j.out
# 자원: ssh.md showque 기준 cpu 파티션(GPU 불필요). 코어 4 × 10.4GB.
# --time 근거: itcerdo에서 같은 점검(고유 문장 27.9만 개 토큰화 + 4개 데이터셋 길이 통계)이 약 1분. CPU 4개 제한과 import 시간을 감안해 30분.
# 제출 (저장소 루트에서): sbatch gec2/slurm/precheck.sbatch
bash gec2/slurm/precheck.sh "$@"
