# plan2.md §3 decoder-only LLM 전체 파인튜닝 (kanana-1.5-2.1b 등). Neuron에서 실행 (gec2/slurm/*.sbatch).
# 사용 (저장소 루트에서): python3 -m gec2.train_llm --model <HF id 또는 로컬 경로> --data korean_learner --run_id lr1e-5_seed0 --lr 1e-5 \
#          --devices 2 --micro_batch 8 --gen_batch_size 32 --gradient_checkpointing
# 기본: 최대 10 epoch, 조기 종료(min_delta 0.2, patience 3, 4 epoch 이후), dropout은 설정 기본값, 손실은 출력 부분에만.
from gec2.core import main

if __name__ == '__main__':
    main('llm')
