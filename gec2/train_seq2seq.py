# plan2.md §3 seq2seq 학습·평가 (KoBART(gec2), pko-t5, mBART-50 등). itcerdo에서 실행.
# 사용 (저장소 루트에서): python3 -m gec2.train_seq2seq --model paust/pko-t5-base --data korean_learner --run_id lr3e-4_seed0 --lr 3e-4
# 기본: 10 epoch 고정(조기 종료 없음), dropout 0.1, 전역 batch 64, bf16-mixed, 빔 4. 최고 검증 GLEU 체크포인트로 테스트.
from gec2.core import main

if __name__ == '__main__':
    main('seq2seq')
