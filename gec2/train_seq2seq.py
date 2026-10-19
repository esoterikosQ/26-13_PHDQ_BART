# KoBART(gec2), pko-t5, mBART-50 모델 학습·평가용 코드
# RTX5090 노드에서 실행
# run_queue.sh를 실행하는 방식으로 작동
from gec2.core import main

if __name__ == '__main__':
    main('seq2seq')
