# 한국어 문법 오류 교정(GEC): KoBART 재현과 대형 모델 비교

이 저장소는 두 가지 실험의 코드와 기록을 담는다.

1. **KoBART 재현** — *Towards Standardizing Korean Grammatical Error Correction: Datasets and Annotation* (Yoon et al., ACL 2023)의 KoBART 실험(Table 6)을 `skt/kobart-base-v1`과 최신 스택(torch 2.11, PyTorch Lightning 2.6, transformers 4.44)으로 재현했다. 7개 조합 모두 논문 수치에 도달했다(테스트 GLEU 3-seed 평균이 논문과 같거나 0.6–1.5점 높음). 재현에 결정적이었던 것은 라이브러리 버전 차이로 달라진 동작 3가지(토큰화 형식, 디코더 첫 토큰 마스킹, 학습 중 dropout)였다.
2. **대형 모델 비교 (`gec2/`)** — 같은 데이터·평가로 KoBART보다 큰 모델 4개(pko-t5-base 276M, mBART-50 611M, pko-t5-large 821M, kanana-1.5-2.1b 2.1B 전체 파인튜닝)를 파인튜닝해, 크기·구조·사전학습 언어에 따른 성능을 비교했다(4개 데이터셋 × 2개 시나리오 × 3 seed). seq2seq는 RTX 5090, LLM은 KISTI Neuron(A100 × 2 / H200 × 1)에서 실행했다.

### 대형 모델 비교 결과 요약

테스트 GLEU, 시나리오 1(개별 데이터셋), 3-seed 평균:

| 모델 | Kor-Learner | Kor-Native | Kor-Lang8 | Kor-Union | 추론 속도(문장/초)* |
| --- | --- | --- | --- | --- | --- |
| KoBART(gec2) 124M | 52.46 | 78.69 | 35.65 | 44.40 | 1177 |
| pko-t5-base 276M | 57.45 | 82.36 | 37.95 | 47.17 | 432 |
| mBART-50 611M | 52.62 | 79.88 | 36.03 | 44.75 | 335 |
| **pko-t5-large 821M** | **59.19** | 82.70 | **39.84** | **49.16** | 213 |
| kanana-1.5-2.1b 2.09B | 58.52 | **82.96** | 39.11 | 48.66 | 54 |
| (참고) KoBART, 논문 recipe 재현 | 45.68 | 68.40 | 29.65 | 34.75 | — |

\* RTX 5090 1장, 빔 4, batch 64에서 4개 테스트셋 전체를 생성한 처리량. 환경에 따라 절대 배율은 달라진다.

- 한국어 전용 사전학습 T5(pko-t5-large)가 가장 좋고, 2.5배 큰 LLM(kanana)은 비슷하거나 약간 낮으면서 약 4배 느리다.
- 모델 계열 효과(KoBART → pko-t5-base +2.3~+5.0)가 같은 계열 안의 크기 효과(pko-t5 base → large 약 +2, native는 포화)보다 크다.
- 표준 학습 코드로 학습한 KoBART가 논문 recipe 재현보다 6–10점 높다.
- 자세한 결과(P/R/F0.5, 시나리오 2, 오류 유형별 점수, 논문과 비교): [`docs/result_gec2.md`](docs/result_gec2.md)

## 문서

| 문서 | 내용 |
| --- | --- |
| [`REPRODUCTION.md`](REPRODUCTION.md) | KoBART 재현: 원본 대비 변경, 환경 준비, 실행 방법 |
| [`docs/result_kobart.md`](docs/result_kobart.md) | 재현 결과(논문 대비), run별 점수, 체크포인트 사용법 |
| [`docs/result_gec2.md`](docs/result_gec2.md) | 대형 모델 비교 결과(3-seed 표, 시나리오 2, 오류 유형별 점수, 추론 속도, 논문과 비교) |
| [`docs/plan.md`](docs/plan.md), [`docs/plan2.md`](docs/plan2.md) | 재현 계획, 대형 모델 비교 계획과 근거 |
| [`docs/log.md`](docs/log.md) | 두 실험의 시간순 작업 기록(명령·결과) |
| [`gec2/NEURON.md`](gec2/NEURON.md) | Neuron에서 LLM 학습을 실행하는 방법 |

## 구성

| 경로 | 내용 |
| --- | --- |
| `run.py`, `model.py`, `dataset.py`, `tokenizer_setup.py`, `tools/` | KoBART 재현 코드 (`--recipe legacy`가 논문 조건) |
| `gec2/` | 대형 모델 비교: 원문 기준 평가(`evaluate.py`), seq2seq·LLM 학습(`train_seq2seq.py`, `train_llm.py`), 사전 점검(`precheck.py`), 집계(`summarize.py`), 오류 유형별 점수(`errtype.py`), 추론 속도(`speed.py`), itcerdo 대기열(`queues/`), Neuron SLURM 스크립트(`slurm/`) |
| `metric/` | GLEU, M² scorer (원 저장소) |
| `KAGAS/`, `get_data/`, `src/`, `eval/` | 원 저장소 코드 (정답 M² 생성, 데이터 준비, 논문 당시 학습 코드) |

## 데이터

데이터셋은 이 저장소에 포함하지 않는다. Kor-Learner·Kor-Native·Kor-Lang8은 원 저자의 [신청서](https://forms.gle/kF9pvJbLGvnh8ZnQ6)를 통해 받으며 **비상업 목적으로만 사용·배포할 수 있다**. 데이터 준비 절차는 원 저장소 README(`docs/ORIGINAL_README.md`)를 따른다.

## 출처와 라이선스

이 저장소는 [soyoung97/Standard_Korean_GEC](https://github.com/soyoung97/Standard_Korean_GEC) (`dfe0af9`)을 바탕으로 만들었다. 원 저장소 코드와 그 파생 코드는 원 저자의 Modified MIT License를 따른다([`LICENSE`](LICENSE)). 원 저장소 README는 [`docs/ORIGINAL_README.md`](docs/ORIGINAL_README.md)에 그대로 보존했다(GitHub 토큰 형태 문자열이 있던 링크 1곳만 제거).

원 논문(원 저장소의 인용 표기):

```
@article{yoon2022towards,
  title={Towards Standardizing Korean Grammatical Error Correction: Datasets and Annotation},
  author={Yoon, Soyoung and Park, Sungjoon and Kim, Gyuwan and Cho, Junhee and Park, Kihyo and Kim, Gyu Tae and Seo, Minjoon and Oh, Alice},
  journal={arXiv preprint arXiv:2210.14389},
  year={2022}
}
```
