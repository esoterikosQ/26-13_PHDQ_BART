# P1. KoBART를 이용한 한국어 GEC 모델 학습

이 프로젝트는 [한국어 GEC](https://github.com/esoterikosQ/Korean-GEC) 연구의 P1 실험과 관련된 코드와 결과를 담고 있습니다.


## 1.1. 개요

[Yoon et al.(2023)](https://arxiv.org/abs/2210.14389)은 KoBART를 추가 학습하는 방식으로 한국어 GEC 모델을 구성하는 방식과 이를 평가하기 위한 KAGAS 체계를 제안했습니다.

이 저장소는 Yoon(2023) 논문에서 제안하는 방식으로 KoBART를 비롯한 토큰 기반의 언어 모델을 활용하여 한국어 GEC 모델을 학습하고 평가하는 코드와 결과를 담고 있습니다.

---

## 1.2. Yoon(2023) 논문 실험 재현

Yoon(2023) 논문에서는 KoBART를 활용한 한국어 GEC 모델 학습 실험의 데이터셋, 평가 방식, 학습 방법 등을 상세하게 기술하며, [논문의 실험을 재현할 수 있는 코드](https://github.com/soyoung97/Standard_Korean_GEC)를 제공하고 있습니다.

본 레포에서는 코드의 호환성 문제를 해결하고 저자의 데이터를 이용하여 KoBART 기반 한국어 GEC 모델을 재현하는 과정을 다룹니다. 구체적으로 아래와 같은 작업을 진행했습니다.

- 저자 재현 코드와 최신의 라이브러리 스택(torch 2.11, PyTorch Lightning 2.6, transformers 4.44) 간 호환성 문제를 해결

- 저자의 데이터를 이용하여 KoBART 기반 한국어 GEC 모델을 재현하고 평가

원본 코드와 수정 코드 간의 차이는 `docs/modification.md`를, KoBART 재현 결과는 `docs/result_kobart.md`를 참조하시기 바랍니다.

---

## 1.3. 언어 모델 확장

Yoon(2023)의 데이터, 학습 방식, 평가 체계를 다른 언어 모델로 확장하여 (1) 모델이 커지는 경우 GEC 성능도 개선되는지, (2) 모델 고유의 특성에 따라 GEC 성능에 차이가 생기는지를 확인하고자 한 실험을 진행했습니다.

실험에 사용한 모델은 아래와 같습니다.

- [mBART-50](https://huggingface.co/facebook/mbart-large-50) : 메타(페이스북)에서 개발한 seq2seq 기반의 다국어 모델. 한국어를 공식 지원하는 모델로, 별도의 사전학습 없이 한국어를 처리할 수 있으며, 다국어 모델의 GEC 성능이 얼마나 다른지를 확인하기 위해 선택했습니다.(파라미터 사이즈 : 611M)

- pko-t5 : 구글의 T5 모델을 기반으로 한국어 데이터를 이용하여 (주)파우스트가 한국어 데이터로 추가 학습한 모델. [base](https://huggingface.co/paust/pko-t5-base), [large](https://huggingface.co/paust/pko-t5-large) 두 종류의 모델을 제공하며, 모델의 크기와 GEC 성능의 관계를 확인하기 위해 선택했습니다.(파라미터 사이즈 : 276M, 821M)

- [kanana-1.5-2.1b-instruct-2505](https://huggingface.co/kanana-1.5-2.1b-instruct-2505) : 카카오에서 제공하는 모델로, 한국어를 기반으로 사전 학습한 모델입니다. LLM을 기반으로 GEC 모델을 생성했을 때의 성능 차이를 확인하기 위해 선택했습니다.(파라미터 사이즈 : 2.09B)

확장 단계의 구현은 [코드 해설](docs/code_comment.md)을, 결과는 [확장 모델 비교](docs/result_gec2.md)를 참조하시기 바랍니다.

---

## 1.4. 결과 요약

페이즈 1.2를 수행한 결과, 전반적인 성능이 논문에서 제시한 수치보다 다소 향상되었습니다. 라이브러리에서 지원하지 않는 기능 역시 최대한 논문에서 제시한 레시피와 유사하게 재현했기 때문에 성능 차이는 라이브러리 환경이 달라지면서 발생한 차이라고 판단됩니다.

페이즈 1.2의 결과는 아래와 같습니다. (3-seed 평균, 괄호는 논문에서 제시한 수치와의 차이)

| 시나리오 | 데이터 | GLEU 재현 / 논문 | Precision | Recall | F0.5 |
| --- | --- | --- | --- | --- | --- |
| 1 개별 | Kor-Learner | **45.68** / 45.06 (+0.62) | 43.31 / 43.35 (−0.04) | 25.46 / 24.54 (+0.92) | 37.98 / 37.58 (+0.40) |
| 1 개별 | Kor-Native | **68.40** / 67.24 (+1.16) | 80.34 / 75.34 (+5.00) | 58.31 / 55.95 (+2.36) | 74.69 / 70.45 (+4.24) |
| 1 개별 | Kor-Lang8 | **29.65** / 28.48 (+1.17) | 39.46 / 37.56 (+1.90) | 13.33 / 11.62 (+1.71) | 28.33 / 25.93 (+2.40) |
| 1 개별 | Kor-Union | **34.75** / 33.70 (+1.05) | 44.38 / 44.75 (−0.37) | 16.09 / 14.64 (+1.45) | 32.80 / 31.70 (+1.10) |
| 2 Union→개별 | Kor-Learner | **44.19** / 42.66 (+1.53) | 52.15 / 53.51 (−1.36) | 23.47 / 21.18 (+2.29) | 41.88 / 41.00 (+0.88) |
| 2 Union→개별 | Kor-Native | **60.96** / 59.71 (+1.25) | 86.28 / 85.47 (+0.81) | 48.73 / 47.38 (+1.35) | 74.70 / 73.63 (+1.07) |
| 2 Union→개별 | Kor-Lang8 | **29.35** / 28.65 (+0.70) | 37.84 / 37.46 (+0.38) | 13.05 / 12.00 (+1.05) | 27.41 / 26.78 (+0.63) |

페이즈 1.3을 수행한 결과, (1) 모델의 고유한 성능에 따라 GEC 성능에 차이가 생기고, (2) 같은 모델이라면 파라미터 사이즈가 큰 모델의 성능이 더 뛰어나다는 점을 확인할 수 있었습니다. (단, (2)의 결과는 T5 모델 한 종류만 놓고 내린 결론이기 때문에 스케일과 성능이 어느 정도 연관성이 있는지는 좀 더 다양한 사이즈 모델과 더 큰 데이터셋을 이용한 비교가 필요합니다.)

페이즈 1.3의 결과는 아래와 같습니다. (GLEU, 개별 데이터셋, 3-seed 평균)

| 모델 | Kor-Learner | Kor-Native | Kor-Lang8 | Kor-Union | 추론 속도(문장/초)* |
| --- | --- | --- | --- | --- | --- |
| KoBART(gec2) 124M | 52.46 | 78.69 | 35.65 | 44.40 | 1177 |
| pko-t5-base 276M | 57.45 | 82.36 | 37.95 | 47.17 | 432 |
| mBART-50 611M | 52.62 | 79.88 | 36.03 | 44.75 | 335 |
| **pko-t5-large 821M** | **59.19** | 82.70 | **39.84** | **49.16** | 213 |
| kanana-1.5-2.1b 2.09B | 58.52 | **82.96** | 39.11 | 48.66 | 54 |
| (참고) KoBART, 논문 recipe 재현 | 45.68 | 68.40 | 29.65 | 34.75 | — |

---

## 1.5. 레포 구조


| 문서 | 내용 |
| --- | --- |
| [`docs/modification.md`](docs/modification.md) | KoBART 재현: 원본 대비 변경, 환경 준비, 실행 방법 |
| [`docs/result_kobart.md`](docs/result_kobart.md) | 재현 결과(논문 대비), run별 점수, 체크포인트 사용법 |
| [`docs/result_gec2.md`](docs/result_gec2.md) | 대형 모델 비교 결과(3-seed 표, 시나리오 2, 오류 유형별 점수, 추론 속도, 논문과 비교) |
| [`docs/plan.md`](docs/plan.md), [`docs/plan2.md`](docs/plan2.md) | 재현 계획, 대형 모델 비교 계획과 근거 |
| [`docs/log.md`](docs/log.md) | 두 실험의 시간순 작업 기록(명령·결과) |
| [`gec2/NEURON.md`](gec2/NEURON.md) | Neuron에서 LLM 학습을 실행하는 방법 |

| 경로 | 내용 |
| --- | --- |
| `run.py`, `model.py`, `dataset.py`, `tokenizer_setup.py`, `tools/` | KoBART 재현 코드 (`--recipe legacy`가 논문 조건) |
| `gec2/` | 대형 모델 비교: 원문 기준 평가(`evaluate.py`), seq2seq·LLM 학습(`train_seq2seq.py`, `train_llm.py`), 사전 점검(`precheck.py`), 집계(`summarize.py`), 오류 유형별 점수(`errtype.py`), 추론 속도(`speed.py`), itcerdo 대기열(`queues/`), Neuron SLURM 스크립트(`slurm/`) |
| `metric/` | GLEU, M² scorer (원 저장소) |
| `KAGAS/`, `get_data/`, `src/`, `eval/` | 원 저장소 코드 (정답 M² 생성, 데이터 준비, 논문 당시 학습 코드) |


데이터셋은 이 저장소에 포함하지 않는다. Kor-Learner·Kor-Native·Kor-Lang8은 원 저자의 [신청서](https://forms.gle/kF9pvJbLGvnh8ZnQ6)를 통해 받으며 **비상업 목적으로만 사용·배포할 수 있다**. 데이터 준비 절차는 원 저장소 README(`docs/ORIGINAL_README.md`)를 따른다.

---

## 1.6. 작업 환경

- KoBART, T5, mBART 추가 학습과 평가 : RTX 5090(Cuda 12.9) / Ubuntu 24.04

- Kanana 추가 학습과 평가 : KISTI Neuron(A100 × 2) / Rocky Linux 9.4

\* RTX 5090 1장, 빔 4, batch 64에서 4개 테스트셋 전체를 생성한 처리량. 환경에 따라 절대 배율은 달라진다.

- 한국어 전용 사전학습 T5(pko-t5-large)가 가장 좋고, 2.5배 큰 LLM(kanana)은 비슷하거나 약간 낮으면서 약 4배 느리다.
- 모델 계열 효과(KoBART → pko-t5-base +2.3~+5.0)가 같은 계열 안의 크기 효과(pko-t5 base → large 약 +2, native는 포화)보다 크다.
- 표준 학습 코드로 학습한 KoBART가 논문 recipe 재현보다 6–10점 높다.
- 자세한 결과(P/R/F0.5, 시나리오 2, 오류 유형별 점수, 논문과 비교): [`docs/result_gec2.md`](docs/result_gec2.md)



___

## 출처 및 라이선스

이 저장소에서 사용한 코드는 [soyoung97/Standard_Korean_GEC](https://github.com/soyoung97/Standard_Korean_GEC) (`dfe0af9`)에 기반하여 만들었습니다. 원 저장소 코드와 그 파생 코드는 원 저자의 Modified MIT License를 따릅니다([`LICENSE`](LICENSE)).

원 논문의 인용 표기는 아래와 같습니다.
```
@article{yoon2022towards,
  title={Towards Standardizing Korean Grammatical Error Correction: Datasets and Annotation},
  author={Yoon, Soyoung and Park, Sungjoon and Kim, Gyuwan and Cho, Junhee and Park, Kihyo and Kim, Gyu Tae and Seo, Minjoon and Oh, Alice},
  journal={arXiv preprint arXiv:2210.14389},
  year={2022}
}
```






1. **KoBART 재현** — *Towards Standardizing Korean Grammatical Error Correction: Datasets and Annotation* (Yoon et al., ACL 2023)의 KoBART 실험(Table 6)을 `skt/kobart-base-v1`과 최신 스택(torch 2.11, PyTorch Lightning 2.6, transformers 4.44)으로 재현했다. 7개 조합 모두 논문 수치에 도달했다(테스트 GLEU 3-seed 평균이 논문과 같거나 0.6–1.5점 높음). 재현에 결정적이었던 것은 라이브러리 버전 차이로 달라진 동작 3가지(토큰화 형식, 디코더 첫 토큰 마스킹, 학습 중 dropout)였다.
2. **대형 모델 비교 (`gec2/`)** — 같은 데이터·평가로 KoBART보다 큰 모델 4개(pko-t5-base 276M, mBART-50 611M, pko-t5-large 821M, kanana-1.5-2.1b 2.1B 전체 파인튜닝)를 파인튜닝해, 크기·구조·사전학습 언어에 따른 성능을 비교했다(4개 데이터셋 × 2개 시나리오 × 3 seed). seq2seq는 RTX 5090, LLM은 KISTI Neuron(A100 × 2 / H200 × 1)에서 실행했다.
