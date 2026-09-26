# KoBART GEC 재현 결과

2026-09-26 · 논문 *Towards Standardizing Korean Grammatical Error Correction* (Yoon et al., ACL 2023)의 KoBART 실험(Table 6)을 `skt/kobart-base-v1`로 재현한 최종 결과. 작업 과정의 시간순 기록은 `log.md`, 계획과 근거는 `plan.md`(특히 §3-6, §3-7)에 있다.

## 1. 요약

- **논문 수치를 7개 조합(시나리오 1: 4개 데이터셋, 시나리오 2: 3개 데이터셋) 모두 재현했다.** 테스트 GLEU는 3-seed 평균으로 논문과 같거나 0.6–1.5점 높다.
- 재현에 결정적이었던 것은 하이퍼파라미터가 아니라 **라이브러리 버전 차이로 조용히 달라진 동작 3가지**였다. 이를 논문 당시 동작으로 맞췄다(§4).
- 평가 파이프라인은 저자 공개 체크포인트로 검증했다. 저자 체크포인트를 우리 파이프라인으로 채점하면 검증 GLEU가 저자 기록과 같은 46.91이 나온다.

## 2. 최종 결과 (테스트셋, 3-seed 평균)

각 run은 검증 GLEU가 가장 높은 epoch의 체크포인트로 테스트했다. 괄호는 논문과의 차이.

| 시나리오 | 데이터 | GLEU 재현 / 논문 | Precision | Recall | F0.5 |
| --- | --- | --- | --- | --- | --- |
| 1 개별 | Kor-Learner | **45.68** / 45.06 (+0.62) | 43.31 / 43.35 (−0.04) | 25.46 / 24.54 (+0.92) | 37.98 / 37.58 (+0.40) |
| 1 개별 | Kor-Native | **68.40** / 67.24 (+1.16) | 80.34 / 75.34 (+5.00) | 58.31 / 55.95 (+2.36) | 74.69 / 70.45 (+4.24) |
| 1 개별 | Kor-Lang8 | **29.65** / 28.48 (+1.17) | 39.46 / 37.56 (+1.90) | 13.33 / 11.62 (+1.71) | 28.33 / 25.93 (+2.40) |
| 1 개별 | Kor-Union | **34.75** / 33.70 (+1.05) | 44.38 / 44.75 (−0.37) | 16.09 / 14.64 (+1.45) | 32.80 / 31.70 (+1.10) |
| 2 Union→개별 | Kor-Learner | **44.19** / 42.66 (+1.53) | 52.15 / 53.51 (−1.36) | 23.47 / 21.18 (+2.29) | 41.88 / 41.00 (+0.88) |
| 2 Union→개별 | Kor-Native | **60.96** / 59.71 (+1.25) | 86.28 / 85.47 (+0.81) | 48.73 / 47.38 (+1.35) | 74.70 / 73.63 (+1.07) |
| 2 Union→개별 | Kor-Lang8 | **29.35** / 28.65 (+0.70) | 37.84 / 37.46 (+0.38) | 13.05 / 12.00 (+1.05) | 27.41 / 26.78 (+0.63) |

검증셋 최고 GLEU(3-seed 평균) vs 논문 Table D.1:

| 시나리오 | Kor-Learner | Kor-Native | Kor-Lang8 | Kor-Union |
| --- | --- | --- | --- | --- |
| 1 개별 | 46.94 / 46.94 | 70.29 / 69.37 | 28.88 / 28.57 | 34.50 / 34.07 |
| 2 Union→개별 | 44.94 / 44.66 | 62.56 / 61.64 | 28.56 / 28.51 | — |

- 시나리오 2(Union 선학습)에서 정밀도가 오르고 재현율이 내려가는 경향도 논문과 같다.
- Kor-Native 시나리오 1의 정밀도는 논문보다 5.0점 높다. 이 조합은 seed 간 GLEU 범위가 2.46이고, 시나리오 2 Kor-Native는 4.83으로 seed 편차가 가장 크다.

## 3. 재현 가중치 위치와 run별 결과

모든 파일은 itcerdo `~/projects/phdq_bart/Standard_Korean_GEC/` 아래에 있다(아래 경로는 이 디렉토리 기준). run마다 검증 GLEU 최고 체크포인트 1개(각 1.4GB)만 남아 있다.

- 체크포인트: `outputs/<data>/<run_id>/model_ckpt/<data>_<lr>_epoch=<NN>.ckpt`
- 테스트 생성 결과: `outputs/generation/<data>/<run_id>/epoch<NN>/test/` (`hypothesis.txt`, `reference.txt`, `source.txt`, `gleu.txt`, `m2score.txt`)
- 실행 로그: `logs/<data>_<run_id>_{check,train,test,m2}.log`, 특수 토큰 기록 `outputs/<data>/<run_id>/special_tokens.json`
- run 진행 기록: `logs/runs.log`

| 데이터 | run_id | 체크포인트 파일 (`outputs/<data>/<run_id>/model_ckpt/`) | best epoch | 검증 GLEU | 테스트 GLEU | P | R | F0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| korean_learner | dropout_seed0 | `korean_learner_3e-05_epoch=03.ckpt` | 3 | 47.19 | 45.38 | 42.88 | 25.38 | 37.68 |
| korean_learner | dropout_seed1 | `korean_learner_3e-05_epoch=06.ckpt` | 6 | 46.67 | 45.59 | 43.89 | 25.19 | 38.21 |
| korean_learner | dropout_seed2 | `korean_learner_3e-05_epoch=05.ckpt` | 5 | 46.97 | 46.08 | 43.16 | 25.81 | 38.04 |
| native | dropout_seed0 | `native_3e-05_epoch=09.ckpt` | 9 | 69.89 | 67.94 | 80.73 | 57.63 | 74.74 |
| native | dropout_seed1 | `native_3e-05_epoch=04.ckpt` | 4 | 69.77 | 67.39 | 78.19 | 57.56 | 72.96 |
| native | dropout_seed2 | `native_3e-05_epoch=06.ckpt` | 6 | 71.20 | 69.86 | 82.10 | 59.75 | 76.38 |
| lang8 | dropout_seed0 | `lang8_3e-05_epoch=04.ckpt` | 4 | 28.59 | 29.40 | 40.27 | 12.71 | 28.09 |
| lang8 | dropout_seed1 | `lang8_3e-05_epoch=07.ckpt` | 7 | 28.89 | 29.55 | 38.18 | 13.46 | 27.93 |
| lang8 | dropout_seed2 | `lang8_3e-05_epoch=05.ckpt` | 5 | 29.16 | 30.00 | 39.93 | 13.82 | 28.98 |
| union | dropout_seed0 | `union_3e-05_epoch=06.ckpt` | 6 | 33.96 | 34.05 | 44.72 | 15.25 | 32.25 |
| union | dropout_seed1 | `union_3e-05_epoch=03.ckpt` | 3 | 35.47 | 35.80 | 44.28 | 17.32 | 33.76 |
| union | dropout_seed2 | `union_3e-05_epoch=05.ckpt` | 5 | 34.08 | 34.40 | 44.15 | 15.69 | 32.40 |
| korean_learner | afterunion_dropout_seed0 | `korean_learner_1e-05_epoch=02.ckpt` | 2 | 44.54 | 43.90 | 53.56 | 23.19 | 42.45 |
| korean_learner | afterunion_dropout_seed1 | `korean_learner_1e-05_epoch=01.ckpt` | 1 | 46.79 | 45.64 | 51.96 | 25.37 | 42.95 |
| korean_learner | afterunion_dropout_seed2 | `korean_learner_1e-05_epoch=00.ckpt` | 0 | 43.50 | 43.04 | 50.93 | 21.86 | 40.23 |
| native | afterunion_dropout_seed0 | `native_1e-05_epoch=09.ckpt` | 9 | 60.12 | 58.83 | 87.33 | 45.64 | 73.84 |
| native | afterunion_dropout_seed1 | `native_1e-05_epoch=05.ckpt` | 5 | 65.08 | 63.65 | 85.81 | 52.20 | 76.02 |
| native | afterunion_dropout_seed2 | `native_1e-05_epoch=06.ckpt` | 6 | 62.48 | 60.39 | 85.71 | 48.36 | 74.24 |
| lang8 | afterunion_dropout_seed0 | `lang8_1e-05_epoch=00.ckpt` | 0 | 28.20 | 28.97 | 37.95 | 12.47 | 26.94 |
| lang8 | afterunion_dropout_seed1 | `lang8_1e-05_epoch=00.ckpt` | 0 | 29.43 | 30.33 | 38.62 | 14.25 | 28.78 |
| lang8 | afterunion_dropout_seed2 | `lang8_1e-05_epoch=00.ckpt` | 0 | 28.05 | 28.75 | 36.96 | 12.43 | 26.50 |

시나리오 2 run(`afterunion_dropout_seedN`)은 같은 seed의 union 체크포인트(`outputs/union/dropout_seedN/model_ckpt/…`)에서 시작했다.

**진단용 run.** 원인 진단 과정의 run(`individual_seed0`, `legacy_seed0`, `legacyfix_seed0–2`, `afterunion_seed0–2`)은 모두 dropout 없이, 일부는 디코더 누설이 있는 상태로 학습됐다. 이 run들의 체크포인트 23개(32GB)는 2026-09-26에 삭제했고, 생성 결과(`outputs/generation/…`)와 로그는 진단 근거로 남겨 두었다. 현재 `outputs/`에는 위 21개 체크포인트만 있다(32GB).

### 체크포인트 사용법

체크포인트는 PyTorch Lightning 형식이며, BART 가중치는 `state_dict`의 `model.` 접두사 아래에 있다. 논문과 같은 출력을 얻으려면 토크나이저와 생성 설정을 아래처럼 맞춘다(저장소 루트에서 `.venv` 활성화 후).

```python
import torch
from transformers import BartForConditionalGeneration
from tokenizer_setup import load_tokenizer

ck = torch.load('outputs/korean_learner/dropout_seed0/model_ckpt/korean_learner_3e-05_epoch=03.ckpt',
                map_location='cpu', weights_only=False)
model = BartForConditionalGeneration.from_pretrained('skt/kobart-base-v1')
model.load_state_dict({k[len('model.'):]: v for k, v in ck['state_dict'].items() if k.startswith('model.')})
model.eval()
tok = load_tokenizer(add_bos_eos=False)   # 논문 실험과 같은 토큰화: <s>…</s>를 붙이지 않음

ids = torch.tensor([tok.encode('지금부터 방콕에 소개한다.')])
out = model.generate(ids, eos_token_id=1, max_length=128, num_beams=4,
                     decoder_start_token_id=0, repetition_penalty=1.0)
print(tok.decode(out[0], skip_special_tokens=True))
```

평가를 다시 돌릴 때는 `run.py --recipe legacy --eval_test --data <data> --run_id <새 id> --model_ckpt_path <ckpt> --batch_size 64 --max_seq_len 128`을 쓴다.

## 4. 재현에 필요했던 수정

논문 실험은 저자 저장소의 레거시 코드(`src/KoBART-gec/train.py`)로 transformers 4.0.0, PyTorch Lightning 1.1.0에서 수행됐다(저자 spreadsheet의 실행 명령, 체크포인트 메타데이터로 확인). 루트 코드를 최신 스택(torch 2.11, PL 2.6.6, transformers 4.44.2)으로 옮기면서 다음 세 가지 동작이 달라져 있었다.

| # | 문제 | 증상 | 수정 | 효과 (Kor-Learner seed 0 검증 GLEU) |
| --- | --- | --- | --- | --- |
| 1 | 토큰화 형식. 루트 코드용 `hyunwoongko/kobart` 형식(`<s>…</s>` 부착)이 논문 실험의 SKT 토크나이저 형식(부착 안 함)과 다름 | — | `--recipe legacy`: 후처리 없음, cosine·warmup 0.1·clip 1.0, 생성 시작 토큰 0, repetition_penalty 없음 | (2와 함께 적용) |
| 2 | 디코더 첫 토큰 마스킹. `ne(0)` 마스크가 디코더 첫 토큰(0)을 가리고, 최신 transformers가 전부 가려진 행을 '전부 보기'로 풀어 **첫 토큰 예측에 정답이 샘**. transformers 4.0.0은 첫 토큰을 마스킹하지 않았음 | 생성 문장 첫머리가 중복되거나 탈락 | `model.make_decoder_attention_mask()`: 첫 디코더 토큰은 마스킹하지 않음 | 32.50 → 42.23 |
| 3 | 학습 중 dropout 꺼짐. `from_pretrained()`는 eval 모드로 반환하고, PL 2.x는 학습 시작 때 `model.train()`을 호출하지 않음(PL 1.1은 호출) | 과적합(학습 loss 0.01), 검증 GLEU 조기 하락, 재현율 부족 | `on_train_epoch_start()`마다 `self.train()` | 42.23 → 47.19 (저자 46.91) |

2와 3은 매 학습에서 자동으로 검사된다. `tools/check_special_tokens.py legacy`의 누설 검사가 실패하거나, 학습 첫 배치에서 모델이 eval 모드면 학습을 시작하지 않는다.

## 5. 검증 근거

| 검증 | 결과 |
| --- | --- |
| self-GLEU (원문을 출력으로 간주) | Kor-Learner·Native·Lang8 검증·테스트 모두 논문과 소수점 둘째 자리까지 일치 → 데이터 분할·GLEU 계산 동일 |
| 토크나이저 | 레거시 SKT 토크나이저 파일과 전 섹션 동일, 전체 데이터 155,547쌍 인코딩·`dataset.py` 출력 동일 |
| 가중치 | 레거시 `kobart_base_cased` = `skt/kobart-base-v1` (최대 절대차 0.0) |
| 모델 계산·빔서치 | transformers 4.0.0 + torch 1.7.1 환경(CPU)에서 같은 체크포인트로 비교: 생성 256/256 토큰 일치, logits 차이 ≤ 9e-5 |
| 정답 M² | 공식 `native_test.m2`를 현재 KAGAS로 재생성해 비교: 편집 2632/2634블록 일치 |
| 평가 파이프라인 | 저자 체크포인트(run 122)를 우리 파이프라인으로 채점: 검증 GLEU 46.91 = 저자 기록 46.9129, 테스트 GLEU 45.30 |
| 학습 설정 | 저자 yaml·체크포인트의 lr 스케줄(step 1244에서 2.25e-5)·AdamW 설정이 legacy recipe와 일치 |

## 6. 실험 설정

- **환경:** itcerdo (RTX 5090 32GB, Ubuntu 24.04), Python 3.12.3, torch 2.11.0+cu128, pytorch-lightning 2.6.6, transformers 4.44.2, numpy 1.26.4
- **모델:** `skt/kobart-base-v1` (123.9M), recipe `legacy`
- **학습:** lr 3e-5(시나리오 1) / 1e-5(시나리오 2), batch 64, 10 epoch, max_seq_len 128, AdamW(`correct_bias=False`, weight decay 0.01), cosine 스케줄 + warmup 0.1, gradient clipping 1.0, dropout 0.1, fp32, 단일 GPU, seed 0/1/2
- **생성:** beam 4, max_length 128, eos 1, 생성 시작 토큰 0, repetition_penalty 없음
- **데이터:** `~/projects/phdq_bart/data/Preprocessed/<data>/<data>_{train,val,test}.txt` (Kor-Learner 19,898/4,264/4,265, Native 12,292/2,634/2,634, Lang8 76,692/16,434/16,434, Union 108,882/23,332/23,333)
- **평가:** GLEU(`metric/gleumodule.py`), M²(`metric/m2scorer`, 정답은 데이터의 공식 `<data>_test.m2`)
- **소요:** 21 run, 2026-09-25 22:42 – 2026-09-26 10:17 (Native 약 9.5분, Kor-Learner 15분, Lang8 53분, Union 75분 / run)

## 7. 재실행 방법

itcerdo에서 tmux 안에서 실행한다.

```bash
cd ~/projects/phdq_bart/Standard_Korean_GEC
source .venv/bin/activate            # 캐시를 ~/projects/phdq_bart/.cache로 고정
bash tools/run_all.sh <tag>       # 21 run 전체: 특수 토큰 점검 → 학습 → 테스트 → M² (완료 run은 건너뜀)
bash tools/run_one.sh <data> <run_id> <seed> <lr> legacy [init_ckpt]   # 개별 run
python3 tools/summarize_runs.py logs/runs.log <tag>                     # 3-seed 평균 vs 논문 표
```

## 8. 코드와 보관 위치

- **코드**: https://github.com/esoterikosQ/26-13_PHDQ_BART (`main`). 첫 커밋은 저자 코드(`soyoung97/Standard_Korean_GEC` `dfe0af9`)를 가져온 것이고, 그 위에 재현 코드와 문서 커밋이 있다. 저자 README에 GitHub 토큰 형태의 문자열이 있어 GitHub push 보호에 막혔기 때문에, 저자 이력은 가져오지 않고 해당 문자열만 지운 상태로 가져왔다(나머지 코드는 동일). 환경 준비·실행 방법은 저장소의 `REPRODUCTION.md`, 이 문서와 `plan.md`·`log.md`는 `docs/`에 있다.
- **실행 스크립트**는 저장소 안 `tools/`로 옮겼다(이전 위치 `~/projects/phdq_bart/tools/`는 삭제). `log.md`의 과거 명령에 나오는 `../tools/…`는 당시 경로다.
- **저자 체크포인트** run 122는 `~/projects/phdq_bart/author_ckpt/122/`에, transformers 4.0.0 비교 환경은 `~/projects/phdq_bart/.venv-tf400`에 있다.
- 체크포인트·데이터·로그는 저장소에 넣지 않았다(itcerdo에만 있음).
