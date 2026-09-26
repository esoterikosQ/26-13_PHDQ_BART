# KoBART GEC 재현 실험 계획서

2026-09-18 · @Someone

## 0. 작업 기록 규칙 (항상 수행)

모든 작업은 gsm의 `phdq_bart2/log.md`에 **날짜·시간순**으로 기록한다. 작업 단위가 끝날 때마다(명령 결과를 확인한 직후) 바로 추가하고, 나중에 몰아서 쓰지 않는다.

- **형식**: 날짜 제목(`## YYYY-MM-DD`) 아래에 `### HH:MM(–HH:MM) · 작업 제목 [gsm/itcerdo]`을 두고, 다음 세 가지를 적는다.
  1. **작업**: 무엇을 왜 했는지 한두 줄
  2. **명령**: 실제로 입력한 명령(코드 블록). 긴 스크립트는 파일 경로와 실행 명령만 적는다
  3. **결과**: 수치·판정·생성 파일·로그 경로. 실패와 예상 밖의 결과, 지정 디렉토리 밖 접근 같은 규칙 위반도 숨기지 않고 적는다
- **시각**: KST. 명령을 실행한 실제 시각을 쓴다.
- **학습·평가 run**: run_id, 데이터셋, seed, 주요 인자, 체크포인트 경로, 검증 최고 GLEU(epoch), 테스트 GLEU, M² P/R/F0.5, 로그 경로를 반드시 남긴다.
- **결정·계획 변경**: 이유와 함께 기록하고, `plan.md`의 해당 절도 같이 고친다.
- itcerdo의 상세 출력은 `Standard_Korean_GEC/logs/`에 파일로 남기고 `log.md`에는 그 경로를 적는다.

## 1. 개요

**목표**: 논문 *Towards Standardizing Korean Grammatical Error Correction* (Yoon et al., ACL 2023)의 KoBART 파인튜닝 절차를 `skt/kobart-base-v1`로 실행하고, 최종 테스트셋 점수가 논문의 보고값과 어느 정도 일치하는지 확인한다. 원본 모델을 같은 환경에서 재학습하는 대조군 실험은 수행하지 않는다.

**작업 위치**: 현재 접속 중인 노드는 gsm이다. gsm에서는 itcerdo 접속과 필요시 데이터 전송만 하고, 저장소 준비·코드 수정·데이터 배치·환경 설치·학습·평가는 모두 itcerdo의 지정 작업 디렉토리 `~/projects/phdq_bart` 아래에서 진행한다. 이 계획의 상대경로 명령은 별도 표기가 없으면 itcerdo의 저장소 루트에서 실행한다. gsm이나 neuron에서 학습하지 않는다.

**원 논문 모델**: 저자 공개 코드(`soyoung97/Standard_Korean_GEC`)는 루트 레벨 코드에서 `hyunwoongko/kobart`, 레거시 코드(`src/KoBART-gec/`)에서 SKT-AI의 `kobart` pip 패키지를 사용한다. 두 체크포인트 모두 동일한 KoBART 아키텍처(BART-base, 123M params, encoder/decoder 각 6 layers, 16 heads, hidden 768, FFN 3072, vocab 30000)이다.

**교체 대상 모델**: `skt/kobart-base-v1` (HuggingFace Hub) — SKT-AI 공식 KoBART v1 체크포인트. 논문 footnote 3은 KoBART 출처로 `SKT-AI/KoBART`를 인용하므로, 교체 모델은 논문이 인용한 출처와 같은 계열이다.

**두 모델의 확인된 설정 차이 (토크나이저 전체 동일성은 추가 검증 필요)**:

| 항목 | `hyunwoongko/kobart` (원본) | `skt/kobart-base-v1` (교체) |
| --- | --- | --- |
| 확인한 special token ID | `<s>`=0, `</s>`=1, `<pad>`=3, `<unk>`=5, `<mask>`=6 | **동일** |
| config `bos_token_id` | 1 | 0 |
| config `eos_token_id` | 1 | 1 |
| config `pad_token_id` | 3 | 3 |
| config `decoder_start_token_id` | 1 | 1 |
| 가중치 계열 | KoBART v2 | KoBART v1 (HF config로 한 번 재확인; 인과 분리를 하지 않으므로 실험 진행에는 영향 없음) |

→ **핵심 시사점**: 확인한 special token ID는 같지만 이것만으로 두 토크나이저가 같다고 볼 수는 없다. 원본과의 어휘·대표 문장 인코딩 비교는 참고용으로 기록만 하고, 통과 조건은 교체 모델이 코드의 하드코딩과 일관적으로 동작하는지로 한다(§3-2). 이 실험은 베이스 모델 교체 효과의 인과적 분리가 아니라 논문 보고값과의 근접도 확인이 목적이다.

**사용할 코드**: 루트 레벨 코드(`run.py`, `model.py`, `dataset.py`)를 기준으로 한다. 모델/토크나이저 로딩 교체 외에도 최신 PyTorch Lightning 호환, 최고 검증 GLEU 체크포인트 저장, 테스트셋 평가 및 M² 계산을 위한 수정이 필요하다(§3-4, §6). 저장소 `requirements.txt`의 구버전 핀은 이 코드와 호환되지 않는다(§2-2).

**핵심 확인 사항**: `skt/kobart-base-v1`의 토크나이저 special token ID가 코드에 하드코딩된 값과 일관적으로 동작하는지 **검증**한다. 단, 아래 §3-2에서 설명하듯 하드코딩 값을 **함부로 바꾸지 않는다** — 원본 코드의 패딩 컨벤션을 유지해야 재현이 성립한다.

## 2. 환경 설정

### 2-1. 저장소 클론

아래 첫 줄은 gsm에서 실행한다. `ssh` 성공 후 나머지 명령은 모두 itcerdo 셸에서 실행한다(itcerdo에서는 `~/projects/phdq_bart` 바깥 경로를 사용하지 않는다). 데이터 전송이 필요한 경우만 §2-4의 gsm 명령을 따로 실행한다.

```bash
ssh itcerdo
cd ~/projects/phdq_bart
git clone https://github.com/soyoung97/Standard_Korean_GEC.git
cd Standard_Korean_GEC
```

이미 저장소가 있다면 다시 클론하지 않고 해당 디렉토리로 이동한다. 원격 코드 수정도 이 체크아웃에서 수행한다(예: VS Code Remote-SSH로 itcerdo의 저장소 열기). 재접속할 때마다 저장소 루트로 이동하고 아래 가상환경을 다시 활성화한다.

### 2-2. 의존성 — 저장소의 내부 모순 주의 (중요)

저장소 `requirements.txt`는 아래처럼 매우 오래된 버전을 고정한다:

```
torch==1.7.1  transformers==4.8.1  pytorch-lightning==1.1.0  numpy==1.23.5  ...
```

그러나 **우리가 사용할 루트 레벨 `run.py`의 Trainer 생성 코드는 최신 PyTorch Lightning API**를 사용한다:

```python
Trainer(accelerator='gpu', devices=get_device_count(), strategy='dp')
```

`accelerator='gpu'` / `devices=` / `strategy=` 인자는 PL **1.6 이상(사실상 2.x)**에서만 유효하다. `pytorch-lightning==1.1.0`에서는 이 인자들이 존재하지 않아 즉시 에러가 난다. 즉 **`requirements.txt`의 구버전 핀은 레거시 코드(`src/KoBART-gec/`)용이며, 루트 코드와 호환되지 않는다.** 루트 코드를 쓰기로 한 이상 최신 스택이 필수다.

**결론적으로 채택할 스택 (권장 — 옵션 A):**

itcerdo의 저장소 루트에서 가상환경을 만들고 활성화한 뒤 아래 패키지를 설치한다. gsm의 Python 환경에는 설치하지 않는다.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

```bash
# CUDA 12.8+ 런타임 (RTX 5090/Blackwell 지원) 기준
pip install torch --index-url https://download.pytorch.org/whl/cu128   # torch >= 2.7
pip install "pytorch-lightning>=2.5"      # torch 2.7과 맞는 세대 (2.3 이하는 torch 2.3 세대 기준)
pip install "transformers>=4.38,<4.45"   # BartForConditionalGeneration / PreTrainedTokenizerFast API 안정
pip install "numpy<2" pandas scikit-learn sentencepiece
pip install konlpy soylemma soynlp jamo pytz   # 전처리·평가용
pip install hunspell spacy tqdm                # KAGAS M² 생성용 — cyhunspell 대신 pyhunspell 사용(§2-5 시스템 패키지 필요, 설치 완료)
```

- 설치 직후 스택 조합을 스모크 테스트한다: `python3 -c 'import torch, pytorch_lightning as pl, transformers; print(torch.__version__, pl.__version__, transformers.__version__)'`
- `BartForConditionalGeneration`, `PreTrainedTokenizerFast.from_pretrained`, `model.generate(...)`의 시그니처는 위 버전 범위에서 안정적이며 `skt/kobart-base-v1` 로딩에 문제가 없다.
- **`strategy='dp'`는 PL 2.x에서 제거되었다.** 단일 GPU로 실행하면 dp가 필요 없으므로, `run.py`의 Trainer 인자를 `strategy='auto'`(또는 단순히 제거)로 바꾸거나 단일 GPU로만 돌린다. 멀티 GPU가 필요하면 `'ddp'`를 쓰되 글로벌 배치 크기가 달라짐에 유의(§5-1).
- 논문과 소프트웨어 버전이 달라지지만, **하이퍼파라미터·모델 수학은 동일하게 맞출 수 있다.** 애초에 저장소가 자기모순 상태라 "논문 원본 버전 그대로"는 루트 코드에선 불가능하다.

GLEU / M² 스코어러는 저장소에 이미 포함되어 있으므로 별도 클론하지 않는다: GLEU는 `metric/gleumodule.py`(`model.py`가 import), M²는 `metric/m2scorer/scripts/m2scorer.py`. Python3 동작만 스모크 테스트한다(G1, §6-2).

KAGAS(`KAGAS/` 디렉토리)도 M² 파일 생성에 필요하다. itcerdo에서 `java -version`으로 KoNLPy/Kkma용 Java 런타임을 확인하고, 클론된 `KAGAS/aff-dic/ko.aff`와 `ko.dic`의 존재를 확인한다(사전은 공개 저장소에 포함). Linux에서는 저장소가 `cyhunspell`을 권장하지만 Python 3.12용 wheel이 없어 itcerdo에서는 설치되지 않는다. 대신 `hunspell`(pyhunspell)을 사용하며, 이에 필요한 시스템 패키지는 §2-5에 따라 설치 완료했다. §6-2의 샘플 실행을 통과시킨다.

### 2-3. GPU 환경 — itcerdo RTX 5090

학습·평가는 itcerdo의 RTX 5090 한 장에서 순차 실행한다. Blackwell(sm_120)은 CUDA 12.8+ / PyTorch 2.7+ 스택이 필요하다. 논문은 TESLA V100 13GB GPU로 학습했다고 적지만, itcerdo에서도 batch 64의 실제 메모리 사용량은 첫 스모크 테스트로 확인한다. neuron/SLURM은 이 계획의 실행 대상이 아니다.

itcerdo에서 가상환경을 활성화한 뒤 학습 전에 CUDA 인식 여부를 확인한다:

```bash
nvidia-smi
python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable")'
```

### 2-4. 데이터셋 — ✅ 배치·검증 완료 (2026-09-24)

학습 데이터는 itcerdo의 `~/projects/phdq_bart/data/`에 있다(저장소 밖, 저장소 루트 기준 `../data`). 학습 입력은 `data/Preprocessed/<data>/<data>_{train,val,test}.txt`(탭 구분 `소스\t타겟`, 헤더 없음)이다. `run.py`는 `--data`만 주면 이 경로를 기본값으로 쓴다(`--data_root` 기본값 `../data/Preprocessed`, 개별 경로는 `--train_data_path` 등으로 덮어쓸 수 있음). 테스트 M² 채점용 정답 `<data>_test.m2`도 같은 폴더에 있다(§6-2).

| `--data` | train | val | test | 합계 | 논문 |
| --- | --- | --- | --- | --- | --- |
| `korean_learner` | 19,898 | 4,264 | 4,265 | 28,427 | 28,426 |
| `native` | 12,292 | 2,634 | 2,634 | 17,560 | 17,559 |
| `lang8` | 76,692 | 16,434 | 16,434 | 109,560 | 109,559 |
| `union` | 108,882 | 23,332 | 23,333 | 155,547 | — |

점검 결과 (`tools/check_data.py`, `tools/check_split_nfkc.py` — 로그 `logs/check_data.log`, `logs/check_split_nfkc.log`):

- **형식**: 탭 개수 이상 0줄. `read_docs()` 필터로 빠지는 줄 없음(빈 줄 1–2개는 파일 끝 개행).
- **분할**: korean_learner·native는 저장소 `train_test_val_split`(`random_state=1`)을 `<data>.txt`에 적용한 결과와 순서까지 동일하다. native 줄 수는 저장소 README의 실행 로그(`train: 12292, val: 2634, test: 2634`)와 같다. 합계가 논문보다 1씩 많은 것은 README 로그와도 같으므로 논문 표기 차이로 본다. lang8은 `derivation.json`상 union 앞부분에서 떼어 만든 것이라 `<data>.txt` 재분할 결과와는 다르며, 그대로 사용한다.
- **union**: 각 split이 세 데이터셋의 같은 split 합과 멀티셋으로 동일하다.
- **M²**: `<data>_{val,test}.m2`의 `S` 줄이 해당 `.txt` 소스와 순서·개수까지 일치한다.
- **정규화**: 학습 `.txt`는 모두 NFKC 정규화 상태이고, 구두점 분리(`punct_split`)는 적용되지 않은 상태다(§4-1).
- **주의(미사용 파일)**: `korean_learner_train_original.txt`/`_corrected.txt`만 처리 상태가 다르다(구두점 분리됨, NFKC 미적용). `union_val_corrected.txt`도 1줄이 끝 공백만 다르다. 이 파일들은 학습·평가에 쓰지 않는다.
- **split 간 동일 쌍**: korean_learner train∩test 15, lang8 100, union 115쌍. 원 코퍼스의 중복 문장 때문이고 원본 분할과 같으므로 그대로 둔다.

### 2-5. sudo가 필요한 작업 (관리자 요청 대상) — ✅ 완료 (2026-09-24)

itcerdo에서 `sudo`는 비밀번호를 요구하므로, 이 계획의 나머지 작업은 모두 sudo 없이 `~/projects/phdq_bart` 안에서 진행한다. 시스템 패키지 설치가 필요한 항목만 아래에 모았다. (2026-09-24 itcerdo 임시 venv에서 직접 설치를 시도해 확인)

**완료 확인 결과 (2026-09-24)**: `python3.12-dev 3.12.3-1ubuntu0.17`, `libhunspell-dev 1.7.2` 설치됨(`/usr/include/python3.12/Python.h`, `/usr/include/hunspell/hunspell.h`, pkg-config `hunspell` 1.7.2). 임시 venv(Python 3.12.3)에서 §2-2 설치 순서대로 설치 시 `hunspell 0.5.5` 빌드 성공, `numpy 1.26.4` 유지(spacy 3.8.16과 호환), `pip check` 이상 없음. 저장소 사전(`KAGAS/aff-dic/ko.dic`, `ko.aff`)으로 `HunSpell(...).suggest('안뇽')` → `['안녕']`, KoNLPy `Kkma().pos(...)` 정상 동작. 아래 sudo 없는 대안은 사용하지 않는다.

**필요한 시스템 패키지**

| 패키지 | 용도 | 없을 때 증상 (실측) |
| --- | --- | --- |
| `python3.12-dev` | C 확장 빌드용 `Python.h` | `pip install hunspell` → `fatal error: Python.h: No such file or directory` |
| `libhunspell-dev` | hunspell 헤더·라이브러리(`/usr/include/hunspell`, pkg-config `hunspell`) | 위 오류가 해결돼도 pyhunspell 빌드 불가. `pip install cyhunspell`은 3.12용 wheel이 없어(최신 2.0.2는 cp39까지) 구버전 소스를 받고 `Package hunspell was not found in the pkg-config search path` → `autoreconf` 없음으로 실패 |

관리자에게 요청할 명령:

```bash
sudo apt-get update
sudo apt-get install -y python3.12-dev libhunspell-dev
```

설치 후 (sudo 불필요) 학습용 `.venv`에서 `cyhunspell` 대신 `hunspell`(pyhunspell)을 설치한다. KAGAS 코드는 `hunspell.HunSpell(dic, aff)`를 먼저 시도하므로(`parallel_to_m2_korean.py` 15–18행) 코드 수정은 필요 없다:

```bash
pip install hunspell spacy tqdm
python3 -c "import hunspell; h=hunspell.HunSpell('KAGAS/aff-dic/ko.dic','KAGAS/aff-dic/ko.aff'); print(h.suggest('안뇽'))"
```

**sudo 없이 가능한 대안 (검증 완료)**: KAGAS 전용 Python 3.9 환경을 만들면 `cyhunspell` wheel이 바로 설치된다. `.venv`에서 `pip install uv` 후 `UV_PYTHON_INSTALL_DIR`와 `UV_CACHE_DIR`를 `~/projects/phdq_bart` 아래로 지정하고 `uv venv -p 3.9 .venv-kagas`와 `uv pip install -p .venv-kagas/bin/python cyhunspell spacy tqdm konlpy soylemma soynlp jamo`를 실행한다. 이 경우 §6-2의 KAGAS 명령은 `.venv-kagas/bin/python`으로 실행한다.

**참고 — hunspell은 M² 점수에 영향이 없다**: KAGAS에서 hunspell은 `align_text_korean.py`의 `hobj.suggest()`로 철자 오류 여부(오류 유형 라벨)를 판정하는 데만 쓰이고, 편집 구간·교정문 추출에는 관여하지 않는다. m2scorer는 오류 유형 필드를 `noop` 판별에만 사용하므로 어떤 바인딩을 쓰든 P/R/F0.5는 같다. 따라서 sudo 요청이 지연되면 대안으로 진행해도 재현 목표에는 지장이 없다.

**sudo가 필요 없는 것으로 확인된 항목** (2026-09-24 기준 설치됨): Java(OpenJDK 21, KoNLPy/Kkma용), `tmux`, `git`, `gcc`, `pkg-config`, NVIDIA 드라이버 570.124.04(CUDA 12.8 지원 — cu128 torch wheel이 CUDA 런타임을 포함하므로 시스템 CUDA toolkit은 불필요).

## 3. 코드 수정 사항

> **라인 번호 주의**: 아래 라인 번호는 현재 main 브랜치 기준 근사치이며 저장소 버전에 따라 다를 수 있다. 라인 번호에 의존하지 말고 문자열 검색(`from_pretrained`, `pad_token_id`, `generate(`)으로 위치를 찾을 것.

### 3-1. 모델/토크나이저 로딩 교체 — ✅ 구현 (`run.py`, 신규 `tokenizer_setup.py`)

`skt/kobart-base-v1` 토크나이저에는 후처리(post_processor)가 없어 `encode()`가 `<s>`/`</s>`를 붙이지 않는다. 원본 `hyunwoongko/kobart`는 `RobertaProcessing` 후처리로 `[0, …, 1]`을 만든다. `dataset.py`가 `tok.encode()` 결과를 그대로 쓰므로, 그대로 두면 인코더 입력에서 `</s>`가 빠지고 라벨 형식도 원본과 달라진다. 그래서 저장소에 `tokenizer_setup.py`를 추가해 원본과 똑같은 `RobertaProcessing(sep=('</s>', 1), cls=('<s>', 0))`을 붙이고, 로딩할 때마다 특수 토큰을 검사한다.

```python
from tokenizer_setup import BASE_MODEL, load_tokenizer, check_model_config, special_token_report
config = BartForConditionalGeneration.from_pretrained(BASE_MODEL).config
check_model_config(config)   # bos=0 eos=1 pad=3 decoder_start=1 forced_eos=1 vocab=30000이 아니면 예외
bart_model = BartForConditionalGeneration.from_pretrained(BASE_MODEL, config=config)
tokenizer = load_tokenizer(BASE_MODEL)   # 후처리 부착 + 특수 토큰·encode 형식 검사, 불일치 시 예외
```

매 실행마다 `special_token_report()` 결과를 `outputs/<data>/<run_id>/special_tokens.json`에 한 줄씩 추가한다. 점수가 이상하면 이 파일부터 run끼리 비교한다.

### 3-2. 특수 토큰 전수 점검 — ✅ 통과 (2026-09-24, 본격 학습 전 재확인)

`tools/check_special_tokens.py`(로그 `logs/check_special_tokens.log`)로 원본 `hyunwoongko/kobart`와 교체 모델(`tokenizer_setup.load_tokenizer`)을 전체 데이터에 대해 비교했다. 코드나 패키지 버전을 바꾸면 학습 전에 저장소 루트에서 `python3 tools/check_special_tokens.py`를 다시 실행한다(실패 항목이 있으면 종료 코드 1).

| 항목 | 결과 |
| --- | --- |
| 토크나이저 normalizer·pre_tokenizer·decoder·added_tokens·model(vocab·merges)·post_processor | 원본과 전부 동일 |
| `special_tokens_map`, `all_special_ids` `[0, 1, 5, 3, 6]`, `len(tokenizer)` 30000 | 동일 |
| 모델 config | 차이는 `bos_token_id`(1→0)와 `kobart_version`(2.0→1.0)뿐. `decoder_start_token_id`=1, `eos`=1, `pad`=3, `forced_eos`=1 동일 |
| 임베딩 행 수 = `vocab_size` = `len(tokenizer)` | 30000 |
| union train/val/test 전체(155,547쌍) `encode()` | 원본과 전부 동일 |
| `dataset.py` 출력(`input_ids`·`decoder_input_ids`·`labels`) 전체 | 원본과 전부 동일 |
| 본문 내 특수 토큰(0/1/3/6), `<unk>` | 0개 |
| 128 토큰 초과(잘림) | 0건 |
| `decode(skip_special_tokens=True)` | 원본과 전부 동일 |

`bos_token_id` 차이는 이 파이프라인에서 쓰이지 않는다. 학습 디코더 입력은 `dataset.py`가 직접 만들고(`[0] + labels[:-1]`), 생성은 `decoder_start_token_id=1`로 시작한다.

**원본 코드의 실제 입력 형식 (교체 후에도 동일, 고치지 않음)**:

- 인코더 입력 `[0, 문장…, 1, 0, 0, …]`: 맨 앞 `<s>`(0)도 패딩과 함께 `ne(0)` 마스크에 가려진다.
- 라벨 `[0, 문장…, 1, 1, -100, …]`: `encode()`가 붙인 `</s>` 뒤에 `dataset.py`가 eos를 한 번 더 붙인다.
- 디코더 입력 `[0, 0, 문장…, 1, 0, …]`: 앞 두 칸이 0이라 디코더 attention mask에도 가려진다.
- 생성은 `decoder_start_token_id=1`로 시작하므로 학습(0으로 시작)과 다르다.

네 가지 모두 원본 코드의 동작 그대로이며, 재현 목적상 고치지 않는다. 점수가 논문과 크게 다를 때 원인 후보로 검토한다.

**하드코딩 값을 pad=3 등으로 "고치지 말 것."**

- `dataset.py`는 입력을 `pad_index=0`(즉 `<s>`)으로 패딩하고 라벨 패딩은 `ignore_index=-100`으로 둔다. `model.py` forward가 `input_ids.ne(self.pad_token_id)`(=`ne(0)`)로 인코더·디코더 `attention_mask`를 만들기 때문에 0-패딩이 가려진다. 실제 패드 ID(3)를 쓰지 않는 것은 원본 코드의 설계다. `pad=3`으로 바꾸면 `dataset.py` 패딩과 `model.py` 마스크가 어긋나거나 원본 학습 조건이 바뀐다.
- `model.py` 생성부의 `eos_token_id=1` 하드코딩은 skt의 eos=1과 같다.
- 확인만 해 둘 위치(값 변경 X, 문자열 검색으로 찾을 것): `model.py`의 `self.pad_token_id = 0`, `generate(..., eos_token_id=1, ...)` / `dataset.py`의 `pad_index=0`, `ignore_index=-100`, `[pad_index] + label_ids[:-1]`, `label_ids.append(self.tok.eos_token_id)`.

### 3-3. 그 외 확인 사항 — ✅ 완료

- `model.py`의 `bos_token='<s>'`, `eos_token='</s>'`는 새 토크나이저와 같다.
- `skt/kobart-base-v1` 로딩과 `generate()` 스모크 테스트 통과(파라미터 123.9M). 파인튜닝 전에는 반복 토큰이 생성되며 이는 정상이다.
- Trainer의 `strategy='dp'` 제거(C3).

### 3-4. 코드 커스터마이징 전체 목록 — ✅ 구현 완료 (itcerdo 저장소 작업 트리, 커밋 전)

변경 파일은 `run.py`, `model.py`, 신규 `tokenizer_setup.py`이며 `dataset.py`는 그대로다. 원본 대비 변경은 itcerdo 저장소에서 `git diff`로 확인한다.

#### (필수) 없으면 실행 자체가 안 되는 것

| # | 위치 | 문제 | 조치 (구현) |
| --- | --- | --- | --- |
| C1 | `run.py` 모델 로딩부 | 베이스 모델을 `hyunwoongko/kobart` 로딩 | `tokenizer_setup`으로 `skt/kobart-base-v1` 로딩 + 원본 후처리 + 특수 토큰 검사 (§3-1) |
| C2 | `model.py` AdamW | `transformers.optimization.AdamW(correct_bias=False)` | **변경하지 않음.** transformers 4.44에 아직 존재해(FutureWarning만 발생) 원본 옵티마이저를 그대로 쓴다. torch AdamW로 바꿀 때 생기던 bias 보정 차이(G2)가 없다 |
| C3 | `run.py` Trainer | `from_argparse_args`·`strategy='dp'`가 PL 2.x에서 제거됨 | `pl.Trainer(accelerator='gpu', devices=1, max_epochs, callbacks, check_val_every_n_epoch, log_every_n_steps, num_sanity_val_steps=0, logger=CSVLogger(run_dir))` |
| C4 | `model.py` | `training_epoch_end` 제거됨 | `on_train_epoch_end(self)` |
| C5 | `model.py` | `validation_epoch_end`/`test_epoch_end` 제거, `outputs` 미전달 | `validation_step`/`test_step`에서 loss를 `self.eval_step_losses`에 축적하고, `on_validation_epoch_end`/`on_test_epoch_end`가 공용 `eval_epoch_end(mode)`를 호출 |

#### (재현 정확도) 논문 점수 비교에 필요한 것

| # | 위치 | 문제 | 조치 (구현) |
| --- | --- | --- | --- |
| R1 | `run.py` `ModelCheckpoint` | `every_n_epochs=10, save_top_k=-1`이라 best-GLEU epoch가 저장되지 않을 수 있음(논문은 GLEU 최고 epoch로 평가) | `monitor='val_gleu', mode='max'` 유지, `save_top_k=1`, `every_n_epochs` 기본값 1. 저장 위치 `outputs/<data>/<run_id>/model_ckpt/<data>_<lr>_epoch=NN.ckpt` |
| R2 | `model.py` M² 블록 | f-string 누락·`command` 미정의·p/r/F0.5 0점 하드코딩 | 블록 제거. 로그에서 가짜 0점도 제거. M²는 테스트 출력에 대해 공식 `<data>_test.m2`로 별도 계산(§6-2) |
| R3 | `run.py` | `--model_ckpt_path`를 주면 `validate()`만 호출 | `--resume_finetune`: 체크포인트 가중치만 불러오고 옵티마이저·스케줄러는 새로 시작해 `fit()` |
| R4 | `model.py` `generate()` | `repetition_penalty` 없음(README는 2.0) | `--repetition_penalty` 인자(기본 2.0)를 `generate()`에 전달 |
| R5 | `run.py` | `trainer.test()` 주석 처리 | `--eval_test`: 체크포인트를 불러와 `trainer.test()`만 실행. 생성 경로의 epoch 번호는 체크포인트의 epoch |
| R6 | `run.py`, `model.py` | seed·단계가 같은 경로를 공유 | 필수 인자 `--run_id`. 체크포인트 `outputs/<data>/<run_id>/model_ckpt/`, 생성 `outputs/generation/<data>/<run_id>/epoch<N>/{val,test}/` |
| P1 | `run.py` 인자 | 데이터 경로 기본값이 wikisql 예시 | `--data`는 `korean_learner`/`native`/`lang8`/`union` 중 필수 선택. 경로 기본값은 `<data_root>/<data>/<data>_<split>.txt`, 파일이 없으면 즉시 종료 |

생성 파일 이름은 원본의 `hypothesis_{loss 텐서}.txt` 대신 `hypothesis.txt`/`reference.txt`/`source.txt`/`gleu.txt`로 고정했다. loss와 실행 인자는 `gleu.txt`에 기록된다.

#### (검증) 확인 결과

- **G1** `metric/gleumodule.py`의 `run_gleu`: Python 3.12에서 동작 확인(스모크 테스트에서 val/test GLEU 산출).
- **G2** 해당 없음(C2에서 원본 AdamW 유지).
- **G3** 해결: skt 토크나이저는 원래 `encode()`에 특수 토큰을 붙이지 않았다. 원본 후처리를 붙였고, 원본에서도 맨 앞 `<s>`(0)가 `ne(0)` 마스크에 가려짐을 확인했다(§3-2).
- **G4** `args.device`, `args.best`, `args.num_workers`는 Trainer에 넘기지 않고 모델·로직에서만 쓴다.

### 3-5. 통합 스모크 테스트 — ✅ 통과 (2026-09-24)

native 앞부분(train 512 / val 128 / test 128줄)으로 전체 흐름을 확인했다(로그 `logs/smoke_{fit,test,resume}.log`, 산출물은 확인 후 삭제).

- `fit` 2 epoch: 매 epoch 검증 GLEU 계산, best 체크포인트 1개만 남음(`native_3e-05_epoch=01.ckpt`), `special_tokens.json` 기록.
- `--eval_test`: 테스트 GLEU 산출, `epoch1/test/`에 파일 4개 생성.
- `--resume_finetune`(lr 1e-5, 1 epoch): 체크포인트에서 이어 학습하고 별도 run_id에 저장.
- M²: 공식 `native_test.m2` 앞 128문장으로 m2scorer 채점 정상.
- 정렬: `source.txt`·`reference.txt`·`hypothesis.txt`·데이터·공식 M²가 128/128 순서 일치, `decode(source)`와 `decode(label)`이 원문과 128/128 동일.

### 3-6. recipe와 디코더 첫 토큰 누설 수정 — ✅ (2026-09-25, 논문 재현 경로 = `legacy`)

첫 run(root 설정)이 논문보다 GLEU 약 6점 낮아 원인을 추적했다(상세는 `log.md` 2026-09-24 21:28 ~ 2026-09-25 09:24).

- **논문 수치는 레거시 코드(`src/KoBART-gec`) 산출물이다.** 레거시 가중치는 `skt/kobart-base-v1`과 완전히 같고(최대 절대차 0.0), 레거시 토크나이저는 후처리가 없어 `<s>…</s>`를 붙이지 않는다. 학습 설정은 cosine, warmup 0.1, gradient clip 1.0이고 repetition_penalty를 쓰지 않는다. transformers 4.0.0에서는 생성이 `bos=0`으로 시작한다.
- **디코더 첫 토큰 누설**: `decoder_attention_mask = ne(0)`이 디코더 첫 토큰(0)을 가려 위치0 행이 전부 가려지고, 최신 transformers가 이를 '전부 보기'로 풀어 첫 토큰 예측이 정답 뒷부분을 본다. transformers 4.0.0은 첫 디코더 토큰을 마스킹하지 않았다(`never mask leading token`). `model.make_decoder_attention_mask()`로 같은 동작을 재현했고, `tools/check_special_tokens.py` §5가 매번 검사한다.
- `run.py --recipe {root,legacy}`: legacy = 후처리 없음, cosine, warmup 0.1, clip 1.0, 생성 시작 0, forced eos 없음, repetition_penalty 1.0. **root 형식은 디코더 입력이 `[0, 0, …]`이라 누설을 피할 수 없어 누설 검사에서 실패하므로 쓰지 않는다.** §3-1의 `<s>…</s>` 후처리 결정과 §5-1의 repetition_penalty 2.0 고정은 이 결과로 대체한다.

| run (korean_learner, seed 0) | 검증 GLEU(best) | 테스트 GLEU | P | R | F0.5 |
| --- | --- | --- | --- | --- | --- |
| root | 39.57 | 38.70 | 25.42 | 21.11 | 24.43 |
| legacy (누설 있음) | 32.50 | 32.12 | 20.57 | 14.01 | 18.81 |
| **legacy + 누설 수정** | **42.23** | **40.86** | **42.02** | **19.04** | **33.85** |
| 논문 (3-seed 평균) | 46.94 | 45.06 | 43.35 | 24.54 | 37.58 |

실행: `bash tools/run_one.sh <data> <run_id> <seed> <lr> legacy [init_ckpt]` (저장소 루트, tmux 안).

### 3-7. 학습 중 dropout 활성화 — ✅ (2026-09-25) · 최종 재현 결과

저자 spreadsheet(공개)의 체크포인트 run 122(Kor-Learner seed 0)를 받아 우리 파이프라인으로 채점하니 검증 GLEU가 저자 기록과 같은 46.91이었다. 평가 파이프라인은 논문과 동일하고 차이는 학습에서 생긴다는 뜻이다. 저자 yaml의 학습 설정(batch 64, warmup 0.1, clip 1.0, lr 3e-5, dropout 0.1, fp32)과 체크포인트의 스케줄러·옵티마이저 상태도 legacy recipe와 같았다.

- **원인**: `from_pretrained()`는 eval 모드로 반환하고, PyTorch Lightning 2.x는 학습 시작 때 `model.train()`을 호출하지 않는다 → **dropout 없이 학습**. PL 1.1(논문)은 학습 epoch마다 `train()`을 호출했다. 이전 학습 로그 23개 전부에 PL 경고 "Found 182 module(s) in eval mode at the start of training"가 있었다.
- **수정**: `model.py`에서 `on_train_epoch_start()`마다 `self.train()`을 호출하고, `training_step` 첫 배치에서 내부 모델이 eval이면 `RuntimeError`로 중단한다.
- **재현에 필요했던 수정 3가지**(모두 라이브러리 버전 차이로 생긴 동작 차이): ① 레거시 토큰화(`<s>…</s>` 미부착, §3-6) ② 디코더 첫 토큰 비마스킹(transformers 4.0.0 동작, §3-6) ③ 학습 중 dropout 활성화(PL 1.1 동작).

**최종 결과** — tag `dropout`, 21 run, 2026-09-25 22:42 – 2026-09-26 10:17, 전부 성공. 3-seed 평균 vs 논문 Table 6, 테스트셋:

| 시나리오 | 데이터 | GLEU 재현 / 논문 | P 재현 / 논문 | R 재현 / 논문 | F0.5 재현 / 논문 | GLEU seed 범위 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 개별 | korean_learner | 45.68 / 45.06 (+0.62) | 43.31 / 43.35 (-0.04) | 25.46 / 24.54 (+0.92) | 37.98 / 37.58 (+0.40) | 0.71 (3 seed) |
| 1 개별 | native | 68.40 / 67.24 (+1.16) | 80.34 / 75.34 (+5.00) | 58.31 / 55.95 (+2.36) | 74.69 / 70.45 (+4.24) | 2.46 (3 seed) |
| 1 개별 | lang8 | 29.65 / 28.48 (+1.17) | 39.46 / 37.56 (+1.90) | 13.33 / 11.62 (+1.71) | 28.33 / 25.93 (+2.40) | 0.60 (3 seed) |
| 1 개별 | union | 34.75 / 33.70 (+1.05) | 44.38 / 44.75 (-0.37) | 16.09 / 14.64 (+1.45) | 32.80 / 31.70 (+1.10) | 1.75 (3 seed) |
| 2 Union→개별 | korean_learner | 44.19 / 42.66 (+1.53) | 52.15 / 53.51 (-1.36) | 23.47 / 21.18 (+2.29) | 41.88 / 41.00 (+0.88) | 2.60 (3 seed) |
| 2 Union→개별 | native | 60.96 / 59.71 (+1.25) | 86.28 / 85.47 (+0.81) | 48.73 / 47.38 (+1.35) | 74.70 / 73.63 (+1.07) | 4.83 (3 seed) |
| 2 Union→개별 | lang8 | 29.35 / 28.65 (+0.70) | 37.84 / 37.46 (+0.38) | 13.05 / 12.00 (+1.05) | 27.41 / 26.78 (+0.63) | 1.58 (3 seed) |

7개 조합 모두 GLEU가 논문과 같거나 0.6–1.5 높고, 시나리오 2의 정밀도↑·재현율↓ 경향도 논문과 같다. 실행: `bash tools/run_all.sh <tag>` (run별 결과는 `logs/runs.log`, 요약은 `python3 tools/summarize_runs.py logs/runs.log <tag>`).

## 4. 데이터 전처리 및 분할

### 4-1. 전처리 — 다시 실행하지 않는다

§2-4의 데이터는 이미 분할·Union 구성이 끝난 상태이므로 `get_data/process_for_training.py`를 다시 돌리지 않는다. 저장소 README의 데이터 준비 절차는 `train_split`(그리고 Union은 `make_union`)만 실행하며, 이 데이터는 그 결과와 일치한다(§2-4). 구두점 분리(`punct_split`)는 README 절차에 없고 현재 데이터에도 적용되어 있지 않으므로 **적용하지 않는다.** 적용하면 입력·정답·공식 M²가 모두 바뀌어 원본 조건과 달라진다.

### 4-2. 데이터 분할 세부

논문은 "seed 0으로 분할"이라 적었지만 코드(`process_for_training.py`)는 `train_test_split(..., random_state=1)`을 쓴다. korean_learner·native는 이 코드로 순서까지 재현됨을 확인했다(§2-4). 분할은 학습 seed 0/1/2와 무관하게 고정이다.

```python
train, val_test = train_test_split(data, test_size=0.3, random_state=1)
val, test = train_test_split(val_test, test_size=0.5, random_state=1)
# 결과: train 70% / val 15% / test 15%
```

### 4-3. 데이터 형식

각 줄이 `소스\t타겟`인 txt이며 `dataset.py`의 `read_docs()`가 파싱한다. 경로는 `--data`로 자동 결정되고, 필요하면 `--data_root` 또는 `--train_data_path`/`--val_data_path`/`--test_data_path`로 덮어쓴다.

## 5. 학습 실행

### 5-1. 하이퍼파라미터 — 논문 vs 코드 대조 (검증 완료)

논문 §5.4 및 Appendix D, 그리고 저장소 실행 명령을 대조한 결과:

| 파라미터 | 값 | 출처 / 주의 |
| --- | --- | --- |
| learning rate | **3e-5** (시나리오1) / **1e-5** (시나리오2 2단계) | 논문 §5.4 명시 |
| max epochs | **10** | 논문 명시. **root 기본값은 100이므로 반드시 명시 전달** |
| batch size | **64** | 논문 명시. **root 기본값은 32이므로 반드시 명시 전달** |
| beam search | **4** | 논문·코드 일치(`generate`에 하드코딩) |
| max\_seq\_len | 128 | **논문 미명시** — 코드 기본값. 전체 데이터 중 128 토큰 초과 0건(§3-2) |
| warmup\_ratio | **0.0** | **논문은 warmup을 언급하지 않음.** 저자 공개 실행 명령도 warmup을 넘기지 않아 root 기본값 0.0이 적용됨 → 재현에는 0.0 사용 |
| repetition\_penalty | **2.0** | README 추론 설정. `--repetition_penalty` 기본값 2.0으로 구현(R4) |
| dropout | 코드 기본값 | 논문은 최종값 미명시 → config 기본값 사용 |
| scheduler | linear | 논문 미명시 → root 코드의 `get_linear_schedule_with_warmup` 사용(§5-4) |
| optimizer | AdamW (`correct_bias=False`, weight_decay 0.01) | 원본 transformers AdamW 그대로(C2) |
| seed | 0, 1, 2 (3회) | 논문은 "3 different seeds"만 명시 → 0/1/2로 가정. 분할 seed와 무관 |

**repetition_penalty**: 논문 본문은 이 옵션을 언급하지 않지만 저자 README/데모의 추론 기본값이므로 **모든 주 실험은 2.0으로 고정**한다. 미적용 비교가 필요하면 **한 데이터셋·한 seed의 테스트 생성에서만** `--eval_test --repetition_penalty 1.0`으로 측정한다(재학습 불필요). 같은 run_id로 돌리면 생성 파일을 덮어쓰므로 `--run_id`를 다르게 주고 `--model_ckpt_path`로 같은 체크포인트를 지정한다.

**GPU**: Trainer는 `devices=1`로 고정되어 있다. `CUDA_VISIBLE_DEVICES=0`으로 실행하면 글로벌 배치가 논문과 같은 64가 된다.

아래 학습·평가 명령은 모두 itcerdo 저장소 루트에서 실행한다. 접속이 끊겨도 학습이 유지되도록 `tmux` 세션 안에서 실행한다. 재접속 후 `ssh itcerdo`와 `tmux attach -t kobart-gec`로 진행 상태를 확인한다. `.venv/bin/activate`가 모든 캐시(HF·pip·torch)를 `~/projects/phdq_bart/.cache`로 고정하므로 반드시 활성화 후 실행한다.

```bash
tmux new -s kobart-gec
# 이후 tmux 셸에서:
cd ~/projects/phdq_bart/Standard_Korean_GEC
source .venv/bin/activate
python3 tools/check_special_tokens.py   # 코드·패키지 변경 후에는 학습 전에 재실행 (종료 코드 0 확인)
```

### 5-2. 시나리오 1 — 개별 데이터셋 파인튜닝

각 데이터셋으로 독립 파인튜닝한다(예: Kor-Learner).

```bash
CUDA_VISIBLE_DEVICES=0 python3 run.py \
  --data korean_learner \
  --run_id individual_seed0 \
  --lr 3e-5 \
  --batch_size 64 \
  --max_epochs 10 \
  --max_seq_len 128 \
  --warmup_ratio 0.0 \
  --seed 0 \
  2>&1 | tee logs/korean_learner_individual_seed0.log
```

`--data`는 `korean_learner`, `native`, `lang8`, `union` 중 하나이며 데이터 경로는 자동으로 정해진다. `--run_id`는 데이터셋·시나리오·seed 조합마다 고유하게 준다(예: `individual_seed0`, `individual_seed1`, `individual_seed2`). 각 데이터셋을 seed 0/1/2로 3회 반복하고, 결과는 테스트 점수의 평균으로 보고한다. 최고 검증 GLEU 체크포인트는 `outputs/<data>/<run_id>/model_ckpt/`에 1개만 남는다.

### 5-3. 시나리오 2 — Kor-Union 선행 파인튜닝 + 개별 파인튜닝

**1단계: Kor-Union 파인튜닝** (lr=3e-5)

```bash
CUDA_VISIBLE_DEVICES=0 python3 run.py \
  --data union --run_id union_pretrain_seed0 \
  --lr 3e-5 --batch_size 64 --max_epochs 10 --max_seq_len 128 --warmup_ratio 0.0 --seed 0 \
  2>&1 | tee logs/union_pretrain_seed0.log
```

**2단계: Union 체크포인트에서 개별 데이터셋으로 추가 파인튜닝** (lr=1e-5)

`--resume_finetune`은 체크포인트의 가중치만 불러오고 옵티마이저·스케줄러는 새로 시작한다(R3).

```bash
CUDA_VISIBLE_DEVICES=0 python3 run.py \
  --data korean_learner --run_id after_union_seed0 \
  --model_ckpt_path outputs/union/union_pretrain_seed0/model_ckpt/<best_gleu_checkpoint>.ckpt \
  --resume_finetune \
  --lr 1e-5 --batch_size 64 --max_epochs 10 --max_seq_len 128 --warmup_ratio 0.0 --seed 0 \
  2>&1 | tee logs/korean_learner_after_union_seed0.log
```

1단계와 2단계는 **같은 seed 번호끼리** 연결하고, 각 2단계 run에 사용한 1단계 체크포인트 경로를 기록한다(경로는 `gleu.txt`의 `ckpt=`에도 남는다).

### 5-4. 스케줄러

루트 `model.py`는 `get_linear_schedule_with_warmup`, 레거시 `src/KoBART-gec/train.py`는 `get_cosine_schedule_with_warmup`을 쓴다. 논문은 스케줄러를 명시하지 않으므로 루트 코드의 linear를 그대로 사용한다. `warmup_ratio=0.0`이면 warmup 구간 없이 선형 감쇠만 적용된다(`num_warmup_steps = int(num_train_steps * warmup_ratio)`).

## 6. 평가

### 6-1. GLEU 평가

`model.py`의 `on_validation_epoch_end()`가 매 epoch 검증 GLEU를 계산한다. `metric/gleumodule.py`의 `run_gleu()`가 source/reference/hypothesis 파일로 점수를 반환한다(스코어러: https://github.com/cnap/gec-ranking).

각 run에서 **검증 GLEU 최고 체크포인트**로 테스트셋을 한 번만 평가한다(R5). 이 분기는 학습하지 않는다.

```bash
CUDA_VISIBLE_DEVICES=0 python3 run.py \
  --eval_test \
  --data korean_learner --run_id individual_seed0 --seed 0 \
  --model_ckpt_path outputs/korean_learner/individual_seed0/model_ckpt/<best_gleu_checkpoint>.ckpt \
  --batch_size 64 --max_seq_len 128 \
  2>&1 | tee logs/korean_learner_individual_seed0_test.log
```

시나리오2는 `--run_id after_union_seed0`과 그 run의 최고 검증 GLEU 체크포인트로 같은 방식으로 테스트한다. 출력은 `outputs/generation/<data>/<run_id>/epoch<N>/test/`에 `hypothesis.txt`, `reference.txt`, `source.txt`, `gleu.txt`로 남는다. 테스트의 `<N>`은 불러온 체크포인트의 epoch(0부터 시작)다.

### 6-2. M² Scorer 평가

M²는 epoch마다 계산하지 않고 **§6-1 테스트 출력**에 대해 한 번만 계산한다. 정답은 데이터에 포함된 공식 `<data>_test.m2`를 그대로 쓴다. 이 파일의 문장 순서가 테스트 `.txt`와 같고(§2-4), 테스트 dataloader는 섞지 않으므로 `hypothesis.txt`의 줄 순서와 1:1로 맞는다(스모크 테스트에서 128/128 확인, §3-5). 정답 M²를 KAGAS로 다시 만들 필요가 없다.

공식 파일 검증: native 테스트셋 원문·정답에 현재 환경의 KAGAS를 다시 돌려 공식 `native_test.m2`와 비교했다(1분 39초, 로그 `logs/kagas_check_native_test.log`). 2634블록 중 2626블록이 완전히 같았다. 편집 구간·교정문은 2632블록이 같았고, 나머지 차이는 오류 유형 라벨(m2scorer 채점에 쓰이지 않음)과 편집 2블록이다. 따라서 공식 파일은 KAGAS 출력과 사실상 같으며, 데이터의 정답으로서 공식 파일을 쓴다.

```bash
G=outputs/generation/korean_learner/individual_seed0/epoch<N>/test
python3 metric/m2scorer/scripts/m2scorer.py \
  $G/hypothesis.txt ../data/Preprocessed/korean_learner/korean_learner_test.m2 \
  > $G/m2score.txt
cat $G/m2score.txt   # Precision / Recall / F_0.5
```

저자 README에 따르면 m2scorer는 몇 시간이 걸릴 수 있다. CPU 작업이므로 다음 GPU 학습과 동시에 별도 tmux 창에서 돌려도 된다.

KAGAS는 테스트 채점에 필요하지 않다. 설치 확인용 샘플 실행은 통과했다(저장소 루트에서 실행, 저자 샘플 출력과 `A` 줄이 오류 유형까지 동일, m2scorer 자기채점 1.0):

```bash
cd KAGAS/
python3 parallel_to_m2_korean.py \
  -orig sample_test_data/orig.txt -cor sample_test_data/corrected.txt \
  -out ../outputs/kagas_smoke.m2 -hunspell ./aff-dic -noprint && test -s ../outputs/kagas_smoke.m2
cd ../
```

각 seed·시나리오의 테스트 GLEU, M² precision/recall/F0.5, 체크포인트 경로, 생성 설정(`gleu.txt`)을 함께 기록한다.

### 6-3. 논문 결과와 비교

논문 Table 6(테스트셋, 3-seed 평균)의 GLEU/M² 점수와 같은 데이터셋·시나리오의 3-seed 평균을 비교해 일치 정도를 보고한다. `skt/kobart-base-v1`(v1)은 논문 코드의 `hyunwoongko/kobart`(v2)와 가중치 계열이 다르고 소프트웨어 스택(§2-2)도 다르므로, 점수 차이를 베이스 모델만의 효과로 해석하지 않는다. 원본 모델의 별도 대조군 학습은 이 계획에 포함하지 않는다.

## 7. 실행 전 체크리스트

**환경/스택**

- [x] gsm에서 itcerdo 접속; 지정 디렉토리 `~/projects/phdq_bart/Standard_Korean_GEC`의 저장소(`dfe0af9`)·가상환경 준비
- [x] 스택 설치 — torch 2.11.0+cu128 / PL 2.6.6 / transformers 4.44.2 / numpy 1.26.4, `pip check` 이상 없음; RTX 5090(sm_120) CUDA 인식
- [x] (sudo) `python3.12-dev libhunspell-dev` 시스템 설치 (§2-5)
- [x] `.venv`에 KAGAS용 `hunspell`, `spacy`, `tqdm` 설치; Java 21, `KAGAS/aff-dic/ko.aff`·`ko.dic` 확인
- [x] `.venv/bin/activate`에서 HF·pip·torch 캐시를 `~/projects/phdq_bart/.cache`로 고정
- [x] 데이터 파일(4개 데이터셋 × train/val/test) 형식·경로·분할·Union 구성·M² 정렬 확인 (§2-4)
- [x] `logs/` 생성(`write_command_logs()` 사용)

**코드 커스터마이징 (§3-4, itcerdo 작업 트리, 커밋 전)**

- [x] C1: `tokenizer_setup.py`로 `skt/kobart-base-v1` 로딩 + 원본 `RobertaProcessing` 후처리 + 특수 토큰 검사
- [x] C2: 원본 transformers AdamW(`correct_bias=False`) 유지 — 4.44에서 동작 확인
- [x] C3: `pl.Trainer(...)` 직접 생성, `devices=1`, `num_sanity_val_steps=0`, CSVLogger
- [x] C4: `on_train_epoch_end`
- [x] C5: `on_validation_epoch_end`/`on_test_epoch_end` + `eval_step_losses` 축적
- [x] R1–R6, P1 구현 (best-GLEU 체크포인트 1개, M² 블록 제거, `--resume_finetune`, `--repetition_penalty 2.0`, `--eval_test`, `--run_id`, `--data` 기반 경로)
- [x] 특수 토큰 전수 점검 `tools/check_special_tokens.py` 통과 (§3-2)
- [x] 통합 스모크 테스트 통과 — fit / eval_test / resume_finetune / M² / 정렬 (§3-5)
- [x] 공식 테스트 M²와 KAGAS 재생성 결과 비교 (§6-2)

**실험 실행**

- [x] 학습 전 `python3 tools/check_special_tokens.py legacy` 종료 코드 0 확인 (`run_one.sh`가 매 run 자동 실행)
- [x] 학습 인자 `--lr`, `--batch_size 64`, `--max_epochs 10` 명시 + `--recipe legacy`(warmup 0.1, cosine, clip 1.0) — §3-6
- [x] 시나리오1: 4개 데이터셋 × seed 0/1/2 학습 → `--eval_test` → M² (tag dropout, §3-7)
- [x] 시나리오2: union seed 0/1/2 → 개별 데이터셋 `--resume_finetune`(같은 seed끼리) → `--eval_test` → M² (tag dropout)
- [x] 논문 Table 6의 테스트 점수와 3-seed 평균 비교 — 전 조합 재현 (§3-7)
- [x] 모든 작업을 `log.md`에 기록 (§0, 계속 유지)
