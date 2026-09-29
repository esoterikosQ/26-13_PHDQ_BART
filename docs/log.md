# KoBART GEC 재현 작업 로그

- 시각은 KST. 위치 표기: **[gsm]** 로컬(`phdq_bart2/`), **[itcerdo]** `~/projects/phdq_bart` (명령의 상대경로는 별도 표기가 없으면 `~/projects/phdq_bart/Standard_Korean_GEC`, `.venv` 활성화 상태).
- 기록 규칙은 `plan.md` §0 참조. 새 기록은 맨 아래에 시간순으로 추가한다.

---

## 2026-09-18

### 11:00–12:00 · 계획서 초안 검토·수정 [gsm]

- 작업: `plan.md` 실현 가능성 검토, 특수 토큰 처리 방침 확정, RTX 5090(itcerdo) 사용 결정, 하이퍼파라미터 재현 검토, 코드 커스터마이징 지점(C1–C5, R1–R6) 도출.
- 결정: 구버전 의존성 고정은 쓰지 않고 최신 스택으로 이식(“논문 핵심 요지의 기능적 재현”이 1차 목표).
- 결과: `plan.md` 갱신. (이 세션의 개별 명령 기록은 남아 있지 않음)

---

## 2026-09-24

### 17:52–18:02 · 원격 상태 확인 및 계획서 최종 검토 [gsm, itcerdo]

```bash
ssh -o BatchMode=yes itcerdo 'ls -la ~/projects/phdq_bart; nvidia-smi; python3 --version; which git tmux nvcc'
git clone https://github.com/soyoung97/Standard_Korean_GEC.git   # 검증용, gsm 스크래치 디렉토리
```

- 결과: itcerdo `~/projects/phdq_bart`는 비어 있음. RTX 5090 32GB, 드라이버 570.124.04, Python 3.12.3, git·tmux 있음, nvcc 없음(cu128 wheel이라 불필요).
- 검토 결과(저장소 `dfe0af9` 코드·논문과 대조): §3-2 attention_mask 서술 오류, PL 버전 상한 문제, 토크나이저 동일성 assert 과도, repetition_penalty 전수 실행 과도, `BEST_CKPT.ckpt` 유령 참조 등 지적.

### 18:05–18:07 · 검토 지적 사항 `plan.md` 반영 [gsm]

- 작업: attention_mask 서술 수정(`model.py` forward가 `ne(0)`로 생성), `pytorch-lightning>=2.5`로 변경 + import 스모크 테스트 추가, 토크나이저 비교는 로그만 남기도록 완화, repetition_penalty 2.0 고정, `BEST_CKPT.ckpt` 삭제, 라인 번호·R1 표현·스코어러 중복 클론 정리, `decoder_start_token_id` 확인 추가.
- 결과: `plan.md` 수정 완료.

### 20:46–20:51 · 사용자 수정본 재점검, hunspell 설치 가능성 확인 [gsm, itcerdo]

```bash
curl -s https://pypi.org/pypi/cyhunspell/json    # 최신 2.0.2, wheel cp36–cp39뿐, 2.x sdist 없음
# itcerdo 임시 venv(~/projects/phdq_bart/_pipcheck, 확인 후 삭제)
pip install cyhunspell   # 실패: pkg-config hunspell 없음 → autoreconf 없음
pip install hunspell     # 실패: Python.h 없음 (python3.12-dev 미설치)
uv venv -p 3.9 k39 && uv pip install -p k39/bin/python cyhunspell   # 성공 (sudo 없는 대안)
```

- 결과: KAGAS는 `hunspell.HunSpell`(pyhunspell)을 먼저 시도하고 cyhunspell로 대체하는 구조 → 어느 쪽이든 가능. hunspell은 오류 유형 라벨(`check_spell`)에만 쓰이고 m2scorer는 유형을 `noop` 판별에만 쓰므로 P/R/F0.5에 영향 없음.
- 확인: itcerdo에 Java(OpenJDK 21) 있음, `sudo`는 비밀번호 필요.

### 20:54 · sudo 필요 작업 정리 [gsm]

- 작업: `plan.md` §2-5 신설(`python3.12-dev`, `libhunspell-dev` 요청 명령, sudo 없는 대안, 점수 무영향 근거).

### 20:56–20:58 · sudo 설치 결과 확인 [itcerdo]

```bash
dpkg -l python3.12-dev libhunspell-dev; pkg-config --modversion hunspell
# 임시 venv에서 plan §2-2 순서대로 설치
pip install "numpy<2" pandas scikit-learn sentencepiece; pip install konlpy soylemma soynlp jamo pytz; pip install hunspell spacy tqdm
python3 -c "import hunspell; h=hunspell.HunSpell('aff-dic/ko.dic','aff-dic/ko.aff'); print(h.suggest('안뇽'))"
```

- 결과: `python3.12-dev 3.12.3-1ubuntu0.17`, `libhunspell-dev 1.7.2` 설치됨. `hunspell 0.5.5` 빌드 성공, `suggest('안뇽')` → `['안녕']`, Kkma 정상, numpy 1.26.4 유지, `pip check` 이상 없음. 임시 venv 삭제.
- `plan.md`: 설치 명령을 `hunspell`로 변경, §2-5 완료 표시.

### 20:58 · 저장소 클론 [itcerdo]

```bash
cd ~/projects/phdq_bart && git clone https://github.com/soyoung97/Standard_Korean_GEC.git
```

- 결과: 커밋 `dfe0af9` (2024-01-02).

### 20:59–21:03 · 가상환경·스택 설치, GPU 확인 [itcerdo]

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && mkdir -p logs
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install "pytorch-lightning>=2.5"; pip install "transformers>=4.38,<4.45"
pip install "numpy<2" pandas scikit-learn sentencepiece; pip install konlpy soylemma soynlp jamo pytz; pip install hunspell spacy tqdm
# 로그: logs/setup_install.log
python3 -c "import torch, pytorch_lightning as pl, transformers, numpy; print(...)"; nvidia-smi; pip check
```

- 결과: torch 2.11.0+cu128 / PL 2.6.6 / transformers 4.44.2 / numpy 1.26.4. RTX 5090 (12, 0) 인식, CUDA 12.8, GPU 연산 정상, `pip check` 이상 없음.
- 문제: 설치 중 pip 캐시가 `~/.cache/pip`, transformers 첫 import 시 `~/.cache/huggingface` 마이그레이션 메시지(옮긴 파일 0개) — 지정 디렉토리 밖 접근. 확인·삭제하지 않음.

### 21:03 · 캐시 경로 고정 [itcerdo]

```bash
cat >> .venv/bin/activate   # XDG_CACHE_HOME / HF_HOME / PIP_CACHE_DIR / TORCH_HOME = ~/projects/phdq_bart/.cache/...
```

- 결과: HF 캐시 `~/projects/phdq_bart/.cache/huggingface/hub`, pip 캐시 `~/projects/phdq_bart/.cache/pip` 확인.

### 21:03–21:04 · 토크나이저·config·모델 1차 점검 [itcerdo]

```bash
python3 ../tools/check_tokenizer.py > logs/check_tokenizer.log 2>&1
```

- 결과: skt pad=3 / eos=1 / bos=0 / vocab=30000, config `decoder_start_token_id=1` — 표와 일치. 모델 123.9M. vocab·special_tokens_map은 원본과 동일.
- **발견**: `encode()` 결과가 원본과 다름 — 원본 `[0, …, 1]`, skt `[…]`(후처리 없음). 원본 post_processor는 `RobertaProcessing`, skt는 `None`.
- 확인: skt 토크나이저에 `<s> $A </s>` 후처리를 붙이면 샘플 문장 인코딩·디코딩이 원본과 동일.

### 21:04 · KAGAS 샘플 스모크 테스트 [itcerdo]

```bash
cd KAGAS && python3 parallel_to_m2_korean.py -orig sample_test_data/orig.txt -cor sample_test_data/corrected.txt \
  -out ../outputs/kagas_smoke.m2 -hunspell ./aff-dic -noprint > ../logs/kagas_smoke.log 2>&1 && cd ..
python3 metric/m2scorer/scripts/m2scorer.py KAGAS/sample_test_data/corrected.txt outputs/kagas_smoke.m2
```

- 결과: M² 83줄 생성, `A` 줄이 저자 샘플 출력(`sample_test_data/output.m2`)과 오류 유형까지 완전 동일. m2scorer 자기채점 P/R/F0.5 = 1.0.
- 문제: 명령에 섞인 불필요한 줄로 itcerdo `/tmp/_x`를 생성 후 즉시 삭제 — 지정 디렉토리 밖 접근.

### 21:10–21:14 · 데이터 구조·무결성 점검 [itcerdo]

```bash
find ~/projects/phdq_bart/data -maxdepth 3; cat data/README.md
python3 ../tools/check_data.py | tee logs/check_data.log
python3 ../tools/check_split_nfkc.py | tee logs/check_split_nfkc.log
```

- 결과: 학습 입력 `data/Preprocessed/<data>/<data>_{train,val,test}.txt` (korean_learner / native / lang8 / union), 형식 오류 0.
  - korean_learner·native 분할이 저장소 `train_test_val_split(random_state=1)`로 순서까지 재현됨. native 12292/2634/2634는 README 로그와 일치.
  - union = 세 데이터셋 합(멀티셋 동일). `<data>_{val,test}.m2`의 S줄이 `.txt` 소스와 순서·개수 일치.
  - 학습 `.txt`는 NFKC 정규화 상태, 구두점 분리 미적용(README 절차에도 punct_split 없음 → 그대로 사용).
  - `korean_learner_train_original/_corrected.txt`만 다른 처리본(구두점 분리됨·NFKC 미적용), 미사용.

### 21:14–21:15 · 특수 토큰 전수 점검 [itcerdo]

```bash
# tokenizer_setup.py(원본과 동일한 RobertaProcessing 부착 + 검사) 저장소에 추가
python3 ../tools/check_special_tokens.py > logs/check_special_tokens.log 2>&1   # exit=0
```

- 결과: 전부 통과. 토크나이저 전 섹션·post_processor 원본과 동일, config 차이는 `bos_token_id`(1→0)·`kobart_version`뿐. union 전체 155,547쌍의 `encode()`와 `dataset.py` 출력(input_ids·decoder_input_ids·labels)이 원본과 전부 동일. `<unk>` 0, 128 토큰 초과 0, decode 동일.

### 21:16–21:18 · 코드 수정 (C1–C5, R1–R6, 경로) [gsm → itcerdo]

```bash
scp itcerdo:~/projects/phdq_bart/Standard_Korean_GEC/{run.py,model.py,dataset.py} <gsm 스크래치>/   # 수정 후 다시 scp
python3 -c "from transformers.optimization import AdamW; AdamW(p, lr=3e-5, correct_bias=False)"   # 4.44에서 동작
git diff --stat   # model.py 181, run.py 82줄 변경, tokenizer_setup.py 신규
```

- 결과: C1 `tokenizer_setup` 로딩, C2 원본 AdamW 유지, C3 `pl.Trainer` 직접 생성(`devices=1`, CSVLogger), C4/C5 PL 2.x 훅, R1 best-GLEU 1개 저장, R2 M² 블록 제거, R3 `--resume_finetune`, R4 `--repetition_penalty 2.0`, R5 `--eval_test`, R6 `--run_id`, P1 `--data` 기반 기본 경로(`../data/Preprocessed`). 생성 파일명 `hypothesis/reference/source/gleu.txt`로 고정. 커밋하지 않음.

### 21:18–21:20 · 통합 스모크 테스트 [itcerdo]

```bash
# ~/projects/phdq_bart/smoke/native: train 512 / val 128 / test 128줄
CUDA_VISIBLE_DEVICES=0 python3 run.py --data native --run_id smoke_fit --data_root ../smoke --lr 3e-5 --batch_size 64 --max_epochs 2 --max_seq_len 128 --warmup_ratio 0.0 --seed 0 > logs/smoke_fit.log 2>&1
CUDA_VISIBLE_DEVICES=0 python3 run.py --eval_test --data native --run_id smoke_fit --data_root ../smoke --model_ckpt_path outputs/native/smoke_fit/model_ckpt/native_3e-05_epoch=01.ckpt --batch_size 64 --max_seq_len 128 --seed 0 > logs/smoke_test.log 2>&1
CUDA_VISIBLE_DEVICES=0 python3 run.py --resume_finetune --data native --run_id smoke_after --data_root ../smoke --model_ckpt_path <위 ckpt> --lr 1e-5 --batch_size 64 --max_epochs 1 --max_seq_len 128 --warmup_ratio 0.0 --seed 0 > logs/smoke_resume.log 2>&1
python3 metric/m2scorer/scripts/m2scorer.py outputs/generation/native/smoke_fit/epoch1/test/hypothesis.txt ../smoke/native/native_test.m2
```

- 결과(스모크용 소규모, 성능 수치 아님): fit val GLEU 27.60 → 32.76, best ckpt 1개만 유지. eval_test GLEU 32.56. resume_finetune val GLEU 34.19. M² P 0.2661 / R 0.1500 / F0.5 0.2304.
- 정렬 확인: source·reference·hypothesis·데이터·공식 M² 128/128 순서 일치, decode(source/label) = 원문 128/128.
- 정리: `smoke/`, `outputs/native`, `outputs/generation` 삭제.

### 21:21–21:23 · 공식 테스트 M² 검증 [itcerdo]

```bash
cd KAGAS && python3 parallel_to_m2_korean.py -orig ../../data/Preprocessed/native/native_test_original.txt -cor ../../data/Preprocessed/native/native_test_corrected.txt -out ../outputs/kagas_check/native_test.m2 -hunspell ./aff-dic -noprint > ../logs/kagas_check_native_test.log 2>&1
```

- 결과: 1분 39초. 공식 `native_test.m2`와 2626/2634블록 완전 동일, 편집(구간·교정문) 2632/2634 동일. 나머지는 오류 유형 라벨 차이와 편집 2블록 → 테스트 M²는 공식 파일 사용.

### 21:22–21:30 · `plan.md` 갱신, 작업 로그 신설 [gsm]

- 작업: §2-4 데이터, §3 코드 수정·특수 토큰 점검·스모크 테스트, §4 전처리(재실행 안 함), §5 실행 명령(`--data` 기반), §6 평가(공식 M²), §7 체크리스트 갱신. `log.md` 신설 및 `plan.md` §0 기록 규칙 추가.

### 21:28–21:44 · 시나리오1 첫 run — korean_learner / individual_seed0 (recipe root) [itcerdo]

```bash
# tools/run_one.sh 신설: 특수 토큰 점검 → 학습 → 최고 GLEU ckpt로 --eval_test → M² (진행 기록 logs/runs.log)
tmux new-session -d -s kobart-gec -c ~/projects/phdq_bart/Standard_Korean_GEC "bash ../tools/run_one.sh korean_learner individual_seed0 0 3e-5; exec bash"
```

- 설정: lr 3e-5, batch 64, 10 epoch, warmup 0.0, linear, `<s>…</s>` 후처리, 생성 시작 1, repetition_penalty 2.0.
- 학습: 21:28:33–21:43:19 (epoch당 약 1.5분, GPU 11.5GB). 검증 GLEU epoch0–9: 38.78 / **39.57** / 39.20 / 38.87 / 38.98 / 38.66 / 38.54 / 38.62 / … → best epoch 1.
- ckpt: `outputs/korean_learner/individual_seed0/model_ckpt/korean_learner_3e-05_epoch=01.ckpt`
- 테스트: **GLEU 38.70**, M² **P 25.42 / R 21.11 / F0.5 24.43** (`outputs/generation/korean_learner/individual_seed0/epoch1/test/`)
- 논문 Table 6 (KoBART, Kor-Learner, test): GLEU 45.06 / P 43.35 / R 24.54 / F0.5 37.58 → **GLEU −6.4, P −17.9**. 논문 Table D.1 검증 GLEU 46.94 vs 39.57.
- 로그: `logs/korean_learner_individual_seed0_{check,train,test,m2}.log`
- 참고: tmux 소켓은 `/tmp/tmux-1000/`에 생성됨(계획서에 명시된 tmux 사용에 따른 것).

### 21:39–21:41 · self-GLEU로 데이터·지표 검증 [itcerdo]

```bash
python3 - # metric.gleumodule.run_gleu(reference=타깃, source=소스, hypothesis=소스), 결과 outputs/selfgleu/
```

- 결과: 논문 self-GLEU와 **소수점 둘째 자리까지 일치** — korean_learner val 25.90 / test 25.54, native 25.92 / 25.71, lang8 19.38 / 20.01, union val 21.66 (union test는 21.96, 논문 표 21.66은 val 값 중복 기재로 판단).
- 결론: 데이터 분할·GLEU 계산은 논문과 동일 → 점수 차이는 모델·학습·생성 설정에서 발생.
- 추가 확인: 전체 데이터에서 `decode(encode(x)) == x` (lang8 train 타깃 1줄만 앞 공백 차이) → 논문 §5.5 “디코딩된 텍스트로 학습·M² 생성”과 현재 데이터·공식 M²가 같은 상태.

### 21:41–21:45 · 논문 실험 코드(레거시) 대조 [gsm, itcerdo]

```bash
# 저자 저장소 src/KoBART-gec/{train.py,dataset.py,Makefile}, SKT-AI/KoBART kobart/pytorch_kobart.py 확인
curl -sSL -o kobart_base_tokenizer_cased_cf74400bce.zip https://huggingface.co/skt/kobart-base-v1/resolve/main/legacy/kobart_base_tokenizer_cased_cf74400bce.zip
curl -sSL -o kobart_base_cased_ff4bda5738.zip https://huggingface.co/skt/kobart-base-v1/resolve/main/legacy/kobart_base_cased_ff4bda5738.zip
# 위치: ~/projects/phdq_bart/.cache/kobart_legacy/
```

- README: “원래 실험 재현은 spreadsheet 및 `src/KoBART_GEC` 참고” → 논문 수치는 레거시 코드 산출물로 판단.
- 레거시 가중치(`get_pytorch_kobart_model`) = `skt/kobart-base-v1`과 **완전 동일**(261개 텐서 최대 절대차 0.0).
- 레거시 토크나이저(`get_kobart_tokenizer`, `emji_tokenizer/model.json`) = skt HF 토크나이저와 전 섹션 동일, **post_processor 없음** → 논문 실험은 `<s>…</s>` 없이 학습.
- 레거시 config에는 `decoder_start_token_id` 없음 → 레거시 requirements의 transformers 4.0.0 BartConfig에도 기본값 없음 → 생성이 `bos_token_id=0`으로 시작(학습 디코더 입력 첫 토큰 0과 일치). 현재 root 설정은 학습 0 / 생성 1로 불일치.
- 레거시 학습 설정: cosine 스케줄러, `warmup_ratio=0.1`, `gradient_clip_val=1.0`, dropout 0.1, 생성에 repetition_penalty 없음(빔 4). batch는 Makefile에 미지정(기본 32)이나 논문 명시값 64 사용.

### 21:44–21:46 · recipe 옵션 추가 (`root` / `legacy`) [gsm → itcerdo]

- `run.py`: `--recipe {root,legacy}` + 개별 인자 `--add_bos_eos --scheduler --warmup_ratio --gradient_clip_val --decoder_start_token_id --forced_eos_token_id --repetition_penalty`(명시값 우선, 생략 시 recipe 값). Trainer에 gradient clipping 연결.
  - root: 후처리 O, linear, warmup 0.0, clip 없음, 생성 시작 1, forced eos 1, rep 2.0 (기존 동작 그대로)
  - legacy: 후처리 X, cosine, warmup 0.1, clip 1.0, 생성 시작 0, forced eos 없음, rep 1.0
- `model.py`: 스케줄러 선택, `generate()`에 decoder_start/forced_eos 전달, `gleu.txt`에 recipe 설정 기록.
- `tokenizer_setup.py`: `load_tokenizer(add_bos_eos)`, recipe별 encode 형식 검사, report에 생성 설정 포함.
- `tools/check_special_tokens.py [root|legacy]`: legacy는 레거시 원본 토크나이저 파일과 비교. `tools/run_one.sh <data> <run_id> <seed> <lr> <recipe> [init_ckpt]`로 변경(실행 중 스크립트 보호 위해 임시 파일 후 `mv`).

### 21:46– · 진단 run — korean_learner / legacy_seed0 (recipe legacy) [itcerdo]

```bash
tmux send-keys -t kobart-gec "bash ../tools/run_one.sh korean_learner legacy_seed0 0 3e-5 legacy" Enter
```

- 특수 토큰 점검(legacy 기준) 전부 통과, `num_warmup_steps 310`. 결과는 완료 후 기록.

- (21:46 run 결과 추가) 학습 21:46:14–22:07:35, 테스트 22:08:19, M² 22:08:56. 검증 GLEU epoch0–9: 29.46 / **32.50** / 31.29 / 29.09 / 30.57 / 30.71 / 28.36 / 30.77 / 30.80 / 30.84 (val loss 0.62→0.82 상승). best epoch 1.
- 테스트: **GLEU 32.12**, M² **P 20.57 / R 14.01 / F0.5 18.81** — root보다 더 나쁨.
- 로그: `logs/korean_learner_legacy_seed0_{check,train,test,m2}.log`
- 기록 누락 정정: root run(individual_seed0) 검증 GLEU epoch 8, 9 = 38.60, 38.57.

---

## 2026-09-25

### 08:36–08:38 · 두 run 생성 결과 분석 [itcerdo]

```bash
python3 - # outputs/generation/korean_learner/{individual_seed0,legacy_seed0}/epoch1/test/ 의 source/reference/hypothesis 비교
```

- 공통 증상: **문장 첫머리만 깨지고 나머지는 대체로 올바르게 교정.**
  - root: 앞 단어 중복 — `이유로 TV 이유로 TV에서…`, `지금부터부터`, `들면 웰 들면 웰빙…` (인접 단어 반복 519문장, 정답은 31)
  - legacy: 앞 단어 탈락 — `TV에서 방영하는…`, `많이 사용을…`, `전업주부가…` (길이비 0.84)

### 08:38–08:39 · 원인 확인 — 디코더 첫 위치 미래 토큰 누설 [gsm, itcerdo]

```bash
curl -s https://raw.githubusercontent.com/huggingface/transformers/v4.0.0/src/transformers/models/bart/modeling_bart.py | sed -n 159,180p
python3 - # skt 모델에서 decoder_input 뒤쪽 토큰만 바꿔 위치0 logits 비교 (sdpa/eager)
```

- `model.py` forward의 `decoder_attention_mask = decoder_input_ids.ne(0)`이 디코더 첫 토큰(0)을 가림 → 위치0 행이 전부 가려짐 → 최신 transformers가 이 행을 '전부 보기'로 풀어 **첫 토큰 예측이 정답 뒷부분을 봄**. 생성 때는 뒷부분이 없으므로 첫머리가 망가짐.
- transformers **4.0.0**(논문 실험 코드 버전) `_prepare_bart_decoder_inputs`에는 `# never mask leading token, even if it is pad` → `decoder_padding_mask[:, 0] = decoder_padding_mask[:, 1]` 처리가 있어 누설이 없었음. 4.8.1에는 없음.
- 재현: `ne(0)` 그대로 → 뒤쪽 토큰만 바꿔도 위치0 logits 최대차 **24.65** (sdpa·eager 동일). 첫 토큰 비마스킹(4.0.0 방식) → **0.0000**.

### 08:39 · 수정 및 점검 추가 [gsm → itcerdo]

- `model.py`: `make_decoder_attention_mask()` 추가(첫 디코더 토큰은 마스킹하지 않음, 4.0.0 동작 재현), forward에서 사용.
- `tools/check_special_tokens.py`: §5 누설 검사 추가(위치0 logits가 뒤쪽 디코더 토큰과 무관해야 통과).

```bash
python3 ../tools/check_special_tokens.py legacy   # exit 0, 누설 logits 최대차 0.000000
python3 ../tools/check_special_tokens.py root     # exit 1, 누설 11.99 (디코더 입력이 [0, 0, …]이라 앞 두 칸 모두 0 → 수정으로도 해소 불가)
```

- 결론: root 형식(`<s>…</s>` 후처리)은 현재 transformers에서 누설을 피할 수 없고 논문 실험 형식도 아님 → **legacy가 논문 재현 경로.**

### 08:39– · 수정 후 진단 run — korean_learner / legacyfix_seed0 (recipe legacy) [itcerdo]

```bash
tmux send-keys -t kobart-gec "bash ../tools/run_one.sh korean_learner legacyfix_seed0 0 3e-5 legacy" Enter
```

- 결과는 완료 후 기록.

- (08:39 run 결과) 학습 08:39:52–08:54:44, 테스트 08:55:26, M² 08:55:28. 검증 GLEU epoch0–9: 41.90 / 40.04 / 41.51 / 42.21 / 41.80 / **42.23** / 41.75 / 41.83 / 41.69 / 41.64. best epoch 5.
- ckpt: `outputs/korean_learner/legacyfix_seed0/model_ckpt/korean_learner_3e-05_epoch=05.ckpt`
- 테스트: **GLEU 40.86**, M² **P 42.02 / R 19.04 / F0.5 33.85** (`outputs/generation/korean_learner/legacyfix_seed0/epoch5/test/`)
- 출력 점검: 첫머리 오류 해소(첫 단어 정답 일치 3311/4265, 소스 그대로일 때 3202), 인접 단어 반복 60(정답 31), 가설==소스 1353, 가설==정답 426, 길이비 0.969.
- 로그: `logs/korean_learner_legacyfix_seed0_{check,train,test,m2}.log`

### 09:24 · 비교 정리, 세션 종료 [gsm, itcerdo]

| run (korean_learner, seed 0) | 검증 GLEU(best) | 테스트 GLEU | P | R | F0.5 |
| --- | --- | --- | --- | --- | --- |
| individual_seed0 (root) | 39.57 (ep1) | 38.70 | 25.42 | 21.11 | 24.43 |
| legacy_seed0 (legacy, 누설 있음) | 32.50 (ep1) | 32.12 | 20.57 | 14.01 | 18.81 |
| **legacyfix_seed0 (legacy + 누설 수정)** | **42.23 (ep5)** | **40.86** | **42.02** | **19.04** | **33.85** |
| 논문 (3-seed 평균) | 46.94 | 45.06 | 43.35 | 24.54 | 37.58 |

```bash
ssh itcerdo 'tmux kill-session -t kobart-gec'
```

- itcerdo tmux 세션 종료, GPU 유휴(1 MiB). 로컬 대기 작업 종료.
- 문제: 로컬 대기 루프가 `pgrep -f "run_one.sh …"`로 종료를 판단했는데, 원격 확인 명령 자신이 같은 문자열을 포함해 자기 자신과 매칭 → run 종료(08:55) 후에도 끝나지 않았고 보고가 늦어짐. 이후에는 `logs/runs.log`의 `M2 DONE`/`FAIL` 표시로 판단한다.

### 09:27– · korean_learner legacy seed 1, 2 (3-seed 평균용) [itcerdo]

- 결정: batch는 논문 명시값 64 유지(레거시 Makefile 기본값 32 비교는 하지 않음 — 사용자 확인).

```bash
tmux new-session -d -s kobart-gec -c ~/projects/phdq_bart/Standard_Korean_GEC "bash ../tools/run_one.sh korean_learner legacyfix_seed1 1 3e-5 legacy && bash ../tools/run_one.sh korean_learner legacyfix_seed2 2 3e-5 legacy; exec bash"
```

- seed 1 시작 09:27:18. 결과는 완료 후 기록.

### 09:28 · 어제 남은 대기 작업 정리 [gsm, itcerdo]

- 로컬 대기 작업 `b5pedtbiv`(2026-09-24 legacy_seed0 대기, `pgrep` 자기 매칭으로 종료되지 않던 것) 종료.
- 확인: itcerdo 사용자 프로세스는 오늘 09:27 시작한 seed 1/2 run(tmux `kobart-gec`)뿐, 어제 것 잔여 없음. 로컬 itcerdo ssh 잔여 없음.
- (09:27 run 결과) seed 1: 09:27:18–09:44:46, best epoch 4 (검증 GLEU 42.91), 테스트 **GLEU 41.63 / P 39.22 / R 20.29 / F0.5 33.05**. seed 2: 09:44:46–10:00:44, best epoch 3 (검증 42.66), 테스트 **GLEU 41.19 / P 39.09 / R 19.68 / F0.5 32.65**.
- 로그: `logs/korean_learner_legacyfix_seed{1,2}_{check,train,test,m2}.log`. 대기 작업 종료 코드 1은 마지막 `grep -l Traceback`이 0건이라 생긴 것(오류 아님).

### 10:02 · korean_learner 3-seed 평균, 세션 종료 [itcerdo]

| korean_learner (legacy + 누설 수정) | 검증 GLEU | 테스트 GLEU | P | R | F0.5 |
| --- | --- | --- | --- | --- | --- |
| seed 0 / 1 / 2 | 42.23 / 42.91 / 42.66 | 40.86 / 41.63 / 41.19 | 42.02 / 39.22 / 39.09 | 19.04 / 20.29 / 19.68 | 33.85 / 33.05 / 32.65 |
| **평균** | **42.60** | **41.23** | **40.11** | **19.67** | **33.18** |
| 논문 | 46.94 | 45.06 | 43.35 | 24.54 | 37.58 |
| 차이 | −4.34 | −3.83 | −3.24 | −4.87 | −4.40 |

- seed 간 테스트 GLEU 범위 0.77 → 차이는 seed 편차가 아니라 체계적.
- 레거시 `train.py` forward(마스크 생성)는 현재 코드와 동일함을 확인.

```bash
ssh itcerdo 'tmux kill-session -t kobart-gec'
```

- tmux 종료, GPU 유휴.

### 10:26– · 나머지 전체 실험 대기열 시작 [itcerdo]

- `tools/run_all.sh` 신설: recipe legacy, batch 64, 10 epoch. 실패 시 `FAIL` 기록 후 다음 run 진행, 완료 run은 `outputs/<data>/<run_id>/.done`으로 표시(재실행 시 건너뜀), 끝나면 `QUEUE DONE`.
- 순서: native seed0–2 (lr 3e-5) → union seed0–2 (3e-5, 시나리오 2 1단계 겸용) → korean_learner·native `afterunion_seed0–2` (union 같은 seed ckpt에서 lr 1e-5) → lang8 seed0–2 (3e-5) → lang8 `afterunion_seed0–2` (1e-5). 총 18 run.
- 디스크: 여유 530GB, 체크포인트 run당 1.4GB.

```bash
tmux new-session -d -s kobart-gec -c ~/projects/phdq_bart/Standard_Korean_GEC "bash ../tools/run_all.sh; exec bash"
```

- 10:26:22 QUEUE START. 결과는 run 완료 시 기록.

### 10:35 · native / legacyfix_seed0 완료 [itcerdo, 대기열]

- 10:26:22–10:35:49. 검증 GLEU epoch0–9: 53.74 / 63.70 / 62.72 / **64.43** / 63.87 / 63.32 / 63.35 / 63.12 / 63.09 / 63.12 → best epoch 3.
- 테스트: **GLEU 63.42, P 76.14 / R 50.59 / F0.5 69.15** (논문 native: 67.24 / 75.34 / 55.95 / 70.45, 검증 69.37). Kor-Learner와 같은 경향(재현율이 낮음).
- run당 약 9.5분 → 대기열 전체 종료 예상 2026-09-25 23시 전후(union·lang8은 데이터 크기 비례 추정).

### 10:40–11:05 · 빔서치·모델 동작을 transformers 4.0.0과 직접 비교 [gsm, itcerdo]

- 소스 대조(4.0.0 vs 4.44.2 `generation_beam_search.py`/`generation_utils.py`/`modeling_bart.py`): 길이 정규화 분모(4.0.0 `hyp.shape[-1]`=시작 토큰+생성 토큰, 4.44 `generated_len`=생성 토큰+eos)와 종료 판정은 결과적으로 동일. 로짓 처리 순서·top-2k 후보도 동일. 4.0.0 BART는 `max_length-1`에서 eos 강제(= 4.44 `forced_eos_token_id=1`).
- 4.0.0 실행 환경 구성(CPU 전용, 프로젝트 안):

```bash
Standard_Korean_GEC/.venv/bin/pip install uv
UV_PYTHON_INSTALL_DIR=~/projects/phdq_bart/.cache/uv-python UV_CACHE_DIR=~/projects/phdq_bart/.cache/uv uv venv -p 3.8 .venv-tf400
uv pip install -p .venv-tf400/bin/python "torch==1.7.1+cpu" --find-links https://download.pytorch.org/whl/torch_stable.html
uv pip install -p .venv-tf400/bin/python "transformers==4.0.0" "tokenizers==0.9.4" "numpy<1.24" six sacremoses filelock requests tqdm regex packaging
# 결과: Python 3.8.20 / torch 1.7.1+cpu / transformers 4.0.0 / tokenizers 0.9.4  (torch 1.13은 SAVE_STATE_WARNING import 오류로 불가)
```

- 비교(`tools/cmp_new.py`, `tools/cmp_old.py`, `tools/cmp_report.py`, 산출물 `outputs/tfcmp/`): `korean_learner/legacyfix_seed0` 체크포인트, 검증 앞 256문장 생성 + 학습 32문장 forward. 4.0.0 쪽은 레거시 config·레거시 `generate(input_ids, eos_token_id=1, max_length=128, num_beams=4)`·레거시 마스크(`ne(0)`) 그대로.

```bash
CUDA_VISIBLE_DEVICES="" python3 ../tools/cmp_new.py outputs/korean_learner/legacyfix_seed0/model_ckpt/korean_learner_3e-05_epoch=05.ckpt
../.venv-tf400/bin/python ../tools/cmp_old.py
python3 ../tools/cmp_report.py
```

- 결과: **생성 토큰 256/256 완전 일치**(forced eos 1/None 모두), 첫 토큰 둘 다 0. forward logits 최대 절대차 8.7e-5(부동소수 오차), loss 0.0116466 vs 0.0116464.
- 결론: 현재 코드(legacy recipe + 첫 토큰 비마스킹)의 **모델 계산과 빔서치는 논문 버전과 동일** → 빔서치는 수정할 필요 없음. 남은 차이는 학습 과정(PyTorch Lightning 1.1 vs 2.6 학습 루프, torch 1.7 vs 2.11 수치, 셔플·seed 등) 쪽.
- 참고: 학습 샘플 loss 0.0116(epoch 5)로 학습 데이터에 강하게 맞춰짐, 검증 loss는 epoch 1 이후 상승 — 과적합 구간에서 검증 GLEU 최고 epoch를 고르는 구조.

### 10:35–22:01 · 대기열 18 run 완료 [itcerdo]

- 전 run 정상 종료(FAIL 0, `.done` 18개). 개별 결과(시각·best epoch·테스트 GLEU/P/R/F0.5)는 `logs/runs.log`, run별 로그는 `logs/<data>_<run_id>_{check,train,test,m2}.log`.
- 소요: native 약 9분, korean_learner 약 14분, union 약 80분, lang8 약 55–68분(run당).

### 22:29 · 진행 확인, 세션 종료, 3-seed 평균 비교 [gsm, itcerdo]

```bash
ssh itcerdo 'tmux kill-session -t kobart-gec'
python3 tools/summarize_runs.py logs/runs.log   # (gsm 사본으로 실행)
```

| 시나리오 | 데이터 | GLEU 재현 / 논문 | P 재현 / 논문 | R 재현 / 논문 | F0.5 재현 / 논문 | GLEU seed 범위 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 개별 | korean_learner | 41.23 / 45.06 (-3.83) | 40.11 / 43.35 (-3.24) | 19.67 / 24.54 (-4.87) | 33.18 / 37.58 (-4.40) | 0.77 |
| 1 개별 | native | 62.57 / 67.24 (-4.67) | 71.41 / 75.34 (-3.93) | 49.87 / 55.95 (-6.08) | 65.70 / 70.45 (-4.75) | 1.50 |
| 1 개별 | lang8 | 26.35 / 28.48 (-2.13) | 30.79 / 37.56 (-6.77) | 9.16 / 11.62 (-2.46) | 20.90 / 25.93 (-5.03) | 1.15 |
| 1 개별 | union | 31.16 / 33.70 (-2.54) | 35.35 / 44.75 (-9.40) | 11.78 / 14.64 (-2.86) | 25.20 / 31.70 (-6.50) | 0.95 |
| 2 Union→개별 | korean_learner | 38.15 / 42.66 (-4.51) | 40.87 / 53.51 (-12.64) | 15.00 / 21.18 (-6.18) | 30.33 / 41.00 (-10.67) | 2.44 |
| 2 Union→개별 | native | 47.33 / 59.71 (-12.38) | 68.44 / 85.47 (-17.03) | 29.48 / 47.38 (-17.90) | 54.13 / 73.63 (-19.50) | 1.15 |
| 2 Union→개별 | lang8 | 25.96 / 28.65 (-2.69) | 29.03 / 37.46 (-8.43) | 8.79 / 12.00 (-3.21) | 19.80 / 26.78 (-6.98) | 1.48 |

- 시나리오 1: 네 데이터셋 모두 논문보다 GLEU 2.1–4.7 낮음, 재현율 일관되게 낮음(Lang8·Union은 정밀도 차이가 더 큼).
- 시나리오 2: **native가 크게 낮음**(GLEU −12.4). Union 선학습 모델이 원문을 그대로 두는 비율이 높음 — native 테스트 가설==소스 44.7%(시나리오 1은 22.1%), korean_learner 48.3%(31.7%). 띄어쓰기 교정 누락이 두드러짐. native after-union은 검증 loss가 더 낮은데(0.23 vs 0.25) GLEU는 46.9 vs 64.4이고, lr 1e-5 추가 학습 중 검증 GLEU가 epoch 0 이후 내려감. 논문도 시나리오 2 native가 시나리오 1보다 낮지만(59.71 vs 67.24) 그 폭이 훨씬 작음.
- 확인: itcerdo tmux 종료, GPU 유휴.

### 22:30–22:38 · 저자 spreadsheet·체크포인트로 원인 분리 [gsm, itcerdo]

```bash
curl -sL "https://docs.google.com/spreadsheets/d/1II_BB10YPijp1Rgw3ZgQElvv6pw7xINOdTpJbAPz484/export?format=csv&gid=0"      # 공개, 권한 불필요
curl -sL "https://docs.google.com/spreadsheets/d/1II_BB10YPijp1Rgw3ZgQElvv6pw7xINOdTpJbAPz484/htmlview/sheet?headers=true&gid=0"   # 체크포인트 링크 추출
curl -sSL -o 122.ckpt "https://drive.usercontent.google.com/download?id=1W3nECVOhAnDoG4zMk-zF4CMj7nVtjlAs&export=download&confirm=t"   # 1.49GB
curl -sSL -o 122.yaml "https://drive.usercontent.google.com/download?id=19j7lDQvawCiaX9ikQrz2HR1hWvs30sXk&export=download&confirm=t"
# 위치: ~/projects/phdq_bart/author_ckpt/122/
```

- spreadsheet: 실행 명령 `python3 src/KoBART-gec/train.py … --batch_size 64 --dropout 0.1` (레거시 코드). 링크 있는 run: 시나리오1 seed0(KL 122, native 123, lang8 124), union 125/131/135, 시나리오2 seed0(KL 137, native 139, lang8 138). spreadsheet의 native 테스트 평균은 65.07(논문 표 67.24와 다름).
- `122.yaml`: batch 64, warmup 0.1, gradient_clip 1.0, lr 3e-5, dropout 0.1, precision 32, max_len 128 → **legacy recipe와 동일**.
- `122.ckpt`: PL 1.1.0, global_step 1244(311×4), ModelCheckpoint 기록 val_gleu 46.9129(epoch 3). 스케줄러 lr 2.25e-5 @1244 = 우리 cosine 계산 2.2485e-5, AdamW eps 1e-6·correct_bias False·wd 0.01/0.0 동일. (`tools/inspect_author_ckpt.py`: 옛 PL 클래스는 대체 객체로 역직렬화)
- **저자 가중치를 우리 파이프라인으로 채점** (`author_122_converted.ckpt`, lm_head=shared 연결):

```bash
python3 run.py --recipe legacy --data korean_learner --run_id author122_val --seed 0 --model_ckpt_path ../author_ckpt/122/author_122_converted.ckpt --batch_size 64 --max_seq_len 128
python3 run.py --recipe legacy --eval_test --data korean_learner --run_id author122_test --seed 0 --model_ckpt_path ../author_ckpt/122/author_122_converted.ckpt --batch_size 64 --max_seq_len 128
```

- 결과: **검증 GLEU 46.91 (저자 기록 46.9129와 일치)**, 테스트 GLEU 45.30 / P 45.08 / R 25.10 / F0.5 38.89. → 평가 파이프라인은 논문과 동일, 차이는 **학습**에서 발생.

### 22:38–22:41 · 원인 확인 — PL 2.x에서 dropout 꺼진 채 학습 [itcerdo]

```bash
CUDA_VISIBLE_DEVICES="" python3 ../tools/check_train_mode.py
grep -l "module(s) in eval mode at the start of training" logs/*_train.log | wc -l   # 23/23
```

- `from_pretrained()`는 eval 모드로 반환 → PL 2.6은 학습 시작 때 `model.train()`을 호출하지 않음 → **training_step 중 내부 BART training=False (dropout 꺼짐)**. PL 경고 "Found 182 module(s) in eval mode at the start of training". PL 1.1은 학습 epoch마다 `model.train()` 호출.
- 지금까지 23개 학습 로그 전부 해당 → 모든 run이 dropout 없이 학습됨(학습 loss 0.0116까지 과적합, 검증 GLEU 조기 하락과 일치).
- 수정: `model.py`에 `on_train_epoch_start()`에서 `self.train()`, `training_step` 첫 배치에서 내부 모델이 eval이면 `RuntimeError`로 중단.
- `tools/run_all.sh <tag>`로 변경(run_id `<tag>_seedN`, `afterunion_<tag>_seedN`), korean_learner 시나리오1 포함 21 run.

### 22:42– · 전체 재실행 대기열 (tag=dropout) [itcerdo]

```bash
tmux new-session -d -s kobart-gec -c ~/projects/phdq_bart/Standard_Korean_GEC "bash ../tools/run_all.sh dropout; exec bash"
```

- 22:42:00 QUEUE START. 첫 run korean_learner dropout_seed0: eval 모드 검사 통과(중단 없음). PL 경고 1회는 훅 이전 시점 검사라 정상.
- 22:49 확인 — korean_learner dropout_seed0 검증 GLEU epoch0–3: 42.61 / 44.84 / 46.68 / **47.19**, 검증 loss 0.71 / 0.64 / 0.64 / **0.67**. 저자 체크포인트(epoch 3) 검증 GLEU 46.91·loss 0.67과 일치 수준(dropout 없던 run 최고 42.23). → 수정 효과 확인, 대기열 계속 진행.

---

## 2026-09-26

### 09:12 · 재실행 대기열(tag=dropout) 중간 확인 [gsm, itcerdo]

```bash
python3 tools/summarize_runs.py logs/runs.log dropout   # tag 인자 지원, 완료 seed만 평균
```

- 진행: 21개 중 19개 완료(FAIL 0). 남은 run: lang8 `afterunion_dropout_seed1`(08:31:40 시작, epoch 6/10), `afterunion_dropout_seed2`. 종료 예상 10:20 전후.
- Union 체크포인트(시나리오 2 시작점): seed0 epoch 6, seed1 epoch 3, seed2 epoch 5.
- 중간 3-seed 평균 vs 논문 (lang8 시나리오 2는 1 seed):

| 시나리오 | 데이터 | GLEU 재현 / 논문 | P 재현 / 논문 | R 재현 / 논문 | F0.5 재현 / 논문 | GLEU seed 범위 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 개별 | korean_learner | 45.68 / 45.06 (+0.62) | 43.31 / 43.35 (-0.04) | 25.46 / 24.54 (+0.92) | 37.98 / 37.58 (+0.40) | 0.71 (3 seed) |
| 1 개별 | native | 68.40 / 67.24 (+1.16) | 80.34 / 75.34 (+5.00) | 58.31 / 55.95 (+2.36) | 74.69 / 70.45 (+4.24) | 2.46 (3 seed) |
| 1 개별 | lang8 | 29.65 / 28.48 (+1.17) | 39.46 / 37.56 (+1.90) | 13.33 / 11.62 (+1.71) | 28.33 / 25.93 (+2.40) | 0.60 (3 seed) |
| 1 개별 | union | 34.75 / 33.70 (+1.05) | 44.38 / 44.75 (-0.37) | 16.09 / 14.64 (+1.45) | 32.80 / 31.70 (+1.10) | 1.75 (3 seed) |
| 2 Union→개별 | korean_learner | 44.19 / 42.66 (+1.53) | 52.15 / 53.51 (-1.36) | 23.47 / 21.18 (+2.29) | 41.88 / 41.00 (+0.88) | 2.60 (3 seed) |
| 2 Union→개별 | native | 60.96 / 59.71 (+1.25) | 86.28 / 85.47 (+0.81) | 48.73 / 47.38 (+1.35) | 74.70 / 73.63 (+1.07) | 4.83 (3 seed) |
| 2 Union→개별 | lang8 | 28.97 / 28.65 (+0.32) | 37.95 / 37.46 (+0.49) | 12.47 / 12.00 (+0.47) | 26.94 / 26.78 (+0.16) | 0.00 (1 seed) |

- 결과: dropout 수정 후 **모든 데이터셋·시나리오가 논문 수치에 도달**(GLEU −0.0 ~ +1.5). 시나리오 2 native도 60.96(논문 59.71)으로 회복(수정 전 47.33). native 시나리오 1은 정밀도가 논문보다 +5.0 높음(seed 범위 2.46).

### 10:28 · 재실행 대기열 완료, 최종 결과 [gsm, itcerdo]

- 09:24:22 lang8 afterunion_dropout_seed1 완료(GLEU 30.33 / P 38.62 R 14.25 F0.5 28.78), 10:17:31 seed2 완료(GLEU 28.75 / P 36.96 R 12.43 F0.5 26.50), **10:17:31 QUEUE DONE tag=dropout — 21/21 완료, FAIL 0.**
- 최종 3-seed 평균 vs 논문 Table 6 (`python3 tools/summarize_runs.py logs/runs.log dropout`):

| 시나리오 | 데이터 | GLEU 재현 / 논문 | P 재현 / 논문 | R 재현 / 논문 | F0.5 재현 / 논문 | GLEU seed 범위 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 개별 | korean_learner | 45.68 / 45.06 (+0.62) | 43.31 / 43.35 (-0.04) | 25.46 / 24.54 (+0.92) | 37.98 / 37.58 (+0.40) | 0.71 (3 seed) |
| 1 개별 | native | 68.40 / 67.24 (+1.16) | 80.34 / 75.34 (+5.00) | 58.31 / 55.95 (+2.36) | 74.69 / 70.45 (+4.24) | 2.46 (3 seed) |
| 1 개별 | lang8 | 29.65 / 28.48 (+1.17) | 39.46 / 37.56 (+1.90) | 13.33 / 11.62 (+1.71) | 28.33 / 25.93 (+2.40) | 0.60 (3 seed) |
| 1 개별 | union | 34.75 / 33.70 (+1.05) | 44.38 / 44.75 (-0.37) | 16.09 / 14.64 (+1.45) | 32.80 / 31.70 (+1.10) | 1.75 (3 seed) |
| 2 Union→개별 | korean_learner | 44.19 / 42.66 (+1.53) | 52.15 / 53.51 (-1.36) | 23.47 / 21.18 (+2.29) | 41.88 / 41.00 (+0.88) | 2.60 (3 seed) |
| 2 Union→개별 | native | 60.96 / 59.71 (+1.25) | 86.28 / 85.47 (+0.81) | 48.73 / 47.38 (+1.35) | 74.70 / 73.63 (+1.07) | 4.83 (3 seed) |
| 2 Union→개별 | lang8 | 29.35 / 28.65 (+0.70) | 37.84 / 37.46 (+0.38) | 13.05 / 12.00 (+1.05) | 27.41 / 26.78 (+0.63) | 1.58 (3 seed) |

- 결론: 논문 수치 재현 완료. 7개 조합 모두 GLEU가 논문과 같거나 0.6–1.5 높음. 시나리오 2의 정밀도↑·재현율↓ 경향도 논문과 일치.
- 재현에 필요했던 수정 3가지(모두 라이브러리 버전 차이로 생긴 동작 차이): ① 레거시 토큰화(`<s>…</s>` 미부착) ② 디코더 첫 토큰 비마스킹(transformers 4.0.0 동작) ③ 학습 epoch마다 `train()`으로 dropout 활성화(PL 1.1 동작).

```bash
ssh itcerdo 'tmux kill-session -t kobart-gec'
```

- tmux 종료, GPU 유휴.

### 10:40 · 결과 문서 작성 [gsm, itcerdo]

- `result_kobart.md` 작성: 최종 결과(테스트·검증 3-seed 평균 vs 논문), 재현 체크포인트 21개 경로와 run별 점수, 체크포인트 사용법, 재현에 필요했던 수정 3가지, 검증 근거, 설정·재실행 방법.
- 재현 가중치 위치: itcerdo `~/projects/phdq_bart/Standard_Korean_GEC/outputs/<data>/{dropout_seedN, afterunion_dropout_seedN}/model_ckpt/*.ckpt` (각 1.4GB). 진단용 run(`individual_seed0`, `legacy_seed0`, `legacyfix_seed*`, `afterunion_seed*`)은 dropout 없이 학습된 것이라 사용 금지로 표기.
- 사용법 예제 검증: `dropout_seed0` 체크포인트를 `BartForConditionalGeneration`에 로드해 생성 — 정상 동작(`페이스북이 유투복 많이 사용을…` → `페이스북이 유투복을 많이 사용을…`).

### 10:55–11:10 · 남은 작업: 체크포인트 정리, 코드 정리·커밋 [gsm, itcerdo]

```bash
# 진단용 run 체크포인트 삭제 (재현 21개 제외, 생성 결과·로그는 유지)
cd ~/projects/phdq_bart/Standard_Korean_GEC/outputs
for d in $(ls -d */*/model_ckpt | grep -vE "/(dropout_seed[012]|afterunion_dropout_seed[012])/"); do rm -rf "$d"; done
# 도구 스크립트를 저장소 안으로 이동, 경로를 저장소 루트 기준으로 변경
cp -r ~/projects/phdq_bart/tools Standard_Korean_GEC/tools && rm -rf ~/projects/phdq_bart/tools
python3 tools/check_special_tokens.py legacy   # 새 위치에서 exit 0 확인
git checkout -b skt-kobart-v1-repro && git add .gitignore REPRODUCTION.md run.py model.py tokenizer_setup.py tools/ && git commit   # e4a163c
```

- 삭제: 진단용 체크포인트 23개(32GB). 남은 체크포인트는 재현 run 21개(`outputs/` 32GB, 디스크 여유 502GB).
- 스크립트 경로: `~/projects/phdq_bart/...` 절대경로 → 저장소 루트 기준(`../data/Preprocessed`, `outputs/tfcmp`, 레거시 파일은 `KOBART_LEGACY_DIR` 또는 `../.cache/kobart_legacy`). `run_one.sh`/`run_all.sh`는 `tools/` 기준. `plan.md`·`result_kobart.md`의 명령도 `tools/…`로 수정.
- 저장소에 넣지 않은 것: `.venv`, `logs/`, `outputs/`, 체크포인트, 데이터, 논문 PDF, `ssh.md`. 커밋 전 문서에서 토큰·비밀번호·IP 패턴 검사(해당 없음).
- 원격: https://github.com/esoterikosQ/26-13_PHDQ_BART (빈 저장소). itcerdo에는 GitHub 인증이 없어 gsm에서 push.

### 11:10–11:25 · GitHub push [gsm, itcerdo]

```bash
git clone -b skt-kobart-v1-repro itcerdo:projects/phdq_bart/Standard_Korean_GEC push_repo   # gsm 스크래치
git remote set-url origin https://github.com/esoterikosQ/26-13_PHDQ_BART.git
git push origin skt-kobart-v1-repro:main   # 거부: GH013 push protection
```

- 거부 원인: **저자 저장소 이력의 README에 GitHub 개인 액세스 토큰(`ghp_…`) 형태의 문자열**(“original agreement form” 링크 자리, 현재 README 101행 포함 5개 커밋). 우리 코드와 무관.
- 조치: 비밀 허용(unblock)은 하지 않고 이력을 새로 구성 — 저자 코드 `dfe0af9`를 토큰 문자열만 설명 문구로 바꿔 단일 커밋으로 가져오고(`acc6dea`), 재현 코드(`7f5c2ec`)·문서(`2bb2094`) 커밋을 그 위에 cherry-pick. 세 커밋 전체에서 토큰 패턴 검사 통과, 원래 브랜치와의 차이는 README 1줄.

```bash
git checkout --orphan main dfe0af9   # README 토큰 링크 치환 후 커밋, 이후 cherry-pick e4a163c 36b7844
git push origin main:main            # 성공, main = 2bb2094
```

- itcerdo 작업 저장소: remote `phdq` 추가, `main`을 `phdq/main` 추적으로 전환(작업 트리는 README 1줄 외 동일). 이전 로컬 브랜치 `skt-kobart-v1-repro`(저자 이력 포함)는 로컬에만 남음.
- 이후 문서 갱신은 추가 커밋으로 올림.

### 11:40 · 후속 실험 계획서 `plan2.md` 작성 [gsm]

```bash
curl -s "https://huggingface.co/api/models/<id>?blobs=true"   # 후보 존재·파라미터(safetensors 또는 fp32 가중치 크기)·라이선스 확인
```

- 확인한 후보(파라미터 추정): pko-t5-base 276M / pko-t5-large 821M (CC-BY-4.0), ke-t5-base 247M / ke-t5-large 783M (Apache-2.0, `KETI-AIR`→`KETI-NLP`로 이전), mbart-large-50 611M (MIT), mt5-large 1.23B / byt5-large 1.23B / mt5-xl 3.74B (Apache-2.0), kanana-1.5-2.1b 2.3B / kanana-1.5-8b 8.0B (Apache-2.0), EXAONE-3.5-2.4B 2.4B (비상업), Qwen2.5-1.5B 1.5B (Apache-2.0). gated: gemma-3-4b-it, HyperCLOVAX-SEED-1.5B.
- 계획 요지: 원문 기준 평가로 통일(다른 토크나이저의 디코딩 차이 배제), 모델별 동일 lr 탐색 예산, 새 표준 학습 스크립트(`gec2/`, 학습 모드·누설 검사 내장), 권장 최소 구성 4개(pko-t5-base, pko-t5-large, mbart-large-50, kanana-1.5-2.1b LoRA).

### 12:10 · `plan2.md` 보완 (사용자 검토 반영) [gsm]

- 사용자 결정: Neuron 작업·데이터 이전은 사용자가 직접 수행(Claude는 코드 준비 → GitHub → 사용자가 pull·실행 → 로그·결과 공유), `plan.md`와 `plan2.md`는 별개 실험, LLM 5 epoch, LLM 디코딩 빔 4, V100 제외(A100×2 → H200×1만), itcerdo torch와 Neuron 드라이버 호환성 확인 완료.
- 반영: §0에 자원 구성·호환성 확인 명시, §0-1 Neuron 작업 인계 방식 신설(작업 스크립트 템플릿, 경로 인자, run 디렉토리 단위 공유 항목), 원칙 3에 트랙별 epoch(seq2seq 10 / LLM 5)와 연장 판단 규칙, 원칙 5 빔 4 통일, 정밀도·메모리·자원 비교에서 V100 삭제, §4·§5의 6시간 기준을 트랙별 epoch로 수정(LLM 약 60분/epoch 목표), §6에 모델–구성 실패 시 제외 규칙과 인계 누락 대응 추가, tmux 문구를 itcerdo 한정으로 수정.

### 12:40 · `plan2.md` 최종 검토 반영 [gsm]

```bash
curl -sL https://huggingface.co/<id>/raw/main/config.json   # 후보 dropout 기본값 확인
```

- 확인: dropout 기본값 pko-t5-base/large·mt5-large `dropout_rate` 0.1, mbart-large-50 `dropout` 0.1, **ke-t5-large 0.0**, kanana-1.5-2.1b는 attention_dropout 0.0 외 없음.
- 사용자 결정: 6시간은 **1 epoch 기준**(공유 노드에서 epoch 도중 재개가 까다로움), 초과 예상 시 별도 가이드와 SLURM 작업 시간을 명시. seq2seq는 조기 종료 없음.
- 반영: 원칙 3(체크포인트 최고값과 조기 종료 기준값 분리, min_delta 0.2·patience 3·4 epoch 이후 중단 예시, 트랙별 규칙 차이 표기), 원칙 4(KoBART(gec2) 기준선 추가, T5 lr 1e-4~5e-4), §3(torch AdamW·wd 0.01·clip 1.0 공통, dropout 정책: seq2seq 0.1 통일·LLM 기본값 기록), §0-1(`--time` 명시·산정식, 공유 항목에 실제 학습 설정·dropout), §4·§5·§6·§7(1 epoch 6시간 기준, 초과 시 별도 가이드 내용, 48시간 초과 시 epoch 경계 분할 제출, LLM은 korean_learner·native 먼저), 하위 목록 들여쓰기 4칸으로 통일(17줄 조정). 수정 전 사본: gsm 스크래치 `plan2.before_final.md`.

### 20:24–20:29 · plan2 단계 1: 원문 기준 평가 스크립트, KoBART 21 run 재채점 [gsm, itcerdo]

- **작업**: `plan2.md` 원칙 2에 따라 source·reference를 데이터 원문으로 쓰는 공통 평가 `gec2/evaluate.py`를 만들고, 재현 KoBART 21 run의 기존 생성 결과(전 epoch 검증 + 테스트)를 다시 채점해 기존 수치와 비교.
- 코드: `gec2/evaluate.py`(원문 읽기 규칙은 `KoBARTGecDataset.read_docs`와 동일: 빈 줄·탭 2칸 아님·빈 칸 제외 / GLEU는 `metric/gleumodule.py`, M²는 `metric/m2scorer` + 공식 `<data>_test.m2` / 결과 `source·reference·hypothesis·gleu·m2score.txt`, `scores.json`), `gec2/rescore_kobart.py`. gsm 스크래치에서 작성 후 rsync로 itcerdo 저장소 `gec2/`에 배치. `.gitignore`에 `outputs2/` 추가.

```bash
cd ~/projects/phdq_bart/Standard_Korean_GEC && source .venv/bin/activate
python3 -m gec2.rescore_kobart > logs/gec2_rescore_kobart.log 2>&1   # 3분 30초
```

- **결과**: 21 run 모두 **테스트 GLEU·P·R·F0.5 최대 절대차 0.0000**, 전 epoch 검증 GLEU 최대 절대차 0.0000, 최고 검증 epoch 21/21 동일. 기존 source/reference(KoBART 디코딩)와 원문이 다른 줄 0개(4개 데이터 val/test 전부). → 원칙 2 검증 통과, 새 평가로 KoBART 기준선 수치가 바뀌지 않음.
- 산출물: `outputs2/kobart_repro/<data>/<run>/epochNN/{val,test}/`, 표 `outputs2/kobart_repro/rescore.md`, `rescore.json`, 로그 `logs/gec2_rescore_kobart.log`.

### 20:30–20:45 · plan2 단계 2: gec2 환경, 모델 다운로드, 사전 점검(§4) [gsm, itcerdo]

- **작업**: 후보 모델 캐시 다운로드, gec2 전용 가상환경 생성, 모델별 사전 점검.
- 기존 `.venv`(transformers 4.44.2, tokenizers 0.19.1)로는 kanana `tokenizer.json`을 읽지 못함(`data did not match any variant of untagged enum ModelWrapper`), mBART-50은 protobuf 없음 → 재현 환경은 그대로 두고 **`.venv-gec2`를 새로 만듦**(캐시 경로 export는 `.venv`와 동일하게 activate에 추가). Neuron에서도 같은 버전을 쓰도록 `gec2/requirements.txt`로 고정.

```bash
python3 -m venv .venv-gec2 && source .venv-gec2/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 && pip install -r gec2/requirements.txt
# → torch 2.11.0+cu128, transformers 4.57.6, tokenizers 0.22.2, pytorch-lightning 2.6.6, accelerate 1.15.0, numpy 2.5.3, scipy 1.18.1
snapshot_download: skt/kobart-base-v1, paust/pko-t5-base(.bin), paust/pko-t5-large(safetensors), facebook/mbart-large-50(.bin+spm),
                   kakaocorp/kanana-1.5-2.1b-instruct-2505(safetensors), Qwen/Qwen2.5-0.5B-Instruct(코드 검증용)   # 캐시 +13GB, 여유 489GB
python3 -m gec2.precheck --model <id> --kind seq2seq|llm --leak [--dropout 0.1]   # 결과 outputs2/precheck/<tag>.json
```

- **사전 점검 결과** (전체 데이터 고유 문장 278,830개, 길이는 모델 입력 형식 그대로 특수 토큰·프롬프트 포함):

| 모델 | 파라미터 | 원문 복원 | `<unk>` | 최대 토큰 src/tgt (union=전체) | 누설 검사 | 형식 |
| --- | --- | --- | --- | --- | --- | --- |
| skt/kobart-base-v1 (gec2) | 123.9M | 99.9996% (앞 공백 1건) | 0 | 118 / 119 | 통과 | `<s>…</s>`로 감쌈(BART 표준), decoder_start 1 |
| paust/pko-t5-base | 275.6M | 100% | 0 | 187 / 188 | 통과 | `…</s>`, decoder_start 0 |
| paust/pko-t5-large | 820.5M | 100% | 0 | 187 / 188 | 통과 | 〃 |
| facebook/mbart-large-50 | 610.9M | 99.08% (공백 27, 기타 2,552) | **2,612 토큰 (0.06%)** | 137 / 138 | 통과 | `[ko_KR] … </s>`, decoder_start 2, forced_bos ko_KR |
| kanana-1.5-2.1b-instruct | 2.087B | 100% | 0 | 181 / 161 (합계 342) | 통과 | 프롬프트+정답+`<|eot_id|>`, pad `<|end_of_text|>` |
| Qwen2.5-0.5B (코드 검증용) | 494M | 100% | 0 | 192 / 174 (합계 366) | 통과 | 〃 `<|im_end|>` |

- mBART-50의 기타·unk는 sentencepiece에 없는 드문 음절(예: `잏본`, `괂찮아` → `<unk>`)로, 출력에서도 이 음절을 만들 수 없음. 평가는 원문 기준이라 점수 계산 방식에는 영향 없음(해당 음절 복원 불가는 모델 특성으로 기록).
- 잘림: 학습 코드는 길이를 자르지 않음(동적 패딩) → 잘림 0건.
- dropout 기본값: kobart `dropout` 0.1, pko-t5 `dropout_rate` 0.1, mBART `dropout` 0.1 (seq2seq는 0.1로 명시 설정), kanana·Qwen은 `attention_dropout` 0.0뿐(LLM은 기본값 유지).
- 학습 전 모델 생성 확인: kanana는 이미 교정문 형태 출력(`왜냐하면 현대 사회에서 …`), 일부 `1. ` 접두 → 출력 파싱 규칙(첫 줄만, 앞뒤 공백 제거)과 이탈 수(`newline_cut`, `empty`)를 run마다 기록.

### 20:45–20:59 · gec2 학습 코드 작성·스모크 [gsm, itcerdo]

- **작업**: `plan2.md` §3의 학습 스크립트 작성(`gec2/core.py` 공통, `train_seq2seq.py`, `train_llm.py`, `memprobe.py`, `run_queue.sh`).
    - HF 표준(실제 pad, attention_mask, 라벨 −100, 모델 내부 shift), 학습 전 누설 검사(`check_leak.json`, 실패 시 중단), 첫 학습 배치의 학습 모드 검사(`check_train_mode.json`, 실패 시 중단), epoch마다 `model.train()`.
    - torch AdamW(bias 보정), wd 0.01(bias·Norm 제외), clip 1.0, warmup 10% + cosine(10 epoch 예정 step으로 고정), bf16-mixed, 전역 batch 64(= micro × GPU × accumulation), 빔 4·샘플링 없음, 최대 생성 길이 `min(512, ceil(2 × batch 최장 입력) + 10)`.
    - 체크포인트: 평가용 최고 검증 GLEU `ckpt/best`(HF 형식), 재개용 `ckpt/last.ckpt`(옵티마이저·스케줄러·step·조기 종료 상태). `--epochs_per_job`로 epoch 경계 분할, `--resume`으로 재개.
    - LLM 조기 종료: 최고값과 기준값 분리, min_delta 0.2 이상일 때만 기준 갱신, patience 3, 4 epoch 이후(`es_update`). 규칙 시뮬레이션: plan 예시(최고 epoch 1, 2·3·4 미갱신) → epoch 4 뒤 중단 ✓, 작은 상승은 최고만 갱신 ✓, 정확히 +0.2는 기준 갱신 ✓, 3 epoch만 있으면 중단 없음 ✓.
    - run 디렉토리(`outputs2/<tag>/<data>/<run_id>/`)에 §0-1 공유 항목 자동 저장: `run_info.json`(커밋·명령·SLURM 변수·nvidia-smi·패키지), `train_config.json`/`train_config_applied.json`(옵티마이저·lr·스케줄·warmup·wd·clip·dropout·전역 batch·정밀도), `model_info.json`(파라미터 수·임베딩 제외 수·특수 토큰), `check_*.json`, `train_steps.jsonl`(step별 loss·lr), `epochs.jsonl`(epoch별 검증 loss·GLEU·학습/검증 시간·GPU별 최대 메모리), `val/epochNN/`, `test/`(hypothesis·gleu·m2score·scores.json·문장/초).
- 스모크 (모두 `outputs2/smoke/`, 로그 `logs/gec2_smoke_*.log`):
    - KoBART native 512줄, 2 epoch를 `--epochs_per_job 1`로 나눠 실행 → `--resume` → 테스트: 재개 후 epoch·step·lr 스케줄 연속(마지막 lr 0.0), 테스트 GLEU 30.25(256줄, 스모크). 정상.
    - DDP 경로: GPU가 1개라 CPU 2프로세스(gloo)로 검증. 홀수 줄(37)에서 DistributedSampler 패딩 중복 제거·원래 순서 복원 정상(37줄). 같은 가중치의 단일 프로세스 테스트와 36/37줄 동일, 1줄은 배치 구성 차이로 빔 결과가 다름(미학습 모델, bf16 autocast) → GPU 수에 따라 생성이 극소수 문장에서 달라질 수 있음을 기록.
    - LLM 경로: Qwen2.5-0.5B(코드 검증용, 실험 아님) native 256줄 2 epoch, gradient checkpointing, accumulation 2 → 누설·학습 모드 검사 통과, 테스트 GLEU 49.44(64줄). 정상.
- **규칙 위반 기록**: DDP 비교 중 itcerdo `/tmp/phdq_ddp_hyp.txt`에 임시 파일 1개를 썼음(지정 디렉토리 밖). 즉시 작업 디렉토리로 옮긴 뒤 삭제, `/tmp`에 남은 것 없음 확인.

### 20:57–21:00 · 메모리 측정, batch 결정, 커밋, lr 탐색 시작 [itcerdo]

```bash
python3 -m gec2.memprobe --model <id> --kind seq2seq --micro_batch <N> --gen_batch_size <M>   # union 최장 문장 batch로 학습 1 step + 빔 생성
```

| 모델 | micro 64 | 선택 (micro × accum) | 학습 최대 | 생성 batch · 최대 |
| --- | --- | --- | --- | --- |
| kobart-base-v1 | 6.8GB | 64 × 1 | 6.8GB | 64 · 3.6GB |
| pko-t5-base | OOM | 32 × 2 | 18.8GB | 64 · 8.9GB |
| pko-t5-large | OOM | 16 × 4 | 25.4GB | 32 · 14.1GB |
| mbart-large-50 | OOM (logits 250k 어휘) | 32 × 2 | 21.1GB | 64 · 12.7GB |

- 전역 batch 64는 모두 동일(accumulation으로 맞춤).
- 커밋: itcerdo 저장소 `7db2e96` “Add gec2 …” (`.gitignore`에 `outputs2/`, `.venv-gec2/` 추가). GitHub push는 아래에서 별도로.

```bash
tmux new-session -d -s gec2 "bash gec2/run_queue.sh gec2/queues/lr_sweep.txt"   # 21:00:36 QUEUE START, 12 run
```

- lr 탐색 대기열(§5 단계 3, korean_learner seed 0): KoBART(gec2) 1e-5/3e-5/5e-5 → pko-t5-base 1e-4/3e-4/5e-4 → mBART-50 1e-5/3e-5/5e-5 → pko-t5-large 1e-4/3e-4/5e-4. 진행 기록 `logs/gec2_runs.log`, run별 로그 `logs/gec2_<tag>_<data>_<run>.log`. 완료 run은 `last.ckpt`를 지우고 `ckpt/best`만 유지.
- 첫 run(KoBART lr 1e-5): epoch당 학습 0.2분 + 검증 생성 0.1분(재현 코드 run당 15분 대비 크게 빠름: bf16 + 동적 패딩).

### 21:01–21:07 · Neuron SLURM 템플릿·안내, 모의 실행, 커밋·push [gsm, itcerdo]

- **작업**: `plan2.md` §0-1의 Neuron 인계 준비. 파티션·`--gres`·`--comment`·CPU 수는 실사례를 받기 전이라 스크립트에 고정하지 않고 제출 명령 인자로 받게 함.
    - `gec2/slurm/llm_smoke.sbatch`(`--time 01:00:00`: 메모리 측정 5조합 + native 512줄 1 epoch), `llm_train.sbatch`(`--time 08:00:00` 임시: 대표 epoch 측정용, 실측 후 교체; `RESUME=1`로 재개; `MICRO`·`GEN_BATCH`·`GC` 환경변수로 조정), `common.sh`(경로·환경·GPU 수=task 수 확인, srun에 `--cpus-per-task` 명시, 작업 스크립트·slurm 로그·nvidia-smi를 run 디렉토리로 복사), `precheck.sh`, `paths.example.sh`(→ `paths.local.sh`, git 제외), 안내 `gec2/NEURON.md`.
    - DDP에 `gradient_as_bucket_view=True`(그래디언트 버킷 중복 메모리 절약).
- 모의 실행: itcerdo에서 가짜 `srun`·`scontrol`(작업 디렉토리 `logs/slurmsim/bin`)과 SLURM 환경변수로 `llm_train.sbatch`를 Qwen0.5B·128줄로 실행 → 1 epoch 분할 중단 → `RESUME=1` 재개 → 테스트, run 디렉토리에 `slurm/`(스크립트·로그·nvidia-smi) 저장까지 정상. 모의 파일·출력은 삭제.

```bash
git commit   # itcerdo 7d3c918 "Add Neuron SLURM templates and guide for the LLM track (plan2 §0-1)"
git clone -b main itcerdo:projects/phdq_bart/Standard_Korean_GEC push_repo   # gsm 스크래치, 토큰 패턴 검사 통과
git push origin main:main   # 838bdc5..7d3c918
```

- 참고: lr 탐색 첫 run은 `7db2e96`에서 시작했고, 이후 run은 `7d3c918` 코드로 실행됨(차이는 DDP 옵션과 slurm 파일뿐, 단일 GPU 학습에는 영향 없음).

### 21:03–21:08 · lr 탐색 첫 결과 확인: KoBART(gec2) lr 1e-5 [itcerdo]

- run: `outputs2/kobart-base-v1/korean_learner/lr1e-5_seed0`, seed 0, lr 1e-5, batch 64×1, 10 epoch, 소요 2분 52초. 로그 `logs/gec2_kobart-base-v1_korean_learner_lr1e-5_seed0.log`.
- 검증 GLEU epoch 0–9: 40.36 / 46.09 / 48.96 / 50.16 / 51.05 / 51.55 / 51.90 / **52.23** / 52.10 / 52.22 → 최고 epoch 7.
- **테스트 GLEU 51.12, P 55.62, R 32.69, F0.5 48.77** (재현 KoBART 3-seed 평균 45.68 / 43.31 / 25.46 / 37.98보다 GLEU +5.4).
- 평가 오류 여부 확인: 출력 4,265줄, 원문 그대로 출력 비율 new 0.201 / old 0.190, 정답과 완전 일치 new 0.204 / old 0.123. 원문을 출력으로 둔 self-GLEU 25.54(P 100, R 0) 정상. 예시에서 재현 모델은 단어를 빠뜨리는 경우(`페이스북이 유투복 많이 …` → `페이스북이 많이 사용했었습니다.`)가 있고 gec2 모델은 `유투브를` 복원. M²는 공식 정답 파일로 따로 계산되므로 GLEU 계산과 독립적으로 같은 방향.
- 해석(잠정): 재현 recipe(`legacy`)의 원본 특이 동작(0-패딩, 특수 토큰 미부착, bias 보정 없는 AdamW, fp32 등)이 성능을 낮추고 있었음. plan2 원칙 4의 KoBART(gec2) 기준선이 필요한 이유가 확인됨. 원인 분해는 하지 않음.

### 21:30–21:45 · Neuron 스크립트를 conda·실제 파티션 기준으로 갱신 [gsm, itcerdo]

- **작업**: 사용자 요청 — Neuron은 venv가 아니라 conda로 환경 관리, `ssh.md`에 Neuron 정보(작업 디렉토리 `/scratch/r984a02/phdq_bart`, `showque`, `showappl`) 추가, 데이터는 `/scratch/r984a02/phdq_bart/data`로 이전 완료.
- 결정: 실제 동작 스크립트 사례 대신 `ssh.md`의 공식 안내 값으로 자원 지시문을 확정(`plan2.md` §0-1에 기록). 첫 스모크로 확인.
    - `gec2/slurm/a100x2_{smoke,train}.sbatch`: `amd_a100nv_8`, `--gres=gpu:2`, `--ntasks-per-node=2`, `--cpus-per-task=8`(GPU당 코어 8/1), `--comment="field=nlp;appl=pytorch-ddp"`.
    - `gec2/slurm/h200x1_{smoke,train}.sbatch`: `amd_h200nv_8`, `--gres=gpu:1`, task 1 × CPU 8, `appl=pytorch`.
    - `--time`: 스모크 01:00:00, 학습 08:00:00(임시, 대표 epoch 측정 후 교체) — 근거를 각 스크립트 주석에 적음.
    - 본문은 `llm_smoke.sh`·`llm_train.sh`로 공유. `paths.sh`에 Neuron 경로 고정(저장소 `…/26-13_PHDQ_BART`, 데이터 `…/data/Preprocessed`가 없으면 `…/data`, 모델 `…/models/kanana-1.5-2.1b-instruct-2505`, conda `…/conda/gec2`, 캐시 `…/.cache`), 덮어쓰기는 `paths.local.sh`(git 제외).
    - `activate_env`: `eval "$(conda shell.bash hook)"; conda activate $CONDA_ENV`. `common.sh`가 시작 전에 4개 데이터셋 파일(train/val/test.txt, test.m2)과 모델 디렉토리를 확인.
    - `NEURON.md`를 conda 기준으로 다시 씀. `requirements.txt` 주석 갱신. 기존 `llm_*.sbatch`, `paths.example.sh` 삭제.
- 모의 실행(itcerdo, 가짜 srun·scontrol, `paths.local.sh`로 venv·Qwen0.5B 지정):
    - 1차(GPU): **실행 중인 lr 탐색 run(mBART lr 1e-5)과 GPU가 겹쳐 모의 쪽이 OOM으로 실패.** 탐색 run은 계속 진행됨을 확인(epoch 1 검증 GLEU 49.03, 최대 25.63GB). 다만 epoch 1 학습 시간이 수 초 정도 영향을 받았을 수 있음.
    - 2차(CPU, `--accelerator cpu`, 16줄): 제출 → 경로·데이터 확인 → 학습 → 테스트 → `slurm/` 저장까지 exit 0. 모의 파일은 삭제.
- 커밋 `a76d291` “Neuron scripts: conda env, fixed partitions and paths from ssh.md”, GitHub push `7d3c918..a76d291`.
- lr 탐색 진행(`logs/gec2_runs.log`): KoBART(gec2) lr 5e-5, pko-t5-base 1e-4/3e-4/5e-4 완료(pko-t5-base 테스트 GLEU 3e-4 57.94, 5e-4 57.41), mBART 진행 중.

### 21:51–22:18 · lr 탐색: mBART 완료, pko-t5-large 실패(OOM) [itcerdo]

- mBART-50 (batch 32×2): lr 1e-5 검증 52.92 / 테스트 GLEU 52.42 · P 58.10 R 36.70 F0.5 52.03 (best epoch 6), lr 3e-5 53.55 / 52.51 · 59.61 37.91 53.49 (8), lr 5e-5 53.40 / 52.15 · 60.67 37.65 54.06 (7).
- **pko-t5-large 3개 run 모두 FAIL** (22:17–22:18): 첫 optimizer step 직후 `torch.OutOfMemoryError`(프로세스 31.3GB). 원인: `memprobe.py`가 학습 1 step만 측정해 Adam 상태가 생긴 뒤의 순전파·역전파 메모리를 빠뜨림(micro 16을 25.4GB로 잘못 판단). 로그는 `logs/failed/gec2_pko-t5-large_korean_learner_lr*_seed0_micro16_oom.log`로 옮김.

### 22:40–22:51 · memprobe 수정, pko-t5-large 재실행 [gsm, itcerdo]

```bash
python3 -m gec2.memprobe ...   # 2 step(clip 포함)으로 수정 후 재측정
```

- 재측정(학습 최대): pko-t5-large micro 16 **OOM**, micro 8 **21.1GB**; pko-t5-base micro 32 20.8GB; mBART micro 32 25.6GB(탐색 run 실측 25.63GB와 일치).
- 대기열 파일에서 pko-t5-large를 8 × accumulation 8로 변경, 실패 run 디렉토리 삭제 후 대기열 재시작(완료 run은 SKIP).

```bash
rm -rf outputs2/pko-t5-large && tmux new-session -d -s gec2 "bash gec2/run_queue.sh gec2/queues/lr_sweep.txt"   # 22:47:40
```

- pko-t5-large lr 1e-4 epoch 0: 검증 GLEU 55.07, 학습 2.8분 + 검증 0.8분, 최대 24.13GB.

### 22:20–22:51 · Neuron 수정 반영: conda 환경 이름, precheck, GitHub 경유 파일 이동 [gsm, itcerdo]

- **사용자 요청**: conda 환경 이름 `phdq_bart`(용량 설정은 사용자가 관리), `precheck.sh` 실행 실패, 스모크 작업 대기 중, 환경 간 이동은 GitHub 경유(Neuron → GitHub → itcerdo), Neuron 파일 수정도 GitHub로.
- 확인: 저장소 `esoterikosQ/26-13_PHDQ_BART`는 **public**(GitHub API). 데이터셋은 저자 README상 신청서(Google form)로 받고 비상업 용도로만 사용·배포 가능 → 데이터셋과 데이터 원문이 들어간 파일은 공개 저장소에 올리면 안 됨. 모델 출력(hypothesis)도 데이터 문장에서 파생되므로 저장소를 비공개로 바꾸기 전에는 올리지 않음.
- precheck 실패 원인(추정, 오류 출력은 받지 못함): ① `paths.sh`가 conda 환경을 경로 `…/conda/gec2`로 찾음(실제 이름은 `phdq_bart`) ② 비대화형 bash는 `~/.bashrc`의 conda init을 읽지 않아 `conda activate` 불가 ③ 저장소 루트가 아닌 곳에서 실행하면 상대경로 실패.
- 수정(`gec2/slurm/`):
    - `paths.sh`: `CONDA_ENV=phdq_bart`. `REPO`는 스크립트 위치에서 자동으로 구함. `activate_env`가 `conda.sh`를 `$CONDA_EXE` → `CONDA_BASE` → 흔한 설치 위치 순으로 찾아 불러온 뒤, torch·transformers·pytorch_lightning import 확인. 실패하면 `오류:` 메시지.
    - `precheck.sh`: `set -e`, 어디서 실행해도 동작, 모델·데이터 경로 확인. `common.sh`: 저장소 루트가 아닌 곳에서 제출하면 중단.
    - `share_results.sh` 신설: `$OUT_ROOT` run 디렉토리(ckpt·safetensors·`source.txt`·`reference.txt`·`gold_*.m2` 제외)를 `results/neuron/runs/`로, `logs/slurm/*.out`을 `results/neuron/slurm_logs/`로 복사해 commit → `pull --rebase` → push. **저장소가 public이면 중단**, 50MB 넘는 파일이 있어도 중단. `outputs2/`는 `.gitignore` 패턴에 걸려서 `runs/`라는 이름을 씀(`git check-ignore`로 확인).
    - `NEURON.md`: conda 이름, 로그 위치, GitHub 규칙, `git pull` 시점(작업 실행 중에는 하지 않음) 반영.
- activate_env 검증(itcerdo, 가짜 `conda.sh`): CONDA_BASE 지정 ✓, CONDA_EXE로 찾기 ✓, 못 찾을 때 오류 메시지 ✓. **규칙 위반 기록**: “못 찾는 경우” 테스트에서 스크립트가 itcerdo 홈의 기존 conda 설치(`~/miniconda3` 등 기본 위치)를 찾아 `conda.sh`를 읽고 실행함(지정 디렉토리 밖 읽기, 쓰기 없음). 테스트용 가짜 파일은 삭제.
- `plan2.md` 수정(수정 전 사본: gsm 스크래치 `plan2.before_github_flow.md`): §0 환경(conda `phdq_bart`, itcerdo `.venv-gec2`), §0 파일 이동 규칙(GitHub 경유, 데이터셋·체크포인트 제외, 비공개 필요, pull 시점), §0-1 실행(로그 위치)·공유(`share_results.sh`), §6 위험(데이터 조건 위반).
- 작업 방식 전환: 코드 수정은 gsm의 clone에서 커밋해 GitHub에 push하고, itcerdo는 `git pull phdq main`으로 받음(이전: itcerdo에 rsync 후 커밋).

```bash
git commit && git push origin main:main          # ea7b6b5 "Neuron: conda env phdq_bart, robust conda activation, GitHub result sharing"
ssh itcerdo 'git checkout -- gec2/memprobe.py gec2/queues/lr_sweep.txt && git pull phdq main'   # itcerdo = ea7b6b5, 작업 트리 깨끗
```

### 22:55–23:00 · Neuron precheck 재실패: 로그인 노드 계산 금지, 사전 점검을 CPU 작업으로 [gsm, itcerdo]

- **사용자 보고**: glogin01에서 `bash gec2/slurm/precheck.sh` 실행 → 스크립트 출력 없이 로그인 때의 Lustre quota 안내(/home01 23.99G/64G, /scratch 1.263T/100T)만 나오고 conda에서 튕김. 사용자 판단: 셸 스크립트가 공유 노드(로그인 노드) 규칙을 위반했을 때 나오는 결과.
- 원인(사용자 판단에 따름): 로그인 노드에서 python 계산(torch import, 고유 문장 27.9만 개 토큰화 — 토크나이저가 기본으로 모든 코어 사용)을 실행. `plan2.md` §0-1의 "로그인 노드나 짧은 작업에서" 문구를 그대로 따른 설계 오류.
- 수정:
    - `precheck.sh`: SLURM 작업 밖이면 즉시 중단(`require_job`). `precheck.sbatch` 신설(`--partition=cpu`, 코어 4, `--comment="field=nlp;appl=pytorch"`, `--time=00:30:00`: itcerdo에서 약 1분). 데이터 줄 수 출력 추가.
    - `llm_smoke.sh`: 시작할 때 같은 사전 점검 수행(단계 0).
    - `common.sh`·`precheck.sh`: conda 활성화는 `set +eu` 상태에서(conda 스크립트가 errexit·nounset에서 중간 종료될 수 있음), `limit_threads`로 OMP·MKL·RAYON 스레드를 `SLURM_CPUS_PER_TASK`로 제한.
    - `NEURON.md`: 로그인 노드에서는 `git pull`, `sbatch`, `share_results.sh`, 모델 내려받기만. `plan2.md` §0-1 문구 수정.
- 모의 검증(itcerdo, 가짜 conda, 저장소 사본 `logs/simrepo`): ① 작업 밖 실행 → "로그인 노드에서 실행하지 말 것" rc=1 ② `SLURM_JOB_ID` 설정 후 `precheck.sbatch` → 스레드 4, kanana 원문 복원 100%, unk 0, 최대 181/161 토큰(이전 결과와 동일) rc=0. 사본 삭제.
- itcerdo 데이터 줄 수(`grep -c ''`): korean_learner 19,898/4,264/4,265, native 12,292/2,634/2,634, lang8 76,692/16,434/16,434, union 108,883/23,333/23,334 → NEURON.md에 비교 기준으로 적음.
- 커밋 `0aa15c1` "Neuron: no computation on login nodes; precheck as a CPU job", push, itcerdo `git pull` 완료.

## 2026-09-27

### 00:38 · lr 탐색 완료 (plan2 §5 단계 3) [itcerdo]

- 00:38:14 QUEUE DONE, tmux 세션 자동 종료 확인. pko-t5-large(8 × accumulation 8): lr 1e-4 검증 59.91 / 테스트 GLEU 59.10 · P 64.02 R 45.69 F0.5 59.26 (best epoch 6), 3e-4 59.85 / 59.54 · 65.14 46.19 60.20 (8), 5e-4 59.43 / 58.86 · 65.23 45.46 60.01 (9). run당 약 37분.
- lr 선택(korean_learner seed 0, **검증 GLEU 최고**, 테스트는 선택에 쓰지 않음):

| 모델 | 1순위 lr (검증 GLEU) | 나머지 | 선택 lr의 테스트 GLEU / F0.5 |
| --- | --- | --- | --- |
| KoBART(gec2) | 5e-5 (54.50) | 3e-5 54.18, 1e-5 52.23 | 52.17 / 52.38 |
| pko-t5-base | 5e-4 (58.57) | 1e-4 58.48, 3e-4 58.45 | 57.41 / 58.45 |
| mBART-50 | 3e-5 (53.55) | 5e-5 53.40, 1e-5 52.92 | 52.51 / 53.49 |
| pko-t5-large | 1e-4 (59.91) | 3e-4 59.85, 5e-4 59.43 | 59.10 / 59.26 |

- 참고: KoBART·pko-t5-base는 탐색 범위 위쪽 끝, pko-t5-large는 아래쪽 끝이 선택됨. pko-t5 두 모델은 세 값의 차이가 0.1–0.5점으로 작음.

### 00:40–07:12 · README 교체, 결과 공유를 push 전용으로 단순화 [gsm, itcerdo]

- **사용자 지적**: ① `share_results.sh`가 rsync·GitHub API 검사 등으로 복잡함. Neuron 로그인 노드에서는 rsync를 쓸 수 없음. `.gitignore`로 걸러서 push만 하면 되고, 대용량·저작권 파일은 사용자가 직접 옮김. ② GitHub 첫 페이지 README가 원 저자(Yoon) 저장소 README 그대로였음. ③ 스모크 로그를 올렸으니 확인할 것.
- 확인: GitHub `main`은 `0aa15c1`(내 마지막 커밋)이고 Neuron에서 올라온 커밋·브랜치는 없음(`git ls-remote`). 원인 추정: `.gitignore`의 `logs/` 때문에 `logs/slurm/*.out`이 add되지 않음.
- 수정:
    - `README.md`를 이 프로젝트 소개로 새로 씀(KoBART 재현 요약, gec2 비교, 문서·구성, 데이터 비포함·비상업 조건, 출처). 원 저자 README는 `docs/ORIGINAL_README.md`로 옮겨 보존하고, 원 저장소의 Modified MIT License 본문을 `LICENSE`로 분리. 인용 표기는 원 저장소 README의 bibtex를 그대로 씀. `REPRODUCTION.md`의 "저자 README" 참조 경로 수정.
    - `.gitignore`: `logs/` → `/logs/*` + `!/logs/slurm/`. Neuron 출력 `neuron_outputs/`에서 `ckpt/`, `*.safetensors`, `*.bin`, `*.pt`, `source.txt`, `reference.txt`, `gold_*.m2` 제외. 가짜 파일로 `git add -n`을 돌려 보니 slurm 로그·epochs.jsonl·hypothesis·scores·precheck json만 잡힘(ckpt·safetensors·원문 사본·logs/failed 제외).
    - `paths.sh`: `OUT_ROOT=$REPO/neuron_outputs`(저장소 안). `share_results.sh`: `git add logs/slurm neuron_outputs` → commit → `pull --rebase` → push만 함. `common.sh`: slurm 로그를 run 디렉토리로 복사하던 부분 삭제.
    - `NEURON.md`, `plan2.md` §0·§0-1·§6 갱신(공개 저장소 검사 내용 삭제, rsync 불가 명시).
- 커밋 `8b0a429` "Project README; push-only result sharing filtered by .gitignore", push, itcerdo `git pull`(작업 트리 깨끗).
- 첫 스모크 run은 이전 설정대로 `/scratch/r984a02/phdq_bart/outputs2/`(저장소 밖)에 저장됐을 것임 → 올리려면 저장소의 `neuron_outputs/`로 옮겨야 함(사용자에게 안내).

### 07:25 · lr 탐색 결과 진단: 크기 효과 vs 데이터 한계 [itcerdo]

- **작업**: 사용자 질문(seq2seq 간 압도적 차이가 없는 이유가 실험 미완인지, 데이터의 구조적 한계인지)에 답하기 위해, 선택 lr run 4개(korean_learner seed 0)의 검증 곡선과 테스트 출력을 비교.

```bash
PYTHONPATH=. python3 logs/diag_s2s.py | tee logs/diag_s2s_korean_learner.txt
```

- 검증 GLEU 곡선: 모든 모델이 epoch 3–6에 평탄해짐. 마지막 학습 loss 0.008–0.028(과적합 영역) → 학습 부족이 원인은 아님. 같은 토크나이저인 pko-t5에서 검증 loss는 large 0.462 < base 0.534.
- 테스트 정답 완전 일치: kobart 0.232, mbart 0.244, t5base 0.279, t5large 0.291. 원문 그대로 출력: 0.173 / 0.149 / 0.140 / 0.124.
- 4개 모델 출력이 모두 같은 문장 22.0%, 그중 정답과 다른 문장 9.2%(4개 모두 원문 유지인데 정답은 수정한 경우 4.1%). 적어도 한 모델이 정답과 일치 41.4%.
- 예시: 정답이 문체 교체(`해요→합니다`, `-니까→-어서`)이거나 비문(`돕는다→돕아준다`)인 경우, 모델 출력이 문법적으로 맞는데도 정답과 달라 감점됨. 단일 참조 정답이 점수 상한을 누르는 구조.

### 07:32 · Neuron 스모크 결과 확인, 배치·시간 설정, .gitignore 복구 [gsm, itcerdo]

- 사용자 push: `ca2767f`(.gitignore에 `.cache/` 추가, slurm 로그), `fdc117c`(병합), `2ab48f1`(스모크 run·precheck). **병합 후 `.gitignore`에 충돌 표시(`<<<<<<<`, `>>>>>>>`)가 남은 채 커밋됨** → 정리함(다른 파일에는 충돌 표시 없음).
- 사전 점검 작업 915979 (cpu06): 데이터 줄 수 itcerdo와 동일(korean_learner 19,898/4,264/4,265, native 12,292/2,634/2,634, lang8 76,692/16,434/16,434, union 108,883/23,333/23,334), 원문 복원 100%, unk 0, 최대 181/161 토큰.
- 스모크 작업 915975 (gpu40, amd_a100nv_8, **A100-SXM4-80GB × 2**, 드라이버 580.105.08/CUDA 13.0, conda `phdq_bart`, 커밋 0aa15c1):
    - 메모리(최장 문장 batch, GPU 1개): 학습 micro 8 GC 없음 42.4GB, micro 16 GC 없음 57.5GB, GC 켬 micro 8·16 38.9GB, micro 32 42.9GB / 생성 batch 32 39.5GB, 64 51.8GB.
    - DDP 2 GPU, micro 8 × 2 × accumulation 4, GC 켬: 누설·학습 모드 검사 통과, native 512줄 학습 9.2초(8 step), 검증 512줄 GLEU 68.99(생성 11.8초, 검증 전체 31.7초: 최고 체크포인트 8GB 저장 포함), 테스트 256줄 GLEU 67.73 · P 83.97 R 60.51 F0.5 77.93, 33.9문장/초, GPU별 최대 40.85GB. 출력 이탈(줄바꿈·빈 출력) 0.
    - 경고: 테스트 때 `ckpt/best`에 재저장한 토크나이저를 불러오며 "incorrect regex pattern … fix_mistral_regex" 경고. itcerdo에서 원본과 재저장본의 인코딩을 전체 고유 문장 278,830개로 비교 → 차이 0(점수 영향 없음). 그래도 테스트는 원본 모델 토크나이저를 쓰도록 변경.
    - 그 밖의 경고(PL "423 modules in eval mode": 첫 배치 학습 모드 검사로 train 확인됨, AccumulateGrad stream, 테스트 DistributedSampler 중복: 코드에서 제거)는 결과에 영향 없음.
- 결정: A100이 80GB라 FSDP 불필요, DDP 사용. 기본값을 GPU당 micro 16, 생성 batch 64, GC 끔으로 변경(속도 우선, 57.5/51.8GB로 80GB 안에 들어감). 대표 epoch 측정 작업의 `--time`을 02:00:00으로(스모크 처리량으로 korean_learner 1 epoch 15–20분 추정, 근거는 sbatch 주석).
- 커밋 `ddb96ea` "Fix .gitignore merge markers; LLM batch defaults from the Neuron smoke", push, itcerdo pull.

### 07:37 · plan2 단계 4 시작: seq2seq 4개 모델 × native·lang8·union, seed 0 [gsm, itcerdo]

- **작업**: 단계 3에서 고른 lr(KoBART(gec2) 5e-5, pko-t5-base 5e-4, mBART-50 3e-5, pko-t5-large 1e-4)로 시나리오 1 seed 0을 나머지 3개 데이터셋에서 실행. korean_learner는 단계 3의 같은 설정 run을 그대로 씀(추가 실행 없음).
- 대기열 `gec2/queues/stage4_seed0.txt` 12 run(native → lang8 → union 순, 데이터셋마다 4개 모델). 커밋 `de86ecf`, push 후 itcerdo `git pull`.

```bash
tmux new-session -d -s gec2 "bash gec2/run_queue.sh gec2/queues/stage4_seed0.txt"   # 07:37:20 QUEUE START
```

- 예상 소요(korean_learner run 시간 × 데이터 크기 비율로 추정): 합계 약 10–11시간, pko-t5-large union이 가장 김(약 3–3.5시간). 디스크 여유 446GB, 현재 `outputs2` 33GB.

### 10:09 · Neuron kanana lr 탐색 1 epoch(대표 epoch 측정) 결과 확인 [gsm]

- 사용자 push `f6eb125`("lr search test done") + 병합 2건(`fcbf048`, `3ac4813`). 충돌 표시 없음 확인.
- 작업 916172/916173/916174 (gpu39, A100×2, 커밋 ddb96ea, micro 16 × accumulation 2, GC 없음, 생성 batch 64), korean_learner seed 0, `--epochs_per_job 1` → 3개 모두 1 epoch 뒤 정상 분할 중단(`stop_reason` 비어 있음, 재개 대기).

| lr | epoch 0 검증 GLEU | 검증 loss | 학습 | 검증(생성) | GPU 최대 |
| --- | --- | --- | --- | --- | --- |
| 1e-5 | 57.83 | 0.285 | 3.7분 | 2.1분 (1.8분) | 68.3 / 65.3GB |
| 3e-5 | 51.63 | 0.364 | 3.7분 | 2.3분 (1.9분) | 〃 |
| 5e-5 | 47.52 | 0.426 | 3.6분 | 2.8분 (2.5분) | 〃 |

- 주의: warmup이 전체 3,110 step의 10%(311 step) = 정확히 1 epoch라, epoch 0 끝이 lr 최고점. epoch 0 순위로 lr을 고르지 않음(10 epoch·조기 종료 규칙대로 끝난 뒤 최고 검증 GLEU로 선택). 출력 이탈(줄바꿈·빈 출력) 0.
- 작업 시간 실측: 준비 1.5분, epoch당 학습+검증 약 6분 + 재개용 체크포인트 저장 1.2–1.3분 ≈ 7.5분, 작업 전체 약 9분. 1 epoch 6시간 기준 충족.
- `--time`: korean_learner 1.5 + 10 × 7.5 + 테스트 4 = 80.5분 × 1.2 = 97분 → `02:00:00` 유지. native·lang8·union 추정값(01:30 / 06:00 / 08:00, 문장 수 비례)을 sbatch 주석에 기록, 제출 때 `--time`으로 덮어씀. 커밋 `5651329`, push.

### 14:12 · Neuron kanana lr 탐색 완료, lr 선택 / itcerdo 단계 4 중간 결과 [gsm]

- 사용자 push `99d35b3`. 재개 작업 916204(1e-5), 916205(3e-5), 916206(5e-5), A100×2, 커밋 5651329. 충돌 표시 없음.

| lr | 검증 GLEU (epoch 0→끝) | 최고 (epoch) | 종료 | 테스트 GLEU / P / R / F0.5 |
| --- | --- | --- | --- | --- |
| **1e-5** | 57.83 56.16 58.83 58.64 58.80 58.92 | **58.92 (5)** | early_stop (기준값 58.83 @2 이후 3회 +0.2 미달, epoch 6 종료) | 58.33 / 63.99 / 46.45 / 59.50 |
| 3e-5 | 51.63 … 55.24 55.27 | 55.27 (9) | max_epochs | 54.62 / 65.32 / 41.68 / 58.67 |
| 5e-5 | 47.52 … 53.05 53.09 | 53.09 (9) | max_epochs | 52.00 / 64.58 / 38.75 / 56.98 |

- 조기 종료 규칙 동작 확인: 1e-5는 epoch 5에서 최고값 갱신(58.92, 기준값 대비 +0.09)이지만 기준값은 유지 → wait 3, 6번째 epoch 뒤 중단(4 epoch 이후 조건 충족). 최고 체크포인트는 epoch 5로 테스트.
- **lr 선택: 1e-5**(검증 GLEU 최고). 탐색 범위 아래쪽 끝이 선택됨. 3e-5·5e-5는 10 epoch 끝까지 오르는 중이었음. 출력 이탈 0, 테스트 속도 약 58–60문장/초(A100×2, 생성 batch 64, 빔 4).
- 재개 작업 시간: 9 epoch + 테스트 49분(916205) → epoch당 약 5.1분(첫 epoch 7.5분보다 짧음: 검증 생성 시간 감소). --time 추정(01:30 / 06:00 / 08:00)은 여유가 충분해 그대로 둠.
- 비교(korean_learner seed 0, 검증 / 테스트 GLEU): kanana-2.1b 58.92 / 58.33, pko-t5-large 59.91 / 59.10, pko-t5-base 58.57 / 57.41, KoBART(gec2) 54.50 / 52.17. P·R도 kanana ≈ pko-t5-large(과교정 징후 없음).
- itcerdo 단계 4(테스트 GLEU / F0.5): native — KoBART 78.76 / 87.02, pko-t5-base 82.54 / 89.30, mBART 79.72 / 87.23, pko-t5-large 82.84 / 89.42. lang8 — 35.58 / 43.49, 37.89 / 44.83, 35.91 / 45.84, 39.89 / 48.12. union — 44.29 / 52.53, 47.22 / 55.23, 44.66 / 50.85, pko-t5-large 진행 중(13:57 시작).

### 15:05 · Neuron kanana native(lr 1e-5, seed 0) 결과 분석 [gsm, itcerdo]

- 사용자 push `28fd336`. 작업 916288 (gpu39, A100×2, 커밋 99d35b3), 14:16:44–14:50:30(34분).
- 검증 GLEU epoch 0–9: 79.41 83.31 83.59 83.23 83.96 84.10 **84.40** 84.27 84.27 84.27 → 최고 epoch 6, 마지막 epoch에서 조기 종료 조건(wait 3) 충족(사실상 10 epoch 모두 학습). 학습 loss 0.238 → 0.000(epoch 7부터), 검증 loss는 epoch 1 0.106이 최저이고 이후 0.155까지 상승(과적합), epoch 7–9 출력 동일.
- **테스트 GLEU 83.02, P 92.51, R 79.40, F0.5 89.56**, 40.9문장/초, 출력 이탈 0, GPU 최대 54.3GB.
- native 비교(seed 0, 검증 / 테스트 GLEU / F0.5): KoBART(gec2) 80.27 / 78.76 / 87.02, mBART-50 81.08 / 79.72 / 87.23, pko-t5-base 84.09 / 82.54 / 89.30, pko-t5-large 83.80 / 82.84 / 89.42, kanana-2.1b 84.40 / 83.02 / 89.56. 상위 3개 차이 0.5점 이내, 재현 KoBART native seed 범위 2.46 → seed 0 한 번으로는 구분 불가.
- 분할 간 겹침 확인(itcerdo, 필터 후 쌍 기준, 테스트 원문이 학습 원문과 같은 비율 / (원문, 정답) 쌍이 같은 비율): korean_learner 0.84% / 0.35%, **native 0.11% / 0.00%**, **lang8 27.80% / 0.61%**, union 19.75% / 0.49%. native 고득점은 누설 때문이 아님. lang8은 같은 원문이 학습셋에 다른 정답으로 들어 있는 경우가 많음(한 원문에 여러 교정) → 정답 모호성으로 점수 상한이 낮은 구조적 요인.
- 시간: epoch당 3.2분(korean_learner 재개 epoch 5.1분 × 문장 수 비 0.62와 일치, 선형 확인) → lang8 `--time=05:00:00`, union `--time=07:00:00`로 확정(계산식은 sbatch 주석). 커밋 `09d484e`, push.
- 참고: 테스트 속도는 run마다 달라짐(korean_learner 58–60, native 41문장/초). 추론 속도 보고(원칙 6)는 단계 6에서 같은 조건으로 따로 측정하는 편이 맞음.

### 17:49 · plan2 단계 4 완료: 5개 모델 × 4개 데이터셋 (seed 0) [gsm, itcerdo]

- Neuron: 사용자 push `8ca319a`. kanana lr 1e-5 seed 0 — lang8(작업 916319): 검증 GLEU 37.52 **38.65** 37.88 37.39 37.28 → epoch 1 최고, 5 epoch 후 조기 종료, 테스트 GLEU 39.42 · P 52.58 R 32.25 F0.5 46.69. union(916320): 47.46 48.27 **48.37** 48.29 48.21 → epoch 2 최고, 5 epoch 후 조기 종료, 테스트 48.39 · 59.30 · 40.42 · 54.24. epoch당 lang8 약 16.5분, union 약 23.5분. 출력 이탈 0, GPU 최대 73GB.
- itcerdo: 17:05:57 QUEUE DONE(stage4_seed0), pko-t5-large union 검증 48.99(epoch 4) / 테스트 49.31 · 61.97 · 40.11 · 55.88. tmux 자동 종료 확인. itcerdo `git pull`로 Neuron 결과까지 받아 표 작성.

| 테스트 GLEU / F0.5 | korean_learner | native | lang8 | union |
| --- | --- | --- | --- | --- |
| KoBART(gec2) 124M | 52.17 / 52.38 | 78.76 / 87.02 | 35.58 / 43.49 | 44.29 / 52.53 |
| mBART-50 611M | 52.51 / 53.49 | 79.72 / 87.23 | 35.91 / 45.84 | 44.66 / 50.85 |
| pko-t5-base 276M | 57.41 / 58.45 | 82.54 / 89.30 | 37.89 / 44.83 | 47.22 / 55.23 |
| pko-t5-large 821M | **59.10** / 59.26 | 82.84 / 89.42 | **39.89** / **48.12** | **49.31** / **55.88** |
| kanana-2.1b 2.09B | 58.33 / **59.50** | **83.02** / **89.56** | 39.42 / 46.69 | 48.39 / 54.24 |

- pko-t5 base→large 테스트 GLEU 차이: korean_learner +1.69, native +0.30, lang8 +2.00, union +2.09 (학습 데이터가 크고 native처럼 포화되지 않은 곳에서 크기 효과가 큼).
- kanana − pko-t5-large: −0.77, +0.18, −0.47, −0.92. 재현율은 4개 데이터셋 모두 kanana가 최고, 정밀도는 lang8(−3.7)·union(−2.7)에서 낮음.
- 재현 KoBART(논문 recipe, 3-seed 평균) 대비 KoBART(gec2) seed 0: +6.5 / +10.4 / +5.9 / +9.5.
- 비교 시 주의: seed 0 한 번. 재현 KoBART의 seed 간 테스트 GLEU 범위(korean_learner 0.71, native 2.46, lang8 0.60, union 1.75)보다 작은 차이는 판단 보류. 트랙별 규칙 차이(seq2seq 10 epoch 고정 / LLM 조기 종료) 표기 필요.

### 17:50–17:56 · plan2 단계 5 시작: 모든 모델에 seed 1·2와 시나리오 2 [gsm, itcerdo]

- **결정(사용자)**: 소요 시간이 예상보다 짧아 단계 5를 5개 모델 모두에 적용(`plan2.md` §5 단계 5, §7-3 갱신).
- 시나리오 2 경로 스모크(itcerdo, KoBART, native 256줄, union seed 0 `ckpt/best`에서 시작, lr 1.6667e-5): 첫 step loss 0.028(union 학습에 native가 포함되어 있으므로 가중치 로딩 확인), 정상 종료. 스모크 출력 삭제.
- 시작 lr = union lr / 3 (4자리 반올림): KoBART 1.6667e-5, pko-t5-base 1.6667e-4, mBART 1e-5, pko-t5-large 3.3333e-5, kanana 3.3333e-6. run_id `s2_lr<union lr>_seed<N>`.
- itcerdo 대기열 `gec2/queues/stage5.txt` 68 run: 시나리오 2 seed 0(12) → seed 1 시나리오 1(16, union 먼저) → 시나리오 2 seed 1(12) → seed 2 시나리오 1(16) → 시나리오 2 seed 2(12). 예상 약 40시간.
- Neuron: `llm_train.sh`에 `S2=<union lr>`(같은 seed union `ckpt/best`에서 시작) 추가, 끝난 run은 재개용 `last.ckpt` 자동 삭제(`ckpt/best`만 유지). `NEURON.md`에 시나리오 2 명령 추가.
- 커밋 `c08800a` "Stage 5: seeds 1-2 and scenario 2 for all models", push, itcerdo pull.

```bash
tmux new-session -d -s gec2 "bash gec2/run_queue.sh gec2/queues/stage5.txt"   # 17:55:17 QUEUE START
```

### 18:04 · Neuron 다른 파티션 사용 준비 (amd_a100_4, amd_h200nv_8) [gsm]

- 사용자 showque(2026-09-27 저녁): amd_a100nv_8 `Run/SubLimit 1/0, CanSubmit 0`(추가 제출 불가; 대기·실행 중 gec2 작업 916459–916461 + blt 1개). amd_a100_4 CanSubmit 2(gpu45, A100 4장, GPU당 코어 16), amd_h200nv_8 CanSubmit 2, amd_h100_2 CanSubmit 2(계획상 A100·H200만 쓰므로 제외), gh200_1 기간 만료(250101–251231).
- amd_a100_4의 GPU 메모리(40/80GB)는 미확인. 40GB면 kanana 전체 파인튜닝 불가(가중치·그래디언트·Adam 상태만 약 33GB, 기본 설정 최대 학습 57.5GB·생성 51.8GB).
- 수정: `llm_train.sh`가 시작 때 GPU 메모리를 확인해 75,000MiB 미만이면 중단. `a100x2_train.sbatch`는 `-p amd_a100_4`로 파티션만 바꿔 제출(자원 지시문 동일, 코어 8은 한도 16 안). `h200x1_train.sbatch`의 `--time`은 H200 실측 전이라 A100×2 실측값 × 1.5(korean_learner 03:00, native 01:30, lang8 07:30, union 10:30). 같은 run의 재개는 시작한 GPU 구성으로. `NEURON.md` 파티션 표 갱신.
- 커밋 `d3135d0`, push.

### 20:44 · itcerdo 대기열 독립 실행 확인 [itcerdo]

- **작업**: 사용자가 약 10시간 뒤 Claude Code 연결을 끊을 예정 → gsm 세션과 무관하게 itcerdo 대기열이 도는지 확인.

```bash
tmux ls; ps -o pid,ppid,sid,etime,cmd -u $USER | grep -E "run_queue|gec2.train|tmux"
```

- 결과: tmux 세션 `gec2` 실행 중. tmux 서버 프로세스의 부모가 PID 1(데몬화)이고, 대기열(`run_queue.sh stage5.txt`)과 학습 프로세스가 그 아래에서 2시간 48분째 실행 중. 이 대화의 ssh 명령은 매번 접속했다 끊기는데도 계속 돌았음 → gsm 연결과 무관.
- stage5 진행: 68 run 중 10개 완료, FAIL 0. 현재 mBART lang8 시나리오 2 seed 0(GPU 96%, 27.5GB). 방금 끝난 pko-t5-base lang8 시나리오 2 seed 0: 검증 37.01(epoch 0) / 테스트 GLEU 37.97 · P 58.16 R 35.71 F0.5 51.67. 디스크 여유 407GB.
- 남은 58 run 예상 약 37시간.

## 2026-09-28

### · Neuron kanana 단계 5 중간 결과 확인 (시각 기록 불가: Bash 사용 불가) [gsm]

- Claude Code auto 모드의 명령 판정 서비스가 응답 없이 실패("classifier gave no verdict")해 Bash를 쓸 수 없었음(7회). 사용자가 `!`로 직접 `git pull` 실행 → `55251af`("260928 kanana finetuning ongoing"). 결과 파일은 Read로만 확인했고, itcerdo 진행 상황과 slurm 로그 목록은 확인하지 못함.
- kanana 시나리오 1 (lr 1e-5) 테스트 GLEU / F0.5, [GPU]:

| 데이터 | seed 0 | seed 1 | seed 2 | 평균 GLEU (범위) |
| --- | --- | --- | --- | --- |
| korean_learner | 58.33 / 59.50 [A100×2] | 58.64 / 59.52 [A100×2] | 58.59 / 58.36 [H200] | 58.52 (0.31) |
| native | 83.02 / 89.56 | 82.93 / 89.62 | 82.95 / 89.38 [A100×2] | 82.97 (0.09) |
| lang8 | 39.42 / 46.69 | 38.77 / 46.34 | 39.14 / 46.46 [H200] | 39.11 (0.65) |
| union | 48.39 / 54.24 | 48.75 / 52.37 | 48.85 / 54.59 [H200] | 48.66 (0.46) |

- kanana 시나리오 2 (union → 개별, lr 3.3333e-6): korean_learner seed 0 59.07 / 59.26 [H200], seed 1 59.29 / 60.21 [A100×2], seed 2 결과 없음. lang8 seed 0 39.21 / 52.22 [A100×2], seed 1 40.07 / 50.81 [H200], seed 2 38.72 / 51.88 [H200]. native seed 0·1·2 결과 없음(run 폴더 자체가 GitHub에 없음).
    - lang8 시나리오 2는 세 seed 모두 epoch 0이 최고(union 학습에 lang8 학습셋이 포함되어 있음), P·R이 모두 시나리오 1보다 높아 F0.5 약 +5(46.5 → 51.6).
- **확인 필요**: native 시나리오 2 seed 0은 2차 묶음(amd_a100_4로 제출 권장)이었는데 결과 폴더가 없음. amd_a100_4 GPU가 80GB 미만이면 `llm_train.sh`의 메모리 검사에서 python 실행 전에 중단되어 run 폴더가 생기지 않음 → slurm 로그 확인 필요.
- H200 실측: epoch당 korean_learner 약 3.8분(A100×2 5.1분), lang8 시나리오 2 약 12.6분(A100×2 약 16.5분), union 약 18.3분(A100×2 약 23.5분) → H200 1장이 A100 2장보다 약 25% 빠름. H200 `--time`은 A100×2 값으로 줄여도 충분(Bash 복구 후 스크립트 주석 갱신 예정).
- 사용자가 `!`로 실행한 요약(push_repo): 위와 동일하게 17개 중 13개 완료, native 시나리오 2 seed 0·1·2와 korean_learner 시나리오 2 seed 2는 run 폴더 없음(진행 중인 run 없음 → 미시작 또는 python 실행 전 중단). 사용자에게 `sacct`와 `grep "오류" logs/slurm/*.out`으로 확인 요청.

### 21:29 기준 · itcerdo 단계 5 진행 확인 (사용자가 `!`로 실행) [itcerdo]

```bash
ssh itcerdo 'grep -c DONE .../logs/gec2_runs.log; grep FAIL ... | tail -3; tail -2 ...'
```

- FAIL은 9/26 lr 탐색 때의 pko-t5-large 3건(OOM, 재실행 완료)뿐 → 단계 5 실패 0.
- 마지막 완료: 21:28:54 mBART native seed 2 검증 81.15(epoch 6) / 테스트 GLEU 80.08 · P 90.61 R 75.87 F0.5 87.22. 진행 중: pko-t5-large native seed 2(21:28:55 시작).
- 대기열 순서 기준 68 run 중 51 완료, 1 진행, 16 남음(seed 2 lang8 4, 시나리오 2 seed 2 12). 예상 종료 9/29 06–07시.

## 2026-09-29

### 06:05–06:09 · 단계 5 결과 확인, 3-seed 집계 [gsm, itcerdo]

- Bash 복구됨(06:05). 사용자 push `8bc8526`("260929 kanana finetuning done").
- kanana 남은 4개 모두 완료(조기 종료): native 시나리오 2 seed 0(작업 917385, amd_a100nv_8 A100×2) 테스트 GLEU 82.30, seed 1(917386, A100×2) 82.31, seed 2(917387, A100×2) 82.88, korean_learner 시나리오 2 seed 2(917390, **amd_h200nv_8 H200**) 58.80. push된 slurm 로그 29개에 "오류" 줄 없음 → native 시나리오 2 seed 0이 이전에 없던 원인은 확인 못 함(재제출로 해결). **kanana 17개 전부 완료**, 출력 이탈 0.
- itcerdo: stage5 68 run 중 67 완료, FAIL 0. 마지막 pko-t5-large lang8 시나리오 2 seed 2(04:44:55 시작, 07시 전후 종료 예상). 디스크 여유 307GB.
- `gec2/summarize.py` 작성(모델 × 데이터셋 × 시나리오, 테스트 GLEU/P/R/F0.5 3-seed 평균 ± 범위/2, 재현 KoBART·논문 행 참고). H200 `--time` 주석을 실측으로 갱신(H200 epoch가 A100×2보다 약 25% 빠름 → A100×2 값 사용). 커밋 `888b5e4`, push, itcerdo pull.

```bash
python3 -m gec2.summarize --json logs/gec2_summary_interim.json > logs/gec2_summary_interim.md   # 누락: pko-t5-large s2 lang8 1개
```

- 시나리오 1 테스트 GLEU (3-seed 평균 ± 범위/2): KoBART(gec2) 52.46±0.33 / 78.69±0.09 / 35.65±0.13 / 44.40±0.14, pko-t5-base 57.45±0.25 / 82.36±0.19 / 37.95±0.11 / 47.17±0.05, mBART-50 52.62±0.13 / 79.88±0.18 / 36.03±0.10 / 44.75±0.09, pko-t5-large **59.19±0.17** / 82.70±0.12 / **39.84±0.15** / **49.16±0.12**, kanana **58.52±0.15** / **82.96±0.04** / 39.11±0.32 / 48.66±0.23 (korean_learner / native / lang8 / union).
- 시나리오 2 테스트 GLEU: KoBART 53.03 / 76.90 / 35.80, pko-t5-base 57.45 / 80.50 / 37.61, mBART 53.80 / 77.38 / 36.10, pko-t5-large 59.41 / 81.78 / 39.71(2 seed), kanana 59.05 / 82.50 / 39.33 (korean_learner / native / lang8).
- 요점: seed 간 흔들림이 ±0.04–0.33으로 작아(재현 KoBART의 범위 0.6–2.5보다 훨씬 작음) 모델 간 0.5점 이상 차이는 안정적. pko-t5-large가 korean_learner·lang8·union 1위, kanana가 native 1위. pko-t5 base→large +1.7 / +0.3 / +1.9 / +2.0. 시나리오 2는 native에서 모든 모델 GLEU 하락(−0.5~−2.5), lang8에서 F0.5 상승(+2~+5.4).
- 표 전체: itcerdo `logs/gec2_summary_interim.md`.

### 10:24 · kanana 체크포인트 복사 확인, 단계 5 최종 집계 [itcerdo]

- itcerdo: 06:55:12 마지막 run(pko-t5-large lang8 시나리오 2 seed 2, 검증 39.07 epoch 0 / 테스트 GLEU 40.04 · P 56.70 R 34.32 F0.5 50.16) 완료, 06:55:13 **QUEUE DONE stage5 — 68/68, FAIL 0**. tmux 세션 자동 종료 확인. 디스크 여유 276GB.
- 사용자가 kanana 시나리오 1 seed 0 `ckpt/best` 4개(korean_learner·native·lang8·union)를 itcerdo `neuron_outputs/kanana-1.5-2.1b/<data>/lr1e-5_seed0/ckpt/best/`로 복사(각 7.8GB, safetensors 2개 + 설정·토크나이저).
- 무결성 확인(스크래치 `ckpt_check.py`를 itcerdo `logs/`에서 실행 후 삭제): 4개 모두 로드 정상, `best_info.json`의 epoch·검증 GLEU가 `test/scores.json`과 일치. 테스트 짧은 문장 64개를 다시 생성해 Neuron 출력과 비교 → korean_learner 63/64, native 64/64, lang8 62/64, union 62/64 동일. 다른 문장은 빔 후보가 거의 같은 경우(예: `안녕합시니까.` → Neuron `안녕합시니까.` / itcerdo `안녕합시다.`)로, GPU(A100 vs RTX 5090)·배치 구성 차이에 의한 것. 결과: `logs/gec2_kanana_ckpt_copy_check.txt`.
- 최종 집계: `python3 -m gec2.summarize --json logs/gec2_summary_final.json > logs/gec2_summary_final.md` → 누락 seed 없음. 중간 집계와 달라진 칸은 pko-t5-large 시나리오 2 lang8(3 seed): GLEU 39.82 ±0.17, F0.5 50.89 ±0.57.

### 07:20–11:20 · plan2 단계 6: 추론 속도 측정 [gsm, itcerdo]

- **작업**: 원칙 6의 추론 속도(논문 Table 6 방식). 논문(`papers/korean_gec.pdf`) 확인: "Generation time (sentence/second) is measured by dividing the total number of sentences by total amount of generation time taken", 모델당 한 값(KoBART 38.25, V100), batch 크기는 적혀 있지 않음.
- `gec2/speed.py` 작성(커밋 `39cd29d`, push, itcerdo pull): RTX 5090 1장(다른 작업 없음), 5개 모델 같은 설정(빔 4, 샘플링 없음, 생성 batch 64, 최대 생성 길이 규칙은 학습 평가와 동일, fp32 가중치 + bf16 autocast, 길이순 batch, 예열 1 batch 제외), 각 데이터셋의 시나리오 1 seed 0 `ckpt/best`, 데이터셋별과 4개 테스트셋 합계(46,666문장).
- 1차(tmux `gec2speed`): seq2seq 4개 완료, **kanana는 batch 64에서 OOM**(32GB 중 8.35GB가 조각난 채 미할당 → 단편화). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`로 kanana 재측정 성공(최대 29.2GB). 조건을 맞추려고 seq2seq도 같은 설정으로 재측정(1차와 차이 1.5% 이내). 로그 `logs/gec2_speed_{seq2seq_b64,seq2seq,kanana}.log`, 합친 결과 `logs/gec2_speed_final.json`.
- 함께 계산한 GLEU가 기록된 seed 0 테스트 점수와 일치(seq2seq는 소수 둘째 자리까지, pko-t5-large korean_learner 59.14 vs 59.10, kanana는 A100과 GPU가 달라 ±0.05) → 측정 설정이 학습 평가와 같음.

| 문장/초 (RTX 5090, batch 64, 빔 4) | korean_learner | native | lang8 | union | 합계(논문 방식) | 최대 메모리 |
| --- | --- | --- | --- | --- | --- | --- |
| KoBART(gec2) 124M | 1138.0 | 1105.4 | 1192.1 | 1183.5 | **1177.5** | 2.0GB |
| pko-t5-base 276M | 384.3 | 393.4 | 450.6 | 434.4 | **432.2** | 4.3GB |
| mBART-50 611M | 304.9 | 306.0 | 347.0 | 336.6 | **335.0** | 6.8GB |
| pko-t5-large 821M | 183.9 | 195.9 | 224.5 | 213.9 | **213.2** | 10.7GB |
| kanana-1.5-2.1b 2.09B | 41.6 | 30.6 | 58.6 | 58.4 | **53.7** | 29.2GB |

- kanana의 데이터셋별 편차 확인(스크래치 `speed_rep.py`, 실행 후 삭제, 결과 `logs/gec2_speed_kanana_repeat.txt`): native를 같은 프로세스에서 두 번 재도 30.7 / 30.7(예열 문제 아님). 가장 긴 입력 batch 하나가 약 20초(중앙값 1.2초)를 차지. lang8도 앞 2,634문장만 재면 38.9로 떨어짐 → 긴 batch 한두 개의 고정 비용이 작은 테스트셋에서 크게 보이는 것. 출력 길이 이상은 없음(출력/입력 토큰 비 최대 1.25–3.93, 평균 출력 길이 ≈ 입력).
- 원인 추정: LLM의 최대 생성 길이가 프롬프트(약 30토큰)를 포함한 입력 길이 × 2 + 10이라 seq2seq(원문 길이 기준)보다 여유가 커서, 빔이 끝나지 않는 문장이 있는 batch는 상한까지 생성함. 정확도 결과에는 같은 규칙이 쓰였으므로 속도도 같은 규칙으로 보고하고, 이 점을 주석으로 남김.
- 논문 수치(KoBART 38.25문장/초, V100)와는 GPU·batch가 달라 절대값 비교 불가. 모델 간 상대 비교만 유효: KoBART(gec2) 대비 pko-t5-base 0.37배, mBART 0.28배, pko-t5-large 0.18배, kanana 0.046배.

### 08:05–13:28 · plan2 단계 6: 오류 유형별 분석, 논문과 비교 [gsm, itcerdo]

- 추론 속도 해석 정리(사용자와 합의): 표의 절대 배율은 RTX 5090·batch 64·빔 4 환경의 값이며 GPU·batch에 따라 달라짐. 로드 시간은 측정에서 제외했고 5개 모델 모두 같은 batch 64(작은 모델이 더 큰 batch를 쓸 여지는 반영 안 됨). 순서와 대략적 크기만 해석. 참고: kanana korean_learner 테스트 속도 A100×2 약 58, H200 약 64, RTX 5090 42문장/초.
- 논문 Table 7 / D.4의 유형별 점수는 계산 방식이 적혀 있지 않음(저자 코드에도 없음). D.4에서 편집이 없는 유형(Kor-Learner WO 0개)이 P/R/F0.5 100·GLEU 0(m2scorer에 빈 집합을 넣은 값)인 점과 유형별 개수가 편집 수라는 점으로 보아, "그 유형의 정답 편집이 있는 테스트 문장만 골라 GLEU·M² 계산"으로 재구성. 정답 M²의 유형별 편집 수가 논문 Table D.3 테스트 개수와 정확히 일치(예: korean_learner INS 517, DEL 205, SPELL 879, PUNCT 3, WO 0).
- `gec2/errtype.py` 작성(커밋 `15a764c`). union_test.m2는 데이터셋 경계 2곳에 빈 줄이 없어 빈 줄 기준 분할 시 블록 23,331개(문장 23,333) → `S ` 줄 기준으로 분할하도록 수정(커밋, push, itcerdo pull; m2scorer도 `S ` 줄 기준이라 기존 채점에는 영향 없음).
- 검증(`--validate`, 재현 KoBART union 3 seed): 유형별 GLEU가 논문 KoBART Kor-Union 행과 평균 0.84(최대 2.47), P 0.89/4.27, R 1.32/2.90, F0.5 1.26/2.80 차이. 전체(TOTAL) 차이가 +1.05(34.75 vs 33.70)라 유형별 차이도 같은 크기이고, 유형 간 순서(SPELL·WS·SHORT 높고 DEL·WO 낮음)가 같음 → 재구성 방식 타당.
- 전체 계산: 5개 모델 × 4개 데이터셋 × 3 seed(시나리오 1), 6분 25초. 결과 itcerdo `logs/gec2_errtype.md`, `logs/gec2_errtype/errtype.json`.
    - union GLEU, KoBART(gec2) 대비 pko-t5-large: SPELL +6.6, WS +5.9, PART +5.4, NOUN +5.1, END +4.6 / WO +0.2, PUNCT +2.2, DEL +2.8, INS +2.9 → 큰 모델의 이득은 철자·형태(조사·어미·명사)·띄어쓰기에 집중, 삽입·삭제·어순·문장부호는 작음.
    - kanana − pko-t5-large(union GLEU): WS +1.6, WO +2.2, SHORT +0.7 / PART −1.9, END −1.8, MOD −2.2, ADJ −2.7, NOUN −1.3 → LLM은 띄어쓰기·어순이 강하고, 조사·어미 등 형태 교정은 한국어 전용 T5가 강함.
    - 재현 KoBART → KoBART(gec2)(union GLEU): WS +18.5로 가장 큼, SHORT +11.9, WO +12.5, PART +10.0 등 전 유형 상승.
- 시나리오 2 − 시나리오 1(3-seed 평균) 논문과 비교:
    - 논문·재현 KoBART: 정밀도 크게 상승(korean_learner +10.2/+8.8, native +10.1/+5.9), 재현율 하락, GLEU 하락(native −7.5/−7.4).
    - gec2 5개 모델: korean_learner ΔGLEU 0~+1.2·ΔP +0.2~+2.8·ΔR −0.1~+1.8, native ΔGLEU −0.5~−2.5·ΔP −1.3~+0.8·ΔR −0.5~−2.8, lang8 ΔGLEU ≈0·ΔP +0.8~+4.3·ΔR +3.2~+6.4(ΔF0.5 +2.0~+5.4).
    - 해석: 논문의 "시나리오 2에서 정밀도↑·재현율↓" 경향은 재현 KoBART(원본 recipe)에서는 재현되지만 표준 학습(gec2)에서는 거의 사라짐 → 그 경향의 상당 부분은 원본 학습 방식의 특성. native에서 GLEU가 내려가는 방향만 공통.

### 08:40–13:31 · plan2 단계 7: 결과 문서, 저장소 정리 [gsm, itcerdo]

- 추가 확인: kanana native seed 1(시나리오 1)이 두 번 실행됨(작업 916461 2026-09-27, 916761 2026-09-28). 두 번째 기록이 같은 `epochs.jsonl`에 덧붙어 14줄. 두 실행의 epoch별 검증 GLEU가 완전히 같음(79.26 … 84.63, 7 epoch 후 조기 종료) → 결과 영향 없음, 같은 seed·GPU 구성의 결정적 재현 확인. 학습 시간 집계는 epoch별 마지막 기록만 사용.
- 학습 시간(3-seed 평균, 학습 + 매 epoch 검증): KoBART(gec2) 3/2/9/13분, pko-t5-base 9/5/31/45분, mBART 12/7/43/62분, pko-t5-large 35/21/127/184분(RTX 5090), kanana 27/27/78/111분(A100×2·H200, 조기 종료 5–10 epoch). 파라미터(전체 / 임베딩 제외): KoBART 123.9M/100.8M, pko-t5-base 275.6M/198.2M, mBART 610.9M/354.8M, pko-t5-large 820.5M/717.4M, kanana 2,087.0M/1,857.1M.
- `result_gec2.md` 작성(요약, 설정, 시나리오 1·2 3-seed 표, 크기·계열 비교, 논문 비교(시나리오 2 차이, 유형별 점수 재구성·검증), 추론 속도·학습 시간, 한계, 파일 위치). 수치 대조 중 요약의 "재현 KoBART 대비 6.8–10.3점"을 "6.0–10.3점"(lang8 +6.0)으로 정정.
- `plan2.md` §5 단계 6 설명 갱신(속도는 환경 조건부 수치, 유형별 점수는 재구성·검증).
- 저장소: `docs/`에 plan2.md·result_gec2.md·log.md 복사(비밀 패턴 검사 통과), README에 결과 요약 표·문서 링크·gec2 구성 추가. 커밋 `be3e2ef` "Add larger-model comparison results and docs (plan2 steps 6-7)", push, itcerdo pull.

### 13:59 · 질문 검토: 재현 KoBART가 논문보다 높은 이유 [gsm]

- 근거 재확인(기존 기록 + 논문 `papers/korean_gec.pdf`):
    - 평가 동일: 저자 체크포인트 run 122(korean_learner seed 0)를 우리 파이프라인으로 채점 → 검증 GLEU 46.91 = 저자 기록 46.9129. self-GLEU가 논문과 소수점 둘째 자리까지 일치.
    - 데이터 크기 동일: 논문 Table 1 문장 쌍 Kor-Learner 28,426 / Kor-Native 17,559 / Kor-Lang8 109,559, 우리 필터 후 train+val+test 28,427 / 17,560 / 109,560(각 1쌍 차이).
    - 같은 seed 비교: 저자 run 122 테스트 GLEU 45.30 vs 재현 seed 0 45.38(+0.08). 논문 평균 45.06과의 차이(+0.62)는 우리 seed 1·2(45.59, 46.08)가 저자 나머지 seed보다 높은 데서 옴.
    - 검증 3-seed 평균 차이(재현 − 논문 Table D.1): korean_learner 0.00, native +0.92, lang8 +0.31, union +0.43 / 테스트 차이: +0.62, +1.16, +1.17, +1.05 → 테스트 차이가 검증 차이보다 일관되게 큼.
    - seed 편차 대비: 3-seed 평균 차이의 표준편차 ≈ √(2/3)·σ_seed(σ ≈ seed 범위/2)로 보면 korean_learner 약 1.9σ, native 약 1.1σ, union 약 1.4σ, lang8 약 4σ → lang8만 seed 운으로 설명하기 어려움. 7개 조합 모두 +방향.
- 결론(잠정): 평가·데이터는 같고 차이는 학습 과정. 대부분은 seed 표본 차이(난수 흐름·수치 연산이 torch 1.7/V100과 달라 같은 seed라도 같은 run이 아님)로 설명되나, lang8 테스트와 전 조합 + 방향은 확인하지 못한 체계적 요인이 있을 수 있음. 확인 방법: 저자 spreadsheet의 다른 체크포인트(특히 lang8·union)를 받아 우리 파이프라인으로 채점.

### 14:18 · native 정밀도 차이 조사: 저자 native 체크포인트 검증 [gsm, itcerdo]

- **작업**: 사용자 지적 — 재현 native(시나리오 1) 테스트 정밀도가 논문보다 +5.0(80.34 vs 75.34), seed 운으로 보기 어렵다.
- 재현 run의 **검증** P/R/F0.5(최고 epoch, 3-seed 평균) vs 논문 Table D.1: korean_learner 44.01/26.54/38.89 vs 43.95/26.35/38.76(일치), **native 80.18/59.34/74.92 vs 75.07/56.81/70.53(P +5.1)**, lang8 39.05/13.11/27.96 vs 37.69/12.64/26.96, 시나리오 2 korean_learner 52.21/24.12/42.31 vs 51.94/23.55/41.83, native 85.62/49.95/74.86 vs 83.95/48.55/73.25, lang8 37.52/12.80/27.04 vs 38.53/12.72/27.40 → 차이가 검증에서 이미 존재(모델 차이).
- 저자 spreadsheet 재확인(`export?format=csv`): native 시나리오 1 검증 GLEU 68.22/70.62/69.27·P 74.94/74.12/76.14(평균 69.37/75.07 = 논문 D.1), 테스트 GLEU 65.89/66.12/63.21·P 75.12/74.51/69.89 → **테스트 평균 65.07/73.17로 논문 Table 6(67.24/75.34)과 다름**(저자 기록끼리 불일치).
- 저자 native seed 0(run 123) 다운로드(`author_ckpt/123/`, yaml 1.5KB·ckpt 1.49GB, spreadsheet 링크 id 1dvw89…·1emzbe…). yaml은 run 122(korean_learner)와 데이터 경로 외 동일(batch 64, lr 3e-5, dropout 0.1, precision 32, max_len 128). 체크포인트: best val_gleu 68.2243(epoch 5), PL 1.1.
- run 122와 같은 방식으로 변환(`author_123_converted.ckpt`, lm_head = shared) 후 레거시 파이프라인으로 채점:

```bash
python3 run.py --recipe legacy --data native --run_id author123_val --seed 0 --model_ckpt_path ../author_ckpt/123/author_123_converted.ckpt --batch_size 64 --max_seq_len 128
python3 run.py --recipe legacy --eval_test --data native --run_id author123_test --seed 0 --model_ckpt_path ../author_ckpt/123/author_123_converted.ckpt --batch_size 64 --max_seq_len 128
```

- 결과: 검증 GLEU 68.22 · P 74.94 R 55.70 F0.5 70.10(**저자 기록과 완전 일치** → native도 평가 동일), 테스트 GLEU 66.55 · P 75.41 R 55.66 F0.5 70.42(저자 spreadsheet 테스트 65.89/75.12/54.67/69.89와 약간 다름).
- 학습 데이터 동일성 확인(스크래치 `memo_check.py`, 실행 후 삭제, 결과 `logs/author_memo_check.txt`): 저자 모델의 우리 학습셋 teacher-forcing loss vs 검증셋 loss(앞 4,000쌍) — run 122 0.107 / 0.665, **run 123 0.002 / 0.284**(우리 학습셋을 외운 상태 → 같은 학습 데이터), 재현 seed 0 korean_learner 0.109 / 0.665, native 0.001 / 0.291.
- 재현 native epoch별 검증 GLEU/P(`logs/native_repro_val_epoch_prec.txt`): GLEU는 epoch 3 이후 69–71로 평탄한데 P는 한 run 안에서 72.7–83.4로 크게 흔들리고 후반 epoch로 갈수록 오름. 같은 epoch 5에서 재현 P 78.5 / 78.4 / 81.4 vs 저자 seed 0 74.94.
- 결론: 데이터·설정·평가가 같고 저자 native 모델만 정밀도가 낮음. native 정밀도는 GLEU가 평탄한 구간에서도 epoch마다 ±5 흔들리는 민감한 지표라 일부는 우연이지만, 같은 epoch에서도 재현 쪽이 3–6 높아 학습 과정의 체계적 차이가 남음(저자 run의 epoch별 기록이 없어 원인 특정 불가). 후보: 수치 환경(V100·torch 1.7·transformers 4.0 eager attention vs RTX 5090·torch 2.11·transformers 4.44 SDPA). 또 논문 native 테스트 수치 자체가 저자 기록과 달라, 논문 대비 차이의 일부는 논문 표의 문제.

### 14:10–14:52 · native 정밀도: eager attention으로 재학습 [gsm, itcerdo]

- **작업**: 사용자 결정 — attention 구현 차이(transformers 4.44 기본 SDPA vs 논문 당시 4.0 eager)가 native 정밀도 차이의 원인인지 확인.
- 확인: 재현 학습은 `BartSdpaAttention`(config `_attn_implementation=sdpa`)으로 돌았음. `run.py`에 `--attn_implementation {eager,sdpa}` 추가(기본값은 기존과 같음, 로그에 실제 구현 출력). 커밋 `40986a7`, push, itcerdo pull.
- 실행: `logs/run_eager_native.sh`(tools/run_one.sh와 같은 순서에 `--attn_implementation eager`만 추가), tmux `eager`, run_id `eager_dropout_seed{0,1,2}`, native 시나리오 1, lr 3e-5, batch 64, 10 epoch, legacy recipe. 로그 `logs/native_eager_dropout_seed*_{train,test,m2}.log`, 진행 `logs/runs.log`. 로그에서 `attention implementation: eager (BartAttention)` 확인. 14:21–14:50, run당 약 9.5분.
- 결과(`logs/native_eager_vs_sdpa.txt`):

| native 시나리오 1 | 검증 GLEU / P | 테스트 GLEU / P / R / F0.5 | 검증 P epoch 5 |
| --- | --- | --- | --- |
| 재현 SDPA (3-seed) | 70.29 / 80.18 | 68.40 / 80.34 / 58.31 / 74.69 | 79.42 |
| 재현 **eager** (3-seed) | 70.53 / 79.17 | 68.79 / 79.24 / 58.41 / 73.96 | 78.50 |
| 저자 (spreadsheet, 3-seed) | 69.37 / 75.07 | 65.07 / 73.17 / 54.24 / 68.39 | (seed 0 best=epoch 5: 74.94) |

- eager seed별 테스트 P 77.94 / 79.07 / 80.70(SDPA 80.73 / 78.19 / 82.10). eager가 정밀도를 약 1점 낮추지만 seed 간 범위(약 3점) 안이고, 저자와의 차이 약 4–5점 중 대부분은 그대로 → **attention 구현은 주원인이 아님**. GLEU는 차이 없음.
- 남는 후보: GPU·torch·cuBLAS 수치 경로(V100·torch 1.7 vs RTX 5090·torch 2.11), PyTorch Lightning 1.1 학습 루프의 확인되지 않은 동작 차이. RTX 5090에서는 torch 1.7을 쓸 수 없어 GPU로 직접 확인 불가.
