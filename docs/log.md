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
