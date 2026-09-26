# KoBART GEC 재현 (skt/kobart-base-v1, 최신 PyTorch 스택)

이 저장소는 [soyoung97/Standard_Korean_GEC](https://github.com/soyoung97/Standard_Korean_GEC) (`dfe0af9`)을 기반으로, 논문 *Towards Standardizing Korean Grammatical Error Correction* (Yoon et al., ACL 2023)의 KoBART 실험(Table 6)을 최신 스택(torch 2.11 / PyTorch Lightning 2.6 / transformers 4.44, RTX 5090)에서 재현한 것이다. 7개 조합(시나리오 1: 4개 데이터셋, 시나리오 2: 3개 데이터셋) 모두 논문 수치를 재현했다(테스트 GLEU 3-seed 평균이 논문과 같거나 0.6–1.5 높음).

- 결과와 체크포인트 목록: [`docs/result_kobart.md`](docs/result_kobart.md)
- 계획과 근거: [`docs/plan.md`](docs/plan.md) (§3-6, §3-7이 재현의 핵심)
- 시간순 작업 기록(명령·결과): [`docs/log.md`](docs/log.md)

## 원본 대비 변경

| 파일 | 내용 |
| --- | --- |
| `run.py` | PL 2.x Trainer, `--recipe {root,legacy}`, `--data` 기반 데이터 경로, 검증 GLEU 최고 체크포인트 1개 저장, `--resume_finetune`(시나리오 2), `--eval_test`, `--run_id`, 실행마다 특수 토큰 기록 |
| `model.py` | PL 2.x 훅(`on_*_epoch_end`), **디코더 첫 토큰 비마스킹**(`make_decoder_attention_mask`), **학습 epoch마다 `train()`으로 dropout 활성화**, 스케줄러·생성 설정 인자화, 고장 난 M² 블록 제거 |
| `tokenizer_setup.py` (신규) | `skt/kobart-base-v1` 토크나이저 로딩(recipe별 `<s>…</s>` 부착 여부)과 특수 토큰·config 검사 |
| `tools/` (신규) | 실행(`run_one.sh`, `run_all.sh`), 점검(`check_special_tokens.py` 등), 요약(`summarize_runs.py`), 진단 스크립트 |

논문 재현에는 `--recipe legacy`를 쓴다. 논문 실험 코드(`src/KoBART-gec`)와 같은 조건이다: SKT 토크나이저 형식(`<s>…</s>` 미부착), cosine 스케줄 + warmup 0.1, gradient clipping 1.0, 생성 시작 토큰 0, repetition_penalty 없음. `root`는 비교용으로만 남겼다. 최신 transformers에서는 디코더 누설을 피할 수 없어 점검에서 실패한다.

## 준비

디렉토리 구성(데이터와 캐시는 저장소 밖):

```
<work>/
├── 26-13_PHDQ_BART/          # 이 저장소 (명령은 여기서 실행)
├── data/Preprocessed/<data>/<data>_{train,val,test}.txt, <data>_test.m2
└── .cache/                    # HF·pip·torch 캐시, 레거시 토크나이저
```

`<data>`는 `korean_learner`, `native`, `lang8`, `union` 중 하나다. 파일은 탭 구분 `소스\t타겟` 형식이며, 저자 README의 데이터 준비 절차(`train_split`, `random_state=1`)로 만든다.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install "pytorch-lightning>=2.5" "transformers>=4.38,<4.45"
pip install "numpy<2" pandas scikit-learn sentencepiece
pip install konlpy soylemma soynlp jamo pytz hunspell spacy tqdm   # hunspell 빌드에 python3.12-dev, libhunspell-dev 필요

# 캐시를 작업 디렉토리 안에 고정 (.venv/bin/activate 끝에 추가)
export XDG_CACHE_HOME="$(cd ..; pwd)/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface" PIP_CACHE_DIR="$XDG_CACHE_HOME/pip" TORCH_HOME="$XDG_CACHE_HOME/torch"

# 레거시 SKT 토크나이저·가중치 (특수 토큰 점검의 기준, 위치는 KOBART_LEGACY_DIR로 변경 가능)
mkdir -p ../.cache/kobart_legacy && cd ../.cache/kobart_legacy
for f in kobart_base_tokenizer_cased_cf74400bce.zip kobart_base_cased_ff4bda5738.zip; do
  curl -sSL -o $f https://huggingface.co/skt/kobart-base-v1/resolve/main/legacy/$f && unzip -oq $f
done
cd -
```

## 실행

```bash
mkdir -p logs
python3 tools/check_special_tokens.py legacy     # 토크나이저·config·디코더 누설 점검 (종료 코드 0이어야 함)
bash tools/run_all.sh <tag>                       # 21 run 전체 (tmux 권장, RTX 5090에서 약 12시간)
bash tools/run_one.sh <data> <run_id> <seed> <lr> legacy [init_ckpt]   # 개별 run: 점검 → 학습 → 테스트 → M²
python3 tools/summarize_runs.py logs/runs.log <tag>                    # 3-seed 평균 vs 논문 Table 6
```

학습 설정: lr 3e-5(시나리오 1) / 1e-5(시나리오 2), batch 64, 10 epoch, max_seq_len 128, seed 0/1/2. 체크포인트는 `outputs/<data>/<run_id>/model_ckpt/`, 생성 결과는 `outputs/generation/<data>/<run_id>/epoch<N>/{val,test}/`에 저장된다.
