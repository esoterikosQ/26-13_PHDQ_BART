# 더 큰 모델로 한국어 GEC 파인튜닝 비교 계획서

2026-09-26 · 후속 실험 계획. 선행 작업(`plan.md`, `result_kobart.md`)에서 재현한 KoBART(`skt/kobart-base-v1`, 124M) 결과를 기준선으로 삼는다.

## 0. 작업 규칙

- `plan.md` §0의 기록 규칙을 그대로 따른다. 모든 작업은 `log.md`에 날짜·시간순으로 **작업 / 명령 / 결과**를 즉시 남긴다.
- seq2seq 학습·평가는 itcerdo(RTX 5090 32GB)의 `~/projects/phdq_bart`에서, LLM 학습·평가는 Neuron에서 실행한다. 각 환경에서 sudo 없이 사용자 권한으로 작업하고, 캐시·출력은 해당 환경의 사용자 쓰기 가능 경로에 둔다. itcerdo의 장시간 작업은 tmux에서 돌리고 끝나면 세션을 닫는다. Neuron 작업은 사용자가 직접 SLURM으로 제출한다(§0-1). Neuron 작업 경로는 `/scratch/r984a02/phdq_bart`(데이터 `…/data`), Python 환경은 conda 환경 `phdq_bart`이고 용량 설정은 사용자가 관리한다(2026-09-26 확인, `ssh.md`). itcerdo의 gec2 환경은 `.venv-gec2`다.
- Neuron LLM 자원은 **A100 × 2 → H200 × 1** 두 구성만 쓴다(배정 가능성 기준, 학습 속도 순위 아님). **V100은 쓰지 않는다.** GPU 수가 실제 요청 노드 수와 같은지는 클러스터 구성에 따라 다르므로 확인한다.
- itcerdo의 torch 버전(2.11+cu128)과 Neuron 드라이버의 호환성은 확인이 끝났다.
- 코드는 GitHub 저장소(`esoterikosQ/26-13_PHDQ_BART`)에 커밋한다.
- **환경 간 파일 이동은 GitHub로만 한다** (2026-09-26 결정).
    - 코드·스크립트: Claude가 GitHub `main`에 올리고, itcerdo와 Neuron은 `git pull`로 받는다. Neuron에서는 추적 파일을 직접 고치지 않고, 필요한 수정은 GitHub로 반영한다. Neuron 전용 값은 git 제외 파일 `gec2/slurm/paths.local.sh`에 둔다.
    - Neuron 결과: run 디렉토리를 저장소 안 `neuron_outputs/`에 만들고, slurm 로그는 `logs/slurm/`에 쌓는다. 로그인 노드에서 `gec2/slurm/share_results.sh`(git add·commit·push만 함)로 올리고, itcerdo는 `git pull`로 받는다. 올라갈 파일은 `.gitignore`로 정한다: 체크포인트·가중치와 데이터 원문 사본(`source.txt`, `reference.txt`, `gold_*.m2`)은 제외.
    - 데이터셋과 체크포인트는 GitHub로 옮기지 않고, 필요하면 사용자가 직접 옮긴다(데이터셋은 신청서 동의·비상업 조건, 체크포인트는 GitHub 파일 한도 100MB 초과). Neuron 로그인 노드에서는 rsync를 쓸 수 없다.
    - `git pull`은 해당 환경의 작업이 대기 중이거나 끝난 뒤에 한다(실행 중인 스크립트·모듈 교체 방지).

### 0-1. Neuron 작업 인계 방식

Neuron 작업(접속, 데이터 이전·관리, 작업 제출)은 사용자가 직접 한다. Claude는 Neuron에서 바로 실행할 수 있는 코드와 작업 스크립트를 GitHub에 준비하고, 사용자가 공유한 로그·결과물을 분석한다.

1. **준비 (Claude)**: `gec2/`의 LLM 학습·평가 코드와 SLURM 작업 스크립트 템플릿(`gec2/slurm/*.sbatch`)을 커밋한다.
    - 사용자가 제공하는 다른 프로젝트의 실제로 동작한 Neuron SLURM 스크립트를 먼저 참고하고, Neuron의 제출 규칙·공식 `sbatch` 문법과 대조한 뒤 이 프로젝트에 맞게 조정한다. 사례를 받기 전에는 파티션·GPU 요청 형식을 임의로 확정하지 않는다.
    - (2026-09-26) 사용자가 `ssh.md`에 Neuron 공식 안내(`showque`·`showappl`)를 추가해, 그 값으로 확정했다: A100 × 2는 `amd_a100nv_8`·`--gres=gpu:2`·task 2 × CPU 8·`--comment="field=nlp;appl=pytorch-ddp"`, H200 × 1은 `amd_h200nv_8`·`--gres=gpu:1`·CPU 8·`appl=pytorch`. 첫 스모크 작업으로 실제 동작을 확인한다.
    - 파티션, GPU 요청(`--gres` 등), `--comment`는 실제 `#SBATCH` 지시문에 값으로 적거나 제출 명령의 인자로 전달한다(`#SBATCH` 안에서는 셸 변수가 확장되지 않음). 작업·데이터·모델 경로 등 실행 중 쓰는 값은 스크립트 상단의 변수로 모은다. 제출 전 `sbatch --test-only`와 짧은 GPU 작업으로 요청 자원·실행 경로를 확인한다.
    - 코드는 데이터 경로와 모델 경로를 인자로 받는다. 모델은 로컬 디렉토리에서도 읽을 수 있게 한다(계산 노드의 인터넷 사용 여부와 무관하게 실행 가능).
    - 시작 전 점검(§4)을 독립 작업(`gec2/slurm/precheck.sbatch`, cpu 파티션)으로도 제공하고, 스모크 작업 시작 때도 수행한다. **Neuron 로그인 노드에서는 python 계산을 하지 않는다**(공유 노드 규칙; 2026-09-26 로그인 노드에서 사전 점검을 실행했다가 세션이 끊김). 로그인 노드에서는 `git pull`, `sbatch`, `share_results.sh`, 모델 내려받기만 한다.
    - 작업 시간(`--time`)은 스크립트에 명시한다. 값은 §5 단계 2에서 실측한 epoch당 시간 × 최대 epoch + 준비·최종 평가 시간 + 여유(약 20%)로 정하고, 근거를 스크립트 주석에 남긴다. 1 epoch가 6시간을 넘을 것으로 추정되면 §5의 별도 가이드를 함께 준다.
2. **실행 (사용자)**: Neuron에서 `git pull` 후 저장소 루트에서 작업을 제출한다. slurm 로그는 저장소 루트의 `logs/slurm/<작업이름>_<jobid>.out`에 쌓인다.
3. **공유 (사용자 → GitHub → Claude)**: 작업이 끝나면 `bash gec2/slurm/share_results.sh`로 push한다. 코드가 아래 항목을 run 디렉토리(`neuron_outputs/…`) 한 곳에 모아 저장하고, `.gitignore`가 체크포인트·데이터 사본을 거른다.
    - 실행 정보: 커밋 해시, 실행 명령·인자, 작업 스크립트, `nvidia-smi` 요약(GPU 모델·메모리·개수), 패키지 버전
    - 실제 적용된 학습 설정: 옵티마이저·lr·스케줄·warmup·weight decay·clipping·**dropout 값**·전역 batch·정밀도
    - 사전 점검 결과(§4)와 특수 토큰·마스크·학습 모드 검사 결과
    - 학습 로그: step별 loss·lr, epoch별 검증 loss·GLEU, epoch별 학습·검증 생성 시간, 최대 GPU 메모리
    - 테스트 결과: `hypothesis.txt`, `gleu.txt`, `m2score.txt` (체크포인트는 필요할 때만 공유)
4. **분석·기록 (Claude)**: 공유받은 결과를 기존 결과와 같은 형식으로 집계하고, `log.md`에 기록한다.

## 1. 목표와 비교 원칙

**목표**: KoBART보다 파라미터가 큰 모델을 같은 데이터로 파인튜닝해, 한국어 GEC 성능(GLEU, M² P/R/F0.5)이 모델 크기·구조·사전학습 언어에 따라 어떻게 달라지는지 비교한다.

**공정한 비교를 위한 원칙**

1. **데이터와 분할은 고정한다.** `data/Preprocessed/<data>/<data>_{train,val,test}.txt`, 4개 데이터셋(korean_learner, native, lang8, union)을 쓴다.
2. **평가는 원문 기준으로 통일한다.** 지금 파이프라인은 GLEU의 source·reference를 *모델 토크나이저로 디코딩한 텍스트*로 만든다. KoBART는 디코딩 결과가 원문과 같아서 문제가 없었다(§3-2 검증). 하지만 다른 토크나이저는 정규화(NFKC 등)나 공백 처리 때문에 원문과 달라질 수 있다. 그래서 source·reference는 데이터 파일의 원문을, hypothesis만 모델 출력을 쓰도록 바꾼다. M²는 계속 공식 `<data>_test.m2`를 쓴다.
3. **체크포인트 선택 규칙은 같다.** seq2seq는 10 epoch를 모두 학습한다(조기 종료 없음). LLM은 **최대 10 epoch**를 학습하고 아래 규칙으로 조기 종료한다(예비 실험 전에 고정).
    - **두 기록을 따로 관리한다.** 테스트에 쓸 최고 체크포인트는 검증 GLEU가 조금이라도 높아지면 갱신한다. 조기 종료 판단의 기준값은 직전 기준값보다 **0.2 GLEU점 이상** 높아졌을 때만 갱신한다(PyTorch Lightning `EarlyStopping`의 `min_delta` 방식).
    - 기준값이 갱신되지 않은 epoch를 처음부터 연속으로 센다. 3회 연속(patience 3)이 되면 중단하되, 중단은 **4번째 epoch를 마친 뒤부터** 가능하다. 예: 최고가 epoch 1이고 epoch 2·3·4가 기준값을 갱신하지 못하면 epoch 4를 마친 뒤 중단한다.
    - 학습한 모든 epoch 중 최고 검증 GLEU 체크포인트로 테스트하며, 테스트셋으로 설정이나 중단 시점을 고르지 않는다. 실제 중단 epoch와 최고 epoch를 기록한다.
    - 결과 표에는 트랙별 규칙 차이(seq2seq 10 epoch 고정, LLM 조기 종료)를 함께 적는다.
4. **하이퍼파라미터 탐색 예산을 모든 모델에 똑같이 준다.** 모델마다 권장 lr이 크게 다르므로(BART 3e-5, T5 1e-4~5e-4, LLM 전체 파인튜닝 1e-5~5e-5), 작은 lr 탐색을 같은 방식으로 한다(§5 단계 3). 옵티마이저 등 나머지 설정은 모든 모델에 같게 둔다(§3 학습 설정). 어댑터 실험은 별도 트랙으로 표기한다.
    - **기준선은 두 가지로 보고한다.** (a) 재현된 KoBART(`result_kobart.md`, 논문 대조용)와 (b) **KoBART(gec2)**: 같은 `skt/kobart-base-v1`을 새 `gec2` 스크립트로 같은 lr 탐색을 거쳐 학습한 결과. 큰 모델과의 차이는 (b)와 비교해, 모델 차이와 코드·마스크·옵티마이저 차이를 분리한다.
5. **디코딩은 같게 한다.** seq2seq와 LLM 모두 **빔 4**, 샘플링 없이 생성한다. 최대 생성 길이는 입력 길이에 비례해 정한다.
6. **보고 항목**: 총 파라미터 수와 임베딩을 뺀 파라미터 수, GLEU, M² P/R/F0.5, 3-seed 평균·범위, 학습 시간, 추론 속도(문장/초, 논문 Table 6과 같은 방식).

## 2. 모델 후보

Hugging Face에서 2026-09-26에 확인한 값이다. 파라미터 수는 공개 safetensors 정보 또는 fp32 가중치 크기로 추정했다.

### A. 한국어 전용 seq2seq (1순위)

| 모델 | 파라미터 | 구조 / 토크나이저 | 라이선스 | 비고 |
| --- | --- | --- | --- | --- |
| `paust/pko-t5-base` | 약 276M | T5 v1.1, 한국어 BBPE (OOV 없음) | CC-BY-4.0 | 나무위키·위키·모두의말뭉치로 사전학습. 크기 비교의 중간 지점 |
| `paust/pko-t5-large` | 약 821M | 〃 | CC-BY-4.0 | **주 후보.** KoBART의 약 6.6배 |
| `KETI-NLP/ke-t5-base` | 약 247M | T5, 한·영 sentencepiece | Apache-2.0 | `-ko` 변형도 있음 (같은 크기) |
| `KETI-NLP/ke-t5-large` | 약 783M | 〃 | Apache-2.0 | pko-t5-large와 같은 급의 대안 |

한국어 전용 사전학습에 KoBART와 같은 encoder-decoder 구조라, "크기만 키웠을 때"의 효과를 보기에 가장 깨끗하다. base와 large를 함께 돌리면 124M → 약 250M → 약 800M의 크기 곡선을 얻는다.

### B. 다국어 seq2seq (2순위)

| 모델 | 파라미터 | 구조 / 토크나이저 | 라이선스 | 비고 |
| --- | --- | --- | --- | --- |
| `facebook/mbart-large-50` | 약 611M | BART-large, 다국어 sentencepiece (ko_KR) | MIT | KoBART와 같은 BART 계열이라 구조 차이가 가장 작음 |
| `google/mt5-large` | 약 1.23B | T5, 250k 다국어 sentencepiece | Apache-2.0 | 임베딩이 파라미터의 큰 비중(약 5억)을 차지함. 크기를 비교할 때 임베딩을 뺀 수치도 함께 보고 |
| `google/byt5-large` | 약 1.23B | T5, **바이트 단위** (토크나이저 없음) | Apache-2.0 | 철자·띄어쓰기 오류에 강할 수 있음. 한글은 1글자에 3바이트라 시퀀스가 길어서 학습·추론이 느림 |

`google/mt5-xl`(약 3.7B)은 seq2seq 실행 환경인 itcerdo의 32GB에서 전체 파인튜닝이 어려워 이번 seq2seq 후보에서 제외한다.

### C. 한국어 성능이 좋은 decoder-only LLM 전체 파인튜닝 (3순위, 탐색적)

| 모델 | 파라미터 | 라이선스 | 비고 |
| --- | --- | --- | --- |
| `kakaocorp/kanana-1.5-2.1b-instruct-2505` | 약 2.3B | Apache-2.0 | **주 후보.** 한국어 특화, 라이선스 제약 적음 |
| `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct` | 약 2.4B | EXAONE 라이선스 (비상업) | 한국어 특화. 연구용으로만 사용 |
| `Qwen/Qwen2.5-1.5B-Instruct` | 약 1.5B | Apache-2.0 | 다국어 대조군 |
| `kakaocorp/kanana-1.5-8b-instruct-2505` | 약 8.0B | Apache-2.0 | 별도 QLoRA 탐색 후보. 전체 파인튜닝 비교에 포함하지 않음 |

1.5–2.4B LLM은 Neuron에서 전체 파인튜닝을 우선한다. 메모리·시간 조건을 만족하지 못해 LoRA로 전환한 결과는 전체 파인튜닝과 섞지 않고 별도 트랙에 기록한다. LLM은 과교정(원문을 필요 이상으로 바꿈) 경향이 알려져 있다. 그래서 정밀도가 낮게 나올 수 있고, seq2seq와 같은 기준에서 비교하되 결과는 별도 트랙으로 해석한다. gated 모델(`google/gemma-3-4b-it`, `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B`)은 HF 토큰과 약관 동의가 필요해 기본 후보에서 뺐다.

### 권장 최소 구성

시간 예산이 제한되면 다음 4개로 시작한다. 크기 곡선(A), 같은 구조의 대형(B), LLM(C)을 하나씩 확인할 수 있다.

1. `paust/pko-t5-base` (276M)
2. `paust/pko-t5-large` (821M)
3. `facebook/mbart-large-50` (611M)
4. `kakaocorp/kanana-1.5-2.1b-instruct-2505` (2.3B, 전체 파인튜닝)

## 3. 구현 방침

기존 코드(`run.py`, `model.py`)에는 논문 재현을 위해 **일부러 남긴 원본의 특이 동작**이 있다. 0-패딩 컨벤션, `[0] + labels[:-1]` 디코더 입력, 생성 시작 토큰 0이 그렇다. 새 모델에 이를 그대로 적용하면 안 된다. 그래서 **별도 학습 스크립트**를 만들고, 데이터·평가·산출물 형식만 공유한다.

- **새 스크립트** `gec2/train_seq2seq.py`: HF 표준 방식을 따른다. 토크나이저의 실제 pad 토큰, `attention_mask` 전달, 라벨 패딩 −100, 디코더 입력은 모델 내부의 `shift_right`가 만든다. 학습 루프는 PL 2.x를 쓴다.
    - 학습 중 dropout이 켜져 있는지는 첫 배치에서 검사한다(`plan.md` §3-7의 교훈).
    - 디코더 첫 위치 누설 검사도 모델별로 수행한다(§3-6의 교훈).
- **LLM 스크립트** `gec2/train_llm.py`: 프롬프트는 `"다음 문장의 문법 오류를 고치세요.\n입력: {src}\n출력: "` 형식으로 하고, 손실은 출력 부분에만 건다. 1.5–2.4B는 전체 파인튜닝한다. A100 40GB처럼 단일 GPU 메모리가 부족하면 FSDP/ZeRO로 가중치·그래디언트·옵티마이저 상태를 샤딩한다(DDP만으로는 메모리가 합쳐지지 않음). 8B QLoRA는 필요할 때만 별도 실험으로 진행한다. 생성은 출력 부분만 잘라낸다. Neuron에서 실행하므로 §0-1의 인계 방식(작업 스크립트 템플릿, 경로 인자, run 디렉토리 단위 결과 저장)을 따른다.
- **공통 평가** `gec2/evaluate.py`: 원문 source·reference 파일과 hypothesis로 GLEU를 계산하고, 공식 M²로 채점한다. 결과는 `outputs2/<model>/<data>/<run_id>/`에 모은다.
    - KoBART 재현 결과도 이 스크립트로 다시 채점해, 기존 수치와 같은지 확인한다(원칙 2 검증).
- **정밀도**: itcerdo와 Neuron의 A100/H200 모두 bf16을 쓴다(V100을 쓰지 않으므로 fp16 경로는 두지 않는다). mT5는 fp16에서 불안정할 수 있으므로 bf16을 유지한다.
- **메모리 예상** (전체 파인튜닝, AdamW, 파라미터당 약 16–18바이트 + 활성화·런타임 여유; 물리 GPU별 기준):
    - 800M급은 약 15GB + 활성화 → batch 32–64 가능
    - 1.2B급은 약 22GB + 활성화 → batch 8–16에 gradient accumulation으로 유효 batch 64
    - 1.5–2.4B LLM 전체 파인튜닝은 약 24–43GB + 활성화. H200 141GB × 1에서는 단일 GPU, A100 80GB × 2에서는 GPU별 적합성을 확인한 뒤 DDP를 우선한다. A100 40GB × 2에서는 FSDP/ZeRO를 검토한다.
    - 8B QLoRA는 약 6GB + 활성화(별도 트랙)
- **자원 비교**: 밀집 BF16 Tensor Core 이론 최고 성능은 A100 × 2 약 624, H200 × 1 약 835–989(NVL/SXM) TFLOPS다([A100](https://www.nvidia.com/en-us/data-center/a100/), [H200](https://www.nvidia.com/en-us/data-center/h200/), [NVIDIA 성능 가이드](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html); 희소성 적용 수치 제외). 실제 학습 시간은 메모리 대역폭·상호 연결·샤딩·검증 생성에 좌우된다. 자원 우선순위(§0)는 배정 가능성 기준이며 H200 × 1의 이론 성능이 A100 × 2보다 높다. 두 구성에서 같은 모델을 돌릴 때는 전역 batch·초기 lr·스케줄·최대 epoch·조기 종료 조건·정밀도를 같게 두고 실제 중단 epoch와 GPU 구성을 run 기록에 남긴다.
- **batch**: 가능하면 유효 global batch 64로 맞추고(KoBART와 같게), GPU 수가 바뀌어도 gradient accumulation으로 동일하게 유지한다.
- **학습률 스케줄**: 재현된 KoBART는 고정 lr이 아니라 시작 lr까지 전체 step의 10% 동안 warmup한 뒤 cosine으로 10 epoch 종료 시 0에 가깝게 줄인다(`result_kobart.md` 학습 설정). 새 모델도 우선 동일한 스케줄을 10 epoch의 예정 총 step으로 설정한다. LLM이 조기 종료되더라도 학습 중 스케줄을 다시 계산하지 않는다.
- **옵티마이저 (모든 모델 공통)**: torch `AdamW`(bias 보정 있음), weight decay 0.01(bias·LayerNorm 제외), gradient clipping 1.0. 재현된 KoBART는 bias 보정 없는 옛 transformers AdamW(`correct_bias=False`)를 썼으므로, 이 차이는 KoBART(gec2) 기준선(원칙 4)으로 흡수한다.
- **dropout 정책**: 기본값이 모델마다 다르다(2026-09-26 확인: pko-t5·mT5 `dropout_rate` 0.1, mBART `dropout` 0.1, **ke-t5-large 0.0**, kanana 등 LLM은 attention dropout 0.0 외 설정 없음). 선행 재현에서 dropout 유무로 검증 GLEU가 약 5점 달라졌으므로 명시적으로 정한다.
    - seq2seq: **0.1로 통일**한다(ke-t5는 0.0 → 0.1로 올림).
    - LLM: 설정 기본값을 그대로 쓰고, 그 사실을 기록한다.
    - 모든 run은 실제 적용된 dropout 값을 run 디렉토리에 저장한다(§0-1).

## 4. 사전 점검 항목 (모델마다)

| 항목 | 방법 | 통과 기준 |
| --- | --- | --- |
| 원문 복원 | 전체 데이터에서 `decode(encode(x)) == x` 비율 | 100%가 아니어도 되지만, 차이 유형과 비율을 기록 (평가는 원문 기준이라 점수 계산에는 영향 없음) |
| `<unk>` 비율 | 전체 데이터 토큰 중 unk 수 | 0에 가까울 것 (mBART·mT5 확인 필요) |
| 길이 분포 | 모델 토크나이저 기준 소스·타깃 토큰 수의 99.9%·최대 | 잘림 0건이 되도록 max_len 설정 (ByT5는 512 이상 예상) |
| 특수 토큰·마스크 | pad/eos/decoder_start 확인, 첫 위치 누설 검사 | 누설 없음 |
| 학습 모드 | 첫 학습 배치에서 `model.training` | True |
| 메모리·처리량 | 후보와 자원 구성별 소규모 스모크(native 512줄, 1 epoch), 이어서 대표 epoch의 학습·전체 검증 생성·저장 측정 | OOM·수치 이상(loss NaN·발산) 없음, **1 epoch(학습+전체 검증 생성+저장) 6시간 이내 예상**. 넘으면 §5의 별도 가이드 |

## 5. 작업 단계

| 단계 | 내용 | 산출물 | 예상 소요 |
| --- | --- | --- | --- |
| 1 | 평가 스크립트를 원문 기준으로 분리, KoBART 21 run을 재채점해 기존 수치와 비교 | `gec2/evaluate.py`, 재채점 표 | 반나절 |
| 2 | 후보별 사전 점검(§4), 환경별 캐시에 모델 다운로드, 자원 배정 및 스모크·대표 epoch 측정으로 epoch당 시간과 SLURM `--time` 산정 | GPU 사양·메모리·max_len·batch·epoch당 학습+검증 시간, `--time` 값, 6시간/epoch 초과 여부 | 배정·측정 후 산정 |
| 3 | lr 탐색: korean_learner, seed 0, 후보당 3개 값(T5 1e-4/3e-4/5e-4, mBART 1e-5/3e-5/5e-5, LLM 전체 파인튜닝 1e-5/3e-5/5e-5, KoBART(gec2) 1e-5/3e-5/5e-5), 검증 GLEU로 선택 | lr 선택표 | 모델별 측정 후 산정 |
| 4 | 본 실험 1차: 시나리오 1, 4개 데이터셋, seed 0 | 모델 × 데이터셋 표 | 모델당 약 0.5–2일 |
| 5 | 본 실험 2차: **모든 모델**(2026-09-27 결정)에 seed 1, 2와 시나리오 2(union → 개별, 2단계 시작 lr은 union 학습에 사용한 시작 lr의 1/3, seed 0·1·2) 추가 | 3-seed 평균 | seq2seq(itcerdo) 약 40시간, kanana(Neuron) 17개 작업 |
| 6 | 분석: 크기–성능 곡선, P/R 경향(과교정 여부), 추론 속도(같은 GPU·batch에서 측정, 환경 조건부 수치로 해석). KAGAS 유형별 점수(논문 Table 7/D.4 방식 재구성, 재현 KoBART로 검증) | `result_gec2.md` | 반나절 |
| 7 | 정리: 코드·문서 커밋, `log.md`·`result_gec2.md` 갱신 | GitHub 커밋 | — |

**소요 추정 근거와 진행 기준**: KoBART는 itcerdo에서 run당 native 약 9.5분, korean_learner 15분, lang8 53분, union 75분이었다(검증 생성 포함). 다른 모델이나 Neuron GPU에 이 시간을 성능 수치 비율로 환산하지 않고, 모델·데이터셋·할당 GPU마다 대표 epoch의 학습+전체 검증 생성+저장 시간을 실측한다.

- **기준은 1 epoch 6시간이다.** 공유 노드에서는 epoch 도중에 끊긴 작업을 체크포인트에서 이어 가기가 까다롭기 때문에, 1 epoch가 한 작업 안에서 끝나야 한다. run 전체 시간에는 상한을 두지 않는다.
- **SLURM 작업 시간(`--time`)은 항상 명시한다.** 실측 epoch당 시간 × 최대 epoch(seq2seq 10, LLM 10) + 준비·최종 평가 시간 + 여유(약 20%)로 정하고, 작업 스크립트 주석에 근거(측정값·계산식)를 남긴다. LLM은 조기 종료가 없다고 가정해 최대 epoch로 잡는다. 이 값이 파티션 최대 작업 시간(Neuron 기본 48시간)을 넘으면, epoch 경계에서 나눠 제출한다. 같은 run을 이어갈 때는 마지막 완료 epoch의 **재개용 체크포인트**(모델·옵티마이저·스케줄러·진행 step·조기 종료 상태)를 사용해 학습률과 중단 판단을 이어가고, 테스트에는 별도로 보존한 **평가용 최고 검증 GLEU 체크포인트**를 사용한다. 시나리오 2의 union → 개별 전환은 새 학습이므로 재개와 달리 가중치만 가져와 옵티마이저·스케줄러를 새로 시작한다.
- **1 epoch가 6시간을 넘을 것으로 추정되면**, 제출 전에 별도 가이드를 준다. 가이드에는 적용한 조정(batch·gradient accumulation·gradient checkpointing·검증 생성 배치 크기), 조정 후 재측정한 epoch 시간, 권장 `--time`, 필요하면 작업을 나눠 제출하는 방법과 epoch 경계 체크포인트에서 재개하는 절차를 담는다.
- **LLM 트랙 진행 순서**: korean_learner·native로 먼저 epoch 시간을 확인한 뒤 lang8·union을 진행한다. lang8·union은 검증셋(1.6만·2.3만 문장)을 매 epoch 빔 4로 생성해야 해서 epoch 시간이 가장 길다.

**검증 생성 비용**: 검증 GLEU 최고 epoch를 고르는 실험에서는 비교 대상인 모든 epoch의 전체 검증셋에 같은 생성 규칙을 적용한다. 생성 비용 때문에 일부 epoch만 평가한다면 선택 규칙이 달라지므로, 해당 결과는 별도 프로토콜로 표기하고 기준선과 직접 비교하지 않는다.

## 6. 위험 요소와 대응

| 위험 | 대응 |
| --- | --- |
| 토크나이저 차이로 평가가 불공정해짐 | 원문 기준 평가(원칙 2), 모델별 복원·unk 점검(§4) |
| lr 등 설정 차이가 모델 차이로 오인됨 | 모든 모델에 같은 탐색 예산(원칙 4), 기준선도 같은 탐색 |
| 라이브러리 기본 동작 차이(dropout·마스크 등)로 조용히 성능 저하 | 새 스크립트에 학습 모드·누설 검사 내장(§3), 첫 스모크에서 확인 |
| LLM 과교정·형식 이탈(설명문 출력 등) | 출력 파싱 규칙과 이탈 비율 기록, 빔 4·샘플링 없음(원칙 5), 필요하면 출력 길이 제한 |
| 메모리 부족 | GPU별 메모리 확인 후 gradient accumulation·checkpointing·FSDP/ZeRO. 전체 파인튜닝이 어려우면 LoRA 결과를 별도 표기하고 전체 파인튜닝 결과로 대체하지 않음 |
| 특정 모델이 특정 GPU 구성에서 실패(수치 불안정·OOM 등) | 원인을 기록하고 해당 모델–구성 조합은 옵션에서 제외(대체 구성으로 억지로 맞추지 않음) |
| 배정 지연·1 epoch 6시간 초과 | §0의 순서로 자원을 요청하고 §5 단계 2에서 구성별 epoch 시간을 실측; 초과가 예상되면 §5의 별도 가이드와 `--time`을 제공; GPU·노드 구성, 통신 방식, 대기 시간 기록 |
| Neuron 결과 인계 누락 | run 디렉토리에 §0-1의 공유 항목을 코드가 자동 저장, 분석 전 항목 누락 여부 확인 |
| 데이터 조건 위반 (저장소에 데이터 게시) | 데이터셋은 GitHub에 올리지 않고 `.gitignore`로 데이터 원문 사본을 제외 |
| 대형 모델 다운로드·디스크 | 환경별 캐시에 저장하고 run마다 평가용 최고 GLEU 체크포인트 1개와 재개용 최신 epoch 전체 상태 체크포인트 1개를 별도 유지; 중복 저장량을 감안해 디스크 여유를 확인 |
| 라이선스 | EXAONE은 비상업 연구용으로만 쓰고, 결과 공개 시 라이선스를 표기 |

## 7. 결정이 필요한 사항

1. 후보 범위: §2의 **권장 최소 구성(4개)**으로 시작하고, epoch 시간 실측 뒤 A·B·C 후보를 얼마나 추가할지
2. Neuron 배정 후 A100의 40/80GB 여부, H200의 SXM/NVL 여부, GPU 간 연결·실제 노드 수 및 사용 가능 시간을 확인 (사용자 확인 후 공유)
3. ~~시나리오 2와 3-seed를 모든 모델에 적용할지, 상위 모델에만 적용할지(§5 단계 5)~~ → 2026-09-27 결정: 모든 모델(5개)에 적용
