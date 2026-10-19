# gec2 공통 학습·평가 코드. HF 표준 방식: 토크나이저의 실제 pad, attention_mask 전달
# 라벨 패딩 -100, 디코더 입력은 모델 내부 shift가 만든다. train_seq2seq.py / train_llm.py가 이 모듈을 쓴다.

import argparse 
import datetime
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import time

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer,
                          get_cosine_schedule_with_warmup)
from gec2.evaluate import REPO_ROOT, evaluate, read_pairs, split_path

# LLM에 입력할 문법 오류 수정 지시 프롬프트
PROMPT = '다음 문장의 문법 오류를 고치세요.\n입력: {src}\n출력: '


# 시각 표시를 KST로 변환하여 문자열로 반환하는 함수
KST = datetime.timezone(datetime.timedelta(hours=9))
def now():
    return datetime.datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')


# ---------slurm 노드 다중 작업용 함수

# 다중 노드, 다중 GPU 환경에서 중복하여 로그를 기록할 수 있으므로 0번 프로세스 로그만 출력하도록 점검하는 global_rank(), rank_zero() 함수가
# log() 함수에 포함되어 실행
def global_rank():
    """Trainer 생성 전에도 쓸 수 있는 rank. PL 자식 프로세스는 LOCAL_RANK, srun 작업은 SLURM_PROCID로 구분한다(단일 노드)."""
    for k in ('RANK', 'LOCAL_RANK', 'SLURM_PROCID'):
        if k in os.environ:
            return int(os.environ[k])
    return 0


def rank_zero():
    return global_rank() == 0


def log(msg):
    if rank_zero():
        print(f'[{now()}] {msg}', flush=True)

# 파이썬 객체를 json 파일로 저장하는 함수
def dump_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)

# epoch 단위로 작성하는 여러 개의 json 파일을 단일 jsonl 파일로 이어쓰는 함수
def append_jsonl(path, obj):
    # 이어쓰기 모드로 열기
    with open(path, 'a', encoding='utf-8') as f:
        # 한 줄에 JSON 하나씩 기록
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + '\n')


# ---------- 토큰화 형식

class Formatter:
    """모델별 입력 형식과 생성 설정을 정의하는 클래스

    kind = seq2seq: 토크나이저 기본 동작을 따르는 클래스(T5: 끝에 </s>, mBART-50: [ko_KR] … </s>, KoBART: <s> … </s>)
    kind = llm: 프롬프트(PROMPT, 토크나이저 기본 BOS 포함) + 정답 + eos를 입력하고, 손실은 정답+eos로만 평가하는 클래스에만 건다(T5: 끝에 </s>, mBART-50: [ko_KR] … </s>).
             KoBART(PreTrainedTokenizerFast)처럼 특수 토큰을 붙이지 않는 토크나이저는 BART 표준대로 <s> … </s>로 감싼다.
    """

    def __init__(self, tok, config, kind):
        self.tok, self.kind, self.model_type = tok, kind, config.model_type

        # generate()에 추가로 넘길 인자
        self.gen_kwargs = {}
        if kind == 'seq2seq':
            # mBART는 소스/타깃 언어 코드를 ko_KR로 지정하고 생성 첫 토큰을 언어 코드로 강제
            if self.model_type == 'mbart':
                tok.src_lang = tok.tgt_lang = 'ko_KR'
                self.gen_kwargs['forced_bos_token_id'] = tok.convert_tokens_to_ids('ko_KR')
            # 토크나이저가 끝에 </s>를 안 붙이면(KoBART 등) 직접 <s> ... </s>로 감싸야 함
            self.wrap = tok('가')['input_ids'][-1] != tok.eos_token_id
        else:
            # LLM은 감싸지 않음
            self.wrap = False
            if tok.pad_token_id is None:
                # pad 토큰이 없으면 eos로 대체
                tok.pad_token = tok.eos_token
            # 생성 종료/패딩 토큰 지정
            self.gen_kwargs.update(eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        # 패딩에 쓸 ID 지정
        self.pad_id = tok.pad_token_id

    # 앞뒤에 BOS/EOS를 붙여주는 메소드
    def _wrap(self, ids):
        return [self.tok.bos_token_id] + ids + [self.tok.eos_token_id] if self.wrap else ids

    # 소스 문장 인코딩 : seq2seq는 BOS/EOS로 감싸고, llm은 프롬프트를 추가하여 인코딩 
    def enc_src(self, texts):
        if self.kind == 'seq2seq':
            # 소스 문장 인코딩 후 필요하면 BOS/EOS로 감쌈
            return [self._wrap(x) for x in self.tok(texts)['input_ids']]
        # LLM: 프롬프트 템플릿에 소스를 끼워 넣어 인코딩
        return self.tok([PROMPT.format(src=s) for s in texts])['input_ids']

    def enc_tgt(self, texts):
        if self.kind == 'seq2seq':
            # 타깃 문장을 text_target 모드로 인코딩 후 필요하면 감쌈
            return [self._wrap(x) for x in self.tok(text_target=texts)['input_ids']]
        # LLM: 특수 토큰 없이 인코딩하고 끝에 eos를 직접 붙임
        return [x + [self.tok.eos_token_id] for x in self.tok(texts, add_special_tokens=False)['input_ids']]

    def decode(self, seqs, prompt_len=0):
        """생성 결과 → 문자열. LLM은 프롬프트 뒤만 잘라 첫 줄만 쓴다(이탈 통계는 stats에)."""
        # 결과 문자열 목록과 파싱 이상 통계(개행으로 잘림/빈 출력)
        outs, stats = [], {'newline_cut': 0, 'empty': 0}
        # 프롬프트 길이 이후 토큰만 디코딩 (특수 토큰 제거)
        for s in self.tok.batch_decode(seqs[:, prompt_len:], skip_special_tokens=True):
            if self.kind == 'llm':
                # 앞뒤 공백 제거 후에도 줄바꿈이 있으면 잘림 횟수 증가
                if '\n' in s.strip():
                    stats['newline_cut'] += 1
                # LLM은 첫 줄만 정답으로 사용
                s = s.strip().split('\n')[0]
            # 남은 개행은 공백으로 바꾸고 양끝 공백 제거
            s = s.replace('\n', ' ').strip()
            # 빈 문자열이면 empty 카운트 증가 (True는 1로 더해짐)
            stats['empty'] += s == ''
            outs.append(s)
        # 문자열 목록과 통계 반환
        return outs, stats


# ---------------------------------------------------------------- 데이터

class PairSet(Dataset):
    def __init__(self, pairs, fmt, sort_by_len=False):
        # 소스/타깃 문장을 토큰 id 리스트로 미리 인코딩
        src, tgt = fmt.enc_src([s for s, _ in pairs]), fmt.enc_tgt([t for _, t in pairs])
        # (원래 인덱스, 소스 ids, 타깃 ids) 튜플 목록
        self.items = [(i, s, t) for i, (s, t) in enumerate(zip(src, tgt))]
        if sort_by_len:   # 생성 효율용. 결과는 원래 순서(idx)로 되돌린다.
            # 소스 길이가 긴 순서로 정렬
            self.items.sort(key=lambda x: -len(x[1]))
        # 최대 소스 토큰 수
        self.max_src = max(len(s) for _, s, _ in self.items)
        # 최대 타깃 토큰 수
        self.max_tgt = max(len(t) for _, _, t in self.items)

    def __len__(self):
        # 샘플 수
        return len(self.items)

    def __getitem__(self, i):
        # i번째 (idx, src, tgt) 반환
        return self.items[i]


def make_collate(fmt, gen):
    # 패딩 토큰 id
    pad = fmt.pad_id

    def pad_right(seqs, value):
        # 배치 내 최대 길이
        n = max(len(s) for s in seqs)
        # 뒤쪽을 value로 채워 텐서로 변환
        return torch.tensor([s + [value] * (n - len(s)) for s in seqs])

    def pad_left(seqs, value):
        # 배치 내 최대 길이
        n = max(len(s) for s in seqs)
        # 앞쪽을 value로 채워 텐서로 변환 (LLM 생성용 왼쪽 패딩)
        return torch.tensor([[value] * (n - len(s)) + s for s in seqs])

    def collate(batch):
        # 원래 문장 인덱스 (정렬된 생성 결과를 원래 순서로 되돌리는 데 사용)
        idx = torch.tensor([b[0] for b in batch])
        # 소스/타깃 토큰 리스트 분리
        src, tgt = [b[1] for b in batch], [b[2] for b in batch]
        out = {'idx': idx}
        if fmt.kind == 'seq2seq':
            # 인코더 입력: 오른쪽 pad
            out['input_ids'] = pad_right(src, pad)
            # 실제 토큰은 1, 패딩은 0
            out['attention_mask'] = pad_right([[1] * len(s) for s in src], 0)
            # 라벨 패딩은 -100 (손실에서 제외)
            out['labels'] = pad_right(tgt, -100)
        else:
            # LLM 학습 입력: 프롬프트 + 정답을 이어 붙임
            seq = [s + t for s, t in zip(src, tgt)]
            out['input_ids'] = pad_right(seq, pad)
            out['attention_mask'] = pad_right([[1] * len(s) for s in seq], 0)
            # 프롬프트 부분은 -100으로 가려 정답+eos에만 손실을 건다
            out['labels'] = pad_right([[-100] * len(s) + t for s, t in zip(src, tgt)], -100)
            if gen:
                # 생성용 입력: 프롬프트만, 왼쪽 패딩
                out['gen_input_ids'] = pad_left(src, pad)
                out['gen_attention_mask'] = pad_left([[1] * len(s) for s in src], 0)
        return out

    # 만든 collate 함수를 DataLoader에 넘기도록 반환
    return collate


# ---------------------------------------------------------------- 모델 로드·검사

def load_model(args, kind, dtype=None):
    # 모델 설정 로드
    config = AutoConfig.from_pretrained(args.model)
    # from_pretrained에 넘길 추가 인자
    kw = {}
    if kind == 'seq2seq' and args.dropout is not None:   # plan2 §3 dropout 정책: seq2seq 0.1 통일
        # T5 계열은 dropout_rate, 그 외는 dropout 키 사용
        key = 'dropout_rate' if config.model_type in ('t5', 'mt5') else 'dropout'
        kw[key] = args.dropout
    # 종류에 맞는 모델 클래스 선택
    cls = AutoModelForSeq2SeqLM if kind == 'seq2seq' else AutoModelForCausalLM
    if dtype is not None:
        # 정밀도 지정 시 전달
        kw['torch_dtype'] = dtype
    # 모델 가중치 로드
    model = cls.from_pretrained(args.model, **kw)
    # 토크나이저 로드
    tok = AutoTokenizer.from_pretrained(args.model)
    # 모델, 토크나이저, 적용된 config 반환
    return model, tok, model.config


def dropout_report(model):
    # 모델 config
    cfg = model.config
    # config에서 확인할 dropout 관련 키 목록
    keys = ['dropout', 'dropout_rate', 'attention_dropout', 'activation_dropout', 'hidden_dropout',
            'attention_probs_dropout_prob', 'hidden_dropout_prob', 'resid_pdrop', 'embd_pdrop', 'attn_pdrop',
            'classifier_dropout', 'mlp_dropout']
    # dropout 확률별 nn.Dropout 모듈 개수
    mods = {}
    # 모든 하위 모듈 순회
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            # p 값을 문자열 키로 어서 개수 누적
            mods[str(m.p)] = mods.get(str(m.p), 0) + 1
    # config 값과 실제 모듈 설정을 함께 반환
    return {'config': {k: getattr(cfg, k) for k in keys if hasattr(cfg, k)}, 'nn.Dropout_p_counts': mods}


def alt_token(tok, cur):
    """누설 검사용으로 cur와 다른 일반 토큰 id."""
    # 후보 글자를 차례로 시도
    for w in ('가', '나', '다'):
        # 후보 글자의 마지막 토큰 id
        t = tok.convert_tokens_to_ids(tok.tokenize(w)[-1])
        # 현재 토큰과 다르면 반환
        if t != cur:
            return t


@torch.no_grad()
def leak_check(model, fmt, batch, device):
    """디코더(또는 LLM 정답 구간) 첫 위치 누설 검사. 정답 토큰 k를 바꿨을 때 k를 예측하는 위치까지의 logits는
    변하지 않고(누설 없음), 그 다음 위치는 변해야(입력으로 쓰임) 한다. eval 모드 fp32에서 계산."""
    # 원래 학습/평가 모드를 기억해 두었다가 마지막에 복원
    was_training = model.training
    # 평가 모드 (dropout 끔)
    model.eval()
    # 배치의 앞 2개 샘플을 장치로 이동
    b = {k: batch[k][:2].to(device) for k in ('input_ids', 'attention_mask', 'labels')}
    # 원본 입력의 logits (기준값)
    base = model(**b).logits.float()
    # 첫 샘플의 라벨
    lab0 = batch['labels'][0]
    # 라벨이 유효(-100 아님)한 위치 목록
    valid = (lab0 != -100).nonzero().flatten().tolist()
    # 검사 결과 저장용 dict
    res = {}
    if fmt.kind == 'seq2seq':
        # 라벨에서 모델이 만드는 디코더 입력의 첫 토큰(시작 토큰) 목록
        start = model.prepare_decoder_input_ids_from_labels(labels=b['labels'])[:, 0].tolist()
        res['decoder_start_ids'] = start
        res['config_decoder_start_token_id'] = model.config.decoder_start_token_id
        # 라벨 첫 위치와 중간 위치 두 곳에서 검사
        for name, k in (('first', valid[0]), ('middle', valid[len(valid) // 2])):
            # 라벨 k번째 토큰을 다른 토큰으로 바꾼 복사본 생성
            lab = b['labels'].clone()
            lab[0, k] = alt_token(fmt.tok, int(lab[0, k]))
            # 바꾼 라벨로 다시 계산한 logits
            out = model(**{**b, 'labels': lab}).logits.float()
            # k번 위치까지의 logits 변화 (0에 가까워야 누설 없음)
            before = (out[0, :k + 1] - base[0, :k + 1]).abs().max().item()
            # k+1번 위치의 logits 변화 (커야 바뀐 토큰이 입력으로 쓰임)
            after = (out[0, k + 1] - base[0, k + 1]).abs().max().item()
            res[name] = {'pos': k, 'max_diff_upto_pos': before, 'max_diff_next': after, 'ok': before <= 1e-4 and after > 1e-4}
        # 두 위치 모두 통과하고 디코더 시작 토큰이 config와 같아야 통과
        res['ok'] = all(res[n]['ok'] for n in ('first', 'middle')) and \
            all(s == model.config.decoder_start_token_id for s in start)
    else:
        # LLM: 정답 구간의 첫 위치와 중간 위치에서 검사
        for name, p in (('first', valid[0]), ('middle', valid[len(valid) // 2])):
            # 입력 토큰 p를 다른 토큰으로 바꾼 복사본
            ids = b['input_ids'].clone()
            ids[0, p] = alt_token(fmt.tok, int(ids[0, p]))
            out = model(input_ids=ids, attention_mask=b['attention_mask']).logits.float()
            # p 이전 위치의 logits는 변하면 안 됨 (인과적 마스크)
            before = (out[0, :p] - base[0, :p]).abs().max().item()
            # p 위치의 logits는 바뀌어야 함 (자기 자신은 보이므로)
            after = (out[0, p] - base[0, p]).abs().max().item()
            res[name] = {'pos': p, 'max_diff_before_pos': before, 'max_diff_at_pos': after, 'ok': before <= 1e-4 and after > 1e-4}
        # 패딩을 제외한 실제 프롬프트 길이 (라벨의 -100 개수 - 패딩 개수)
        prompt_len = int((batch['labels'][0] == -100).sum() - (batch['attention_mask'][0] == 0).sum())
        # 손실이 걸리는 토큰 수
        res['loss_tokens_example0'] = len(valid)
        # 손실 구간이 프롬프트 바로 다음부터 시작하는지
        res['loss_starts_after_prompt'] = valid[0] == prompt_len
        res['ok'] = res['first']['ok'] and res['middle']['ok'] and res['loss_starts_after_prompt']
    # 원래 모드로 복원
    model.train(was_training)
    return res


# ---------------------------------------------------------------- 실행 정보

def sh(cmd):
    try:
        # 쉘 명령을 최대 60초 실행하고 표준 출력을 문자열로 반환
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:   # noqa: BLE001
        # 실패해도 스크립트를 중단하지 않고 오류 문자열을 반환
        return f'ERR {e}'


def run_info(args):
    # 버전 기록용 import (함수 안에서만 불러옴)
    import transformers, tokenizers, accelerate   # noqa: E401
    return {
        # 실행 시각, 호스트, 작업 디렉터리
        'time_kst': now(), 'host': os.uname().nodename, 'cwd': os.getcwd(),
        # 실행 명령과 파싱된 인자 전체
        'command': ' '.join([sys.executable] + sys.argv), 'args': vars(args),
        # 현재 git 커밋 해시
        'git_commit': sh(f'git -C {REPO_ROOT} rev-parse HEAD'),
        # gec2 폴더의 커밋되지 않은 변경 사항
        'git_dirty': sh(f'git -C {REPO_ROOT} status --porcelain -- gec2'),
        # SLURM 관련 환경변수 전부
        'slurm': {k: v for k, v in os.environ.items() if k.startswith('SLURM_')},
        # GPU 목록/메모리/드라이버 버전
        'nvidia_smi': sh('nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader'),
        # 보이는 GPU 지정 값
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        # 주요 라이브러리 버전
        'versions': {'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda,
                     'transformers': transformers.__version__, 'tokenizers': tokenizers.__version__,
                     'pytorch_lightning': pl.__version__, 'accelerate': accelerate.__version__},
        # 설치된 패키지 전체 목록 (재현용)
        'pip_freeze': sh(f'{sys.executable} -m pip freeze').split('\n'),
    }


# ---------------------------------------------------------------- 체크포인트 선택·조기 종료 (plan2 원칙 3)

def es_update(st, gleu, ep, args):
    """최고 체크포인트(조금이라도 높으면 갱신)와 조기 종료 기준값(min_delta 이상 높을 때만 갱신)을 따로 관리.
    기준값이 갱신되지 않은 epoch가 patience회 연속이면 중단하되, min_epochs(4)번째 epoch를 마친 뒤부터만.
    반환: (최고 갱신 여부, 중단 여부). ep는 0부터."""
    # 점수가 조금이라도 높거나 처음이면 최고 점수 갱신 대상
    improved = st['best_gleu'] is None or gleu > st['best_gleu']
    if improved:
        # 최고 GLEU와 그 epoch 기록
        st['best_gleu'], st['best_epoch'] = gleu, ep
    # 조기 종료 기준값이 없거나 min_delta 이상 올랐으면 (부동소수점 오차 보정 1e-9)
    if st['es_ref'] is None or gleu - st['es_ref'] >= args.min_delta - 1e-9:
        # 기준값과 기준 epoch를 갱신하고 대기 횟수를 0으로 초기화
        st['es_ref'], st['es_ref_epoch'], st['wait'] = gleu, ep, 0
    else:
        # 개선이 충분하지 않으면 대기 횟수 증가
        st['wait'] += 1
    # 완료한 epoch 수
    st['epochs_done'] = ep + 1
    # 조기 종료 켜짐 & patience 초과 & 최소 epoch 완료 시 중단
    stop = bool(args.early_stop and st['wait'] >= args.patience and ep + 1 >= args.min_epochs)
    # (최고 갱신 여부, 중단 여부) 반환
    return improved, stop


# ---------------------------------------------------------------- Lightning 모듈

class GecModule(pl.LightningModule):
    def __init__(self, args, model, fmt, run_dir, kind):
        super().__init__()
        # 인자, 모델, 포맷터, 실행 디렉터리, 종류 보관
        self.args, self.model, self.fmt, self.run_dir, self.kind = args, model, fmt, run_dir, kind
        # 최고 점수/조기 종료 상태 (체크포인트에 함께 저장되어 재개 시 복원)
        self.st = {'best_gleu': None, 'best_epoch': None, 'es_ref': None, 'es_ref_epoch': None, 'wait': 0,
                   'stop_reason': None, 'epochs_done': 0}
        # 이번 작업(job)에서 진행한 epoch 수
        self.jobs_epochs = 0
        # 학습 모드 검사를 이미 했는지
        self.mode_checked = False
        # 평가 누적 변수 초기화
        self._reset_eval()

    def _reset_eval(self):
        # 생성 결과, 손실, 파싱 통계, 생성 소요 시간 초기화
        self.ev_hyp, self.ev_loss, self.ev_stats, self.ev_gen_time = [], [], {'newline_cut': 0, 'empty': 0}, 0.0

    def on_save_checkpoint(self, ckpt):
        # 체크포인트에 우리 상태 추가 저장
        ckpt['gec2_state'] = dict(self.st)

    def on_load_checkpoint(self, ckpt):
        # 체크포인트에서 상태 복원
        self.st = ckpt['gec2_state']

    # ---- 학습
    def configure_optimizers(self):
        # 가중치 감쇠를 적용/미적용할 파라미터 분류
        decay, no_decay = [], []
        # LayerNorm 류 모듈의 파라미터 id 집합
        norm_params = set()
        for m in self.model.modules():
            # 클래스 이름에 'norm'이 들어 있으면 정규화 모듈로 간주
            if 'norm' in type(m).__name__.lower():
                norm_params.update(id(p) for p in m.parameters(recurse=False))
        for n, p in self.model.named_parameters():
            # 학습 대상이 아닌 파라미터는 건너뜀
            if not p.requires_grad:
                continue
            # bias와 정규화 파라미터는 감쇠 없음, 나머지는 감쇠 적용
            (no_decay if n.endswith('bias') or id(p) in norm_params else decay).append(p)
        # AdamW 옵티마이저 (두 파라미터 그룹에 서로 다른 weight_decay)
        opt = torch.optim.AdamW([{'params': decay, 'weight_decay': self.args.weight_decay},
                                 {'params': no_decay, 'weight_decay': 0.0}], lr=self.args.lr, betas=(0.9, 0.999), eps=1e-8)
        total = self.trainer.estimated_stepping_batches   # max_epochs(=10)의 예정 총 step. 조기 종료·재개와 무관하게 고정
        # 웜업 step 수
        warm = int(self.args.warmup_ratio * total)
        # 웜업 후 코사인 감소 스케줄러
        sch = get_cosine_schedule_with_warmup(opt, warm, total)
        # 스케줄 정보와 파라미터 수를 나중에 기록하기 위해 보관
        self.sched_info = {'total_steps': total, 'warmup_steps': warm,
                           'n_params_decay': sum(p.numel() for p in decay), 'n_params_no_decay': sum(p.numel() for p in no_decay)}
        # 스케줄러는 매 step마다 갱신
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sch, 'interval': 'step'}}

    def on_train_epoch_start(self):
        self.model.train()   # from_pretrained는 eval 모드로 반환하고 PL 2.x는 train()을 부르지 않음 (plan.md §3-7)
        if torch.cuda.is_available():
            # 이번 epoch의 GPU 최대 메모리 측정을 초기화
            torch.cuda.reset_peak_memory_stats()
        # 학습 시작 시각
        self.t_train = time.time()
        # epoch 내 step별 손실 목록
        self.train_losses = []

    def on_train_batch_start(self, batch, batch_idx):
        # 첫 배치에서만 학습 모드를 검사
        if not self.mode_checked:
            self.mode_checked = True
            # p>0인 Dropout 모듈 수
            n_drop = sum(1 for m in self.model.modules() if isinstance(m, torch.nn.Dropout) and m.p > 0)
            # 그 중 학습 모드인 모듈 수
            n_drop_train = sum(1 for m in self.model.modules() if isinstance(m, torch.nn.Dropout) and m.p > 0 and m.training)
            # 검사 결과 기록 (모델이 학습 모드이고 모든 dropout이 켜져 있어야 통과)
            res = {'time_kst': now(), 'epoch': self.current_epoch, 'model.training': self.model.training,
                   'dropout_modules_p>0': n_drop, 'dropout_modules_p>0_training': n_drop_train,
                   'ok': self.model.training and n_drop_train == n_drop}
            if rank_zero():
                dump_json(f'{self.run_dir}/check_train_mode.json', res)
            log(f'학습 모드 검사: {res}')
            # 통과하지 못하면 학습을 중단
            if not res['ok']:
                raise RuntimeError(f'첫 학습 배치에서 학습 모드가 아님: {res}')

    def training_step(self, batch, batch_idx):
        # 모델 순전파 (labels를 주면 내부에서 손실과 디코더 입력 shift를 처리)
        out = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        # 타큰 손실
        loss = out.loss
        # 로그용으로 그래프에서 분리해 저장
        self.train_losses.append(loss.detach())
        # 역전파를 위해 손실 반환
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # gradient accumulation 중간 배치는 로그 생략
        if (batch_idx + 1) % self.trainer.accumulate_grad_batches:
            return
        # 실제 optimizer step 횟수
        step = self.trainer.global_step
        # 0번 프로세스에서 log_every step마다(및 첫 step) 기록
        if rank_zero() and (step % self.args.log_every == 0 or step == 1):
            # 현재 학습률
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            # 최근 accumulation 묶음의 평균 손실
            loss = float(torch.stack(self.train_losses[-self.trainer.accumulate_grad_batches:]).mean())
            # 로그 레코드 구성
            rec = {'step': step, 'epoch': self.current_epoch, 'loss': loss, 'lr': lr, 'elapsed_s': round(time.time() - self.t_train, 1)}
            append_jsonl(f'{self.run_dir}/train_steps.jsonl', rec)
            # 화면 출력은 더 드물게 (log_every의 10배마다)
            if step % (self.args.log_every * 10) == 0 or step == 1:
                # epoch당 배치 수
                n = self.trainer.num_training_batches
                log(f'epoch {self.current_epoch} step {step} batch {batch_idx + 1}/{n} loss {loss:.4f} lr {lr:.3e}')

    # ---- 검증·테스트 (생성)
    def _generate(self, batch):
        # 생성 시작 시각
        t = time.time()
        if self.kind == 'seq2seq':
            # 인코더 입력을 그대로 사용, 프롬프트 길이 0
            ids, mask, plen = batch['input_ids'], batch['attention_mask'], 0
        else:
            # LLM은 왼쪽 패딩된 프롬프트만 입력
            ids, mask = batch['gen_input_ids'], batch['gen_attention_mask']
            # 출력 앞부분(프롬프트) 길이 = 입력 길이
            plen = ids.shape[1]
        # 최대 생성 길이: 배치 내 최장 입력 기반으로 계산하고 상한(max_gen_len)을 둔다
        max_new = min(self.args.max_gen_len, math.ceil(self.args.gen_len_ratio * int(mask.sum(1).max())) + self.args.gen_len_add)
        # 빔 서치(무작위 샘플링 없음)로 생성
        seqs = self.model.generate(input_ids=ids, attention_mask=mask, num_beams=self.args.num_beams, do_sample=False,
                                   max_new_tokens=max_new, early_stopping=True, use_cache=True, **self.fmt.gen_kwargs)
        if self.kind == 'seq2seq':
            seqs = seqs[:, 1:]   # 디코더 시작 토큰 제거 (skip_special_tokens가 어차피 지움)
        # 토큰 id를 문자열로 변환하고 통계 수집
        hyps, stats = self.fmt.decode(seqs, plen)
        if torch.cuda.is_available():
            # GPU 작업이 끝날 때까지 기다렸다가 시간 측정
            torch.cuda.synchronize()
        # 생성 소요 시간 누적
        self.ev_gen_time += time.time() - t
        # 통계를 누적
        for k in stats:
            self.ev_stats[k] += stats[k]
        # (원래 문장 인덱스, 생성 문장) 쌍을 누적
        self.ev_hyp += list(zip(batch['idx'].tolist(), hyps))

    def _eval_step(self, batch):
        # 정답 라벨을 넣고 손실 계산
        out = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        # 손실이 계산된 토큰 수 (패딩 제외)
        n = int((batch['labels'] != -100).sum())
        # 토큰 가중 평균을 위해 (손실 × 토큰 수, 토큰 수) 보관
        self.ev_loss.append((float(out.loss) * n, n))
        # 이어서 문장 생성
        self._generate(batch)

    def validation_step(self, batch, batch_idx):
        # 검증 단계도 공통 평가 로직 사용
        self._eval_step(batch)

    def test_step(self, batch, batch_idx):
        # 테스트 단계도 공통 평가 로직 사용
        self._eval_step(batch)

    def on_validation_epoch_start(self):
        # 검증 시작 시각
        self.t_val = time.time()
        # 학습 시간 = 검증 시작 - 학습 시작 (학습 기록이 없으면 0)
        self.train_time = self.t_val - getattr(self, 't_train', self.t_val)
        # 평가 누적 변수 초기화
        self._reset_eval()

    def on_test_epoch_start(self):
        # 테스트 시작 시각
        self.t_val = time.time()
        self._reset_eval()

    def _gather(self):
        """모든 rank의 (idx, hyp)·손실·통계를 모아 rank 0에서 원래 순서로 정렬 (DistributedSampler 중복 제거)."""
        # 이 프로세스의 평가 결과와 최대 GPU 메모리(GB)
        local = {'hyp': self.ev_hyp, 'loss': self.ev_loss, 'stats': self.ev_stats, 'gen_time': self.ev_gen_time,
                 'peak_mem_gb': torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0}
        # 분산 학습이면 모든 rank의 결과를 모으고, 아니면 로컬 결과만 사용
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            allv = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(allv, local)
        else:
            allv = [local]
        # idx를 키로 한 dict로 합쳐 중복(패딩된 샘플) 제거
        hyp = dict(h for v in allv for h in v['hyp'])
        # 손실이 계산된 전체 토큰 수
        n_tok = sum(n for v in allv for _, n in v['loss'])
        # 원래 순서대로 문장, 토큰 가중 평균 손실, 통계 합계, 시간/메모리 요약 반환
        return {'hyps': [hyp[i] for i in range(len(hyp))],
                'loss': sum(l for v in allv for l, _ in v['loss']) / max(n_tok, 1),
                'stats': {k: sum(v['stats'][k] for v in allv) for k in allv[0]['stats']},
                'gen_time_max_rank': max(v['gen_time'] for v in allv),
                'peak_mem_gb_per_rank': [round(v['peak_mem_gb'], 2) for v in allv]}

    def on_validation_epoch_end(self):
        # 모든 rank의 검증 결과를 합침
        g = self._gather()
        # 현재 epoch (0부터)
        ep = self.current_epoch
        # 점수는 rank 0에서만 계산
        gleu = None
        if rank_zero():
            # 생성 결과를 정답과 비교해 GLEU 등 점수 계산 및 저장
            s = evaluate(self.args.data_root, self.args.data, 'val', g['hyps'], f'{self.run_dir}/val/epoch{ep:02d}',
                         extra={'loss': g['loss'], 'parse': g['stats']}, limit=self.args.limit_val)
            gleu = s['gleu']
        # rank 0의 GLEU를 모든 rank에 공유 (조기 종료 판단이 rank마다 같아야 함)
        gleu = self.trainer.strategy.broadcast(gleu, src=0)
        st = self.st
        # 최고 점수와 조기 종료 상태 갱신
        improved, es_stop = es_update(st, gleu, ep, self.args)
        # 이번 작업에서 처리한 epoch 수 증가
        self.jobs_epochs += 1
        # 최고 점수가 갱신되면 모델 저장
        if improved:
            self._save_best()
        # 검증 소요 시간
        val_time = time.time() - self.t_val
        # epoch 요약 레코드
        rec = {'time_kst': now(), 'epoch': ep, 'global_step': self.trainer.global_step,
               'train_loss_mean': float(torch.stack(self.train_losses).mean()) if getattr(self, 'train_losses', None) else None,
               'val_loss': g['loss'], 'val_gleu': gleu, 'best_gleu': st['best_gleu'], 'best_epoch': st['best_epoch'],
               'es_ref': st['es_ref'], 'es_ref_epoch': st['es_ref_epoch'], 'es_wait': st['wait'],
               'train_time_s': round(self.train_time, 1), 'val_time_s': round(val_time, 1),
               'val_gen_time_s': round(g['gen_time_max_rank'], 1), 'peak_mem_gb_per_rank': g['peak_mem_gb_per_rank'],
               'val_parse': g['stats'], 'lr_end': self.trainer.optimizers[0].param_groups[0]['lr']}
        # 중단 사유 결정: 조기 종료 > 최대 epoch 도달 > 작업 분할 순
        if es_stop:
            st['stop_reason'] = 'early_stop'
            self.trainer.should_stop = True
        elif ep + 1 >= self.args.max_epochs:
            st['stop_reason'] = 'max_epochs'
        elif self.args.epochs_per_job and self.jobs_epochs >= self.args.epochs_per_job:
            self.trainer.should_stop = True   # 작업 분할: 재개용 체크포인트에서 이어감 (stop_reason은 비워 둠)
            rec['job_split_stop'] = True
        # 중단 사유를 레코드에 반영
        rec['stop_reason'] = st['stop_reason']
        if rank_zero():
            # epoch 기록을 jsonl에 추가
            append_jsonl(f'{self.run_dir}/epochs.jsonl', rec)
        # 한 줄 요약 로그 출력
        log(f"EPOCH {ep} val_loss {g['loss']:.4f} val_GLEU {gleu:.2f} (best {st['best_gleu']:.2f} @ {st['best_epoch']}, "
            f"es_ref {st['es_ref']:.2f} wait {st['wait']}) train {self.train_time / 60:.1f}m val {val_time / 60:.1f}m "
            f"mem {g['peak_mem_gb_per_rank']}GB stop={st['stop_reason'] or ('job_split' if rec.get('job_split_stop') else '-')}")

    def _save_best(self):
        # 저장은 0번 프로세스만
        if rank_zero():
            # 최종 저장 경로와 임시 경로 (저장 중 비정상 종료로 깨진 파일이 남지 않게)
            d = f'{self.run_dir}/ckpt/best'
            tmp = d + '.tmp'
            # 이전 임시 폴더 정리
            shutil.rmtree(tmp, ignore_errors=True)
            # 모델 가중치를 safetensors로 저장
            self.model.save_pretrained(tmp, safe_serialization=True)
            # 토크나이저도 함께 저장
            self.fmt.tok.save_pretrained(tmp)
            # 어느 epoch/점수의 체크포인트인지 기록
            dump_json(f'{tmp}/best_info.json', {'epoch': self.current_epoch, 'val_gleu': self.st['best_gleu'], 'time_kst': now()})
            # 기존 best를 삭제하고 임시 폴더 이름을 바꿔 교체
            shutil.rmtree(d, ignore_errors=True)
            os.rename(tmp, d)
        # 모든 프로세스가 저장 완료를 기다리도록 동기화
        self.trainer.strategy.barrier()

    def on_test_epoch_end(self):
        # 모든 rank의 테스트 결과 합침
        g = self._gather()
        if rank_zero():
            # 테스트 문장 수
            n = len(g['hyps'])
            # 생성 속도 지표
            speed = {'test_gen_time_s': round(g['gen_time_max_rank'], 1), 'test_total_time_s': round(time.time() - self.t_val, 1),
                     'sentences_per_s': round(n / g['gen_time_max_rank'], 2), 'gen_batch_size': self.args.gen_batch_size,
                     'n_gpus': self.trainer.world_size, 'num_beams': self.args.num_beams}
            # 테스트 점수 계산 (M2 평가 포함) 및 결과 저장
            s = evaluate(self.args.data_root, self.args.data, 'test', g['hyps'], f'{self.run_dir}/test', m2=True,
                         extra={'loss': g['loss'], 'parse': g['stats'], 'speed': speed,
                                'best_epoch': self.st.get('best_epoch'), 'best_val_gleu': self.st.get('best_gleu')},
                         limit=self.args.limit_test)
            # 최종 테스트 점수 로그
            log(f"TEST GLEU {s['gleu']:.2f} P {s['p']:.2f} R {s['r']:.2f} F0.5 {s['f05']:.2f} "
                f"({speed['sentences_per_s']} 문장/초, {speed['test_gen_time_s']}s)")


# ---------------------------------------------------------------- 실행

def build_parser(kind):
    # 인자 파서 생성
    ap = argparse.ArgumentParser()
    # 학습할 모델 (HF 허브 id 또는 로컬 경로)
    ap.add_argument('--model', required=True, help='HF id 또는 로컬 디렉토리')
    # 결과 폴더 이름 (없으면 모델 경로의 마지막 이름)
    ap.add_argument('--model_tag', default=None, help='출력 디렉토리 이름 (기본: 모델 경로의 마지막 이름)')
    # 사용할 데이터셋
    ap.add_argument('--data', required=True, choices=['korean_learner', 'native', 'lang8', 'union'])
    # 전처리 데이터 루트 경로
    ap.add_argument('--data_root', default=os.path.join(REPO_ROOT, '..', 'data', 'Preprocessed'))
    # 실행 결과 저장 루트 경로
    ap.add_argument('--out_root', default=os.path.join(REPO_ROOT, 'outputs2'))
    # 실행 식별자 (결과 폴더 이름에 사용)
    ap.add_argument('--run_id', required=True)
    # 난수 시드
    ap.add_argument('--seed', type=int, default=0)
    # 학습률 (필수)
    ap.add_argument('--lr', type=float, required=True)
    # 이전 학습한 모델의 가중치로 초기화
    ap.add_argument('--init_weights', default=None, help='시나리오 2: union 최고 체크포인트(ckpt/best) 디렉토리. 가중치만 가져옴')
    # 전체(유효) 배치 크기
    ap.add_argument('--global_batch', type=int, default=64)
    # GPU 하나가 한 번에 처리하는 배치 크기
    ap.add_argument('--micro_batch', type=int, default=64, help='GPU당 한 번에 넣는 batch. global = micro × GPU수 × accumulation')
    # 검증/테스트 문장 생성 배치 크기
    ap.add_argument('--gen_batch_size', type=int, default=64)
    # 최대 학습 epoch 수
    ap.add_argument('--max_epochs', type=int, default=10)
    # 전체 step 중 웜업 비율
    ap.add_argument('--warmup_ratio', type=float, default=0.1)
    # AdamW 가중치 감쇠
    ap.add_argument('--weight_decay', type=float, default=0.01)
    # 그래디언트 클리핑 최대 노름
    ap.add_argument('--clip', type=float, default=1.0)
    # 정밀도 (bf16 혼합 정밀도)
    ap.add_argument('--precision', default='bf16-mixed')
    # 빔 서치 크기
    ap.add_argument('--num_beams', type=int, default=4)
    # 최대 생성 길이 계산에 쓰는 배율
    ap.add_argument('--gen_len_ratio', type=float, default=2.0, help='최대 생성 길이 = ceil(ratio × batch 내 최장 입력) + add')
    # 최대 생성 길이에 더하는 상수
    ap.add_argument('--gen_len_add', type=int, default=10)
    # 최대 생성 길이 절대 상한
    ap.add_argument('--max_gen_len', type=int, default=512)
    # 노드당 GPU 수
    ap.add_argument('--devices', type=int, default=1)
    # 장치 종류 (기본 gpu)
    ap.add_argument('--accelerator', default='gpu', help='코드 검증용으로만 cpu 사용')
    # 노드 수
    ap.add_argument('--num_nodes', type=int, default=1)
    # DataLoader 워커 수
    ap.add_argument('--num_workers', type=int, default=2)
    # 학습 로그를 남길 step 간격
    ap.add_argument('--log_every', type=int, default=10)
    # 학습 데이터 앞 N쌍만 사용 (스모크 테스트용)
    ap.add_argument('--limit_train', type=int, default=None, help='스모크용: 앞 N쌍만 사용')
    # 검증 데이터 앞 N쌍만 사용
    ap.add_argument('--limit_val', type=int, default=None)
    # 테스트 데이터 앞 N쌍만 사용
    ap.add_argument('--limit_test', type=int, default=None)
    # 한 작업에서 학습할 최대 epoch 수 (긴 학습을 작업 여러 개로 나눠 제출)
    ap.add_argument('--epochs_per_job', type=int, default=None, help='이 작업에서 학습할 최대 epoch 수 (작업 분할 제출용)')
    # 마지막 체크포인트에서 이어서 학습
    ap.add_argument('--resume', action='store_true', help='run 디렉토리의 ckpt/last.ckpt에서 이어서 학습')
    # 학습 없이 최고 체크포인트로 테스트만
    ap.add_argument('--test_only', action='store_true', help='학습 없이 ckpt/best로 테스트만')
    # 학습 후 테스트를 건너뜀
    ap.add_argument('--no_test', action='store_true')
    # 그래디언트 체크포인팅 (메모리 절약)
    ap.add_argument('--gradient_checkpointing', action='store_true')
    if kind == 'seq2seq':
        # seq2seq는 dropout 0.1로 통일
        ap.add_argument('--dropout', type=float, default=0.1, help='plan2 §3: seq2seq는 0.1로 통일')
        # seq2seq는 기본이 조기 종료 없음 (옵션을 주면 켜짐)
        ap.add_argument('--early_stop', action='store_true', help='(seq2seq 기본: 조기 종료 없음)')
    else:
        # LLM은 모델 기본 dropout 사용 (None)
        ap.set_defaults(dropout=None)
        # LLM은 기본이 조기 종료 켜짐, 이 옵션으로 끔
        ap.add_argument('--no_early_stop', dest='early_stop', action='store_false')
        ap.set_defaults(early_stop=True)
    # 조기 종료: 개선으로 인정하는 최소 GLEU 증가량
    ap.add_argument('--min_delta', type=float, default=0.2)
    # 조기 종료: 개선 없이 기다릴 epoch 수
    ap.add_argument('--patience', type=int, default=3)
    # 조기 종료를 허용하기 전 반드시 학습할 최소 epoch 수
    ap.add_argument('--min_epochs', type=int, default=4)
    return ap


def main(kind):
    # 인자 파싱 후 모델 종류를 인자에 기록
    args = build_parser(kind).parse_args()
    args.kind = kind
    # 결과 폴더 이름에 쓰는 모델 태그
    tag = args.model_tag or os.path.basename(args.model.rstrip('/'))
    # 결과 저장 디렉터리: out_root/모델/데이터셋/run_id
    run_dir = os.path.abspath(os.path.join(args.out_root, tag, args.data, args.run_id))
    # 전체 GPU 수
    world = args.devices * args.num_nodes
    # global = micro × GPU수 × accumulation이 정수로 맞아떨어지는지 확인
    assert args.global_batch % (args.micro_batch * world) == 0, 'global_batch는 micro_batch × GPU수의 배수여야 함'
    # gradient accumulation 횟수
    accum = args.global_batch // (args.micro_batch * world)
    # python/numpy/torch 난수 시드 고정 (DataLoader 워커 포함)
    pl.seed_everything(args.seed, workers=True)
    # 행렬곱 정밀도를 높여 속도 향상 (TF32 허용)
    torch.set_float32_matmul_precision('high')

    # 모델, 토크나이저, config 로드
    model, tok, config = load_model(args, kind)
    if args.init_weights:
        # 이전 학습 체크포인트에서 가중치만 가져와 초기화
        init = AutoModelForSeq2SeqLM if kind == 'seq2seq' else AutoModelForCausalLM
        src = init.from_pretrained(args.init_weights)
        model.load_state_dict(src.state_dict())
        # 임시로 불러온 모델 해제
        del src
    if args.gradient_checkpointing:
        # 중간 활성값을 저장하지 않고 다시 계산해 메모리 절약
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        # 체크포인팅과 충돌하는 KV 캐시 끄기
        model.config.use_cache = False
    # 모델별 입력/라벨 형식 정의
    fmt = Formatter(tok, config, kind)

    def pairs(split, limit):
        # 해당 분할의 (소스, 타깃) 쌍 읽기
        p = read_pairs(split_path(args.data_root, args.data, split))
        # limit이 지정되면 앞쪽만 사용
        return p[:limit] if limit else p

    # 데이터 인코딩 시작 시각
    t0 = time.time()
    # 검증/테스트는 길이순 정렬 (생성 효율)
    val_set = PairSet(pairs('val', args.limit_val), fmt, sort_by_len=True)
    test_set = PairSet(pairs('test', args.limit_test), fmt, sort_by_len=True)
    # test_only면 학습 데이터는 만들지 않음
    train_set = PairSet(pairs('train', args.limit_train), fmt) if not args.test_only else None
    log(f'데이터 인코딩 {time.time() - t0:.1f}s: train {len(train_set) if train_set else "-"} val {len(val_set)} test {len(test_set)}')

    if rank_zero():
        # 결과 디렉터리 생성
        os.makedirs(run_dir, exist_ok=True)
        # 실행 환경 정보 수집
        info = run_info(args)
        info['run_dir'] = run_dir
        # 재개/테스트 전용 실행이면 파일명에 시각을 붙여 기존 기록을 덮어쓰지 않음
        dump_json(f'{run_dir}/run_info{"_resume_" + now().replace(" ", "_") if args.resume or args.test_only else ""}.json', info)
        # 전체 파라미터 수
        n_params = sum(p.numel() for p in model.parameters())
        # 입력 임베딩 파라미터 수
        emb = model.get_input_embeddings().weight.numel()
        out_emb = model.get_output_embeddings()
        # 입력/출력 임베딩이 같은 메모리를 공유(tied)하는지
        tied = out_emb is not None and out_emb.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
        # 임베딩 총 파라미터 수 (tied면 중복 제외)
        n_emb = emb + (0 if tied or out_emb is None else out_emb.weight.numel())
        # 분할별 최대 소스/타깃 토큰 길이
        lens = {'max_src_tokens': {'train': train_set.max_src if train_set else None, 'val': val_set.max_src, 'test': test_set.max_src},
                'max_tgt_tokens': {'train': train_set.max_tgt if train_set else None, 'val': val_set.max_tgt, 'test': test_set.max_tgt}}
        # 모델 정보 저장 (크기, dropout, 특수 토큰 id, 생성 설정 등)
        dump_json(f'{run_dir}/model_info.json', {'model': args.model, 'model_type': config.model_type, 'n_params': n_params,
                                                  'n_params_non_embedding': n_params - n_emb, 'tied_embeddings': tied,
                                                  'dropout': dropout_report(model), 'format_wrap_bos_eos': fmt.wrap,
                                                  'gen_kwargs': fmt.gen_kwargs, 'pad_id': fmt.pad_id,
                                                  'eos_id': tok.eos_token_id, 'bos_id': tok.bos_token_id,
                                                  'decoder_start_token_id': getattr(config, 'decoder_start_token_id', None),
                                                  'prompt': PROMPT if kind == 'llm' else None, **lens})

    # 학습용(생성 입력 없음)과 평가용(생성 입력 포함) 배치 구성 함수
    collate_tr, collate_ev = make_collate(fmt, gen=False), make_collate(fmt, gen=True)
    # 검증/테스트 로더 (순서 고정)
    val_loader = DataLoader(val_set, batch_size=args.gen_batch_size, shuffle=False, collate_fn=collate_ev, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.gen_batch_size, shuffle=False, collate_fn=collate_ev, num_workers=args.num_workers)

    # 누설 검사 (학습 전, rank 0, eval 모드 fp32)
    if rank_zero() and not args.test_only:
        # 검사를 실행할 장치 (GPU 없으면 CPU)
        dev = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
        model.to(dev)
        # 검증 세트의 샘플 두 개로 누설 검사 실행
        chk = leak_check(model, fmt, collate_ev([val_set.items[-1], val_set.items[-2]]), dev)
        # 학습 시작 전에 모델을 CPU로 되돌리고 GPU 캐시 비움
        model.cpu()
        torch.cuda.empty_cache()
        dump_json(f'{run_dir}/check_leak.json', chk)
        log(f'누설 검사: ok={chk["ok"]} {chk}')
        # 누설이 감지되면 학습을 시작하지 않음
        if not chk['ok']:
            raise RuntimeError(f'누설 검사 실패: {chk}')

    # 최고 점수 모델은 별도 저장하므로 마지막(last) 체크포인트만 유지 (재개용)
    ckpt_cb = ModelCheckpoint(dirpath=f'{run_dir}/ckpt', filename='epoch{epoch:02d}', save_top_k=0, save_last=True,
                              save_on_train_epoch_end=True, auto_insert_metric_name=False)

    def make_trainer(callbacks):
        # 다중 GPU면 DDP, 아니면 자동 선택
        strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True) if world > 1 else 'auto'
        # 로그/프로그레스바/요약은 끄고 우리 코드의 로그를 사용
        return pl.Trainer(accelerator=args.accelerator, devices=args.devices, num_nodes=args.num_nodes, strategy=strategy,
                          precision=args.precision, max_epochs=args.max_epochs, accumulate_grad_batches=accum,
                          gradient_clip_val=args.clip, gradient_clip_algorithm='norm', num_sanity_val_steps=0,
                          logger=False, enable_progress_bar=False, enable_model_summary=False, callbacks=callbacks,
                          default_root_dir=run_dir, use_distributed_sampler=True)

    # 학습/평가 로직을 담은 Lightning 모듈
    module = GecModule(args, model, fmt, run_dir, kind)
    if not args.test_only:
        # 학습 로더 (매 epoch 셔플)
        train_loader = DataLoader(train_set, batch_size=args.micro_batch, shuffle=True, collate_fn=collate_tr,
                                  num_workers=args.num_workers, drop_last=False)
        trainer = make_trainer([ckpt_cb])
        # 재개 시 사용할 마지막 체크포인트 경로
        last = f'{run_dir}/ckpt/last.ckpt'
        ckpt_path = last if args.resume and os.path.exists(last) else None
        # 재개를 요청했지만 체크포인트가 없으면 오류
        if args.resume and ckpt_path is None:
            raise FileNotFoundError(last)
        if rank_zero():
            # 학습 설정 전체를 기록해 두는 dict (재현용)
            cfg = {'optimizer': 'torch.optim.AdamW (bias correction)', 'lr': args.lr, 'betas': [0.9, 0.999], 'eps': 1e-8,
                   'weight_decay': args.weight_decay, 'weight_decay_excluded': 'bias, *Norm* modules',
                   'schedule': f'linear warmup {args.warmup_ratio} + cosine to 0 over {args.max_epochs} planned epochs',
                   'gradient_clip_norm': args.clip, 'precision': args.precision, 'global_batch': args.global_batch,
                   'micro_batch_per_gpu': args.micro_batch, 'n_gpus': world, 'grad_accumulation': accum,
                   'max_epochs': args.max_epochs, 'early_stop': args.early_stop,
                   'early_stop_rule': {'min_delta': args.min_delta, 'patience': args.patience, 'min_epochs': args.min_epochs} if args.early_stop else None,
                   'dropout': dropout_report(model), 'gradient_checkpointing': args.gradient_checkpointing,
                   'generation': {'num_beams': args.num_beams, 'do_sample': False, 'max_new_tokens': f'min({args.max_gen_len}, ceil({args.gen_len_ratio} × 최장 입력) + {args.gen_len_add})', **fmt.gen_kwargs},
                   'seed': args.seed, 'init_weights': args.init_weights, 'resume_from': ckpt_path,
                   'train_size': len(train_set), 'micro_batches_per_epoch_per_gpu': math.ceil(len(train_set) / world / args.micro_batch)}
            # 재개일 때는 파일명에 _resume을 붙여 저장
            dump_json(f'{run_dir}/train_config{"_resume" if ckpt_path else ""}.json', cfg)
        log(f'학습 시작 run_dir={run_dir} accum={accum} resume={ckpt_path}')
        # 학습 실행 (각 epoch 끝에 검증이 호출됨)
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
        if rank_zero():
            # 실제 적용된 스케줄과 정밀도 플러그인 정보 저장
            dump_json(f'{run_dir}/train_config_applied.json', {**module.sched_info, 'accumulate_grad_batches': trainer.accumulate_grad_batches,
                                                               'precision_plugin': type(trainer.precision_plugin).__name__})
        # 최종 학습 상태
        state = dict(module.st)
        log(f'학습 종료: {state}')
        if rank_zero():
            dump_json(f'{run_dir}/train_state.json', state)
        # 중단 사유가 없으면 작업 분할로 중간에 멈춘 것이므로 테스트 없이 종료
        if state['stop_reason'] is None:
            log('작업 분할로 중단: 다음 작업에서 --resume으로 이어서 실행')
            return
        # 학습에 쓴 객체를 정리해 테스트용 메모리 확보
        del trainer, module, model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        # test_only: 이전 학습의 상태 파일을 읽어 옴
        state = json.load(open(f'{run_dir}/train_state.json'))
    # 테스트를 건너뛰는 옵션
    if args.no_test:
        return
    # 최고 GLEU 체크포인트 경로
    best_dir = f'{run_dir}/ckpt/best'
    cls = AutoModelForSeq2SeqLM if kind == 'seq2seq' else AutoModelForCausalLM
    # 최고 체크포인트 모델 로드
    best = cls.from_pretrained(best_dir)
    # 토크나이저는 원본 모델 것을 쓴다 (학습 중 인코딩과 같고, 재저장본 로딩 경고를 피함)
    tm = GecModule(args, best, Formatter(AutoTokenizer.from_pretrained(args.model), best.config, kind), run_dir, kind)
    # 학습 상태(최고 점수 등)를 테스트 모듈에 전달
    tm.st = state
    # 최고 모델로 테스트 세트 평가 실행
    make_trainer([]).test(tm, dataloaders=test_loader)
