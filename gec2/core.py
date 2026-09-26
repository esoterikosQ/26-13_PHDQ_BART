# gec2 공통 학습·평가 코드 (plan2.md §3). HF 표준 방식: 토크나이저의 실제 pad, attention_mask 전달,
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

PROMPT = '다음 문장의 문법 오류를 고치세요.\n입력: {src}\n출력: '
KST = datetime.timezone(datetime.timedelta(hours=9))


def now():
    return datetime.datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')


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


def dump_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str)


def append_jsonl(path, obj):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + '\n')


# ---------------------------------------------------------------- 토큰화 형식

class Formatter:
    """모델별 입력·라벨 형식과 생성 설정.

    seq2seq: 토크나이저 기본 동작을 따른다(T5: 끝에 </s>, mBART-50: [ko_KR] … </s>).
             KoBART(PreTrainedTokenizerFast)처럼 특수 토큰을 붙이지 않는 토크나이저는 BART 표준대로 <s> … </s>로 감싼다.
    llm:     프롬프트(PROMPT, 토크나이저 기본 BOS 포함) + 정답 + eos. 손실은 정답+eos에만 건다.
    """

    def __init__(self, tok, config, kind):
        self.tok, self.kind, self.model_type = tok, kind, config.model_type
        self.gen_kwargs = {}
        if kind == 'seq2seq':
            if self.model_type == 'mbart':
                tok.src_lang = tok.tgt_lang = 'ko_KR'
                self.gen_kwargs['forced_bos_token_id'] = tok.convert_tokens_to_ids('ko_KR')
            self.wrap = tok('가')['input_ids'][-1] != tok.eos_token_id
        else:
            self.wrap = False
            if tok.pad_token_id is None:   # 라벨은 위치로 가리므로 pad=eos여도 eos 학습에 영향 없음
                tok.pad_token = tok.eos_token
            self.gen_kwargs.update(eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
        self.pad_id = tok.pad_token_id

    def _wrap(self, ids):
        return [self.tok.bos_token_id] + ids + [self.tok.eos_token_id] if self.wrap else ids

    def enc_src(self, texts):
        if self.kind == 'seq2seq':
            return [self._wrap(x) for x in self.tok(texts)['input_ids']]
        return self.tok([PROMPT.format(src=s) for s in texts])['input_ids']

    def enc_tgt(self, texts):
        if self.kind == 'seq2seq':
            return [self._wrap(x) for x in self.tok(text_target=texts)['input_ids']]
        return [x + [self.tok.eos_token_id] for x in self.tok(texts, add_special_tokens=False)['input_ids']]

    def decode(self, seqs, prompt_len=0):
        """생성 결과 → 문자열. LLM은 프롬프트 뒤만 잘라 첫 줄만 쓴다(이탈 통계는 stats에)."""
        outs, stats = [], {'newline_cut': 0, 'empty': 0}
        for s in self.tok.batch_decode(seqs[:, prompt_len:], skip_special_tokens=True):
            if self.kind == 'llm':
                if '\n' in s.strip():
                    stats['newline_cut'] += 1
                s = s.strip().split('\n')[0]
            s = s.replace('\n', ' ').strip()
            stats['empty'] += s == ''
            outs.append(s)
        return outs, stats


# ---------------------------------------------------------------- 데이터

class PairSet(Dataset):
    def __init__(self, pairs, fmt, sort_by_len=False):
        src, tgt = fmt.enc_src([s for s, _ in pairs]), fmt.enc_tgt([t for _, t in pairs])
        self.items = [(i, s, t) for i, (s, t) in enumerate(zip(src, tgt))]
        if sort_by_len:   # 생성 효율용. 결과는 원래 순서(idx)로 되돌린다.
            self.items.sort(key=lambda x: -len(x[1]))
        self.max_src = max(len(s) for _, s, _ in self.items)
        self.max_tgt = max(len(t) for _, _, t in self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collate(fmt, gen):
    pad = fmt.pad_id

    def pad_right(seqs, value):
        n = max(len(s) for s in seqs)
        return torch.tensor([s + [value] * (n - len(s)) for s in seqs])

    def pad_left(seqs, value):
        n = max(len(s) for s in seqs)
        return torch.tensor([[value] * (n - len(s)) + s for s in seqs])

    def collate(batch):
        idx = torch.tensor([b[0] for b in batch])
        src, tgt = [b[1] for b in batch], [b[2] for b in batch]
        out = {'idx': idx}
        if fmt.kind == 'seq2seq':
            out['input_ids'] = pad_right(src, pad)
            out['attention_mask'] = pad_right([[1] * len(s) for s in src], 0)
            out['labels'] = pad_right(tgt, -100)
        else:
            seq = [s + t for s, t in zip(src, tgt)]
            out['input_ids'] = pad_right(seq, pad)
            out['attention_mask'] = pad_right([[1] * len(s) for s in seq], 0)
            out['labels'] = pad_right([[-100] * len(s) + t for s, t in zip(src, tgt)], -100)
            if gen:
                out['gen_input_ids'] = pad_left(src, pad)
                out['gen_attention_mask'] = pad_left([[1] * len(s) for s in src], 0)
        return out

    return collate


# ---------------------------------------------------------------- 모델 로드·검사

def load_model(args, kind, dtype=None):
    config = AutoConfig.from_pretrained(args.model)
    kw = {}
    if kind == 'seq2seq' and args.dropout is not None:   # plan2 §3 dropout 정책: seq2seq 0.1 통일
        key = 'dropout_rate' if config.model_type in ('t5', 'mt5') else 'dropout'
        kw[key] = args.dropout
    cls = AutoModelForSeq2SeqLM if kind == 'seq2seq' else AutoModelForCausalLM
    if dtype is not None:
        kw['torch_dtype'] = dtype
    model = cls.from_pretrained(args.model, **kw)
    tok = AutoTokenizer.from_pretrained(args.model)
    return model, tok, model.config


def dropout_report(model):
    cfg = model.config
    keys = ['dropout', 'dropout_rate', 'attention_dropout', 'activation_dropout', 'hidden_dropout',
            'attention_probs_dropout_prob', 'hidden_dropout_prob', 'resid_pdrop', 'embd_pdrop', 'attn_pdrop',
            'classifier_dropout', 'mlp_dropout']
    mods = {}
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            mods[str(m.p)] = mods.get(str(m.p), 0) + 1
    return {'config': {k: getattr(cfg, k) for k in keys if hasattr(cfg, k)}, 'nn.Dropout_p_counts': mods}


def alt_token(tok, cur):
    """누설 검사용으로 cur와 다른 일반 토큰 id."""
    for w in ('가', '나', '다'):
        t = tok.convert_tokens_to_ids(tok.tokenize(w)[-1])
        if t != cur:
            return t


@torch.no_grad()
def leak_check(model, fmt, batch, device):
    """디코더(또는 LLM 정답 구간) 첫 위치 누설 검사. 정답 토큰 k를 바꿨을 때 k를 예측하는 위치까지의 logits는
    변하지 않고(누설 없음), 그 다음 위치는 변해야(입력으로 쓰임) 한다. eval 모드 fp32에서 계산."""
    was_training = model.training
    model.eval()
    b = {k: batch[k][:2].to(device) for k in ('input_ids', 'attention_mask', 'labels')}
    base = model(**b).logits.float()
    lab0 = batch['labels'][0]
    valid = (lab0 != -100).nonzero().flatten().tolist()
    res = {}
    if fmt.kind == 'seq2seq':
        start = model.prepare_decoder_input_ids_from_labels(labels=b['labels'])[:, 0].tolist()
        res['decoder_start_ids'] = start
        res['config_decoder_start_token_id'] = model.config.decoder_start_token_id
        for name, k in (('first', valid[0]), ('middle', valid[len(valid) // 2])):
            lab = b['labels'].clone()
            lab[0, k] = alt_token(fmt.tok, int(lab[0, k]))
            out = model(**{**b, 'labels': lab}).logits.float()
            before = (out[0, :k + 1] - base[0, :k + 1]).abs().max().item()
            after = (out[0, k + 1] - base[0, k + 1]).abs().max().item()
            res[name] = {'pos': k, 'max_diff_upto_pos': before, 'max_diff_next': after, 'ok': before <= 1e-4 and after > 1e-4}
        res['ok'] = all(res[n]['ok'] for n in ('first', 'middle')) and \
            all(s == model.config.decoder_start_token_id for s in start)
    else:
        for name, p in (('first', valid[0]), ('middle', valid[len(valid) // 2])):
            ids = b['input_ids'].clone()
            ids[0, p] = alt_token(fmt.tok, int(ids[0, p]))
            out = model(input_ids=ids, attention_mask=b['attention_mask']).logits.float()
            before = (out[0, :p] - base[0, :p]).abs().max().item()
            after = (out[0, p] - base[0, p]).abs().max().item()
            res[name] = {'pos': p, 'max_diff_before_pos': before, 'max_diff_at_pos': after, 'ok': before <= 1e-4 and after > 1e-4}
        prompt_len = int((batch['labels'][0] == -100).sum() - (batch['attention_mask'][0] == 0).sum())
        res['loss_tokens_example0'] = len(valid)
        res['loss_starts_after_prompt'] = valid[0] == prompt_len
        res['ok'] = res['first']['ok'] and res['middle']['ok'] and res['loss_starts_after_prompt']
    model.train(was_training)
    return res


# ---------------------------------------------------------------- 실행 정보

def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:   # noqa: BLE001
        return f'ERR {e}'


def run_info(args):
    import transformers, tokenizers, accelerate   # noqa: E401
    return {
        'time_kst': now(), 'host': os.uname().nodename, 'cwd': os.getcwd(),
        'command': ' '.join([sys.executable] + sys.argv), 'args': vars(args),
        'git_commit': sh(f'git -C {REPO_ROOT} rev-parse HEAD'),
        'git_dirty': sh(f'git -C {REPO_ROOT} status --porcelain -- gec2'),
        'slurm': {k: v for k, v in os.environ.items() if k.startswith('SLURM_')},
        'nvidia_smi': sh('nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader'),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'versions': {'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda,
                     'transformers': transformers.__version__, 'tokenizers': tokenizers.__version__,
                     'pytorch_lightning': pl.__version__, 'accelerate': accelerate.__version__},
        'pip_freeze': sh(f'{sys.executable} -m pip freeze').split('\n'),
    }


# ---------------------------------------------------------------- 체크포인트 선택·조기 종료 (plan2 원칙 3)

def es_update(st, gleu, ep, args):
    """최고 체크포인트(조금이라도 높으면 갱신)와 조기 종료 기준값(min_delta 이상 높을 때만 갱신)을 따로 관리.
    기준값이 갱신되지 않은 epoch가 patience회 연속이면 중단하되, min_epochs(4)번째 epoch를 마친 뒤부터만.
    반환: (최고 갱신 여부, 중단 여부). ep는 0부터."""
    improved = st['best_gleu'] is None or gleu > st['best_gleu']
    if improved:
        st['best_gleu'], st['best_epoch'] = gleu, ep
    if st['es_ref'] is None or gleu - st['es_ref'] >= args.min_delta - 1e-9:
        st['es_ref'], st['es_ref_epoch'], st['wait'] = gleu, ep, 0
    else:
        st['wait'] += 1
    st['epochs_done'] = ep + 1
    stop = bool(args.early_stop and st['wait'] >= args.patience and ep + 1 >= args.min_epochs)
    return improved, stop


# ---------------------------------------------------------------- Lightning 모듈

class GecModule(pl.LightningModule):
    def __init__(self, args, model, fmt, run_dir, kind):
        super().__init__()
        self.args, self.model, self.fmt, self.run_dir, self.kind = args, model, fmt, run_dir, kind
        self.st = {'best_gleu': None, 'best_epoch': None, 'es_ref': None, 'es_ref_epoch': None, 'wait': 0,
                   'stop_reason': None, 'epochs_done': 0}
        self.jobs_epochs = 0
        self.mode_checked = False
        self._reset_eval()

    def _reset_eval(self):
        self.ev_hyp, self.ev_loss, self.ev_stats, self.ev_gen_time = [], [], {'newline_cut': 0, 'empty': 0}, 0.0

    def on_save_checkpoint(self, ckpt):
        ckpt['gec2_state'] = dict(self.st)

    def on_load_checkpoint(self, ckpt):
        self.st = ckpt['gec2_state']

    # ---- 학습
    def configure_optimizers(self):
        decay, no_decay = [], []
        norm_params = set()
        for m in self.model.modules():
            if 'norm' in type(m).__name__.lower():
                norm_params.update(id(p) for p in m.parameters(recurse=False))
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if n.endswith('bias') or id(p) in norm_params else decay).append(p)
        opt = torch.optim.AdamW([{'params': decay, 'weight_decay': self.args.weight_decay},
                                 {'params': no_decay, 'weight_decay': 0.0}], lr=self.args.lr, betas=(0.9, 0.999), eps=1e-8)
        total = self.trainer.estimated_stepping_batches   # max_epochs(=10)의 예정 총 step. 조기 종료·재개와 무관하게 고정
        warm = int(self.args.warmup_ratio * total)
        sch = get_cosine_schedule_with_warmup(opt, warm, total)
        self.sched_info = {'total_steps': total, 'warmup_steps': warm,
                           'n_params_decay': sum(p.numel() for p in decay), 'n_params_no_decay': sum(p.numel() for p in no_decay)}
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sch, 'interval': 'step'}}

    def on_train_epoch_start(self):
        self.model.train()   # from_pretrained는 eval 모드로 반환하고 PL 2.x는 train()을 부르지 않음 (plan.md §3-7)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.t_train = time.time()
        self.train_losses = []

    def on_train_batch_start(self, batch, batch_idx):
        if not self.mode_checked:
            self.mode_checked = True
            n_drop = sum(1 for m in self.model.modules() if isinstance(m, torch.nn.Dropout) and m.p > 0)
            n_drop_train = sum(1 for m in self.model.modules() if isinstance(m, torch.nn.Dropout) and m.p > 0 and m.training)
            res = {'time_kst': now(), 'epoch': self.current_epoch, 'model.training': self.model.training,
                   'dropout_modules_p>0': n_drop, 'dropout_modules_p>0_training': n_drop_train,
                   'ok': self.model.training and n_drop_train == n_drop}
            if rank_zero():
                dump_json(f'{self.run_dir}/check_train_mode.json', res)
            log(f'학습 모드 검사: {res}')
            if not res['ok']:
                raise RuntimeError(f'첫 학습 배치에서 학습 모드가 아님: {res}')

    def training_step(self, batch, batch_idx):
        out = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        loss = out.loss
        self.train_losses.append(loss.detach())
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if (batch_idx + 1) % self.trainer.accumulate_grad_batches:
            return
        step = self.trainer.global_step
        if rank_zero() and (step % self.args.log_every == 0 or step == 1):
            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            loss = float(torch.stack(self.train_losses[-self.trainer.accumulate_grad_batches:]).mean())
            rec = {'step': step, 'epoch': self.current_epoch, 'loss': loss, 'lr': lr, 'elapsed_s': round(time.time() - self.t_train, 1)}
            append_jsonl(f'{self.run_dir}/train_steps.jsonl', rec)
            if step % (self.args.log_every * 10) == 0 or step == 1:
                n = self.trainer.num_training_batches
                log(f'epoch {self.current_epoch} step {step} batch {batch_idx + 1}/{n} loss {loss:.4f} lr {lr:.3e}')

    # ---- 검증·테스트 (생성)
    def _generate(self, batch):
        t = time.time()
        if self.kind == 'seq2seq':
            ids, mask, plen = batch['input_ids'], batch['attention_mask'], 0
        else:
            ids, mask = batch['gen_input_ids'], batch['gen_attention_mask']
            plen = ids.shape[1]
        max_new = min(self.args.max_gen_len, math.ceil(self.args.gen_len_ratio * int(mask.sum(1).max())) + self.args.gen_len_add)
        seqs = self.model.generate(input_ids=ids, attention_mask=mask, num_beams=self.args.num_beams, do_sample=False,
                                   max_new_tokens=max_new, early_stopping=True, use_cache=True, **self.fmt.gen_kwargs)
        if self.kind == 'seq2seq':
            seqs = seqs[:, 1:]   # 디코더 시작 토큰 제거 (skip_special_tokens가 어차피 지움)
        hyps, stats = self.fmt.decode(seqs, plen)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.ev_gen_time += time.time() - t
        for k in stats:
            self.ev_stats[k] += stats[k]
        self.ev_hyp += list(zip(batch['idx'].tolist(), hyps))

    def _eval_step(self, batch):
        out = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        n = int((batch['labels'] != -100).sum())
        self.ev_loss.append((float(out.loss) * n, n))
        self._generate(batch)

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch)

    def test_step(self, batch, batch_idx):
        self._eval_step(batch)

    def on_validation_epoch_start(self):
        self.t_val = time.time()
        self.train_time = self.t_val - getattr(self, 't_train', self.t_val)
        self._reset_eval()

    def on_test_epoch_start(self):
        self.t_val = time.time()
        self._reset_eval()

    def _gather(self):
        """모든 rank의 (idx, hyp)·손실·통계를 모아 rank 0에서 원래 순서로 정렬 (DistributedSampler 중복 제거)."""
        local = {'hyp': self.ev_hyp, 'loss': self.ev_loss, 'stats': self.ev_stats, 'gen_time': self.ev_gen_time,
                 'peak_mem_gb': torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0}
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            allv = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(allv, local)
        else:
            allv = [local]
        hyp = dict(h for v in allv for h in v['hyp'])
        n_tok = sum(n for v in allv for _, n in v['loss'])
        return {'hyps': [hyp[i] for i in range(len(hyp))],
                'loss': sum(l for v in allv for l, _ in v['loss']) / max(n_tok, 1),
                'stats': {k: sum(v['stats'][k] for v in allv) for k in allv[0]['stats']},
                'gen_time_max_rank': max(v['gen_time'] for v in allv),
                'peak_mem_gb_per_rank': [round(v['peak_mem_gb'], 2) for v in allv]}

    def on_validation_epoch_end(self):
        g = self._gather()
        ep = self.current_epoch
        gleu = None
        if rank_zero():
            s = evaluate(self.args.data_root, self.args.data, 'val', g['hyps'], f'{self.run_dir}/val/epoch{ep:02d}',
                         extra={'loss': g['loss'], 'parse': g['stats']}, limit=self.args.limit_val)
            gleu = s['gleu']
        gleu = self.trainer.strategy.broadcast(gleu, src=0)
        st = self.st
        improved, es_stop = es_update(st, gleu, ep, self.args)
        self.jobs_epochs += 1
        if improved:
            self._save_best()
        val_time = time.time() - self.t_val
        rec = {'time_kst': now(), 'epoch': ep, 'global_step': self.trainer.global_step,
               'train_loss_mean': float(torch.stack(self.train_losses).mean()) if getattr(self, 'train_losses', None) else None,
               'val_loss': g['loss'], 'val_gleu': gleu, 'best_gleu': st['best_gleu'], 'best_epoch': st['best_epoch'],
               'es_ref': st['es_ref'], 'es_ref_epoch': st['es_ref_epoch'], 'es_wait': st['wait'],
               'train_time_s': round(self.train_time, 1), 'val_time_s': round(val_time, 1),
               'val_gen_time_s': round(g['gen_time_max_rank'], 1), 'peak_mem_gb_per_rank': g['peak_mem_gb_per_rank'],
               'val_parse': g['stats'], 'lr_end': self.trainer.optimizers[0].param_groups[0]['lr']}
        if es_stop:
            st['stop_reason'] = 'early_stop'
            self.trainer.should_stop = True
        elif ep + 1 >= self.args.max_epochs:
            st['stop_reason'] = 'max_epochs'
        elif self.args.epochs_per_job and self.jobs_epochs >= self.args.epochs_per_job:
            self.trainer.should_stop = True   # 작업 분할: 재개용 체크포인트에서 이어감 (stop_reason은 비워 둠)
            rec['job_split_stop'] = True
        rec['stop_reason'] = st['stop_reason']
        if rank_zero():
            append_jsonl(f'{self.run_dir}/epochs.jsonl', rec)
        log(f"EPOCH {ep} val_loss {g['loss']:.4f} val_GLEU {gleu:.2f} (best {st['best_gleu']:.2f} @ {st['best_epoch']}, "
            f"es_ref {st['es_ref']:.2f} wait {st['wait']}) train {self.train_time / 60:.1f}m val {val_time / 60:.1f}m "
            f"mem {g['peak_mem_gb_per_rank']}GB stop={st['stop_reason'] or ('job_split' if rec.get('job_split_stop') else '-')}")

    def _save_best(self):
        if rank_zero():
            d = f'{self.run_dir}/ckpt/best'
            tmp = d + '.tmp'
            shutil.rmtree(tmp, ignore_errors=True)
            self.model.save_pretrained(tmp, safe_serialization=True)
            self.fmt.tok.save_pretrained(tmp)
            dump_json(f'{tmp}/best_info.json', {'epoch': self.current_epoch, 'val_gleu': self.st['best_gleu'], 'time_kst': now()})
            shutil.rmtree(d, ignore_errors=True)
            os.rename(tmp, d)
        self.trainer.strategy.barrier()

    def on_test_epoch_end(self):
        g = self._gather()
        if rank_zero():
            n = len(g['hyps'])
            speed = {'test_gen_time_s': round(g['gen_time_max_rank'], 1), 'test_total_time_s': round(time.time() - self.t_val, 1),
                     'sentences_per_s': round(n / g['gen_time_max_rank'], 2), 'gen_batch_size': self.args.gen_batch_size,
                     'n_gpus': self.trainer.world_size, 'num_beams': self.args.num_beams}
            s = evaluate(self.args.data_root, self.args.data, 'test', g['hyps'], f'{self.run_dir}/test', m2=True,
                         extra={'loss': g['loss'], 'parse': g['stats'], 'speed': speed,
                                'best_epoch': self.st.get('best_epoch'), 'best_val_gleu': self.st.get('best_gleu')},
                         limit=self.args.limit_test)
            log(f"TEST GLEU {s['gleu']:.2f} P {s['p']:.2f} R {s['r']:.2f} F0.5 {s['f05']:.2f} "
                f"({speed['sentences_per_s']} 문장/초, {speed['test_gen_time_s']}s)")


# ---------------------------------------------------------------- 실행

def build_parser(kind):
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='HF id 또는 로컬 디렉토리')
    ap.add_argument('--model_tag', default=None, help='출력 디렉토리 이름 (기본: 모델 경로의 마지막 이름)')
    ap.add_argument('--data', required=True, choices=['korean_learner', 'native', 'lang8', 'union'])
    ap.add_argument('--data_root', default=os.path.join(REPO_ROOT, '..', 'data', 'Preprocessed'))
    ap.add_argument('--out_root', default=os.path.join(REPO_ROOT, 'outputs2'))
    ap.add_argument('--run_id', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lr', type=float, required=True)
    ap.add_argument('--init_weights', default=None, help='시나리오 2: union 최고 체크포인트(ckpt/best) 디렉토리. 가중치만 가져옴')
    ap.add_argument('--global_batch', type=int, default=64)
    ap.add_argument('--micro_batch', type=int, default=64, help='GPU당 한 번에 넣는 batch. global = micro × GPU수 × accumulation')
    ap.add_argument('--gen_batch_size', type=int, default=64)
    ap.add_argument('--max_epochs', type=int, default=10)
    ap.add_argument('--warmup_ratio', type=float, default=0.1)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--precision', default='bf16-mixed')
    ap.add_argument('--num_beams', type=int, default=4)
    ap.add_argument('--gen_len_ratio', type=float, default=2.0, help='최대 생성 길이 = ceil(ratio × batch 내 최장 입력) + add')
    ap.add_argument('--gen_len_add', type=int, default=10)
    ap.add_argument('--max_gen_len', type=int, default=512)
    ap.add_argument('--devices', type=int, default=1)
    ap.add_argument('--accelerator', default='gpu', help='코드 검증용으로만 cpu 사용')
    ap.add_argument('--num_nodes', type=int, default=1)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--log_every', type=int, default=10)
    ap.add_argument('--limit_train', type=int, default=None, help='스모크용: 앞 N쌍만 사용')
    ap.add_argument('--limit_val', type=int, default=None)
    ap.add_argument('--limit_test', type=int, default=None)
    ap.add_argument('--epochs_per_job', type=int, default=None, help='이 작업에서 학습할 최대 epoch 수 (작업 분할 제출용)')
    ap.add_argument('--resume', action='store_true', help='run 디렉토리의 ckpt/last.ckpt에서 이어서 학습')
    ap.add_argument('--test_only', action='store_true', help='학습 없이 ckpt/best로 테스트만')
    ap.add_argument('--no_test', action='store_true')
    ap.add_argument('--gradient_checkpointing', action='store_true')
    if kind == 'seq2seq':
        ap.add_argument('--dropout', type=float, default=0.1, help='plan2 §3: seq2seq는 0.1로 통일')
        ap.add_argument('--early_stop', action='store_true', help='(seq2seq 기본: 조기 종료 없음)')
    else:
        ap.set_defaults(dropout=None)
        ap.add_argument('--no_early_stop', dest='early_stop', action='store_false')
        ap.set_defaults(early_stop=True)
    ap.add_argument('--min_delta', type=float, default=0.2)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--min_epochs', type=int, default=4)
    return ap


def main(kind):
    args = build_parser(kind).parse_args()
    args.kind = kind
    tag = args.model_tag or os.path.basename(args.model.rstrip('/'))
    run_dir = os.path.abspath(os.path.join(args.out_root, tag, args.data, args.run_id))
    world = args.devices * args.num_nodes
    assert args.global_batch % (args.micro_batch * world) == 0, 'global_batch는 micro_batch × GPU수의 배수여야 함'
    accum = args.global_batch // (args.micro_batch * world)
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision('high')

    model, tok, config = load_model(args, kind)
    if args.init_weights:
        init = AutoModelForSeq2SeqLM if kind == 'seq2seq' else AutoModelForCausalLM
        src = init.from_pretrained(args.init_weights)
        model.load_state_dict(src.state_dict())
        del src
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.config.use_cache = False
    fmt = Formatter(tok, config, kind)

    def pairs(split, limit):
        p = read_pairs(split_path(args.data_root, args.data, split))
        return p[:limit] if limit else p

    t0 = time.time()
    val_set = PairSet(pairs('val', args.limit_val), fmt, sort_by_len=True)
    test_set = PairSet(pairs('test', args.limit_test), fmt, sort_by_len=True)
    train_set = PairSet(pairs('train', args.limit_train), fmt) if not args.test_only else None
    log(f'데이터 인코딩 {time.time() - t0:.1f}s: train {len(train_set) if train_set else "-"} val {len(val_set)} test {len(test_set)}')

    if rank_zero():
        os.makedirs(run_dir, exist_ok=True)
        info = run_info(args)
        info['run_dir'] = run_dir
        dump_json(f'{run_dir}/run_info{"_resume_" + now().replace(" ", "_") if args.resume or args.test_only else ""}.json', info)
        n_params = sum(p.numel() for p in model.parameters())
        emb = model.get_input_embeddings().weight.numel()
        out_emb = model.get_output_embeddings()
        tied = out_emb is not None and out_emb.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
        n_emb = emb + (0 if tied or out_emb is None else out_emb.weight.numel())
        lens = {'max_src_tokens': {'train': train_set.max_src if train_set else None, 'val': val_set.max_src, 'test': test_set.max_src},
                'max_tgt_tokens': {'train': train_set.max_tgt if train_set else None, 'val': val_set.max_tgt, 'test': test_set.max_tgt}}
        dump_json(f'{run_dir}/model_info.json', {'model': args.model, 'model_type': config.model_type, 'n_params': n_params,
                                                  'n_params_non_embedding': n_params - n_emb, 'tied_embeddings': tied,
                                                  'dropout': dropout_report(model), 'format_wrap_bos_eos': fmt.wrap,
                                                  'gen_kwargs': fmt.gen_kwargs, 'pad_id': fmt.pad_id,
                                                  'eos_id': tok.eos_token_id, 'bos_id': tok.bos_token_id,
                                                  'decoder_start_token_id': getattr(config, 'decoder_start_token_id', None),
                                                  'prompt': PROMPT if kind == 'llm' else None, **lens})

    collate_tr, collate_ev = make_collate(fmt, gen=False), make_collate(fmt, gen=True)
    val_loader = DataLoader(val_set, batch_size=args.gen_batch_size, shuffle=False, collate_fn=collate_ev, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.gen_batch_size, shuffle=False, collate_fn=collate_ev, num_workers=args.num_workers)

    # 누설 검사 (학습 전, rank 0, eval 모드 fp32)
    if rank_zero() and not args.test_only:
        dev = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
        model.to(dev)
        chk = leak_check(model, fmt, collate_ev([val_set.items[-1], val_set.items[-2]]), dev)
        model.cpu()
        torch.cuda.empty_cache()
        dump_json(f'{run_dir}/check_leak.json', chk)
        log(f'누설 검사: ok={chk["ok"]} {chk}')
        if not chk['ok']:
            raise RuntimeError(f'누설 검사 실패: {chk}')

    ckpt_cb = ModelCheckpoint(dirpath=f'{run_dir}/ckpt', filename='epoch{epoch:02d}', save_top_k=0, save_last=True,
                              save_on_train_epoch_end=True, auto_insert_metric_name=False)

    def make_trainer(callbacks):
        strategy = DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True) if world > 1 else 'auto'
        return pl.Trainer(accelerator=args.accelerator, devices=args.devices, num_nodes=args.num_nodes, strategy=strategy,
                          precision=args.precision, max_epochs=args.max_epochs, accumulate_grad_batches=accum,
                          gradient_clip_val=args.clip, gradient_clip_algorithm='norm', num_sanity_val_steps=0,
                          logger=False, enable_progress_bar=False, enable_model_summary=False, callbacks=callbacks,
                          default_root_dir=run_dir, use_distributed_sampler=True)

    module = GecModule(args, model, fmt, run_dir, kind)
    if not args.test_only:
        train_loader = DataLoader(train_set, batch_size=args.micro_batch, shuffle=True, collate_fn=collate_tr,
                                  num_workers=args.num_workers, drop_last=False)
        trainer = make_trainer([ckpt_cb])
        last = f'{run_dir}/ckpt/last.ckpt'
        ckpt_path = last if args.resume and os.path.exists(last) else None
        if args.resume and ckpt_path is None:
            raise FileNotFoundError(last)
        if rank_zero():
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
            dump_json(f'{run_dir}/train_config{"_resume" if ckpt_path else ""}.json', cfg)
        log(f'학습 시작 run_dir={run_dir} accum={accum} resume={ckpt_path}')
        trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
        if rank_zero():
            dump_json(f'{run_dir}/train_config_applied.json', {**module.sched_info, 'accumulate_grad_batches': trainer.accumulate_grad_batches,
                                                               'precision_plugin': type(trainer.precision_plugin).__name__})
        state = dict(module.st)
        log(f'학습 종료: {state}')
        if rank_zero():
            dump_json(f'{run_dir}/train_state.json', state)
        if state['stop_reason'] is None:
            log('작업 분할로 중단: 다음 작업에서 --resume으로 이어서 실행')
            return
        del trainer, module, model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        state = json.load(open(f'{run_dir}/train_state.json'))
    if args.no_test:
        return
    best_dir = f'{run_dir}/ckpt/best'
    cls = AutoModelForSeq2SeqLM if kind == 'seq2seq' else AutoModelForCausalLM
    best = cls.from_pretrained(best_dir)
    tm = GecModule(args, best, Formatter(AutoTokenizer.from_pretrained(best_dir), best.config, kind), run_dir, kind)
    tm.st = state
    make_trainer([]).test(tm, dataloaders=test_loader)
