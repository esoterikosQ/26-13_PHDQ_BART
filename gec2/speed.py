# plan2.md 원칙 6 / §5 단계 6: 추론 속도 측정 (논문 Table 6 방식: 전체 문장 수 ÷ 전체 생성 시간, 문장/초).
# 모든 모델을 같은 GPU 1장, 같은 생성 설정(빔 4, 샘플링 없음, 생성 batch 동일, 학습 코드와 같은 최대 길이 규칙)으로 잰다.
# 체크포인트: 각 데이터셋의 시나리오 1 seed 0 최고 검증 체크포인트(ckpt/best). 문장은 길이순으로 묶는다(학습 코드의 평가와 같음).
# 사용 (저장소 루트, GPU를 비운 상태에서): python3 -m gec2.speed [--gen_batch_size 64] [--out logs/gec2_speed]
import argparse
import json
import math
import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from gec2.core import Formatter, PairSet, make_collate
from gec2.evaluate import REPO_ROOT, evaluate, read_pairs, split_path

# (이름, 원본 모델(토크나이저), 결과 루트, 시나리오 1 lr, 종류)
MODELS = [
    ('KoBART(gec2)', 'skt/kobart-base-v1', 'outputs2/kobart-base-v1', '5e-5', 'seq2seq'),
    ('pko-t5-base', 'paust/pko-t5-base', 'outputs2/pko-t5-base', '5e-4', 'seq2seq'),
    ('mBART-50', 'facebook/mbart-large-50', 'outputs2/mbart-large-50', '3e-5', 'seq2seq'),
    ('pko-t5-large', 'paust/pko-t5-large', 'outputs2/pko-t5-large', '1e-4', 'seq2seq'),
    ('kanana-1.5-2.1b', 'kakaocorp/kanana-1.5-2.1b-instruct-2505', 'neuron_outputs/kanana-1.5-2.1b', '1e-5', 'llm'),
]
DATAS = ('korean_learner', 'native', 'lang8', 'union')


@torch.no_grad()
def generate(model, fmt, batch, kind, a):
    """core.GecModule._generate와 같은 규칙."""
    if kind == 'seq2seq':
        ids, mask, plen = batch['input_ids'], batch['attention_mask'], 0
    else:
        ids, mask = batch['gen_input_ids'], batch['gen_attention_mask']
        plen = ids.shape[1]
    ids, mask = ids.cuda(), mask.cuda()
    max_new = min(a.max_gen_len, math.ceil(a.gen_len_ratio * int(mask.sum(1).max())) + a.gen_len_add)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        seqs = model.generate(input_ids=ids, attention_mask=mask, num_beams=a.num_beams, do_sample=False,
                              max_new_tokens=max_new, early_stopping=True, use_cache=True, **fmt.gen_kwargs)
    if kind == 'seq2seq':
        seqs = seqs[:, 1:]
    return fmt.decode(seqs.cpu(), plen)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default=os.path.join(REPO_ROOT, '..', 'data', 'Preprocessed'))
    ap.add_argument('--out', default=os.path.join(REPO_ROOT, 'logs', 'gec2_speed'))
    ap.add_argument('--gen_batch_size', type=int, default=64)
    ap.add_argument('--num_beams', type=int, default=4)
    ap.add_argument('--gen_len_ratio', type=float, default=2.0)
    ap.add_argument('--gen_len_add', type=int, default=10)
    ap.add_argument('--max_gen_len', type=int, default=512)
    ap.add_argument('--models', nargs='*', default=None, help='이름 일부로 거르기')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_float32_matmul_precision('high')
    rows = []
    for name, base, root, lr, kind in MODELS:
        if a.models and not any(m in name for m in a.models):
            continue
        tok = AutoTokenizer.from_pretrained(base)
        tot_n, tot_t = 0, 0.0
        for d in DATAS:
            ckpt = os.path.join(REPO_ROOT, root, d, f'lr{lr}_seed0', 'ckpt', 'best')
            cls = AutoModelForSeq2SeqLM if kind == 'seq2seq' else AutoModelForCausalLM
            model = cls.from_pretrained(ckpt).cuda().eval()
            fmt = Formatter(tok, model.config, kind)
            ps = PairSet(read_pairs(split_path(a.data_root, d, 'test')), fmt, sort_by_len=True)
            loader = DataLoader(ps, batch_size=a.gen_batch_size, shuffle=False, collate_fn=make_collate(fmt, gen=True))
            generate(model, fmt, next(iter(loader)), kind, a)   # 예열(시간에서 제외)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            hyp, t_gen = {}, 0.0
            for b in loader:
                torch.cuda.synchronize()
                t = time.time()
                out = generate(model, fmt, b, kind, a)
                torch.cuda.synchronize()
                t_gen += time.time() - t
                hyp.update(zip(b['idx'].tolist(), out))
            n = len(hyp)
            s = evaluate(a.data_root, d, 'test', [hyp[i] for i in range(n)], os.path.join(a.out, name, d), m2=False)
            r = {'model': name, 'data': d, 'n': n, 'gen_time_s': round(t_gen, 2), 'sentences_per_s': round(n / t_gen, 2),
                 'peak_mem_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), 'gleu_check': s['gleu'], 'ckpt': ckpt}
            rows.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
            tot_n += n
            tot_t += t_gen
            del model
            torch.cuda.empty_cache()
        rows.append({'model': name, 'data': 'ALL', 'n': tot_n, 'gen_time_s': round(tot_t, 2), 'sentences_per_s': round(tot_n / tot_t, 2)})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    info = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'gen_batch_size': a.gen_batch_size,
            'num_beams': a.num_beams, 'max_new_tokens': f'min({a.max_gen_len}, ceil({a.gen_len_ratio} × batch 최장 입력) + {a.gen_len_add})',
            'precision': 'fp32 가중치 + bf16 autocast (학습 코드의 평가와 같음)', 'order': '길이 내림차순 batch',
            'rows': rows}
    with open(os.path.join(a.out, 'speed.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print('| 모델 | ' + ' | '.join(DATAS) + ' | 합계 (논문 방식) |')
    print('| --- | ' + ' | '.join('---' for _ in DATAS) + ' | --- |')
    for name, *_ in MODELS:
        rr = {r['data']: r for r in rows if r['model'] == name}
        if rr:
            print(f'| {name} | ' + ' | '.join(f"{rr[d]['sentences_per_s']:.1f}" for d in DATAS) + f" | {rr['ALL']['sentences_per_s']:.1f} |")


if __name__ == '__main__':
    main()
