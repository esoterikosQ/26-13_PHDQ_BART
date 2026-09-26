# plan2.md §4 사전 점검 (모델마다): 원문 복원, <unk> 비율, 길이 분포, 특수 토큰·마스크(누설 검사).
# 학습 모드 검사는 학습 첫 배치에서 자동 수행된다(check_train_mode.json). 제출 전 로그인 노드나 짧은 작업에서도 실행 가능.
# 사용 (저장소 루트에서): python3 -m gec2.precheck --model <HF id 또는 로컬 경로> --kind seq2seq|llm [--leak] [--out outputs2/precheck]
import argparse
import collections
import json
import os
import unicodedata

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

from gec2.core import PROMPT, Formatter, PairSet, dropout_report, leak_check, load_model, make_collate
from gec2.evaluate import REPO_ROOT, read_pairs, split_path

DATAS = ('korean_learner', 'native', 'lang8', 'union')


def classify(x, d):
    if d == x:
        return 'same'
    if ' '.join(d.split()) == ' '.join(x.split()):
        return 'whitespace'
    if d == unicodedata.normalize('NFKC', x) or ' '.join(d.split()) == ' '.join(unicodedata.normalize('NFKC', x).split()):
        return 'nfkc'
    return 'other'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--model_tag', default=None)
    ap.add_argument('--kind', required=True, choices=['seq2seq', 'llm'])
    ap.add_argument('--data_root', default=os.path.join(REPO_ROOT, '..', 'data', 'Preprocessed'))
    ap.add_argument('--out', default=os.path.join(REPO_ROOT, 'outputs2', 'precheck'))
    ap.add_argument('--leak', action='store_true', help='모델 가중치를 올려 누설 검사까지 수행 (GPU 있으면 GPU)')
    ap.add_argument('--dropout', type=float, default=None)
    a = ap.parse_args()
    tag = a.model_tag or os.path.basename(a.model.rstrip('/'))
    tok = AutoTokenizer.from_pretrained(a.model)
    config = AutoConfig.from_pretrained(a.model)
    fmt = Formatter(tok, config, a.kind)
    res = {'model': a.model, 'kind': a.kind, 'model_type': config.model_type, 'tokenizer': type(tok).__name__,
           'vocab': len(tok), 'special': {k: (getattr(tok, k), getattr(tok, k + '_id')) for k in ('pad_token', 'eos_token', 'bos_token', 'unk_token')},
           'decoder_start_token_id': getattr(config, 'decoder_start_token_id', None), 'format_wrap_bos_eos': fmt.wrap,
           'gen_kwargs': fmt.gen_kwargs}
    ex_s, ex_t = '지금부터 방콕에 소개한다.', '지금부터 방콕을 소개하겠다.'
    res['example'] = {'src_ids': fmt.enc_src([ex_s])[0], 'tgt_ids': fmt.enc_tgt([ex_t])[0]}
    res['example']['src_tokens'] = tok.convert_ids_to_tokens(res['example']['src_ids'])
    res['example']['tgt_tokens'] = tok.convert_ids_to_tokens(res['example']['tgt_ids'])

    # 원문 복원·unk: 전체 데이터의 고유 문장
    texts = set()
    for d in DATAS:
        for sp in ('train', 'val', 'test'):
            for s, t in read_pairs(split_path(a.data_root, d, sp)):
                texts.add(s)
                texts.add(t)
    texts = sorted(texts)
    ids = tok(texts, add_special_tokens=False)['input_ids']
    dec = tok.batch_decode(ids, skip_special_tokens=False)
    cats, examples = collections.Counter(), collections.defaultdict(list)
    for x, dd in zip(texts, dec):
        c = classify(x, dd)
        cats[c] += 1
        if c != 'same' and len(examples[c]) < 5:
            examples[c].append({'orig': x, 'decoded': dd})
    unk = tok.unk_token_id
    n_tok = sum(len(i) for i in ids)
    n_unk = sum(i.count(unk) for i in ids) if unk is not None else 0
    res['roundtrip'] = {'n_unique_texts': len(texts), 'categories': dict(cats),
                        'same_ratio': cats['same'] / len(texts), 'examples': examples}
    res['unk'] = {'unk_id': unk, 'n_unk': n_unk, 'n_tokens': n_tok, 'ratio': n_unk / max(n_tok, 1),
                  'note': 'unk_id가 pad_id와 같으면(pko-t5) 텍스트 인코딩에 나온 수만 셈' if unk is not None and unk == tok.pad_token_id else None}

    # 길이 분포 (모델 입력 형식 그대로, 특수 토큰·프롬프트 포함)
    res['lengths'] = {}
    for d in DATAS:
        row = {}
        for sp in ('train', 'val', 'test'):
            pairs = read_pairs(split_path(a.data_root, d, sp))
            s = np.array([len(x) for x in fmt.enc_src([p[0] for p in pairs])])
            t = np.array([len(x) for x in fmt.enc_tgt([p[1] for p in pairs])])
            row[sp] = {'src_p99.9': float(np.percentile(s, 99.9)), 'src_max': int(s.max()),
                       'tgt_p99.9': float(np.percentile(t, 99.9)), 'tgt_max': int(t.max()),
                       'tgt/src_max_ratio': float((t / s).max()), 'src_mean': float(s.mean())}
            if a.kind == 'llm':
                row[sp]['total_max'] = int((s + t).max())
                row[sp]['total_p99.9'] = float(np.percentile(s + t, 99.9))
        res['lengths'][d] = row
    res['prompt'] = PROMPT if a.kind == 'llm' else None

    if a.leak:
        class A:  # load_model 인자
            model, dropout = a.model, a.dropout
        model, tok2, _ = load_model(A, a.kind)
        fmt2 = Formatter(tok2, model.config, a.kind)
        vs = PairSet(read_pairs(split_path(a.data_root, 'korean_learner', 'val'))[:8], fmt2)
        batch = make_collate(fmt2, gen=True)(vs.items[:2])
        dev = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
        model.to(dev)
        res['leak_check'] = leak_check(model, fmt2, batch, dev)
        res['dropout'] = dropout_report(model)
        res['n_params'] = sum(p.numel() for p in model.parameters())
        # 학습 전 모델의 생성 형식 확인 (빔 4, 2문장)
        model.eval()
        with torch.no_grad():
            if a.kind == 'seq2seq':
                ids_, mask, plen = batch['input_ids'].to(dev), batch['attention_mask'].to(dev), 0
            else:
                ids_, mask = batch['gen_input_ids'].to(dev), batch['gen_attention_mask'].to(dev)
                plen = ids_.shape[1]
            out = model.generate(input_ids=ids_, attention_mask=mask, num_beams=4, do_sample=False, max_new_tokens=64, **fmt2.gen_kwargs)
            if a.kind == 'seq2seq':
                out = out[:, 1:]
            res['untrained_generation'] = fmt2.decode(out.cpu(), plen)[0]
    os.makedirs(a.out, exist_ok=True)
    with open(f'{a.out}/{tag}.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
    summ = {'model': a.model, 'roundtrip_same': f"{res['roundtrip']['same_ratio']:.4%}", 'roundtrip_cats': dict(cats),
            'unk': n_unk, 'max_len': {d: {k: max(res['lengths'][d][sp][k] for sp in res['lengths'][d]) for k in ('src_max', 'tgt_max')} for d in DATAS},
            'leak_ok': res.get('leak_check', {}).get('ok')}
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == '__main__':
    main()
