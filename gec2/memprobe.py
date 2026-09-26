# 최장 문장 batch로 학습 1 step + 빔 생성 1회의 최대 GPU 메모리를 잰다 (micro_batch·gen_batch_size 결정용).
# 사용: python3 -m gec2.memprobe --model <id|경로> --kind seq2seq|llm --micro_batch 64 --gen_batch_size 64 [--gradient_checkpointing]
import argparse
import types

import torch

from gec2.core import Formatter, PairSet, load_model, make_collate
from gec2.evaluate import REPO_ROOT, read_pairs, split_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--kind', required=True, choices=['seq2seq', 'llm'])
    ap.add_argument('--data_root', default=f'{REPO_ROOT}/../data/Preprocessed')
    ap.add_argument('--micro_batch', type=int, default=64)
    ap.add_argument('--gen_batch_size', type=int, default=64)
    ap.add_argument('--gradient_checkpointing', action='store_true')
    a = ap.parse_args()
    args = types.SimpleNamespace(model=a.model, dropout=0.1 if a.kind == 'seq2seq' else None)
    model, tok, cfg = load_model(args, a.kind)
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    fmt = Formatter(tok, cfg, a.kind)
    pairs = read_pairs(split_path(a.data_root, 'union', 'train'))
    ps = PairSet(pairs, fmt, sort_by_len=True)   # union은 모든 데이터를 포함 → 최장 문장
    dev = torch.device('cuda')
    model.to(dev).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    b = make_collate(fmt, gen=True)(ps.items[:a.micro_batch])
    torch.cuda.reset_peak_memory_stats()
    # 2 step: 두 번째 step은 옵티마이저 상태가 이미 있는 상태에서 순전파·역전파 (실제 학습의 최대치)
    for _ in range(2):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(input_ids=b['input_ids'].to(dev), attention_mask=b['attention_mask'].to(dev), labels=b['labels'].to(dev)).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    train_peak = torch.cuda.max_memory_allocated() / 2 ** 30
    model.eval()
    g = make_collate(fmt, gen=True)(ps.items[:a.gen_batch_size])
    ids = g['input_ids'] if a.kind == 'seq2seq' else g['gen_input_ids']
    mask = g['attention_mask'] if a.kind == 'seq2seq' else g['gen_attention_mask']
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        model.generate(input_ids=ids.to(dev), attention_mask=mask.to(dev), num_beams=4, do_sample=False,
                       max_new_tokens=int(2 * mask.sum(1).max()) + 10, use_cache=True, **fmt.gen_kwargs)
    gen_peak = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f'{a.model} micro_batch {a.micro_batch} (최장 입력 {b["input_ids"].shape[1]} 토큰) 학습 최대 {train_peak:.1f}GB / '
          f'gen_batch {a.gen_batch_size} 생성 최대 {gen_peak:.1f}GB / GPU {torch.cuda.get_device_properties(0).total_memory / 2 ** 30:.0f}GB')


if __name__ == '__main__':
    main()
