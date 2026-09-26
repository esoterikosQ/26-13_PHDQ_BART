# cmp_new.py / cmp_old.py 결과 비교 (저장소 루트에서 .venv로 실행)
import os, sys, json, torch
sys.path.insert(0, os.getcwd())
from tokenizer_setup import load_tokenizer

O = 'outputs/tfcmp'
new = json.load(open(f'{O}/gen_new.json'))
old = json.load(open(f'{O}/gen_old.json'))
ref = old['gen']
strip = lambda x: [t for t in x if t != 3]  # 배치 패딩(pad=3) 제거
for k in ('forced_eos_1', 'forced_eos_none'):
    same = sum(strip(a) == strip(b) for a, b in zip(new[k], ref))
    print(f'{k}: 4.0.0과 토큰 완전 일치 {same}/{len(ref)}')
print('첫 토큰 4.0.0:', sorted({x[0] for x in ref}), '/ 현재:', sorted({x[0] for x in new['forced_eos_1']}))
lo = torch.load(f'{O}/logits_old.bin')
ln = torch.load(f'{O}/logits_new.bin')
print('forward logits 최대 절대차:', (lo - ln).abs().max().item(), '/ loss 4.0.0', old['loss'], '현재', new['loss'])
tok = load_tokenizer(add_bos_eos=False)
diff = [(i, a, b) for i, (a, b) in enumerate(zip(new['forced_eos_1'], ref)) if strip(a) != strip(b)]
for i, a, b in diff[:3]:
    print(i, '\n  현재 :', tok.decode(strip(a), skip_special_tokens=True), '\n  4.0.0:', tok.decode(strip(b), skip_special_tokens=True))
