# 특수 토큰 전수 점검. 실행: 저장소 루트에서 python3 tools/check_special_tokens.py [root|legacy]
#   root  : 기준 = hyunwoongko/kobart (루트 코드),       교체 = skt/kobart-base-v1 + RobertaProcessing
#   legacy: 기준 = SKT kobart 레거시 토크나이저 파일(논문 실험 코드), 교체 = skt/kobart-base-v1 (후처리 없음)
import json, os, sys
import numpy as np
sys.path.insert(0, os.getcwd())
from transformers import PreTrainedTokenizerFast, BartConfig, BartForConditionalGeneration, GenerationConfig
from tokenizer_setup import load_tokenizer, check_model_config, special_token_report
from dataset import KoBARTGecDataset

RECIPE = sys.argv[1] if len(sys.argv) > 1 else 'root'
assert RECIPE in ('root', 'legacy')
ADD = RECIPE == 'root'
NEW = 'skt/kobart-base-v1'
ORIG = 'hyunwoongko/kobart' if ADD else os.path.abspath(os.environ.get('KOBART_LEGACY_DIR', '../.cache/kobart_legacy'))
print(f'recipe={RECIPE}, 기준={ORIG}')
DATA = os.path.abspath('../data/Preprocessed')
MAX_LEN = 128
fails = []

def check(name, ok, detail=''):
    print(f'  [{"OK" if ok else "FAIL"}] {name}' + (f' — {detail}' if detail else ''))
    if not ok:
        fails.append(name)

if ADD:
    orig = PreTrainedTokenizerFast.from_pretrained(ORIG)
else:  # SKT kobart 패키지 get_kobart_tokenizer()와 동일한 로딩
    orig = PreTrainedTokenizerFast(tokenizer_file=f'{ORIG}/emji_tokenizer/model.json', bos_token='<s>', eos_token='</s>',
                                   unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
new = load_tokenizer(NEW, add_bos_eos=ADD)  # 내부 검사 실패 시 여기서 예외

print('== 1. 토크나이저 구성 (post_processor 외 전 섹션)')
jo, jn = json.loads(orig.backend_tokenizer.to_str()), json.loads(new.backend_tokenizer.to_str())
for sec in ['normalizer', 'pre_tokenizer', 'decoder', 'added_tokens', 'truncation', 'padding']:
    check(f'{sec} 동일', jo.get(sec) == jn.get(sec), '' if jo.get(sec) == jn.get(sec) else f'원본 {jo.get(sec)} / 교체 {jn.get(sec)}')
mo, mn = jo['model'], jn['model']
check('model 타입·설정 동일', {k: v for k, v in mo.items() if k != 'vocab'} == {k: v for k, v in mn.items() if k != 'vocab'})
check('vocab(토큰→ID) 동일', orig.get_vocab() == new.get_vocab())
check('merges 동일', mo.get('merges') == mn.get('merges'))
check('post_processor 동일', jo['post_processor'] == jn['post_processor'], f'교체: {jn["post_processor"]}')
check('special_tokens_map 동일', orig.special_tokens_map == new.special_tokens_map)
check('all_special_ids 동일', orig.all_special_ids == new.all_special_ids, f'{new.all_special_ids}')
check('len(tokenizer) 동일', len(orig) == len(new), f'{len(orig)} / {len(new)}')

print('\n== 2. 모델 config·generation_config')
co = BartConfig.from_pretrained(ORIG if ADD else f'{ORIG}/kobart_from_pretrained')
cn = BartConfig.from_pretrained(NEW)
check_model_config(cn)
check('교체 config 특수 토큰 기대값', True, str(special_token_report(new, cn)['config']))
do, dn = co.to_dict(), cn.to_dict()
skip = {'_name_or_path', 'transformers_version', 'architectures'}
diff = {k: (do.get(k), dn.get(k)) for k in sorted(set(do) | set(dn)) if k not in skip and do.get(k) != dn.get(k)}
print('  config 차이 (원본, 교체):')
for k, v in diff.items():
    print(f'    {k}: {v}')
for name in ((ORIG if ADD else f'{ORIG}/kobart_from_pretrained'), NEW):
    try:
        g = GenerationConfig.from_pretrained(name).to_dict()
    except OSError:
        g = GenerationConfig.from_model_config(BartConfig.from_pretrained(name)).to_dict()
        g['_출처'] = 'config에서 유도'
    keys = ['decoder_start_token_id', 'bos_token_id', 'eos_token_id', 'pad_token_id', 'forced_bos_token_id',
            'forced_eos_token_id', 'num_beams', 'max_length', 'no_repeat_ngram_size', 'length_penalty',
            'early_stopping', 'repetition_penalty', '_출처']
    print(f'  generation_config {name}: ' + str({k: g.get(k) for k in keys if k in g}))
m = BartForConditionalGeneration.from_pretrained(NEW)
emb = m.get_input_embeddings().weight.shape[0]
check('임베딩 행 수 = vocab_size = len(tokenizer)', emb == cn.vocab_size == len(new), f'{emb} / {cn.vocab_size} / {len(new)}')
del m

print('\n== 3. 전체 데이터 인코딩 비교 (union = 세 데이터셋 전체)')
for sp in ['train', 'val', 'test']:
    path = f'{DATA}/union/union_{sp}.txt'
    ds_o = KoBARTGecDataset(path, orig, MAX_LEN)
    ds_n = KoBARTGecDataset(path, new, MAX_LEN)
    srcs = [s for s, t in ds_n.docs]; tgts = [t for s, t in ds_n.docs]
    eo_s, en_s = orig(srcs)['input_ids'], new(srcs)['input_ids']
    eo_t, en_t = orig(tgts)['input_ids'], new(tgts)['input_ids']
    bad = [i for i in range(len(srcs)) if eo_s[i] != en_s[i] or eo_t[i] != en_t[i]]
    check(f'{sp}: encode 전수 동일 ({len(srcs)}쌍)', not bad, f'불일치 {len(bad)}건 예: {bad[:5]}')
    # dataset.py __getitem__ 결과 전수 비교 (input_ids / decoder_input_ids / labels)
    bad_item = []
    for i in range(len(ds_n)):
        a, b = ds_o[i], ds_n[i]
        if any(not np.array_equal(a[k], b[k]) for k in a):
            bad_item.append(i)
    check(f'{sp}: dataset 출력 전수 동일', not bad_item, f'불일치 {len(bad_item)}건 예: {bad_item[:5]}')
    # 형식 점검: root는 encode 앞뒤 <s>/</s>, legacy는 특수 토큰 없음
    if ADD:
        fmt_bad = [i for i in range(len(srcs)) if en_s[i][0] != 0 or en_s[i][-1] != 1 or en_t[i][0] != 0 or en_t[i][-1] != 1]
        check(f'{sp}: encode 앞뒤 <s>/</s>', not fmt_bad, f'{len(fmt_bad)}건')
    body = (lambda e: e[1:-1]) if ADD else (lambda e: e)
    inner_special = sum(1 for e in en_s + en_t for x in body(e) if x in (0, 1, 3, 6))
    unk = sum(e.count(5) for e in en_s + en_t)
    ntok = sum(len(e) for e in en_s + en_t)
    print(f'  {sp}: 본문 내 특수토큰(0/1/3/6) {inner_special}개, <unk> {unk}개 ({unk / ntok:.5%})')
    over_s = sum(len(e) > MAX_LEN for e in en_s); over_t = sum(len(e) + 1 > MAX_LEN for e in en_t)
    print(f'  {sp}: {MAX_LEN} 토큰 초과(잘림) 소스 {over_s}건 ({over_s / len(srcs):.3%}), 라벨 {over_t}건 ({over_t / len(tgts):.3%})')
    dec_bad = sum(orig.decode(eo_t[i], skip_special_tokens=True) != new.decode(en_t[i], skip_special_tokens=True)
                  for i in range(len(tgts)))
    check(f'{sp}: decode(skip_special) 전수 동일', dec_bad == 0, f'{dec_bad}건')

print('\n== 4. 코드 하드코딩과의 정합성')
item = KoBARTGecDataset(f'{DATA}/native/native_val.txt', new, MAX_LEN)[0]
print('  예시 input_ids[:12]         ', item['input_ids'][:12].tolist())
print('  예시 decoder_input_ids[:12] ', item['decoder_input_ids'][:12].tolist())
print('  예시 labels[:12]            ', item['labels'][:12].tolist())
n_real = int((item['labels'] != -100).sum())
if ADD:
    check('labels 끝이 </s></s> (루트 원본 형식)', item['labels'][n_real - 2:n_real].tolist() == [1, 1])
else:
    check('labels 끝이 </s> 1개 (레거시 형식)', item['labels'][n_real - 1] == 1 and item['labels'][n_real - 2] != 1)
check('decoder_input_ids = [0] + labels[:-1]', item['decoder_input_ids'][1:n_real].tolist() == item['labels'][:n_real - 1].tolist() and item['decoder_input_ids'][0] == 0)

print('\n== 5. 디코더 첫 위치 미래 토큰 누설 검사 (model.make_decoder_attention_mask)')
import torch
from model import make_decoder_attention_mask
m = BartForConditionalGeneration.from_pretrained(NEW).eval()
it = KoBARTGecDataset(f'{DATA}/native/native_val.txt', new, MAX_LEN)[0]
enc = torch.tensor(it['input_ids'])[None]
d1 = torch.tensor(it['decoder_input_ids'])[None]
d2 = d1.clone(); n = int((d1 != 0).sum()) + 2
d2[0, 2:n] = torch.randint(7, 29000, (n - 2,))  # 첫 두 위치 이후 토큰만 교체
with torch.no_grad():
    lo = [m(input_ids=enc, attention_mask=enc.ne(0).float(), decoder_input_ids=d,
            decoder_attention_mask=make_decoder_attention_mask(d)).logits[0, 0] for d in (d1, d2)]
diff = (lo[0] - lo[1]).abs().max().item()
check('위치0 출력이 뒤쪽 디코더 토큰과 무관 (누설 없음)', diff < 1e-4, f'logits 최대차 {diff:.6f}')

print(f'\n결과: {"전부 통과" if not fails else f"실패 {len(fails)}건: {fails}"}')
sys.exit(1 if fails else 0)
