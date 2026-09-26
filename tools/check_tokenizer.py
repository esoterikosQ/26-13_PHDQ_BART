# §3-2 / §3-3 검증 스크립트: skt/kobart-base-v1 토크나이저·config·모델 스모크 테스트
import torch
from transformers import PreTrainedTokenizerFast, BartConfig, BartForConditionalGeneration

NEW, ORIG = 'skt/kobart-base-v1', 'hyunwoongko/kobart'
SAMPLES = ['한국어 문장을 고칩니다.', '띄어쓰기 와 문장부호!']

print('=== [게이트] skt 토크나이저 특수 토큰')
tok = PreTrainedTokenizerFast.from_pretrained(NEW)
print(f'pad_token_id: {tok.pad_token_id} (기대 3)')
print(f'eos_token_id: {tok.eos_token_id} (기대 1)')
print(f'bos_token_id: {tok.bos_token_id} (기대 0)')
print(f'vocab_size:   {tok.vocab_size} (기대 30000)')
print(f'bos/eos 문자열: {tok.bos_token!r} / {tok.eos_token!r} (model.py 기대 <s> / </s>)')

print('\n=== [게이트] skt config')
for name in (NEW, ORIG):
    cfg = BartConfig.from_pretrained(name)
    print(f'{name}: bos={cfg.bos_token_id} eos={cfg.eos_token_id} pad={cfg.pad_token_id} '
          f'decoder_start={cfg.decoder_start_token_id} forced_bos={cfg.forced_bos_token_id} '
          f'forced_eos={cfg.forced_eos_token_id} vocab={cfg.vocab_size}')

print('\n=== [G3] encode가 붙이는 특수 토큰 (dataset.py: tok.encode(source))')
for s in SAMPLES:
    ids = tok.encode(s)
    print(s, ids, '| ID 0 포함:', 0 in ids, '| 디코드:', tok.convert_ids_to_tokens(ids))

print('\n=== [참고 로그] 원본 토크나이저와 비교 (중단 조건 아님)')
orig = PreTrainedTokenizerFast.from_pretrained(ORIG)
print('vocab 동일:', orig.get_vocab() == tok.get_vocab())
print('special_tokens_map 동일:', orig.special_tokens_map == tok.special_tokens_map)
print('원본 special_tokens_map:', orig.special_tokens_map)
print('skt  special_tokens_map:', tok.special_tokens_map)
for s in SAMPLES:
    print(s, '인코딩 동일:', orig.encode(s) == tok.encode(s), orig.encode(s), tok.encode(s))

print('\n=== [§3-3] 모델 로딩 + generate 스모크 테스트')
config = BartForConditionalGeneration.from_pretrained(NEW).config
model = BartForConditionalGeneration.from_pretrained(NEW, config=config).cuda().eval()
print(f'파라미터 수: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M (기대 약 123M)')
inp = torch.tensor([tok.encode(SAMPLES[0])]).cuda()
with torch.no_grad():
    out = model.generate(inp, eos_token_id=1, max_length=128, num_beams=4, repetition_penalty=2.0)
print('생성 ID:', out[0].tolist())
print('생성 첫 토큰(decoder_start):', out[0][0].item())
print('디코드:', tok.decode(out[0], skip_special_tokens=True))
