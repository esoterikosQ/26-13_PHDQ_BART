'''
skt/kobart-base-v1 토크나이저 로딩 및 특수 토큰 검사.

- add_bos_eos=True (recipe root): 루트 코드가 쓰던 hyunwoongko/kobart와 같은 RobertaProcessing
  후처리를 붙여 encode() 결과를 [<s>=0] + ids + [</s>=1]로 만든다.
- add_bos_eos=False (recipe legacy): 논문 실험 코드(src/KoBART-gec)가 쓴 SKT kobart 토크나이저와
  같다. 후처리가 없어 encode()가 특수 토큰을 붙이지 않는다 (skt/kobart-base-v1 그대로).
'''
from tokenizers.processors import RobertaProcessing
from transformers import PreTrainedTokenizerFast

BASE_MODEL = 'skt/kobart-base-v1'

# (토큰 문자열, ID) — model.py/dataset.py 하드코딩(pad_index=0, eos_token_id=1)의 전제
EXPECTED_SPECIAL_TOKENS = {
    'bos': ('<s>', 0),
    'eos': ('</s>', 1),
    'pad': ('<pad>', 3),
    'unk': ('<unk>', 5),
    'mask': ('<mask>', 6),
}
EXPECTED_CONFIG = {'bos_token_id': 0, 'eos_token_id': 1, 'pad_token_id': 3,
                   'decoder_start_token_id': 1, 'forced_bos_token_id': None,
                   'forced_eos_token_id': 1, 'vocab_size': 30000}


def load_tokenizer(name=BASE_MODEL, add_bos_eos=True):
    tok = PreTrainedTokenizerFast.from_pretrained(name)
    if add_bos_eos:
        tok._tokenizer.post_processor = RobertaProcessing(
            sep=('</s>', 1), cls=('<s>', 0), trim_offsets=True, add_prefix_space=False)
    check_tokenizer(tok, add_bos_eos)
    return tok


def check_tokenizer(tok, add_bos_eos=True):
    errors = []
    for key, (token, idx) in EXPECTED_SPECIAL_TOKENS.items():
        got = (getattr(tok, f'{key}_token'), getattr(tok, f'{key}_token_id'))
        if got != (token, idx):
            errors.append(f'{key}: 기대 {(token, idx)}, 실제 {got}')
    if tok.vocab_size != 30000:
        errors.append(f'vocab_size: 기대 30000, 실제 {tok.vocab_size}')
    ids = tok.encode('문장')
    if add_bos_eos and (ids[0] != 0 or ids[-1] != 1 or 0 in ids[1:-1] or 1 in ids[1:-1]):
        errors.append(f'encode 형식: 기대 [0, ..., 1], 실제 {ids}')
    if not add_bos_eos and any(x in (0, 1, 3, 5, 6) for x in ids):
        errors.append(f'encode 형식: 기대 특수 토큰 없음, 실제 {ids}')
    if errors:
        raise ValueError('토크나이저 특수 토큰 검사 실패:\n  ' + '\n  '.join(errors))


def check_model_config(config):
    errors = [f'config.{k}: 기대 {v}, 실제 {getattr(config, k, None)}'
              for k, v in EXPECTED_CONFIG.items() if getattr(config, k, None) != v]
    if errors:
        raise ValueError('모델 config 특수 토큰 검사 실패:\n  ' + '\n  '.join(errors))


def special_token_report(tok, config, args=None):
    '''학습 로그에 남길 요약. 결과가 이상할 때 이 줄부터 비교한다.'''
    gen = {} if args is None else {k: getattr(args, k) for k in
                                   ('recipe', 'decoder_start_token_id', 'forced_eos_token_id', 'repetition_penalty')}
    return {
        'generate': gen,
        'tokenizer': {k: (getattr(tok, f'{k}_token'), getattr(tok, f'{k}_token_id'))
                      for k in EXPECTED_SPECIAL_TOKENS},
        'post_processor': type(tok.backend_tokenizer.post_processor).__name__ if tok.backend_tokenizer.post_processor else None,
        'encode_sample': tok.encode('한국어 문장을 고칩니다.'),
        'config': {k: getattr(config, k, None) for k in EXPECTED_CONFIG},
    }
