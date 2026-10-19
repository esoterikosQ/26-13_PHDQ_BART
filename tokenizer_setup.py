'''
skt/kobart-base-v1 토크나이저 로딩 및 특수 토큰 검사.

- add_bos_eos=True (recipe root): 루트 코드가 쓰던 hyunwoongko/kobart와 같은 RobertaProcessing
  후처리를 붙여 encode() 결과를 [<s>=0] + ids + [</s>=1]로 만든다.
- add_bos_eos=False (recipe legacy): 논문 실험 코드(src/KoBART-gec)가 쓴 SKT kobart 토크나이저와
  같다. 후처리가 없어 encode()가 특수 토큰을 붙이지 않는다 (skt/kobart-base-v1 그대로).
'''
from tokenizers.processors import RobertaProcessing  # <s> ... </s> 형태로 감싸는 후처리기
from transformers import PreTrainedTokenizerFast  # Rust 기반 고속 토크나이저 래퍼

# 기본으로 불러올 모델/토크나이저 이름
BASE_MODEL = 'skt/kobart-base-v1'
 
# (토큰 문자열, ID) — model.py/dataset.py 하드코딩(pad_index=0, eos_token_id=1)의 전제
EXPECTED_SPECIAL_TOKENS = {
    # 시작 토큰
    'bos': ('<s>', 0),
    # 끝 토큰
    'eos': ('</s>', 1),
    # 패딩 토큰 (주의: dataset.py의 pad_index 기본값 0과는 다름)
    'pad': ('<pad>', 3),
    # 미등록 토큰
    'unk': ('<unk>', 5),
    # 마스크 토큰
    'mask': ('<mask>', 6),
}
# 모델 config가 가져야 하는 특수 토큰 관련 기대값
EXPECTED_CONFIG = {'bos_token_id': 0, 'eos_token_id': 1, 'pad_token_id': 3,
                   'decoder_start_token_id': 1, 'forced_bos_token_id': None,
                   'forced_eos_token_id': 1, 'vocab_size': 30000}


def load_tokenizer(name=BASE_MODEL, add_bos_eos=True):
    # 사전학습 토크나이저 로드 (tokenizer.json 기반)
    tok = PreTrainedTokenizerFast.from_pretrained(name)
    # 시작/끝 토큰을 자동으로 붙이려면
    if add_bos_eos:
        # 내부 Rust 토크나이저의 후처리기를 교체 (encode 결과가 [0, ..., 1]이 됨)
        tok._tokenizer.post_processor = RobertaProcessing(
            sep=('</s>', 1), cls=('<s>', 0), trim_offsets=True, add_prefix_space=False)
    # 특수 토큰 ID와 encode 형식이 기대와 다르면 여기서 예외 발생
    check_tokenizer(tok, add_bos_eos)
    # 검사를 통과한 토크나이저 반환
    return tok


def check_tokenizer(tok, add_bos_eos=True):
    # 불일치 항목을 모아 둔 리스트
    errors = []
    # 기대하는 특수 토큰마다 확인
    for key, (token, idx) in EXPECTED_SPECIAL_TOKENS.items():
        # 토큰 문자열과 ID를 속성에서 읽음 (예: tok.bos_token, tok.bos_token_id)
        got = (getattr(tok, f'{key}_token'), getattr(tok, f'{key}_token_id'))
        # 기대값과 다르면 오류로 기록
        if got != (token, idx):
            errors.append(f'{key}: 기대 {(token, idx)}, 실제 {got}')
    # 어휘 크기 확인
    if tok.vocab_size != 30000:
        errors.append(f'vocab_size: 기대 30000, 실제 {tok.vocab_size}')
    # 샘플 문장을 인코딩해 형식 확인
    ids = tok.encode('문장')
    # bos/eos 모드: 앞이 0, 뒤가 1이고 중간에는 0/1이 없어야 함
    if add_bos_eos and (ids[0] != 0 or ids[-1] != 1 or 0 in ids[1:-1] or 1 in ids[1:-1]):
        errors.append(f'encode 형식: 기대 [0, ..., 1], 실제 {ids}')
    # 후처리 없는 모드: 특수 토큰 ID(0,1,3,5,6)가 하나도 없어야 함
    if not add_bos_eos and any(x in (0, 1, 3, 5, 6) for x in ids):
        errors.append(f'encode 형식: 기대 특수 토큰 없음, 실제 {ids}')
    # 오류가 하나라도 있으면 모아서 한 번에 예외 발생
    if errors:
        raise ValueError('토크나이저 특수 토큰 검사 실패:\n  ' + '\n  '.join(errors))


def check_model_config(config):
    # 기대값과 다른 config 속성만 메시지로 수집 (속성이 없으면 None으로 비교)
    errors = [f'config.{k}: 기대 {v}, 실제 {getattr(config, k, None)}'
              for k, v in EXPECTED_CONFIG.items() if getattr(config, k, None) != v]
    # 불일치가 있으면 예외 발생
    if errors:
        raise ValueError('모델 config 특수 토큰 검사 실패:\n  ' + '\n  '.join(errors))


def special_token_report(tok, config, args=None):
    '''학습 로그에 남길 요약. 결과가 이상할 때 이 줄부터 비교한다.'''
    # args가 있으면 생성 관련 설정값만 추출, 없으면 빈 dict
    gen = {} if args is None else {k: getattr(args, k) for k in
                                   ('recipe', 'decoder_start_token_id', 'forced_eos_token_id', 'repetition_penalty')}
    return {
        # 생성 설정
        'generate': gen,
        # 토크나이저의 특수 토큰 문자열과 ID
        'tokenizer': {k: (getattr(tok, f'{k}_token'), getattr(tok, f'{k}_token_id'))
                      for k in EXPECTED_SPECIAL_TOKENS},
        # 후처리기 클래스 이름 (없으면 None)
        'post_processor': type(tok.backend_tokenizer.post_processor).__name__ if tok.backend_tokenizer.post_processor else None,
        # 실제 인코딩 결과 샘플
        'encode_sample': tok.encode('한국어 문장을 고칩니다.'),
        # config의 특수 토큰 관련 값
        'config': {k: getattr(config, k, None) for k in EXPECTED_CONFIG},
    }
