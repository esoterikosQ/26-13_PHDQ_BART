# 특수 토큰 전수 점검. 실행: 저장소 루트에서 python3 tools/check_special_tokens.py [root|legacy]
#   root  : 기준 = hyunwoongko/kobart (루트 코드),       교체 = skt/kobart-base-v1 + RobertaProcessing
#   legacy: 기준 = SKT kobart 레거시 토크나이저 파일(논문 실험 코드), 교체 = skt/kobart-base-v1 (후처리 없음)
import json, os, sys  # tokenizer.json 파싱, 경로/환경변수, 인자/종료코드
import numpy as np  # 배열 동일성 비교

# 저장소 루트를 모듈 검색 경로 맨 앞에 추가 (tokenizer_setup, dataset, model import용)
sys.path.insert(0, os.getcwd())

# HF 토크나이저/모델/생성 설정 클래스
from transformers import PreTrainedTokenizerFast, BartConfig, BartForConditionalGeneration, GenerationConfig
# 토크나이저 로드와 특수 토큰 검사 함수
from tokenizer_setup import load_tokenizer, check_model_config, special_token_report
# 학습에 쓰는 데이터셋 클래스
from dataset import KoBARTGecDataset

# 첫 번째 인자로 레시피 선택 (없으면 root)
RECIPE = sys.argv[1] if len(sys.argv) > 1 else 'root'
# root/legacy 외의 값이면 중단
assert RECIPE in ('root', 'legacy')
# root일 때만 토크나이저에 bos/eos 후처리기를 붙임
ADD = RECIPE == 'root'
# 교체 대상(새로 쓸) 모델/토크나이저
NEW = 'skt/kobart-base-v1'
# 비교 기준: root는 HF 허브 모델, legacy는 로컬 레거시 디렉터리(환경변수로 지정 가능)
ORIG = 'hyunwoongko/kobart' if ADD else os.path.abspath(os.environ.get('KOBART_LEGACY_DIR', '../.cache/kobart_legacy'))
# 실행 설정 출력
print(f'recipe={RECIPE}, 기준={ORIG}')

# 전처리 데이터 루트 경로
DATA = os.path.abspath('../data/Preprocessed')
# 데이터셋의 최대 시퀀스 길이
MAX_LEN = 128
# 실패한 검사 항목 이름 모음
fails = []

def check(name, ok, detail=''):
    # [OK]/[FAIL] 표시와 상세 설명 출력
    print(f'  [{"OK" if ok else "FAIL"}] {name}' + (f' — {detail}' if detail else ''))
    # 실패하면 목록에 기록 (마지막에 종료코드에 반영)
    if not ok:
        fails.append(name)

# 기준 토크나이저 로드
if ADD:
    # root: HF 허브에서 바로 로드
    orig = PreTrainedTokenizerFast.from_pretrained(ORIG)
else:  # SKT kobart 패키지 get_kobart_tokenizer()와 동일한 로딩
    # legacy: 로컬 tokenizer 파일에서 특수 토큰을 직접 지정해 생성
    orig = PreTrainedTokenizerFast(tokenizer_file=f'{ORIG}/emji_tokenizer/model.json', bos_token='<s>', eos_token='</s>',
                                   unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
# 교체 토크나이저 로드
new = load_tokenizer(NEW, add_bos_eos=ADD)  # 내부 검사 실패 시 여기서 예외

print('== 1. 토크나이저 구성 (post_processor 외 전 섹션)')
# 두 토크나이저의 전체 설정을 JSON dict로 변환
jo, jn = json.loads(orig.backend_tokenizer.to_str()), json.loads(new.backend_tokenizer.to_str())
# 어휘/병합 규칙을 제외한 하위 섹션을 하나씩 비교
for sec in ['normalizer', 'pre_tokenizer', 'decoder', 'added_tokens', 'truncation', 'padding']:
    # 동일하면 OK, 다르면 양쪽 값을 상세에 출력
    check(f'{sec} 동일', jo.get(sec) == jn.get(sec), '' if jo.get(sec) == jn.get(sec) else f'원본 {jo.get(sec)} / 교체 {jn.get(sec)}')
# model 섹션(어휘 포함)을 따로 꺼냄
mo, mn = jo['model'], jn['model']
# vocab을 제외한 model 타입·설정이 같은지
check('model 타입·설정 동일', {k: v for k, v in mo.items() if k != 'vocab'} == {k: v for k, v in mn.items() if k != 'vocab'})
# 토큰→ID 매핑이 같은지
check('vocab(토큰→ID) 동일', orig.get_vocab() == new.get_vocab())
# BPE 병합 규칙이 같은지
check('merges 동일', mo.get('merges') == mn.get('merges'))
# 후처리기 설정 비교 (root에서는 일부러 붙였으므로 동일해야 정상)
check('post_processor 동일', jo['post_processor'] == jn['post_processor'], f'교체: {jn["post_processor"]}')
# 특수 토큰 매핑이 같은지
check('special_tokens_map 동일', orig.special_tokens_map == new.special_tokens_map)
# 특수 토큰 ID 목록이 같은지
check('all_special_ids 동일', orig.all_special_ids == new.all_special_ids, f'{new.all_special_ids}')
# 토크나이저 전체 크기가 같은지
check('len(tokenizer) 동일', len(orig) == len(new), f'{len(orig)} / {len(new)}')

print('\n== 2. 모델 config·generation_config')
# 기준 모델 config (legacy는 로컬 하위 디렉터리)
co = BartConfig.from_pretrained(ORIG if ADD else f'{ORIG}/kobart_from_pretrained')
# 교체 모델 config
cn = BartConfig.from_pretrained(NEW)
# 특수 토큰 관련 config가 기대값과 다르면 예외 발생
check_model_config(cn)
# 위 검사를 통과했으므로 항상 OK, 실제 값을 상세에 표시
check('교체 config 특수 토큰 기대값', True, str(special_token_report(new, cn)['config']))

# 두 config를 dict로 변환
do, dn = co.to_dict(), cn.to_dict()
# 비교에서 제외할 메타 키
skip = {'_name_or_path', 'transformers_version', 'architectures'}
# 값이 다른 키만 (원본, 교체) 쌍으로 수집
diff = {k: (do.get(k), dn.get(k)) for k in sorted(set(do) | set(dn)) if k not in skip and do.get(k) != dn.get(k)}
print('  config 차이 (원본, 교체):')
# 차이 항목 출력
for k, v in diff.items():
    print(f'    {k}: {v}')

# 기준/교체 모델의 generation_config를 차례로 출력
for name in ((ORIG if ADD else f'{ORIG}/kobart_from_pretrained'), NEW):
    try:
        # generation_config.json이 있으면 그대로 로드
        g = GenerationConfig.from_pretrained(name).to_dict()
    except OSError:
        # 파일이 없으면 config에서 유도
        g = GenerationConfig.from_model_config(BartConfig.from_pretrained(name)).to_dict()
        # 유도한 값임을 표시
        g['_출처'] = 'config에서 유도'

    # 출력할 생성 설정 키 목록
    keys = ['decoder_start_token_id', 'bos_token_id', 'eos_token_id', 'pad_token_id', 'forced_bos_token_id',
            'forced_eos_token_id', 'num_beams', 'max_length', 'no_repeat_ngram_size', 'length_penalty',
            'early_stopping', 'repetition_penalty', '_출처']
    # 존재하는 키만 추려 출력
    print(f'  generation_config {name}: ' + str({k: g.get(k) for k in keys if k in g}))

# 교체 모델 가중치 로드 (임베딩 크기 확인용)
m = BartForConditionalGeneration.from_pretrained(NEW)
# 입력 임베딩의 행 수
emb = m.get_input_embeddings().weight.shape[0]
# 임베딩 행 수, config의 vocab_size, 토크나이저 크기가 모두 같아야 함
check('임베딩 행 수 = vocab_size = len(tokenizer)', emb == cn.vocab_size == len(new), f'{emb} / {cn.vocab_size} / {len(new)}')
# 메모리 확보를 위해 모델 객체 삭제
del m

print('\n== 3. 전체 데이터 인코딩 비교 (union = 세 데이터셋 전체)')
# train/val/test 분할별로 검사
for sp in ['train', 'val', 'test']:
    # 해당 분할의 union TSV 경로
    path = f'{DATA}/union/union_{sp}.txt'
    # 기준/교체 토크나이저로 각각 데이터셋 생성
    ds_o = KoBARTGecDataset(path, orig, MAX_LEN)
    ds_n = KoBARTGecDataset(path, new, MAX_LEN)

    # 소스/타깃 문장 목록 분리
    srcs = [s for s, t in ds_n.docs]; tgts = [t for s, t in ds_n.docs]
    # 소스 문장을 두 토크나이저로 배치 인코딩 (원본, 교체)
    eo_s, en_s = orig(srcs)['input_ids'], new(srcs)['input_ids']
    # 타깃 문장도 동일하게 인코딩
    eo_t, en_t = orig(tgts)['input_ids'], new(tgts)['input_ids']
    # 소스 또는 타깃 인코딩이 다른 샘플의 인덱스
    bad = [i for i in range(len(srcs)) if eo_s[i] != en_s[i] or eo_t[i] != en_t[i]]
    check(f'{sp}: encode 전수 동일 ({len(srcs)}쌍)', not bad, f'불일치 {len(bad)}건 예: {bad[:5]}')

    # dataset.py __getitem__ 결과 전수 비교 (input_ids / decoder_input_ids / labels)
    bad_item = []
    # 모든 샘플의 데이터셋 출력을 비교
    for i in range(len(ds_n)):
        a, b = ds_o[i], ds_n[i]
        # 세 키(input_ids, decoder_input_ids, labels) 중 하나라도 다르면 불일치
        if any(not np.array_equal(a[k], b[k]) for k in a):
            bad_item.append(i)
    check(f'{sp}: dataset 출력 전수 동일', not bad_item, f'불일치 {len(bad_item)}건 예: {bad_item[:5]}')

    # 형식 점검: root는 encode 앞뒤 <s>/</s>, legacy는 특수 토큰 없음
    if ADD:
        # 시작이 0이 아니거나 끝이 1이 아닌 소스/타깃의 인덱스
        fmt_bad = [i for i in range(len(srcs)) if en_s[i][0] != 0 or en_s[i][-1] != 1 or en_t[i][0] != 0 or en_t[i][-1] != 1]
        check(f'{sp}: encode 앞뒤 <s>/</s>', not fmt_bad, f'{len(fmt_bad)}건')
    # 본문 추출 함수: root는 앞뒤 특수 토큰을 떼고, legacy는 그대로
    body = (lambda e: e[1:-1]) if ADD else (lambda e: e)
    # 본문 안에 들어 있는 특수 토큰(0/1/3/6) 개수
    inner_special = sum(1 for e in en_s + en_t for x in body(e) if x in (0, 1, 3, 6))
    # <unk>(5) 개수
    unk = sum(e.count(5) for e in en_s + en_t)
    # 전체 토큰 수
    ntok = sum(len(e) for e in en_s + en_t)
    print(f'  {sp}: 본문 내 특수토큰(0/1/3/6) {inner_special}개, <unk> {unk}개 ({unk / ntok:.5%})')

    # MAX_LEN을 넘어 잘리는 소스/라벨 개수 (라벨은 데이터셋이 EOS를 하나 더 붙이므로 +1)
    over_s = sum(len(e) > MAX_LEN for e in en_s); over_t = sum(len(e) + 1 > MAX_LEN for e in en_t)
    print(f'  {sp}: {MAX_LEN} 토큰 초과(잘림) 소스 {over_s}건 ({over_s / len(srcs):.3%}), 라벨 {over_t}건 ({over_t / len(tgts):.3%})')

    # 타깃을 특수 토큰 제외하고 디코딩했을 때 두 토크나이저 결과가 다른 개수
    dec_bad = sum(orig.decode(eo_t[i], skip_special_tokens=True) != new.decode(en_t[i], skip_special_tokens=True)
                  for i in range(len(tgts)))
    check(f'{sp}: decode(skip_special) 전수 동일', dec_bad == 0, f'{dec_bad}건')

print('\n== 4. 코드 하드코딩과의 정합성')
# native 검증 데이터의 첫 샘플 하나를 교체 토크나이저로 변환
item = KoBARTGecDataset(f'{DATA}/native/native_val.txt', new, MAX_LEN)[0]
# 앞 12개 토큰 출력
print('  예시 input_ids[:12]         ', item['input_ids'][:12].tolist())
print('  예시 decoder_input_ids[:12] ', item['decoder_input_ids'][:12].tolist())
print('  예시 labels[:12]            ', item['labels'][:12].tolist())

# 패딩(-100)을 제외한 실제 라벨 길이
n_real = int((item['labels'] != -100).sum())
if ADD:
    # root: 라벨 끝이 </s></s> (encode가 넣은 </s> + 데이터셋이 추가한 </s>)
    check('labels 끝이 </s></s> (루트 원본 형식)', item['labels'][n_real - 2:n_real].tolist() == [1, 1])
else:
    # legacy: 라벨 끝이 </s> 하나뿐
    check('labels 끝이 </s> 1개 (레거시 형식)', item['labels'][n_real - 1] == 1 and item['labels'][n_real - 2] != 1)
# 디코더 입력이 [0] + 라벨[:-1] (한 칸 밀림) 구조인지
check('decoder_input_ids = [0] + labels[:-1]', item['decoder_input_ids'][1:n_real].tolist() == item['labels'][:n_real - 1].tolist() and item['decoder_input_ids'][0] == 0)

print('\n== 5. 디코더 첫 위치 미래 토큰 누설 검사 (model.make_decoder_attention_mask)')
import torch  # 텐서 생성/추론용
# 디코더 어텐션 마스크를 만드는 함수
from model import make_decoder_attention_mask

# 추론 모드로 교체 모델 로드
m = BartForConditionalGeneration.from_pretrained(NEW).eval()
# 검증 신호용 샘플 하나
it = KoBARTGecDataset(f'{DATA}/native/native_val.txt', new, MAX_LEN)[0]
# 인코더 입력 (배치 차원 추가)
enc = torch.tensor(it['input_ids'])[None]
# 원본 디코더 입력
d1 = torch.tensor(it['decoder_input_ids'])[None]
# d2는 d1의 복사본, n은 실제 토큰 수 + 2
d2 = d1.clone(); n = int((d1 != 0).sum()) + 2
# 인덱스 2 이후의 토큰을 무작위 토큰으로 교체
d2[0, 2:n] = torch.randint(7, 29000, (n - 2,))  # 첫 두 위치 이후 토큰만 교체

# 기울기 계산 없이 추론
with torch.no_grad():
    # d1, d2 두 입력으로 각각 순전파해 위치 0의 logits만 추출
    lo = [m(input_ids=enc, attention_mask=enc.ne(0).float(), decoder_input_ids=d,
            decoder_attention_mask=make_decoder_attention_mask(d)).logits[0, 0] for d in (d1, d2)]
# 두 logits의 최대 절대 차이
diff = (lo[0] - lo[1]).abs().max().item()
# 차이가 거의 없으면 뒤쪽 토큰이 위치 0에 영향을 주지 않는 것 (누설 없음)
check('위치0 출력이 뒤쪽 디코더 토큰과 무관 (누설 없음)', diff < 1e-4, f'logits 최대차 {diff:.6f}')

# 최종 결과 요약
print(f'\n결과: {"전부 통과" if not fails else f"실패 {len(fails)}건: {fails}"}')
# 실패가 있으면 종료코드 1
sys.exit(1 if fails else 0)
