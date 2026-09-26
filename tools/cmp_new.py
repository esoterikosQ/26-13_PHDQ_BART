# transformers 4.0.0 vs 현재(4.44) 비교 — 현재 환경 쪽. 저장소 루트에서 .venv로 실행.
# 1) 체크포인트 가중치·입력을 torch 1.7이 읽을 수 있는 형식으로 저장  2) 현재 버전으로 생성·forward 결과 저장
import os, sys, json, torch
sys.path.insert(0, os.getcwd())
from transformers import BartForConditionalGeneration
from tokenizer_setup import load_tokenizer
from dataset import KoBARTGecDataset
from model import make_decoder_attention_mask

torch.set_num_threads(4)
OUT = 'outputs/tfcmp'
os.makedirs(OUT, exist_ok=True)
CKPT = sys.argv[1]
N_GEN, N_FWD = 256, 32

ck = torch.load(CKPT, map_location='cpu', weights_only=False)
sd = {k[len('model.'):]: v for k, v in ck['state_dict'].items() if k.startswith('model.')}
torch.save(sd, f'{OUT}/weights.bin', _use_new_zipfile_serialization=False)

tok = load_tokenizer(add_bos_eos=False)
data = os.path.abspath('../data/Preprocessed/korean_learner')
val = KoBARTGecDataset(f'{data}/korean_learner_val.txt', tok, 128)
trn = KoBARTGecDataset(f'{data}/korean_learner_train.txt', tok, 128)
gen_ids = torch.tensor([val[i]['input_ids'] for i in range(N_GEN)])
fwd = {k: torch.tensor([trn[i][k] for i in range(N_FWD)]) for k in ('input_ids', 'decoder_input_ids', 'labels')}
torch.save({'gen_ids': gen_ids, **fwd}, f'{OUT}/inputs.bin', _use_new_zipfile_serialization=False)

m = BartForConditionalGeneration.from_pretrained('skt/kobart-base-v1')
m.load_state_dict(sd)
m.eval()
res = {}
with torch.no_grad():
    for name, feos in (('forced_eos_1', 1), ('forced_eos_none', None)):
        outs = []
        for b in range(0, N_GEN, 64):
            o = m.generate(gen_ids[b:b + 64], eos_token_id=1, max_length=128, num_beams=4, repetition_penalty=1.0,
                           decoder_start_token_id=0, forced_eos_token_id=feos)
            outs += [[t for t in x.tolist()] for x in o]
        res[name] = outs
    out = m(input_ids=fwd['input_ids'], attention_mask=fwd['input_ids'].ne(0).float(),
            decoder_input_ids=fwd['decoder_input_ids'],
            decoder_attention_mask=make_decoder_attention_mask(fwd['decoder_input_ids']), labels=fwd['labels'])
res['loss'] = out.loss.item()
torch.save(out.logits, f'{OUT}/logits_new.bin', _use_new_zipfile_serialization=False)
json.dump(res, open(f'{OUT}/gen_new.json', 'w'))
print('new: loss', res['loss'], 'gen', len(res['forced_eos_1']))
