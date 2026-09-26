# transformers 4.0.0 vs 현재(4.44) 비교 — 4.0.0 쪽 (~/projects/phdq_bart/.venv-tf400, torch 1.7.1 CPU).
# 레거시 코드(src/KoBART-gec/train.py)와 같은 config·forward·generate 호출로 결과를 만들어 현재 버전과 비교한다.
import os, json, torch
from transformers import BartConfig, BartForConditionalGeneration

torch.set_num_threads(4)
OUT = 'outputs/tfcmp'
LEG = os.path.abspath(os.environ.get('KOBART_LEGACY_DIR', '../.cache/kobart_legacy') + '/kobart_from_pretrained')

cfg = BartConfig.from_pretrained(LEG)
print('4.0.0 config: decoder_start', cfg.decoder_start_token_id, 'bos', cfg.bos_token_id, 'eos', cfg.eos_token_id,
      'pad', cfg.pad_token_id, 'num_beams', cfg.num_beams, 'early_stopping', cfg.early_stopping,
      'length_penalty', cfg.length_penalty, 'min_length', cfg.min_length, 'no_repeat', cfg.no_repeat_ngram_size)
m = BartForConditionalGeneration(cfg)
sd = torch.load(f'{OUT}/weights.bin', map_location='cpu')
missing, unexpected = m.load_state_dict(sd, strict=False)
print('missing', missing, 'unexpected', unexpected)
m.eval()
inp = torch.load(f'{OUT}/inputs.bin', map_location='cpu')
gen_ids = inp['gen_ids']
outs = []
with torch.no_grad():
    for b in range(0, len(gen_ids), 64):
        o = m.generate(gen_ids[b:b + 64], eos_token_id=1, max_length=128, num_beams=4)  # 레거시 train.py와 동일한 호출
        outs += [x.tolist() for x in o]
    out = m(input_ids=inp['input_ids'], attention_mask=inp['input_ids'].ne(0).float(),
            decoder_input_ids=inp['decoder_input_ids'],
            decoder_attention_mask=inp['decoder_input_ids'].ne(0).float(),  # 레거시 forward 그대로
            labels=inp['labels'], return_dict=True)
torch.save(out.logits, f'{OUT}/logits_old.bin')
json.dump({'gen': outs, 'loss': out.loss.item()}, open(f'{OUT}/gen_old.json', 'w'))
print('old: loss', out.loss.item(), 'gen', len(outs))
