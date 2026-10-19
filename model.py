'''
https://github.com/soyoung97/Standard_Korean_GEC
Modified MIT License

Software Copyright (c) 2022 Soyoung Yoon

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
The above copyright notice and this permission notice need not be included
with content created by the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.
'''
import pytorch_lightning as pl
import pandas as pd
import logging
from pathlib import Path
import os
import re
import numpy as np
import torch
from pytorch_lightning import loggers as pl_loggers                                                                                                                            
from torch.utils.data import DataLoader, Dataset                                                                                                                               
from dataset import KoBARTGecDataset                                                                                                                                           
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast                                                                                                 
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from metric.gleumodule import run_gleu                                                                                                                                         
from transformers import set_seed                                                                                                                                              
from pprint import pprint
import time                                                                                                                                                                    

#■■ [추가] 실험 환경과 루트/레거시 환경 차이 때문에 생기는 디코더 마스크의 첫 위치 처리 문제를 해결하는 함수
def make_decoder_attention_mask(decoder_input_ids, pad_token_id=0):
    mask = decoder_input_ids.ne(pad_token_id).float()
    # 마스크가 생성되어으면 0번 위치를 1번 위치로 덮어쓰기
    if mask.shape[1] > 1:
        mask[:, 0] = mask[:, 1]
    return mask


class KoBARTConditionalGeneration(pl.LightningModule):
    def __init__(self, args, model, tokenizer, datamodules):
        super().__init__()
        self.assign_attributes(args, model, tokenizer, datamodules)

    def assign_attributes(self, args, model, tokenizer, datamodules):
        self.bos_token = '<s>'                                                                                                                                            
        self.eos_token = '</s>'                                               
        self.pad_token_id = 0                                                 
        self.epoch = 0                                                                                                                                             
        self.outputs = []                                                                                                                                            
        self.decoded_labels = []                        
        self.origs = []                                       
        self.step = 0                                         
        self.args = args
        self.max_len = self.args.max_seq_len                                      
        self.scores = {}                                                      
        self.generation_time = 0
        self.model = model
        self.tokenizer = tokenizer
        self.dm = datamodules
        #■■ [추가] pl 2.0 버전에서 추가된 파라미터
        self.eval_step_losses = []

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.args.lr, correct_bias=False)
        data_len = len(self.dm.train_dataloader().dataset)
        num_train_steps = int(data_len * self.args.max_epochs / self.args.batch_size)
        if data_len < self.args.batch_size:
            num_train_steps = self.args.max_epochs
        print(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.args.warmup_ratio)

        print(f'num_warmup_steps : {num_warmup_steps}')
        #■■ [추가] 레시피에 따라 웜업 스케줄러를 선택하도록 수정
        # recipe root: linear (루트 코드) / legacy: cosine (논문 실험 코드 src/KoBART-gec)
        schedule_fn = get_cosine_schedule_with_warmup if self.args.scheduler == 'cosine' else get_linear_schedule_with_warmup
        scheduler = schedule_fn(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler,
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        self.scheduler = scheduler
        return [optimizer], [lr_scheduler]

    def forward(self, inputs):                                           
        attention_mask = inputs['input_ids'].ne(self.pad_token_id).float()
        #■■ [변경] ne(0)으로 일괄적으로 마스크를 생성하던 코드를 선택적으로 0번 위치 마스크를 1번으로 수정하는 함수로 수정
        # 원본 96행: decoder_attention_mask = inputs['decoder_input_ids'].ne(self.pad_token_id).float()
        decoder_attention_mask = make_decoder_attention_mask(inputs['decoder_input_ids'], self.pad_token_id)
        return self.model(input_ids=inputs['input_ids'],                      
                attention_mask=attention_mask,                                                                                                               
                decoder_input_ids=inputs['decoder_input_ids'],                
                decoder_attention_mask=decoder_attention_mask,            
                labels=inputs['labels'], return_dict=True)                                                                                                   
                                                                                                                                                             
    #■■ [추가] PL 2.x에서는 학습 시작 시 model.train()을 호출하지 않기 때문에 dropout 없이 학습이 진행되어 과적합되는 문제가 발생하기 때문에 매 학습 epoch 시작 시 train 모드로 되돌리는 단계를 추가
    def on_train_epoch_start(self):
        self.train()

    def training_step(self, batch, batch_idx):
        #■■ [추가] BART가 eval 모드로 학습되어 dropout이 꺼진 상태일 수 있기 때문에 첫 배치에서 이를 확인하는 단계 추가
        if batch_idx == 0 and not self.model.training:
            raise RuntimeError('내부 BART가 eval 모드로 학습 중 (dropout 꺼짐) — on_train_epoch_start 확인')
        outs = self(batch)
        loss = outs.loss
        self.log('train_loss', loss, prog_bar=False)
        self.step += 1
        return loss

    #■■ [추가] PL 2.x에서는 training_epoch_end 대신 on_train_epoch_end 훅을 사용해야 함
    # 원본 110행: def training_epoch_end(self, _):
    def on_train_epoch_end(self):
        # 원본 111행: self.scores[self.epoch]['generation_time'] = self.generation_time
        self.scores.setdefault(self.epoch, {})['generation_time'] = self.generation_time
        print(f"\nGeneration time: {self.generation_time}")
        self.generation_time = 0
        self.epoch += 1

    def generate(self, input_ids, labels):
        self.model.eval()
        start = time.time()
        #■■ [추가] skt/kobart와 맞는 인자값을 직접 입력하도록 설정 추가
        # 원본 119행: output = self.model.generate(input_ids, eos_token_id=1, max_length=self.max_len, num_beams=4)
        output = self.model.generate(input_ids, eos_token_id=1, max_length=self.max_len, num_beams=4,
                                     #■■ [추가] 레시피에 따라 루트 2.0, 레거시 1.0
                                     repetition_penalty=self.args.repetition_penalty,
                                     #■■ [추가] 레시피에 따라 루트 1, 레거시 0
                                     decoder_start_token_id=self.args.decoder_start_token_id,
                                     #■■ [추가] 레시피에 따라 루트 1, 레거시 None
                                     forced_eos_token_id=self.args.forced_eos_token_id)
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        end = time.time()
        self.generation_time += end - start
        decoded_label = self.tokenizer.batch_decode(labels.masked_fill(labels == -100, 1), skip_special_tokens=True)
        self.outputs += [x.replace('\n', '') for x in output]
        self.decoded_labels += decoded_label
        self.origs += self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    def should_generate(self):
        return True

    def validation_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs['loss']
        if self.should_generate():
            self.generate(batch['input_ids'], batch['labels'])
        #■■ [추가] pl 2.x에서는 validation_step에서 loss를 직접 축적하도록 변경
        self.eval_step_losses.append(loss.detach())
        return (loss)

    def test_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs['loss']
        if self.should_generate():
            self.generate(batch['input_ids'], batch['labels'])
        #■■ [추가] pl 2.x에서는 test_step에서 loss를 직접 축적하도록 변경
        self.eval_step_losses.append(loss.detach())
        return (loss)

    #■■ [추가] pl 2.x에서는 validation_epoch_end/test_epoch_end 대신 on_*_epoch_end를 사용하고, step 출력은 직접 축적하도록 변경
    # 원본 144행: def test_epoch_end(self, outputs):
    def on_test_epoch_end(self):
        # 원본 145행: return self.validation_epoch_end(outputs, mode='test')
        return self.eval_epoch_end(mode='test')

    #■■ [추가]
    def on_validation_epoch_end(self):
        return self.eval_epoch_end(mode='val')

    #■■ [변경] pl 2.x에서는 loss를 self.log에 직접 기록하기 때문에 validation_step에서 self.log('val_loss', loss)를 호출하면 자동으로 손실함수 처리와 로그 기록을 수행하지만, 
    # 루트/레거시 코드의 pl 1.x의 동작을 그대로 유지하면서 pl 2.x 환경에서도 동일하게 동작하도록 구현하기 위해 validation_step(), test_step에서 loss를 직접 축적하는 기능을 추가하고,
    # eval_epoch_end()에서 직접 loss를 축적하는 코드를 삭제
    # 원본 147행: def validation_epoch_end(self, outputs, mode='val'):
    def eval_epoch_end(self, mode='val'):
        #■■ [삭제] 원본 148행: losses = []
        #■■ [삭제] 원본 149행: for loss in outputs:
            #■■ [삭제] 원본 150행: losses.append(loss)
        #■■ [변경] 원본 151행: total_loss = torch.stack(losses).mean()
        total_loss = torch.stack(self.eval_step_losses).mean()
        #■■ [추가]
        self.eval_step_losses = []
        if self.should_generate():
            # Make generation output directory and file
            #■■ [변경] 원본 154행: directory = f"outputs/generation/epoch{self.epoch}/{mode}"
            directory = f"outputs/generation/{self.args.data}/{self.args.run_id}/epoch{self.epoch}/{mode}"
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            #■■ [변경] 원본 157행: with open(directory + f"/hypothesis_{total_loss}.txt", 'w', encoding='utf-8') as f:
            with open(directory + "/hypothesis.txt", 'w', encoding='utf-8') as f:
                f.write("\n".join(self.outputs))
            #■■ [변경] 원본 159행: with open(directory + f"/reference_{total_loss}.txt", "w", encoding='utf-8') as f:
            with open(directory + "/reference.txt", "w", encoding='utf-8') as f:
                f.write("\n".join(self.decoded_labels))
            #■■ [변경] 원본 161행: with open(directory + f"/source_{total_loss}.txt", "w", encoding='utf-8') as f:
            with open(directory + "/source.txt", "w", encoding='utf-8') as f:
                f.write("\n".join(self.origs))
            self.outputs = []
            self.decoded_labels = []
            self.origs = []
            #■■ [변경] 원본 166행: gleu_out = run_gleu(reference=directory + f"/reference_{total_loss}.txt", source=directory + f"/source_{total_loss}.txt", hypothesis=directory + f"/hypothesis_{total_loss}.txt")
            gleu_out = run_gleu(reference=directory + "/reference.txt", source=directory + "/source.txt", hypothesis=directory + "/hypothesis.txt")
            logging.info(f"\ngleu_value: {gleu_out}\n")
            #■■ [변경] 원본 168행: with open(directory + f"/gleu_{gleu_out}.txt", "w", encoding='utf-8') as f:
            with open(directory + "/gleu.txt", "w", encoding='utf-8') as f:
                #■■ [변경] 원본 169행: f.write(f"data: {self.args.data}, epoch: {self.epoch}, gleu_out: {gleu_out}, val_loss: {total_loss}\nhparams:{self.hparams}")
                f.write(f"data: {self.args.data}, run_id: {self.args.run_id}, epoch: {self.epoch}, gleu_out: {gleu_out}, {mode}_loss: {total_loss.item()}\n"
                        #■■ [추가] 레시피 방식으로 입력하는 인자값 기록 추가
                        f"args: lr={self.args.lr} batch_size={self.args.batch_size} max_epochs={self.args.max_epochs} seed={self.args.seed} "
                        f"recipe={self.args.recipe} scheduler={self.args.scheduler} warmup={self.args.warmup_ratio} clip={self.args.gradient_clip_val} "
                        f"decoder_start={self.args.decoder_start_token_id} forced_eos={self.args.forced_eos_token_id} "
                        f"repetition_penalty={self.args.repetition_penalty} ckpt={self.args.model_ckpt_path}")
            #■■ [삭제] 원본의 KAGAS 호출(f-string 누락·command 미정의)과 0점 대체 기록 제거 / M²는 최고 검증 GLEU 체크포인트의 테스트 출력에 대해 공식 m2로 별도 계산
            #■■ 원본 170행: # make m2 file
            #■■ 원본 171행: if not os.path.exists('get_data/{self.args.data}_{mode}.m2'):
            #■■ 원본 172행: print("Making m2 file since we don't have one..")
            #■■ 원본 173행: command = f"cd KAGAS/ && python3 parallel_to_m2_korean.py -orig ../{directory}/source_{total_loss}.txt -cor ../{directory}/reference_{total_loss}.txt -out get_data/{self.args.data}_{mode}.m2 -noprint && cd ../"
            #■■ 원본 174행: os.system(command)
            #■■ 원본 175행: # output m2scores
            #■■ 원본 176행: system_command = f"python3 ./metric/m2scorer/scripts/m2scorer.py {directory}/hypothesis_{total_loss}.txt get_data/{self.args.data}_{mode}.m2 > {directory}/m2score.txt"
            #■■ 원본 177행: print(f"Processing: {system_command}")
            #■■ 원본 178행: #out = os.system(system_command)
            #■■ 원본 179행: #os.system(f"cat {directory}/m2score.txt")
            #■■ 원본 180행: #with open(f"{directory}/m2score.txt", encoding='utf-8') as f:
            #■■ 원본 181행: #     try:
            #■■ 원본 182행: #         p, r, f_score = [x.split(": ")[1].strip() for x in f.read().strip().split("\n")]
            #■■ 원본 183행: #    except:
            #■■ 원본 184행: #        print("Reading m2 scores failed")
            #■■ 원본 185행: #       p, r, f_score = 0, 0, 0
        #■■ 원본 186행: p, r, f_score = 0,0,0
        gleuscore = float(gleu_out) * 100
        #■■ 원본 188행: self.scores[self.epoch] = {'precision': p, 'recall': r, 'f_score': f_score, 'gleu': gleuscore, 'loss': total_loss.item()}
        self.scores.setdefault(self.epoch, {}).update({'gleu': gleuscore, 'loss': total_loss.item()})
        #■■ 원본 189행: print(f"\n\nEPOCH {self.epoch} / VAL_LOSS {round(total_loss.item(), 2)} / GLEU {round(gleuscore, 2)}\n\n")
        print(f"\n\nEPOCH {self.epoch} / {mode.upper()}_LOSS {round(total_loss.item(), 2)} / GLEU {round(gleuscore, 2)}\n\n")
        self.log(f'{mode}_loss', total_loss, prog_bar=False)
        self.log(f'{mode}_gleu', gleuscore)
        if self.args.best['gleu'] < gleuscore:
            self.args.best['gleu'] = gleuscore
            #■■ 원본 194행: self.args.best['f0.5'] = f_score
            #■■ 원본 195행: self.args.best['prec'] = p
            #■■ 원본 196행: self.args.best['rec'] = r
        #■■ 원본 197행: print(f"Print ordering of: precision, recall, f0.5 score, gleu score, val loss, generation_time. The last epoch is test epoch.")
        print(f"Print ordering of: gleu score, loss, generation_time.")
        pprint(self.scores)
