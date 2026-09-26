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

def make_decoder_attention_mask(decoder_input_ids, pad_token_id=0):
    """디코더 attention mask. 논문 실험 버전(transformers 4.0.0 _prepare_bart_decoder_inputs)과 같이
    첫 디코더 토큰은 패딩 값(0)이어도 마스킹하지 않는다. 4.8 이후에는 이 처리가 없어서 첫 위치 행이
    전부 가려지고, 라이브러리가 그 행을 '전부 보기'로 풀어 버려 첫 토큰 예측에 정답 뒷부분이 샌다
    (plan.md §3-6)."""
    mask = decoder_input_ids.ne(pad_token_id).float()
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
        decoder_attention_mask = make_decoder_attention_mask(inputs['decoder_input_ids'], self.pad_token_id)
        return self.model(input_ids=inputs['input_ids'],                      
                attention_mask=attention_mask,                                                                                                               
                decoder_input_ids=inputs['decoder_input_ids'],                
                decoder_attention_mask=decoder_attention_mask,            
                labels=inputs['labels'], return_dict=True)                                                                                                   
                                                                                                                                                             
    # PL 2.x는 학습 시작 때 model.train()을 호출하지 않고, from_pretrained()는 eval 모드로 반환한다.
    # 그대로 두면 dropout 없이 학습된다. PL 1.1(논문 실험)처럼 매 학습 epoch 시작에 train 모드로 되돌린다 (plan.md §3-7).
    def on_train_epoch_start(self):
        self.train()

    def training_step(self, batch, batch_idx):
        if batch_idx == 0 and not self.model.training:
            raise RuntimeError('내부 BART가 eval 모드로 학습 중 (dropout 꺼짐) — on_train_epoch_start 확인')
        outs = self(batch)
        loss = outs.loss
        self.log('train_loss', loss, prog_bar=False)
        self.step += 1
        return loss

    # PL 2.x: training_epoch_end -> on_train_epoch_end (C4). 검증은 이 훅보다 먼저 끝난다.
    def on_train_epoch_end(self):
        self.scores.setdefault(self.epoch, {})['generation_time'] = self.generation_time
        print(f"\nGeneration time: {self.generation_time}")
        self.generation_time = 0
        self.epoch += 1

    def generate(self, input_ids, labels):
        self.model.eval()
        start = time.time()
        # R4 + recipe: repetition_penalty, 생성 시작 토큰, forced eos를 run.py 인자로 받는다
        output = self.model.generate(input_ids, eos_token_id=1, max_length=self.max_len, num_beams=4,
                                     repetition_penalty=self.args.repetition_penalty,
                                     decoder_start_token_id=self.args.decoder_start_token_id,
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
        self.eval_step_losses.append(loss.detach())
        return (loss)

    def test_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs['loss']
        if self.should_generate():
            self.generate(batch['input_ids'], batch['labels'])
        self.eval_step_losses.append(loss.detach())
        return (loss)

    # PL 2.x: validation_epoch_end/test_epoch_end -> on_*_epoch_end, step 출력은 직접 축적 (C5)
    def on_test_epoch_end(self):
        return self.eval_epoch_end(mode='test')

    def on_validation_epoch_end(self):
        return self.eval_epoch_end(mode='val')

    def eval_epoch_end(self, mode='val'):
        total_loss = torch.stack(self.eval_step_losses).mean()
        self.eval_step_losses = []
        if self.should_generate():
            # Make generation output directory and file (R6: 데이터셋·run별 분리)
            directory = f"outputs/generation/{self.args.data}/{self.args.run_id}/epoch{self.epoch}/{mode}"
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            with open(directory + "/hypothesis.txt", 'w', encoding='utf-8') as f:
                f.write("\n".join(self.outputs))
            with open(directory + "/reference.txt", "w", encoding='utf-8') as f:
                f.write("\n".join(self.decoded_labels))
            with open(directory + "/source.txt", "w", encoding='utf-8') as f:
                f.write("\n".join(self.origs))
            self.outputs = []
            self.decoded_labels = []
            self.origs = []
            gleu_out = run_gleu(reference=directory + "/reference.txt", source=directory + "/source.txt", hypothesis=directory + "/hypothesis.txt")
            logging.info(f"\ngleu_value: {gleu_out}\n")
            with open(directory + "/gleu.txt", "w", encoding='utf-8') as f:
                f.write(f"data: {self.args.data}, run_id: {self.args.run_id}, epoch: {self.epoch}, gleu_out: {gleu_out}, {mode}_loss: {total_loss.item()}\n"
                        f"args: lr={self.args.lr} batch_size={self.args.batch_size} max_epochs={self.args.max_epochs} seed={self.args.seed} "
                        f"recipe={self.args.recipe} scheduler={self.args.scheduler} warmup={self.args.warmup_ratio} clip={self.args.gradient_clip_val} "
                        f"decoder_start={self.args.decoder_start_token_id} forced_eos={self.args.forced_eos_token_id} "
                        f"repetition_penalty={self.args.repetition_penalty} ckpt={self.args.model_ckpt_path}")
            # R2: 원본의 KAGAS 호출(f-string 누락·command 미정의)과 0점 대체 기록 제거.
            # M²는 최고 검증 GLEU 체크포인트의 테스트 출력에 대해 공식 m2로 별도 계산한다 (plan.md §6-2).
        gleuscore = float(gleu_out) * 100
        self.scores.setdefault(self.epoch, {}).update({'gleu': gleuscore, 'loss': total_loss.item()})
        print(f"\n\nEPOCH {self.epoch} / {mode.upper()}_LOSS {round(total_loss.item(), 2)} / GLEU {round(gleuscore, 2)}\n\n")
        self.log(f'{mode}_loss', total_loss, prog_bar=False)
        self.log(f'{mode}_gleu', gleuscore)
        if self.args.best['gleu'] < gleuscore:
            self.args.best['gleu'] = gleuscore
        print(f"Print ordering of: gleu score, loss, generation_time.")
        pprint(self.scores)
