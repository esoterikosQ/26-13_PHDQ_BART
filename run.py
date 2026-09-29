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
import multiprocessing
import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from tokenizer_setup import BASE_MODEL, load_tokenizer, check_model_config, special_token_report
import json
from dataset import KoBARTGecDataset, GecDataModule
from model import KoBARTConditionalGeneration
from pprint import pprint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import datetime
import argparse
from pytz import timezone


def parse_common_args():
    parser = argparse.ArgumentParser(description='Common arguments')
    parser.add_argument('--data', type=str, required=True, choices=['korean_learner', 'native', 'lang8', 'union'], help='Type of dataset to train')
    parser.add_argument('--run_id', type=str, required=True, help='Unique id per dataset/scenario/seed (e.g. individual_seed0). Separates checkpoints and generations')
    parser.add_argument('--max_epochs', type=int, default=100, help='Total number of epochs to train')
    parser.add_argument('--debug', action='store_true', help='If true, reduces the number of validation & test dataset for faster loading and debugging')

    # mode-specific arguments

    parser.add_argument('--name', type=str, default='default_name', help='Name tag, just like memo, that is included in the output file save')
    parser.add_argument('--max_seq_len', type=int, default=128, help='maximum token length for the tokenizer')
    parser.add_argument('--seed', type=int, default=0, help='seed to train model and split data')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-05, help='Set learning rate for optimizer')
    # dataset arguments
    parser.add_argument('--data_root', type=str, default='../data/Preprocessed', help='Directory holding <data>/<data>_{train,val,test}.txt')
    parser.add_argument('--train_data_path', type=str, default='', help='train data path (default: <data_root>/<data>/<data>_train.txt)')
    parser.add_argument('--val_data_path', type=str, default='', help='validation data path (default: <data_root>/<data>/<data>_val.txt)')
    parser.add_argument('--test_data_path', type=str, default='', help='test data path (default: <data_root>/<data>/<data>_test.txt)')
    # model checkpoint arguments
    parser.add_argument('--every_n_epochs', type=int, default=1, help='Save model checkpoint every n epochs')
    parser.add_argument("--model_ckpt_path", type=str, default='', help='Path to load model checkpoint, empty string in default')
    parser.add_argument('--resume_finetune', action='store_true', help='Load weights from --model_ckpt_path and fine-tune (scenario 2, stage 2). Optimizer/scheduler start fresh')
    parser.add_argument('--eval_test', action='store_true', help='Load weights from --model_ckpt_path and run the test set only')
    # recipe: root = 루트 코드(hyunwoongko/kobart 형식), legacy = 논문 실험 코드(src/KoBART-gec + transformers 4.0.0).
    # 아래 None 기본값 인자는 명시하면 그 값을, 생략하면 RECIPES[recipe] 값을 쓴다.
    parser.add_argument('--recipe', type=str, default='root', choices=list(RECIPES), help='Preset for tokenizer/scheduler/generation settings')
    parser.add_argument('--add_bos_eos', type=int, default=None, choices=[0, 1], help='Attach <s>...</s> in tokenizer.encode()')
    parser.add_argument('--scheduler', type=str, default=None, choices=['linear', 'cosine'], help='LR scheduler')
    parser.add_argument('--gradient_clip_val', type=float, default=None, help='Gradient clipping (0 = off)')
    parser.add_argument('--decoder_start_token_id', type=int, default=None, help='First decoder token for generate()')
    parser.add_argument('--forced_eos_token_id', type=int, default=None, help='forced_eos_token_id for generate() (-1 = off)')
    # generation settings
    parser.add_argument('--repetition_penalty', type=float, default=None, help='repetition_penalty for generate()')
    parser.add_argument('--attn_implementation', type=str, default=None, choices=['eager', 'sdpa'], help='BART attention 구현 (기본: transformers 기본값 = 4.44에서 sdpa; 논문 당시 transformers 4.0은 eager)')
    # logging settings
    parser.add_argument('--log_every_n_steps', type=int, default=50, help='do logging at every n steps.')
    # eval settings
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='validate at every n epochs.')
    # training settings
    parser.add_argument('--warmup_ratio', type=float, default=None, help='warmup steps ratio')
    return parser


RECIPES = {
    # 루트 코드: RobertaProcessing 후처리, linear/warmup 0, clip 없음, config의 decoder_start=1/forced_eos=1, README rep 2.0
    'root': {'add_bos_eos': 1, 'scheduler': 'linear', 'warmup_ratio': 0.0, 'gradient_clip_val': 0.0,
             'decoder_start_token_id': 1, 'forced_eos_token_id': 1, 'repetition_penalty': 2.0},
    # 논문 실험 코드: SKT 토크나이저(후처리 없음), cosine/warmup 0.1, clip 1.0,
    # transformers 4.0.0에서 레거시 config로 생성 시작 = bos(0), forced eos 없음, repetition_penalty 없음
    'legacy': {'add_bos_eos': 0, 'scheduler': 'cosine', 'warmup_ratio': 0.1, 'gradient_clip_val': 1.0,
               'decoder_start_token_id': 0, 'forced_eos_token_id': -1, 'repetition_penalty': 1.0},
}


def apply_recipe(args):
    for k, v in RECIPES[args.recipe].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.forced_eos_token_id == -1:
        args.forced_eos_token_id = None
    return args

def current_time():
    current_time = datetime.datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d_%H-%M-%S')
    return current_time

def write_command_logs():
    command_line = ' '.join(sys.argv)
    command_line = "python3 " + command_line
    cur_time = current_time()
    try:
        devices = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        devices = ','.join([str(x) for x in list(range(torch.cuda.device_count()))])
    with open("logs/command_logs.txt", 'a') as f:
        f.write(f"[{cur_time}]: CUDA_VISIBLE_DEVICES={devices} {command_line}\n")


def cli_main():
    write_command_logs()
    parser = parse_common_args()
    args = apply_recipe(parser.parse_args())
    for split in ['train', 'val', 'test']:
        if getattr(args, f'{split}_data_path') == '':
            setattr(args, f'{split}_data_path', os.path.join(args.data_root, args.data, f'{args.data}_{split}.txt'))
        if not os.path.exists(getattr(args, f'{split}_data_path')):
            parser.error(f'{split} data not found: {getattr(args, f"{split}_data_path")}')
    if (args.resume_finetune or args.eval_test) and args.model_ckpt_path == '':
        parser.error('--resume_finetune/--eval_test require --model_ckpt_path')
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.num_sanity_val_steps = 0
    if args.debug:
        args.num_workers = 0
        args.batch_size = 16
    else:
        args.num_workers = int(multiprocessing.cpu_count()/2)
    args.best = {'gleu': 0, 'prec': 0, 'rec': 0, 'f0.5': 0}
    pl.seed_everything(args.seed)
    print("Arguments: ")
    pprint(vars(args))
    run_mode(args)

def get_device_count():
    if not torch.cuda.is_available():
        return 0
    try:
        devices = len(os.environ['CUDA_VISIBLE_DEVICES'].split(","))
    except KeyError:
        devices = torch.cuda.device_count()
    return devices

def run_mode(args):

    # ------------
    # model & tokenizer setup
    # ------------

    # C1: skt/kobart-base-v1. 토크나이저는 원본(hyunwoongko/kobart)과 같은 <s>...</s> 후처리를 붙여 로딩하고,
    # 특수 토큰·config를 매 실행마다 검사해 run 디렉토리에 기록한다 (tokenizer_setup.py).
    config = BartForConditionalGeneration.from_pretrained(BASE_MODEL).config
    check_model_config(config)
    attn_kw = {'attn_implementation': args.attn_implementation} if args.attn_implementation else {}
    bart_model = BartForConditionalGeneration.from_pretrained(BASE_MODEL, config=config, **attn_kw)
    print(f"attention implementation: {bart_model.config._attn_implementation} ({type(bart_model.model.encoder.layers[0].self_attn).__name__})")
    tokenizer = load_tokenizer(BASE_MODEL, add_bos_eos=bool(args.add_bos_eos))
    run_dir = f'outputs/{args.data}/{args.run_id}'
    os.makedirs(run_dir, exist_ok=True)
    report = special_token_report(tokenizer, config, args)
    print("Special tokens: ")
    pprint(report)
    with open(f'{run_dir}/special_tokens.json', 'a', encoding='utf-8') as f:
        f.write(json.dumps({'time': current_time(), 'argv': sys.argv, **report}, ensure_ascii=False) + '\n')
    dm = GecDataModule(args, tokenizer, KoBARTGecDataset)
    model = KoBARTConditionalGeneration(args, bart_model, tokenizer, dm)

    # ------------
    # Defining Callbacks
    # ------------

    # R1/R6: run별 디렉토리에 검증 GLEU 최고 체크포인트 1개만 저장
    ckpt_callback = ModelCheckpoint(
        monitor='val_gleu',
        dirpath=f'{run_dir}/model_ckpt',
        mode='max',
        verbose=True,
        save_last=False,
        save_top_k=1,
        every_n_epochs=args.every_n_epochs,
        filename=f'{args.data}_{args.lr}_' + '{epoch:02d}'
        )
    # make sure log steps are smaller than step per train
    data_len = len(dm.train_dataloader().dataset)
    args.log_every_n_steps = max(min(int(data_len / (args.batch_size*2)), args.log_every_n_steps), 1)
    print(f"Adjusted args:")
    pprint(vars(args))
    # ------------
    # Calling Trainer
    # ------------
    # C3: from_argparse_args/strategy='dp'는 PL 2.x에서 제거됨. 단일 GPU로 글로벌 배치 = batch_size
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=args.max_epochs,
        callbacks=[ckpt_callback],
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        log_every_n_steps=args.log_every_n_steps,
        num_sanity_val_steps=args.num_sanity_val_steps,
        gradient_clip_val=args.gradient_clip_val if args.gradient_clip_val > 0 else None,
        logger=CSVLogger(save_dir=run_dir, name='logs'),
        default_root_dir=run_dir)

    if args.model_ckpt_path == '':
        trainer.fit(model, dm)
    else:
        print(f"Loading model from {args.model_ckpt_path}...")
        ckpt = torch.load(args.model_ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        if args.resume_finetune:
            # R3: 시나리오2 2단계 — 가중치만 이어받고 옵티마이저/스케줄러는 새로 시작
            trainer.fit(model, dm)
        elif args.eval_test:
            # R5: 최고 검증 GLEU 체크포인트로 테스트셋만 평가 (생성 경로의 epoch 번호는 체크포인트 epoch)
            model.epoch = ckpt['epoch']
            trainer.test(model, datamodule=dm)
        else:
            trainer.validate(model, datamodule=dm)


if __name__ == '__main__':
    cli_main()
