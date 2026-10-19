#■■ 저자 원본(soyoung97/Standard_Korean_GEC @ dfe0af9)의 run.py와 달라진 줄마다 바로 위에 표시: 추가 / 변경(원본 줄) / 삭제(원본 줄). 표시 없는 코드 줄은 원본과 같음.
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

# Yoon et al.의 실행 코드를 현재의 작업 환경에 맞도록 일부 수정한 파인튜닝 수행용 코드
# 수정한 부분에는 #■■ 주석으로 표시함

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
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from dataset import KoBARTGecDataset, GecDataModule
from model import KoBARTConditionalGeneration
from pprint import pprint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import datetime
import argparse
from pytz import timezone

#■■ [추가] 원본에 없었던 추가 라이브러리
from pytorch_lightning.loggers import CSVLogger
from tokenizer_setup import BASE_MODEL, load_tokenizer, check_model_config, special_token_report
import json


def parse_common_args():
    parser = argparse.ArgumentParser(description='Common arguments')
    #■■ [수정] 별도로 정한 데이터셋의 인자 이름을 사용하기 위해 수정
    #■■ 원본 48행: parser.add_argument('--data', type=str, default='mtop', help='Type of dataset to train')
    parser.add_argument('--data', type=str, required=True, choices=['korean_learner', 'native', 'lang8', 'union'], help='Type of dataset to train')
    #■■ [추가] 각 seed마다 체크포인트와 생성 결과를 각각 다른 디렉토리에 저장하기 위해 새롭게 추가한 인자값
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
    #■■ [추가] 입력하는 데이터의 루트 경로를 작업 환경에 맞게 수정
    parser.add_argument('--data_root', type=str, default='../data/Preprocessed', help='Directory holding <data>/<data>_{train,val,test}.txt')
    #■■ [수정] 데이터 파일명 및 저장 경로 재설정
    # 원본 60행: parser.add_argument('--train_data_path', type=str, default='data/wikisql/1000/wikisql_train.jsonl', help='wikisql train data path')
    parser.add_argument('--train_data_path', type=str, default='', help='train data path (default: <data_root>/<data>/<data>_train.txt)')
    #■■ 원본 61행: parser.add_argument('--val_data_path', type=str, default='data/wikisql/1000/wikisql_val.jsonl', help='wikisql validation data path')
    parser.add_argument('--val_data_path', type=str, default='', help='validation data path (default: <data_root>/<data>/<data>_val.txt)')
    #■■ 원본 62행: parser.add_argument('--test_data_path', type=str, default='data/wikisql/1000/wikisql_test.jsonl', help='wikisql test data path')
    parser.add_argument('--test_data_path', type=str, default='', help='test data path (default: <data_root>/<data>/<data>_test.txt)')
    # model checkpoint arguments
    #■■ [수정] 체크포인트 저장 주기를 매 epoch 마다 저장하도록 수정(원본은 10 epoch마다 저장)
    #■■ 원본 64행: parser.add_argument('--every_n_epochs', type=int, default=10, help='Save model checkpoint every n epochs')
    parser.add_argument('--every_n_epochs', type=int, default=1, help='Save model checkpoint every n epochs')
    parser.add_argument("--model_ckpt_path", type=str, default='', help='Path to load model checkpoint, empty string in default')

    #■■ [추가] 중단된 작업을 이어서 진행하기 위한 인자값
    parser.add_argument('--resume_finetune', action='store_true', help='Load weights from --model_ckpt_path and fine-tune (scenario 2, stage 2). Optimizer/scheduler start fresh')

    #■■ [추가] 체크포인트에 저장한 가중치로 테스트셋 데이터에 대해서만 평가하기 위한 인자값
    parser.add_argument('--eval_test', action='store_true', help='Load weights from --model_ckpt_path and run the test set only')
    #■■ [추가] 레시피 조합을 레거시 코드(논문 원본) 또는 루트 코드(저자 공개 코드) 중 하나로 선택하기 위한 인자값
    # recipe: root = 루트 코드(hyunwoongko/kobart 형식), legacy = 논문 실험 코드(src/KoBART-gec + transformers 4.0.0).
    #■■ [추가] 레시피 조합에서 값을 따로 조정할 수 있도록 만든 인자값, 기본값 인자는 명시하면 그 값을, 생략하면 RECIPES[recipe] 값을 씀
    # 레시피 선택 (RECIPES의 키를 선택지로 사용)
    parser.add_argument('--recipe', type=str, default='root', choices=list(RECIPES), help='Preset for tokenizer/scheduler/generation settings')
    # 인코딩에 <s>...</s> 붙이기 여부
    parser.add_argument('--add_bos_eos', type=int, default=None, choices=[0, 1], help='Attach <s>...</s> in tokenizer.encode()')
    # 학습률 스케줄러 종류
    parser.add_argument('--scheduler', type=str, default=None, choices=['linear', 'cosine'], help='LR scheduler')
    # 그래디언트 클리핑 값 (0이면 끔)
    parser.add_argument('--gradient_clip_val', type=float, default=None, help='Gradient clipping (0 = off)')
    # 생성 시 디코더 첫 토큰 id
    parser.add_argument('--decoder_start_token_id', type=int, default=None, help='First decoder token for generate()')
    # 생성 시 강제 종료 토큰 id (-1이면 끔)
    parser.add_argument('--forced_eos_token_id', type=int, default=None, help='forced_eos_token_id for generate() (-1 = off)')

    
    # generation settings
    #■■ [추가] 생성 시 반복 페널티 / 루트 코드의 README.md에 명시된 값을 반영하기 위해서 추가
    parser.add_argument('--repetition_penalty', type=float, default=None, help='repetition_penalty for generate()')
    #■■ [추가] BART 어텐션 연산 방식 선택 (eager 또는 sdpa) / 실험 환경(transformers=4.44)에서는 sdpa가 기본값, 레거시 환경(transformers=4.0)은 eager이 기본값
    # 두 연산 방식의 차이가 실제 성능에 미치는 영향을 테스트하기 위한 용도로 적용
    parser.add_argument('--attn_implementation', type=str, default=None, choices=['eager', 'sdpa'], help='BART attention 구현 (기본: transformers 기본값 = 4.44에서 sdpa; 논문 당시 transformers 4.0은 eager)')
    # logging settings
    parser.add_argument('--log_every_n_steps', type=int, default=50, help='do logging at every n steps.')
    # eval settings
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='validate at every n epochs.')
    # training settings
    #■■ [변경] 웜업 단계의 비율 / 레거시(0.1)와 루트(0.0)의 비율이 각각 상이해서 기본값을 None으로 설정하도록 변경
    #■■  원본 71행: parser.add_argument('--warmup_ratio', type=float, default=0.0, help='warmup steps ratio')
    parser.add_argument('--warmup_ratio', type=float, default=None, help='warmup steps ratio')

    return parser


#■■ [추가] 루트/레거시 코드의 레시피 구성
RECIPES = {
    # 루트 코드: RobertaProcessing 후처리, linear/warmup 0, clip 없음, config의 decoder_start=1/forced_eos=1, README rep 2.0
    'root': {'add_bos_eos': 1, 'scheduler': 'linear', 'warmup_ratio': 0.0, 'gradient_clip_val': 0.0,
             'decoder_start_token_id': 1, 'forced_eos_token_id': 1, 'repetition_penalty': 2.0},
    # 레거시 코드: SKT 토크나이저(후처리 없음), cosine/warmup 0.1, clip 1.0,
    # transformers 4.0.0에서 레거시 config로 생성 시작 = bos(0), forced eos 없음, repetition_penalty 없음
    'legacy': {'add_bos_eos': 0, 'scheduler': 'cosine', 'warmup_ratio': 0.1, 'gradient_clip_val': 1.0,
               'decoder_start_token_id': 0, 'forced_eos_token_id': -1, 'repetition_penalty': 1.0},
}


#■■ [추가] 선택한 레시피를 인자에 적용하는 함수
def apply_recipe(args):
    # 선택한 레시피의 설정값을 하나씩 확인
    for k, v in RECIPES[args.recipe].items():
        # 사용자가 지정하지 않은(None) 인자만 레시피 값으로 채움
        if getattr(args, k) is None:
            setattr(args, k, v)
    # -1은 "끈다"는 표시이므로 generate()에 넘길 None으로 변환
    if args.forced_eos_token_id == -1:
        args.forced_eos_token_id = None
    return args

# 한국 시간 기준 현재 시각을 문자열로 생성하는 함수
def current_time():
    current_time = datetime.datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d_%H-%M-%S')
    return current_time

# 파이썬 실행 명령을 기록/저장하는 함수
# 경로 : logs/command_logs.txt
def write_command_logs():
    command_line = ' '.join(sys.argv)
    command_line = "python3 " + command_line
    cur_time = current_time()
    try:
        # 지정된 GPU 목록 확인
        devices = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        # GPU 지정이 안됐으면 없으면 보이는 모든 GPU 번호를 사용
        devices = ','.join([str(x) for x in list(range(torch.cuda.device_count()))])
    # 명령 이력 파일에 이어쓰기 (logs 디렉터리가 없으면 오류)
    with open("logs/command_logs.txt", 'a') as f:
        f.write(f"[{cur_time}]: CUDA_VISIBLE_DEVICES={devices} {command_line}\n")


# 실제 실행 함수
def cli_main():
    # 실행 명령을 기록
    write_command_logs()
    # 인자 파싱 후 레시피 적용
    parser = parse_common_args()
    #■■ [변경] 레시피 단위로 인자값을 읽고 적용하기 위해 변경
    # 원본 93행: args = parser.parse_args()
    args = apply_recipe(parser.parse_args())
    # train/val/test 각각에 대해

    #■■ [추가] 명령어 입력시 별도로 지정하지 않아도 데이터의 경로를 자동 설정
    for split in ['train', 'val', 'test']:
        # 데이터 경로를 별도로 지정하지 않았으면 data_root/data/data_split.txt로 자동 설정
        # 주어진 경로에 데이터 파일이 없으면 에러를 출력하고 종료
        if getattr(args, f'{split}_data_path') == '':
            setattr(args, f'{split}_data_path', os.path.join(args.data_root, args.data, f'{args.data}_{split}.txt'))
        if not os.path.exists(getattr(args, f'{split}_data_path')):
            parser.error(f'{split} data not found: {getattr(args, f"{split}_data_path")}')

    #■■ [추가] 학습이나 평가를 중단했다가 이어서 진행할 때에 체크포인트가 없으면 에러를 출력하고 종료
    if (args.resume_finetune or args.eval_test) and args.model_ckpt_path == '':
        parser.error('--resume_finetune/--eval_test require --model_ckpt_path')

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.num_sanity_val_steps = 0

    if args.debug:
        args.num_workers = 0
        args.batch_size = 16
    else:
        args.num_workers = int(multiprocessing.cpu_count()/2)

    # 최고 점수 기록용 초기값
    args.best = {'gleu': 0, 'prec': 0, 'rec': 0, 'f0.5': 0}

    # 난수 시드 고정
    pl.seed_everything(args.seed)

    # 최종 인자값 표시
    print("Arguments: ")
    pprint(vars(args))

    # 학습/평가 실행
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

    #■■ 추가
    # C1: skt/kobart-base-v1. 토크나이저는 원본(hyunwoongko/kobart)과 같은 <s>...</s> 후처리를 붙여 로딩하고,
    #■■ 추가
    # 특수 토큰·config를 매 실행마다 검사해 run 디렉토리에 기록한다 (tokenizer_setup.py).

    #■■ [변경] 모델의 기본 설정을 skt/kobart-base-v1으로 변경
    # 원본 122행: config = BartForConditionalGeneration.from_pretrained('hyunwoongko/kobart').config
    config = BartForConditionalGeneration.from_pretrained(BASE_MODEL).config

    #■■ [추가] 특수 토큰 관련 설정에 오류가 없는지 검사
    check_model_config(config)

    #■■ [추가] 어텐션 구현 방법을 별도로 지정한 경우 인자값으로 전달
    attn_kw = {'attn_implementation': args.attn_implementation} if args.attn_implementation else {}
    
    #■■ [변경] BART 모델을 skt/kobart-base-v1로 변경
    # 원본 123행: bart_model = BartForConditionalGeneration.from_pretrained('hyunwoongko/kobart', config=config)
    bart_model = BartForConditionalGeneration.from_pretrained(BASE_MODEL, config=config, **attn_kw)
    
    #■■ [추가] 실제 적용된 어텐션 구현 출력
    print(f"attention implementation: {bart_model.config._attn_implementation} ({type(bart_model.model.encoder.layers[0].self_attn).__name__})")

    #■■ [변경] 토크나이저 로드
    # 원본 124행: tokenizer = PreTrainedTokenizerFast.from_pretrained('hyunwoongko/kobart')
    tokenizer = load_tokenizer(BASE_MODEL, add_bos_eos=bool(args.add_bos_eos))

    #■■ [추가] 결과 저장용 디렉터리 생성
    # outputs/<데이터셋>/<run_id>
    run_dir = f'outputs/{args.data}/{args.run_id}'
    os.makedirs(run_dir, exist_ok=True)

    #■■ [추가] 특수 토큰 설정 보고서 생성
    report = special_token_report(tokenizer, config, args)
    print("Special tokens: ")
    pprint(report)
    with open(f'{run_dir}/special_tokens.json', 'a', encoding='utf-8') as f:
        f.write(json.dumps({'time': current_time(), 'argv': sys.argv, **report}, ensure_ascii=False) + '\n')

    # train/val/test 데이터를 한 곳에서 관리하는 데이터 모듈
    dm = GecDataModule(args, tokenizer, KoBARTGecDataset)

    # 학습/평가 로직을 담은 Lightning 모델
    model = KoBARTConditionalGeneration(args, bart_model, tokenizer, dm)

    # ------------
    # Defining Callbacks
    # ------------

    ckpt_callback = ModelCheckpoint(
        monitor='val_gleu',

        #■■ [변경] 체크포인트 저장 경로 변경
        # 원본 134행: dirpath=f'outputs/{args.data}/',
        dirpath=f'{run_dir}/model_ckpt',

        mode='max',
        verbose=True,
        save_last=False,

        #■■ [변경] GLEU 최상위 1개만 저장
        # 원본 138행: save_top_k=-1,
        save_top_k=1,

        every_n_epochs=args.every_n_epochs,

        #■■ [변경] 파일 이름 설정
        # 원본 140행: filename=f'model_ckpt/{args.data}_{args.lr}_' + '{epoch:02d}'
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

    #■■ [변경] PyTorch Lightning 버전 변경으로 인해 원본 코드의 기능을 재현하고 레시피 설정을 반영할 수 있는 방식으로 Trainer 설정 수정
 
    trainer = pl.Trainer(
        accelerator='gpu',

        #■■ [변경] GPU는 1개만 사용(RTX 5090 하나만 있는 환경)
        # 원본 154행: devices=get_device_count(),
        devices=1,

        #■■ [추가] 최대 epoch 수 설정
        max_epochs=args.max_epochs,

        callbacks=[ckpt_callback],
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        
        #■■ [추가] 매 스텝마다 로그를 기록하도록 설정
        log_every_n_steps=args.log_every_n_steps,
        
        #■■ [추가] 학습 전 검증 샘플 스텝 수 설정(0이면 생략)
        num_sanity_val_steps=args.num_sanity_val_steps,
        
        #■■ [추가] gradient clipping 설정(0보다 크면 적용)
        gradient_clip_val=args.gradient_clip_val if args.gradient_clip_val > 0 else None,
        
        
        #■■ [추가] 학습 지표를 CSV로 기록
        logger=CSVLogger(save_dir=run_dir, name='logs'),
        
        #■■ [추가] 기본 저장 경로 설정
        default_root_dir=run_dir)
        
    # 기존 체크포인트가 없으면 처음부터 학습 시작 
    if args.model_ckpt_path == '':
        trainer.fit(model, dm)

    # 체크포인트가 있으면 체크포인트를 로딩하고 학습을 진행
    else:
        print(f"Loading model from {args.model_ckpt_path}...")
        ckpt = torch.load(args.model_ckpt_path, map_location='cpu', weights_only=False)
        
        #■■ [추가] 가중치 로딩
        model.load_state_dict(ckpt['state_dict'])
        #■■ [추가] 이어서 학습하는 경우, 가중치만 로딩하고 옵티마이저/스케줄러는 새롭게 초기화
        if args.resume_finetune:
            trainer.fit(model, dm)
        #■■ [추가] 평가하는 경우 최고 GLEU 체크포인트로 테스트셋에 대한 평가만 진행
        elif args.eval_test:
            model.epoch = ckpt['epoch']
            trainer.test(model, datamodule=dm)
        #■■ [추가] 검증만 하는 경우, 학습 없이 검증만 수행
        else:
            trainer.validate(model, datamodule=dm)


if __name__ == '__main__':
    cli_main()
