# Neuron 경로 설정 예시. 복사해서 paths.local.sh로 저장한 뒤 값을 채운다 (paths.local.sh는 git에 올라가지 않음).
#   cp gec2/slurm/paths.example.sh gec2/slurm/paths.local.sh
REPO=$HOME/26-13_PHDQ_BART                          # git clone 위치 (저장소 루트)
VENV=$REPO/.venv-gec2                               # gec2/requirements.txt로 만든 가상환경 (conda면 아래 activate 줄을 바꿈)
DATA_ROOT=/scratch/$USER/phdq_bart/data/Preprocessed   # <data>/<data>_{train,val,test}.txt 와 <data>_test.m2 가 있는 곳
OUT_ROOT=/scratch/$USER/phdq_bart/outputs2          # run 디렉토리가 만들어질 곳 (체크포인트 포함, run당 최대 약 35GB)
MODEL_DIR=/scratch/$USER/phdq_bart/models/kanana-1.5-2.1b-instruct-2505   # 로그인 노드에서 미리 받은 모델
export HF_HOME=/scratch/$USER/phdq_bart/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1      # 계산 노드 인터넷 여부와 무관하게 로컬 모델만 사용
activate_env() { source "$VENV/bin/activate"; }     # 예: module load ... && conda activate gec2
