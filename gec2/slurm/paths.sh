# Neuron 경로·환경 (ssh.md: 작업 디렉토리 /scratch/r984a02/phdq_bart). 값을 바꾸려면 같은 폴더에 paths.local.sh를
# 만들어 덮어쓴다(git에 올라가지 않음).
BASE=/scratch/r984a02/phdq_bart
REPO=$BASE/26-13_PHDQ_BART                               # git clone 위치 (저장소 루트)
DATA_ROOT=$BASE/data/Preprocessed                        # <data>/<data>_{train,val,test}.txt, <data>_test.m2
[ -d "$DATA_ROOT" ] || DATA_ROOT=$BASE/data              # data/ 바로 아래에 데이터셋 폴더를 둔 경우
OUT_ROOT=$BASE/outputs2                                  # run 디렉토리 (체크포인트 포함, run당 최대 약 35GB)
MODEL_DIR=$BASE/models/kanana-1.5-2.1b-instruct-2505     # 로그인 노드에서 미리 받은 모델
CONDA_ENV=$BASE/conda/gec2                               # conda create -p 로 만든 환경 (홈 용량을 쓰지 않도록 scratch에 둠)
export HF_HOME=$BASE/.cache/huggingface PIP_CACHE_DIR=$BASE/.cache/pip
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1           # 계산 노드 인터넷 여부와 무관하게 로컬 모델만 사용
activate_env() {
  # 로그인 셸에서 conda를 module로 불러온다면 여기에 같은 module load 줄을 넣는다.
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
}
