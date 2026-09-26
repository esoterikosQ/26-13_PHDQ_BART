# Neuron 경로·환경 (ssh.md: 작업 디렉토리 /scratch/r984a02/phdq_bart). 값을 바꾸려면 같은 폴더에 paths.local.sh를
# 만들어 덮어쓴다(git에 올라가지 않음). Neuron에서는 추적 파일을 직접 고치지 않는다(수정은 GitHub로, plan2 §0).
BASE=/scratch/r984a02/phdq_bart
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)   # 이 파일이 있는 저장소 루트 (clone 위치와 무관)
DATA_ROOT=$BASE/data/Preprocessed                        # <data>/<data>_{train,val,test}.txt, <data>_test.m2
[ -d "$DATA_ROOT" ] || DATA_ROOT=$BASE/data              # data/ 바로 아래에 데이터셋 폴더를 둔 경우
OUT_ROOT=$BASE/outputs2                                  # run 디렉토리 (체크포인트 포함)
MODEL_DIR=$BASE/models/kanana-1.5-2.1b-instruct-2505     # 로그인 노드에서 미리 받은 모델
CONDA_ENV=phdq_bart                                      # conda 환경 이름
export HF_HOME=$BASE/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1           # 계산 노드 인터넷 여부와 무관하게 로컬 모델만 사용

activate_env() {
  # 비대화형 bash(sbatch·bash 스크립트)는 ~/.bashrc의 conda init을 읽지 않으므로 conda.sh를 직접 찾는다.
  if ! command -v conda >/dev/null 2>&1 || [ "$(type -t conda)" != function ]; then
    local sh=""
    [ -n "${CONDA_EXE:-}" ] && sh="$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
    if [ ! -f "$sh" ]; then
      for d in "${CONDA_BASE:-}" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" "$BASE/miniconda3"; do
        [ -n "$d" ] && [ -f "$d/etc/profile.d/conda.sh" ] && { sh="$d/etc/profile.d/conda.sh"; break; }
      done
    fi
    [ -f "$sh" ] || { echo "오류: conda.sh를 찾지 못함. paths.local.sh에 CONDA_BASE=\$(conda info --base) 값을 적을 것"; return 1; }
    source "$sh"
  fi
  conda activate "$CONDA_ENV" || { echo "오류: conda activate $CONDA_ENV 실패"; return 1; }
  python3 -c "import torch, transformers, pytorch_lightning" 2>/dev/null \
    || { echo "오류: $CONDA_ENV 환경에 torch/transformers/pytorch_lightning이 없음 (gec2/requirements.txt)"; return 1; }
}
