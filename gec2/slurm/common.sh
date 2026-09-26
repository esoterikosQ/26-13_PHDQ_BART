# sbatch 스크립트 공통: 경로 불러오기, conda 환경 활성화, 데이터·GPU·task 수 확인, 실행 정보 저장.
set -euo pipefail
HERE=${SLURM_SUBMIT_DIR:-$(pwd)}   # 제출 위치 = 저장소 루트여야 함 (--output=logs/slurm/... 도 이 기준)
[ -f "$HERE/gec2/slurm/paths.sh" ] || { echo "오류: 저장소 루트에서 sbatch로 제출할 것 (현재 $HERE)"; exit 1; }
source "$HERE/gec2/slurm/paths.sh"
[ -f "$HERE/gec2/slurm/paths.local.sh" ] && source "$HERE/gec2/slurm/paths.local.sh"
cd "$REPO"
set +u; activate_env || exit 1; set -u   # conda activate 스크립트는 정의되지 않은 변수를 참조함
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
for d in korean_learner native lang8 union; do
  for f in ${d}_train.txt ${d}_val.txt ${d}_test.txt ${d}_test.m2; do
    [ -f "$DATA_ROOT/$d/$f" ] || { echo "오류: 데이터 없음 $DATA_ROOT/$d/$f"; exit 1; }
  done
done
[ -f "$MODEL_DIR/config.json" ] || { echo "오류: 모델 없음 $MODEL_DIR"; exit 1; }
NGPU=$(nvidia-smi -L | wc -l)
NTASK=${SLURM_NTASKS_PER_NODE:-${SLURM_NTASKS:-1}}
echo "[$(TZ=Asia/Seoul date '+%F %T')] job ${SLURM_JOB_ID:-} on $(hostname) partition=${SLURM_JOB_PARTITION:-?} GPU=$NGPU tasks=$NTASK commit=$(git rev-parse --short HEAD) python=$(which python3)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
if [ "$NTASK" != "$NGPU" ]; then
  echo "오류: --ntasks-per-node($NTASK)는 GPU 수($NGPU)와 같아야 함 (Lightning이 srun task마다 GPU 1개 사용)"; exit 1
fi
SRUN=(srun)   # 최근 Slurm은 srun이 sbatch의 --cpus-per-task를 물려받지 않으므로 명시
[ -n "${SLURM_CPUS_PER_TASK:-}" ] && SRUN+=(--cpus-per-task="$SLURM_CPUS_PER_TASK")
save_job_files() {   # $1 = run 디렉토리. 작업 스크립트·slurm 로그·nvidia-smi를 run 디렉토리에 복사
  mkdir -p "$1/slurm"
  scontrol write batch_script "$SLURM_JOB_ID" "$1/slurm/job_${SLURM_JOB_ID}.sbatch" 2>/dev/null || true
  nvidia-smi > "$1/slurm/nvidia-smi_${SLURM_JOB_ID}.txt" 2>&1 || true
  cp "$HERE/logs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out" "$1/slurm/" 2>/dev/null || true
}
