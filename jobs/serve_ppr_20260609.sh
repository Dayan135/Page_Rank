#!/bin/bash
#SBATCH --job-name=ppr_server
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=jobs/logs/%j.out
#SBATCH --error=jobs/logs/%j.err

# Long-running FastAPI/uvicorn PPR server on a GPU node. Reach it from a laptop with:
#   ssh -N -L 8000:<this-node>:8000 dayanb@slurm.bgu.ac.il
# (the node name is printed below and in squeue). No `set -u` (conda/.bashrc).
set -eo pipefail

echo "[serve] node=$SLURMD_NODENAME job=$SLURM_JOB_ID $(date)"
echo "[serve] TUNNEL TARGET -> $SLURMD_NODENAME:8000"

cd "$SLURM_SUBMIT_DIR"
mkdir -p jobs/logs

source ~/.bashrc
conda activate pageRank_312
module load cuda/12.5
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${PYTHONPATH:-}:$SLURM_SUBMIT_DIR/Backend/graph_link

python -c "import torch; print('[serve] cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

cd Backend/server
echo "[serve] starting uvicorn on 0.0.0.0:8000 ..."
exec uvicorn app:app --host 0.0.0.0 --port 8000
