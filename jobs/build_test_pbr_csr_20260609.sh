#!/bin/bash
#SBATCH --job-name=glink_pbr_csr
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=jobs/logs/%j.out
#SBATCH --error=jobs/logs/%j.err

# NOTE: intentionally no `set -u` — sourcing conda/.bashrc trips on unbound vars.
set -eo pipefail

echo "[job] started $(date)  job_id=$SLURM_JOB_ID  node=$SLURMD_NODENAME"

cd "$SLURM_SUBMIT_DIR"
mkdir -p jobs/logs

# --- conda env ---
source ~/.bashrc
conda activate pageRank_312

# --- CUDA toolkit via the cluster module system (no /usr/local/cuda here) ---
# Pick 12.5 to match torch's cu121 build; the node default nvcc (13.x) is too new.
command -v module >/dev/null 2>&1 || source /etc/profile.d/modules.sh 2>/dev/null || true
module unload cuda 2>/dev/null || true
module load cuda/12.5

# --- build toolchain (conda compilers + module CUDA) ---
export CC=$(which x86_64-conda-linux-gnu-cc)
export CXX=$(which x86_64-conda-linux-gnu-c++)
export CUDAHOSTCXX=$CXX
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"   # derive from the loaded module
export PATH=$CUDA_HOME/bin:$PATH

# The compiled .so needs CXXABI_1.3.15 from the conda libstdc++, not the base system one.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# Editable install registers graph_link_core (.so) + the graph_link package via this path.
export PYTHONPATH=${PYTHONPATH:-}:$SLURM_SUBMIT_DIR/Backend/graph_link

echo "[job] CUDA_HOME=$CUDA_HOME"
nvidia-smi
echo "[job] nvcc: $(nvcc --version | tail -1)"

# --- rebuild the CUDA extension (picks up the PBR + CSR-remainder sync) ---
echo "[job] building graph_link extension"
pip install -e Backend/graph_link/ --no-build-isolation

# --- correctness tests ---
echo "[job] running SpMM + PPR correctness tests"
pytest Tests/test_spmm.py Tests/test_ppr_accuracy.py -v

echo "[job] done $(date)"
