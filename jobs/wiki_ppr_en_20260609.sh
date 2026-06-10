#!/bin/bash
#SBATCH --job-name=wiki_ppr_en
#SBATCH --partition=rtx3090
#SBATCH --account=erant
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=jobs/logs/%j.out
#SBATCH --error=jobs/logs/%j.err

# enwiki-scale PPR + torch benchmark. Bigger --mem/--time than the generic job
# (pandas load + CSR build of ~hundreds of millions of edges is host-RAM heavy).
# Usage: sbatch jobs/wiki_ppr_en_20260609.sh <path-to-graph.csv.gz>
set -eo pipefail

GRAPH="${1:?usage: sbatch wiki_ppr_en_*.sh <graph.csv.gz>}"
echo "[wiki-en] node=$SLURMD_NODENAME job=$SLURM_JOB_ID graph=$GRAPH $(date)"

cd "$SLURM_SUBMIT_DIR"
mkdir -p jobs/logs

source ~/.bashrc
conda activate pageRank_312
module load cuda/12.5
export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${PYTHONPATH:-}:$SLURM_SUBMIT_DIR/Backend/graph_link

python Tests/wiki_ppr.py --graph "$GRAPH" --num-seeds 4 --topk 20 --benchmark-torch

echo "[wiki-en] done $(date)"
