#!/bin/bash

#SBATCH --job-name=hugh-train2-fixedpca
#SBATCH --partition=pscomp,pscompl
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --exclude=h13
#SBATCH --time=16:00:00
#SBATCH -o /data101/makinen/juplogs/hugh-train2-fixedpca.o%j
#SBATCH -e /data101/makinen/juplogs/hugh-train2-fixedpca.e%j
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module load cuda/11.8
module load intelpython/3-2022.2.1

export XLA_FLAGS=--xla_gpu_cuda_data_dir=${CUDA_PATH}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

source /home/makinen/venvs/fastjax/bin/activate
cd /home/makinen/repositories/hugh/torch_unet

echo "Started on $(hostname) at $(date)"
echo "Python: $(which python)"
python --version
nvidia-smi || true

CONFIG=config_train2_fixedpca.json
PCA_PATH=/data101/makinen/hirax_sims/accelerator/train2_fixedpca/pca_components_nfg11.pt

# Reuse already-computed PCA components from the earlier run if present.
SHARED_PCA=/data101/makinen/hirax_sims/accelerator/train2_recommended/pca_components_nfg11.pt
if [[ -f "${PCA_PATH}" ]]; then
  echo "Using existing PCA components at ${PCA_PATH}"
elif [[ -f "${SHARED_PCA}" ]]; then
  mkdir -p "$(dirname "${PCA_PATH}")"
  cp "${SHARED_PCA}" "${PCA_PATH}"
  echo "Copied existing PCA components from ${SHARED_PCA} -> ${PCA_PATH}"
else
  echo "Precomputing PCA components from 100 samples"
  python precompute_pca.py --config "${CONFIG}" --num-samples 100 --device cpu
fi

echo "Running train2.py with corrected PCA preprocessing"
python train2.py --config "${CONFIG}"

echo "Finished at $(date)"
