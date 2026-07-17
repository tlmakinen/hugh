#!/bin/bash

#SBATCH --job-name=hugh-train2-v5-v100
#SBATCH --partition=pscomp,pscompl
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=23:00:00
#SBATCH -o /data101/makinen/juplogs/hugh-train2-v5-v100.o%j
#SBATCH -e /data101/makinen/juplogs/hugh-train2-v5-v100.e%j
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module load cuda/11.8
module load intelpython/3-2022.2.1

export XLA_FLAGS=--xla_gpu_cuda_data_dir=${CUDA_PATH}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export CUBLAS_WORKSPACE_CONFIG=:4096:8

source /home/makinen/venvs/fastjax/bin/activate
cd /home/makinen/repositories/hugh/torch_unet

echo "Started on $(hostname) at $(date)"
echo "Python: $(which python)"
python --version
nvidia-smi || true

CONFIG=config_train2_v5_v100.json

SOURCE_PCA=/data101/makinen/hirax_sims/accelerator/train2_fixedpca_v2/pca_components_nfg11.pt
DEST_PCA=/data101/makinen/hirax_sims/accelerator/train2_v5_v100/pca_components_nfg11.pt

if [[ -f "${DEST_PCA}" ]]; then
  echo "Using existing PCA components at ${DEST_PCA}"
elif [[ -f "${SOURCE_PCA}" ]]; then
  mkdir -p "$(dirname "${DEST_PCA}")"
  cp "${SOURCE_PCA}" "${DEST_PCA}"
  echo "Copied PCA components from ${SOURCE_PCA} -> ${DEST_PCA}"
fi

echo "Running train2.py with v5 config"
echo "  - UNet3dV2: filters=24, levels=3"
echo "  - conv_kernel_size: (3, 7, 1)  — wider freq receptive field, no baseline locality"
echo "  - strides: [[2,2,1],[2,2,2],[2,2,2]]  — baseline held at full res through first stage"
echo "  - params: ~9.8M (fewer than v4's ~12.2M despite wider freq kernels)"
echo "  - fp16, gradient checkpointing, AdamW + OneCycleLR, EMA"
python train2.py --config "${CONFIG}"

echo "Finished at $(date)"
