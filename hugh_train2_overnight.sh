#!/bin/bash

#SBATCH --job-name=hugh-train2-overnight
#SBATCH --partition=pscomp,pscompl
#SBATCH --gres=gpu
#SBATCH --time=16:00:00
#SBATCH -o /data101/makinen/juplogs/hugh-train2-overnight.o%j
#SBATCH -e /data101/makinen/juplogs/hugh-train2-overnight.e%j
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module load cuda/11.8
module load intelpython/3-2022.2.1

export XLA_FLAGS=--xla_gpu_cuda_data_dir=${CUDA_PATH}
export PYTHONUNBUFFERED=1

source /home/makinen/venvs/fastjax/bin/activate
cd /home/makinen/repositories/hugh/torch_unet

echo "Started on $(hostname) at $(date)"
echo "Python: $(which python)"
python --version
nvidia-smi || true

CONFIG=config_train2_overnight.json
PCA_PATH=/data101/makinen/hirax_sims/accelerator/train2_overnight/pca_components_nfg11.pt

if [[ -f "${PCA_PATH}" ]]; then
  echo "Using existing PCA components at ${PCA_PATH}"
else
  echo "Precomputing PCA components"
  python precompute_pca.py --config "${CONFIG}" --num-samples 32 --device cpu
fi

echo "Running train2.py"
python train2.py --config "${CONFIG}"

echo "Finished at $(date)"
