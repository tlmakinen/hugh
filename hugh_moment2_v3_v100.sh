#!/bin/bash

#SBATCH --job-name=hugh-moment2-v3-v100
#SBATCH --partition=pscomp,pscompl
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=23:00:00
#SBATCH -o /data101/makinen/juplogs/hugh-moment2-v3-v100.o%j
#SBATCH -e /data101/makinen/juplogs/hugh-moment2-v3-v100.e%j
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

CONFIG=config_moment2_v3_v100.json

MEAN_CKPT=/data101/makinen/hirax_sims/accelerator/train2_v3_v100/ema_state_dict.pt
if [[ ! -f "${MEAN_CKPT}" ]]; then
  echo "ERROR: mean model checkpoint not found at ${MEAN_CKPT}"
  exit 1
fi
echo "Mean model checkpoint: ${MEAN_CKPT}"

echo "Running train_moment2_v3.py"
echo "  - frozen first moment: v3 EMA (${MEAN_CKPT})"
echo "  - variance network: fresh UNet3dV2, same architecture (filters=24, levels=3)"
echo "  - target: (model_1(x) - y)^2 in arcsinh residual space"
echo "  - fp16, gradient checkpointing, AdamW + OneCycleLR, EMA decay=0.999"
python train_moment2_v3.py --config "${CONFIG}"

echo "Finished at $(date)"
