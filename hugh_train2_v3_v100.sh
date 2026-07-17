#!/bin/bash

#SBATCH --job-name=hugh-train2-v3-v100
#SBATCH --partition=pscomp,pscompl
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=23:00:00
#SBATCH -o /data101/makinen/juplogs/hugh-train2-v3-v100.o%j
#SBATCH -e /data101/makinen/juplogs/hugh-train2-v3-v100.e%j
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module load cuda/11.8
module load intelpython/3-2022.2.1

export XLA_FLAGS=--xla_gpu_cuda_data_dir=${CUDA_PATH}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# Volta-friendly allocator settings (smaller cuBLAS workspace, expandable
# segments to fight fragmentation).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export CUBLAS_WORKSPACE_CONFIG=:4096:8

source /home/makinen/venvs/fastjax/bin/activate
cd /home/makinen/repositories/hugh/torch_unet

echo "Started on $(hostname) at $(date)"
echo "Python: $(which python)"
python --version
nvidia-smi || true

CONFIG=config_train2_v3_v100.json

# Reuse v2's PCA components (same N_FG = 11, same data layout).
SOURCE_PCA=/data101/makinen/hirax_sims/accelerator/train2_fixedpca_v2/pca_components_nfg11.pt
DEST_PCA=/data101/makinen/hirax_sims/accelerator/train2_v3_v100/pca_components_nfg11.pt

if [[ -f "${DEST_PCA}" ]]; then
  echo "Using existing PCA components at ${DEST_PCA}"
elif [[ -f "${SOURCE_PCA}" ]]; then
  mkdir -p "$(dirname "${DEST_PCA}")"
  cp "${SOURCE_PCA}" "${DEST_PCA}"
  echo "Copied PCA components from ${SOURCE_PCA} -> ${DEST_PCA}"
else
  echo "Precomputing PCA components from 100 samples"
  python precompute_pca.py --config "${CONFIG}" --num-samples 100 --device cpu
fi

echo "Running train2.py with V100 v3 config"
echo "  - UNet3dV2 (levels=3, filters=24; narrower than H100 v3 to fit 32 GB)"
echo "  - per-step batch=1, grad_accum=2  (effective batch = 2, same as H100)"
echo "  - fp16 mixed precision (V100 has no bf16)"
echo "  - torch.compile DISABLED, gradient checkpointing ENABLED"
python train2.py --config "${CONFIG}"

echo "Finished at $(date)"
